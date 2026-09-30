"""Turn raw RFC 822 bytes into Email records without ever raising on bad mail."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser

from .message import Email, ParseResult

MAX_TEXT = 200_000
KEEP_HEADERS = ("from", "reply-to", "to", "subject", "date", "x-mailer", "user-agent",
                "content-type", "message-id", "return-path", "list-unsubscribe")
URL_RE = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"'()]+""")
# The lookbehind stops the local part from restarting inside a run of word characters,
# which made the unanchored pattern quadratic on long unbroken lines.
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_FALLBACK_CHARSETS = ("utf-8", "cp1252", "latin-1")
_BLOCK_TAGS = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"}


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href.strip())
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, list[str]]:
    parser = _HTMLText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # keep whatever was parsed before the markup broke
        pass
    return "".join(parser.parts), parser.links


def _decode_header(value) -> str:
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value)))).strip()
    except Exception:
        return str(value).strip()


def _decode_part(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        inner = part.get_payload()
        return inner if isinstance(inner, str) else ""
    for charset in (part.get_content_charset(), *_FALLBACK_CHARSETS):
        if not charset:
            continue
        try:
            return payload.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("latin-1", errors="replace")


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if not 1980 <= parsed.year <= 2100:
        return None
    return parsed


def _normalize_for_hash(subject: str, text: str) -> str:
    s = f"{subject}\n{text}".lower()
    s = URL_RE.sub(" u ", s)
    s = _EMAIL_RE.sub(" e ", s)
    s = _NUM_RE.sub("0", s)
    return _WS_RE.sub(" ", s).strip()


def norm_hash(subject: str, text: str, raw_sha256: str) -> str:
    """Hash that ignores numbers, URLs, addresses, case and spacing. Short content falls back to the raw hash."""
    normalized = _normalize_for_hash(subject, text)
    if len(normalized) < 20:  # too little content to call anything a near-duplicate
        return raw_sha256
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_bytes(raw: bytes, *, source: str = "", folder: str = "", label: str | None = None) -> ParseResult:
    if not raw or not raw.strip():
        return ParseResult(None, "empty")
    try:
        msg = BytesParser(policy=policy.compat32).parsebytes(raw)
    except Exception as exc:
        return ParseResult(None, f"unparseable: {type(exc).__name__}")
    if not msg.keys():
        return ParseResult(None, "not-an-email")
    try:
        return ParseResult(_build(msg, raw, source, folder, label))
    except Exception as exc:
        return ParseResult(None, f"extract-failed: {type(exc).__name__}")


def _build(msg, raw: bytes, source: str, folder: str, label: str | None) -> Email:
    plain: list[str] = []
    html: list[str] = []
    n_attachments = 0
    attachment_types: list[str] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        disposition = (part.get("Content-Disposition") or "").lower()
        filename = part.get_filename()
        if "attachment" in disposition or (filename and ctype not in ("text/plain", "text/html")):
            n_attachments += 1
            attachment_types.append(filename.rsplit(".", 1)[-1].lower() if filename and "." in filename else ctype)
            continue
        if ctype == "text/plain":
            plain.append(_decode_part(part))
        elif ctype == "text/html":
            html.append(_decode_part(part))

    html_text, links = html_to_text("\n".join(html)) if html else ("", [])
    plain_text = "\n".join(plain)
    total = len(plain_text) + len(html_text)
    html_share = len(html_text) / total if total else 0.0
    text = (plain_text if plain_text.strip() else html_text)[:MAX_TEXT]
    web_links = [u for u in links if u.lower().startswith(("http://", "https://", "www."))]
    urls = list(dict.fromkeys(web_links + URL_RE.findall(text)))[:200]

    headers = {k: _decode_header(msg.get(k)) for k in KEEP_HEADERS if msg.get(k) is not None}
    subject = headers.get("subject", "")
    raw_sha = hashlib.sha256(raw).hexdigest()
    return Email(
        id=raw_sha[:16],
        raw_sha256=raw_sha,
        norm_body_hash=norm_hash(subject, text, raw_sha),
        source=source,
        folder=folder,
        label=label,
        date=_parse_date(headers.get("date", "")),
        subject=subject,
        from_addr=parseaddr(headers.get("from", ""))[1].lower(),
        reply_to=parseaddr(headers.get("reply-to", ""))[1].lower(),
        to=headers.get("to", ""),
        headers=headers,
        text=text,
        html_share=round(html_share, 4),
        urls=urls,
        n_attachments=n_attachments,
        attachment_types=attachment_types,
    )


def email_from_text(subject: str, body: str, *, source: str = "", folder: str = "", label: str | None = None,
                    date: datetime | None = None, raw_id: bytes | None = None) -> Email:
    """Build an Email from subject + body only (e.g. Enron-Spam, which has no headers)."""
    raw = raw_id if raw_id is not None else f"{subject}\n{body}".encode("utf-8")
    raw_sha = hashlib.sha256(raw).hexdigest()
    body = body[:MAX_TEXT]
    return Email(
        id=raw_sha[:16],
        raw_sha256=raw_sha,
        norm_body_hash=norm_hash(subject, body, raw_sha),
        source=source,
        folder=folder,
        label=label,
        date=date,
        subject=subject,
        text=body,
        urls=list(dict.fromkeys(URL_RE.findall(body)))[:200],
    )
