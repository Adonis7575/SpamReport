"""Outlook .pst files via libpff-python (pypff). Junk Email is spam."""

from __future__ import annotations

from datetime import timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.parser import HeaderParser
from email.utils import format_datetime
from pathlib import Path
from typing import Callable, Iterator

from .base import FolderRules

PST_RULES = FolderRules(
    spam=("junk email", "junk e-mail", "junk", "spam"),
    skip=("deleted items", "drafts", "outbox", "sent items", "sync issues", "conflicts", "local failures",
          "server failures", "calendar", "contacts", "tasks", "notes", "journal", "rss feeds",
          "conversation history"),
)
_DROP_HEADERS = {"content-type", "content-transfer-encoding", "mime-version"}


def _load_pypff():
    try:
        import pypff
    except ImportError as exc:
        raise ImportError('Outlook .pst support needs libpff-python: pip install "spamfilter[pst]"') from exc
    return pypff


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        for charset in ("utf-8", "cp1252"):
            try:
                return value.decode(charset)
            except UnicodeDecodeError:
                continue
        return value.decode("latin-1")
    return str(value)


def message_to_bytes(message) -> bytes:
    """Rebuild an RFC 822 message from a pypff message's transport headers and bodies."""
    headers = _as_text(message.transport_headers)
    plain = _as_text(message.plain_text_body)
    html = _as_text(message.html_body)
    if plain and html:
        out = MIMEMultipart("alternative")
        out.attach(MIMEText(plain, "plain", "utf-8"))
        out.attach(MIMEText(html, "html", "utf-8"))
    else:
        out = MIMEText(plain or html, "html" if html and not plain else "plain", "utf-8")
    if headers.strip():
        for key, value in HeaderParser().parsestr(headers).items():
            if key.lower() not in _DROP_HEADERS:
                out[key] = value
    else:
        if message.subject:
            out["Subject"] = _as_text(message.subject)
        if message.sender_name:
            out["From"] = _as_text(message.sender_name)
    if "Date" not in out and message.delivery_time is not None:
        when = message.delivery_time
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        out["Date"] = format_datetime(when)
    try:
        return out.as_bytes()
    except (UnicodeEncodeError, ValueError):
        return out.as_string().encode("utf-8", errors="replace")


class PstImporter:
    name = "pst"
    rules = PST_RULES

    def __init__(self, opener: Callable[[Path], object] | None = None) -> None:
        self._opener = opener

    def _root(self, path: Path):
        if self._opener is not None:
            return self._opener(path)
        pypff = _load_pypff()
        pst = pypff.file()
        pst.open(str(path))
        return pst.get_root_folder()

    def iter_raw(self, path: Path) -> Iterator[tuple[str, bytes]]:
        yield from self._walk(self._root(path), "")

    def _walk(self, folder, prefix: str) -> Iterator[tuple[str, bytes]]:
        name = _as_text(folder.name)
        here = f"{prefix}/{name}" if prefix and name else (name or prefix)
        for i in range(folder.number_of_sub_messages):
            try:
                yield here, message_to_bytes(folder.get_sub_message(i))
            except Exception:  # one corrupt item must not stop the import; parse reports it as "empty"
                yield here, b""
        for i in range(folder.number_of_sub_folders):
            yield from self._walk(folder.get_sub_folder(i), here)
