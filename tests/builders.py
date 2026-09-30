"""Synthetic mail for tests. No real messages are ever stored in the repo."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

from spamfilter.message import HAM, SPAM, Email
from spamfilter.parse import parse_bytes


def make_raw(
    subject: str = "Hello",
    body: str | None = "Hi there, see you tomorrow.",
    *,
    html: str | None = None,
    sender: str = "Alice <alice@example.com>",
    reply_to: str | None = None,
    to: str = "bob@example.org",
    date: str | None = "Mon, 01 Jan 2024 10:00:00 +0000",
    attachments: tuple[tuple[str, bytes], ...] | list = (),
    headers: dict[str, str] | None = None,
) -> bytes:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    if date is not None:
        msg["Date"] = date
    if reply_to:
        msg["Reply-To"] = reply_to
    for key, value in (headers or {}).items():
        msg[key] = value
    if body is not None:
        msg.set_content(body)
    if html is not None:
        if body is None:
            msg.set_content(html, subtype="html")
        else:
            msg.add_alternative(html, subtype="html")
    for name, data in attachments:
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=name)
    return msg.as_bytes()


SPAM_WORDS = "free winner prize cash offer click discount guaranteed urgent credit loan casino bonus limited deal cheap pills".split()
HAM_WORDS = "meeting project report schedule lunch review draft agenda notes team budget deadline thanks attached minutes quarter".split()
COMMON = "the a and to of for you your this that with on is".split()


def _synthetic_raw(i: int, spam: bool, rng: random.Random, when: datetime) -> bytes:
    vocab = SPAM_WORDS if spam else HAM_WORDS
    words = [rng.choice(vocab if rng.random() < 0.6 else COMMON) for _ in range(40)]
    subject = " ".join(rng.choice(vocab) for _ in range(4)) + f" #{i}"
    sender = f"deals{i}@promo{i % 7}.biz" if spam else f"colleague{i % 11}@example.com"
    body = " ".join(words) + f"\nref {i}"
    html = f"<p>{body}</p><a href='http://promo{i % 5}.biz/x'>click</a>" if spam else None
    return make_raw(subject, body, html=html, sender=sender, date=format_datetime(when))


def synthetic_emails(
    n_spam: int,
    n_ham: int,
    *,
    source: str = "synthetic",
    seed: int = 0,
    start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc),
) -> list[Email]:
    """Easily separable spam/ham with hourly dates, in shuffled order."""
    rng = random.Random(seed)
    flags = [True] * n_spam + [False] * n_ham
    rng.shuffle(flags)
    out = []
    for i, spam in enumerate(flags):
        raw = _synthetic_raw(i + seed * 100_000, spam, rng, start + timedelta(hours=i))
        result = parse_bytes(raw, source=source, folder="Junk" if spam else "Inbox", label=SPAM if spam else HAM)
        assert result.email is not None, result.error
        out.append(result.email)
    return out


def write_mbox(path: Path, messages: list[bytes], *, crlf: bool = False) -> Path:
    """Write messages in mboxo format, escaping body lines that start with 'From '."""
    chunks = []
    for raw in messages:
        lines = raw.replace(b"\r\n", b"\n").split(b"\n")
        lines = [b">" + line if line.startswith(b"From ") else line for line in lines]
        chunks.append(b"From MAILER-DAEMON Mon Jan  1 00:00:00 2024\n" + b"\n".join(lines).rstrip(b"\n") + b"\n\n")
    data = b"".join(chunks)
    if crlf:
        data = data.replace(b"\n", b"\r\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
