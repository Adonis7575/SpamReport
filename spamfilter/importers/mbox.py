"""mbox importer: Thunderbird profiles, Apple Mail exports, single files, and Gmail Takeout."""

from __future__ import annotations

import csv
import mailbox
from email.header import decode_header, make_header
from email.parser import BytesHeaderParser
from pathlib import Path
from typing import Iterator

from .base import FolderRules

MBOX_RULES = FolderRules(
    spam=("junk", "spam", "bulk", "bulk mail", "junk e-mail", "junk email"),
    skip=("trash", "deleted", "deleted items", "deleted messages", "drafts", "sent", "sent items",
          "sent mail", "sent messages", "templates", "outbox"),
)
TAKEOUT_RULES = FolderRules(spam=("spam",), skip=("trash", "drafts", "chat", "sent"))

_NOT_MBOX_SUFFIXES = {".msf", ".dat", ".json", ".sqlite", ".html", ".txt", ".plist", ".eml", ".pst"}


def is_mbox_file(p: Path) -> bool:
    if not p.is_file() or p.suffix.lower() in _NOT_MBOX_SUFFIXES or p.stat().st_size == 0:
        return False
    with p.open("rb") as fh:
        return fh.read(5) == b"From "


def _folder_name(file: Path, root: Path) -> str:
    parts = list(file.relative_to(root).parts)
    if len(parts) > 1 and parts[-1] == "mbox":  # Apple Mail: Inbox.mbox/mbox
        parts = parts[:-1]
    cleaned = []
    for part in parts:
        low = part.lower()
        if low.endswith(".sbd"):
            part = part[:-4]
        elif low.endswith(".mbox"):
            part = part[:-5]
        cleaned.append(part)
    return "/".join(cleaned)


def find_mbox_files(path: Path) -> list[tuple[str, Path]]:
    """(folder name, file) pairs for a single mbox file or every mbox file under a directory."""
    if path.is_file():
        root = path.parent.parent if path.name == "mbox" else path.parent
        return [(_folder_name(path, root), path)]
    return [(_folder_name(p, path), p) for p in sorted(path.rglob("*")) if is_mbox_file(p)]


def is_takeout(file: Path) -> bool:
    with file.open("rb") as fh:
        head = fh.read(256 * 1024).lower()
    return b"\nx-gmail-labels:" in head.replace(b"\r\n", b"\n")


def gmail_labels(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        value = str(make_header(decode_header(value)))
    except Exception:
        pass
    flat = value.replace("\r", "").replace("\n", "")
    return [label.strip() for label in next(csv.reader([flat], skipinitialspace=True)) if label.strip()]


def takeout_folder(labels: list[str]) -> str:
    low = {label.lower() for label in labels}
    if "spam" in low:
        return "Spam"
    if "trash" in low:
        return "Trash"
    if "drafts" in low:
        return "Drafts"
    if "chat" in low:
        return "Chat"
    if "sent" in low:  # you wrote it, even if the thread is in the inbox
        return "Sent"
    if "inbox" in low:
        return "Inbox"
    return "Archived"


class MboxImporter:
    def __init__(self, takeout: bool | None = None) -> None:
        self.takeout = takeout
        self.name = "mbox"
        self.rules = MBOX_RULES
        self._configured = False

    def configure(self, path: Path) -> None:
        if self.takeout is None:
            self.takeout = any(is_takeout(f) for _, f in find_mbox_files(path))
        if self.takeout:
            self.name = "takeout"
            self.rules = TAKEOUT_RULES
        self._configured = True

    def iter_raw(self, path: Path) -> Iterator[tuple[str, bytes]]:
        if not self._configured:
            self.configure(path)
        header_parser = BytesHeaderParser()
        for folder, file in find_mbox_files(path):
            box = mailbox.mbox(file, create=False)
            try:
                for key in box.iterkeys():
                    raw = box.get_bytes(key)
                    if self.takeout:
                        labels = gmail_labels(header_parser.parsebytes(raw).get("X-Gmail-Labels"))
                        yield takeout_folder(labels), raw
                    else:
                        yield folder, raw
            finally:
                box.close()
