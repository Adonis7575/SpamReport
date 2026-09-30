"""The Email record every part of spamfilter passes around."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

SPAM = "spam"
HAM = "ham"
SKIP = "skip"


@dataclass
class Email:
    id: str
    raw_sha256: str
    norm_body_hash: str
    source: str = ""
    folder: str = ""
    label: str | None = None
    date: datetime | None = None
    subject: str = ""
    from_addr: str = ""
    reply_to: str = ""
    to: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""
    html_share: float = 0.0
    urls: list[str] = field(default_factory=list)
    n_attachments: int = 0
    attachment_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["date"] = self.date.isoformat() if self.date else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Email:
        data = dict(data)
        data["date"] = datetime.fromisoformat(data["date"]) if data.get("date") else None
        return cls(**data)


@dataclass
class ParseResult:
    email: Email | None
    error: str | None = None
