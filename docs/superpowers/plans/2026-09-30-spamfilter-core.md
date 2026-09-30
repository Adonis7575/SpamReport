# spamfilter Core Engine Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `spamfilter` Python package's core. It parses real email and imports it from four export formats. It prepares the SpamAssassin and Enron-Spam base corpora, and trains and evaluates a calibrated spam model with false-positive-controlled thresholds. All of this is exposed through a `spamfilter` CLI (import / train / evaluate / predict).

**Architecture:** Data moves in one direction:
- Importers and corpus loaders turn raw mail into `Email` records.
- `dataset.py` stores them (gzipped JSONL), removes duplicates, and splits them. Grouped splits are used for the corpora, and time-based splits for personal mail.
- `features.py` + `model.py` build a TF-IDF + structural-signal pipeline with calibrated linear classifiers.
- `evaluate.py` does metrics and cross-validated model selection.
- `training.py` orchestrates everything into a single-file model bundle that the CLI loads.

**Tech Stack:** Python 3.14; numpy, scipy, pandas, scikit-learn 1.9, joblib; optional `libpff-python` for Outlook `.pst`; pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-spamfilter-design.md`. Plan 1 covers spec sections 4, 5.1–5.7, 6 (the rows that apply to the core) and 7 (parse/importers/dataset/features/model). Plans 2–4 cover IMAP (§5.8), the web UI (§5.9), the benchmark notebook (§9) and the full README.

## Global Constraints

- Repo: `E:\CODING\CODING\Git\SpamReport`, branch `spamfilter`. Every git command is `git -c safe.directory=E:/CODING/CODING/Git/SpamReport …`. It's written below as `GIT` for short: in Git Bash, run `GIT="git -c safe.directory=E:/CODING/CODING/Git/SpamReport"` once per shell. **Push after every task's commit** (`$GIT push`).
- Commands run from the repo root in Git Bash. Python is `.venv/Scripts/python`; tests are `.venv/Scripts/python -m pytest`.
- Python `>=3.12` (developed on 3.14.7). Runtime dependencies are limited to numpy, scipy, pandas, scikit-learn, joblib. `libpff-python` is the only optional runtime extra (`[pst]`). Nothing else without updating the spec.
- Local only. The only network access is `spamfilter corpora download` (the pinned URLs below). No telemetry.
- Cache root: `%LOCALAPPDATA%\spamfilter\`, overridable with `SPAMFILTER_HOME`. Subdirs `corpora/`, `datasets/`, `models/`. Settings live in `config.json`.
- Tests never touch the network or the real cache. An autouse fixture points `SPAMFILTER_HOME` at a temp dir, and corpus tests use `file://` URLs.
- Body text is truncated to **200,000** characters. Char n-grams use subject + the first **2,000** body characters. This is a deliberate deviation from the spec's 5,000: 5-fold cross-validation over ~39k emails with 5,000-char heads took too long on a laptop CPU for the "minutes, not hours" requirement. The value is the `CHAR_HEAD` constant.
- Defaults: flag FP target **0.005**, move FP target **0.001**, user-weight grid **{1, 3, 10}** (fallback **3**), **5**-fold CV, test **20%**, validation **15%**, time-split minimum **200** dated messages, threshold-on-personal-ham minimum **200** ham.
- `.gitignore` must block `*.mbox`, `*.pst`, `*.eml`, `*.model`. Test fixtures are built in code; there are no mail files in the repo.
- Labels are the strings `"spam"`, `"ham"`, `"skip"` (constants `SPAM`, `HAM`, `SKIP` in `spamfilter/message.py`).
- The CLI must never crash on a non-ASCII subject or folder name in a cp1252 Windows console.

## Review Focus

1. **Non-ASCII text in a cp1252 Windows console:** printing a Cyrillic folder name or an emoji subject must print `?` rather than raise `UnicodeEncodeError`. Test added in Task 13.
2. **Missing, garbage or impossible `Date:` headers** (`not a date`, month 32, year 1901, no timezone): parse must succeed with `date=None` or UTC, never raise. Tests added in Task 2.
3. **mbox files with CRLF line endings and `>From ` escaped body lines** (Windows-copied Thunderbird folders): messages must split correctly and keep their text. Test added in Task 4.
4. **`X-Gmail-Labels` with quoted commas (`"Clients, 2024"`) and RFC 2047 encoded label names:** labels must be split correctly so spam is never read as ham. Tests added in Task 4.
5. **Re-importing an existing dataset name, or an import that crashes midway:** the old dataset must survive (without `--force`), and no half-written dataset may remain. Tests added in Task 6.

---

## File Structure

```
SpamReport/
├── .gitignore                      # Task 1
├── pyproject.toml                  # Task 1
├── README.md                       # Task 1 (stub), Task 14 (usage + measured results)
├── project/                        # Task 1: deleted (superseded)
├── spamfilter/
│   ├── __init__.py                 # __version__
│   ├── config.py                   # cache paths, settings
│   ├── message.py                  # Email, ParseResult, label constants
│   ├── parse.py                    # raw bytes -> Email; HTML->text; hashing
│   ├── importers/
│   │   ├── __init__.py             # detect_importer, UnsupportedFormat, re-exports
│   │   ├── base.py                 # FolderRules, Overrides, resolve_label, LabelReport, iter_labeled
│   │   ├── mbox.py                 # MboxImporter (Thunderbird, Apple, Gmail Takeout)
│   │   ├── eml.py                  # EmlImporter
│   │   └── pst.py                  # PstImporter
│   ├── dataset.py                  # store (write/read/list/delete), dedupe, splits
│   ├── corpora.py                  # pinned downloads, SpamAssassin/Enron loaders, Spambase loader
│   ├── features.py                 # normalize_text, structural features, make_features
│   ├── model.py                    # make_classifier, LinearTextModel, Reason, Bundle
│   ├── evaluate.py                 # thresholds, metrics, reports, Candidate, select_model, formatting
│   ├── training.py                 # TrainOptions, train()
│   └── cli.py                      # `spamfilter` entry point
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # isolated SPAMFILTER_HOME; synthetic dataset stores
│   ├── builders.py                 # make_raw, synthetic_emails, write_mbox
│   ├── test_config.py  test_parse.py  test_importers_base.py  test_mbox.py
│   ├── test_eml_pst_detect.py  test_dataset_store.py  test_dedupe_split.py
│   ├── test_corpora.py  test_features.py  test_model.py  test_evaluate.py
│   ├── test_training.py  test_cli.py
└── docs/benchmarks/2026-09-30-base-model.md   # Task 14: real measured results
```

---

### Task 1: Package scaffold, environment, config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `spamfilter/__init__.py`, `spamfilter/config.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`
- Modify: `README.md`
- Delete: `project/full_spam_pipeline.py`, `project/spambase.csv`

**Interfaces:**
- Produces: `spamfilter.__version__: str`; `config.home() -> Path`, `config.corpora_dir() -> Path`, `config.datasets_dir() -> Path`, `config.models_dir() -> Path`, `config.load_settings() -> dict`, `config.save_settings(data: dict) -> None`; pytest fixture `spam_home` (autouse; returns the temp home `Path`).

- [ ] **Step 1: Create the repo virtualenv**

```bash
cd /e/CODING/CODING/Git/SpamReport
"/c/Program Files/Python314/python.exe" -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "spamfilter"
version = "0.1.0"
description = "Local, personalizable spam filter trained on public corpora and your own mail exports"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "pandas>=2.2",
    "scikit-learn>=1.6",
    "joblib>=1.4",
]

[project.optional-dependencies]
pst = ["libpff-python>=20231205"]
dev = ["pytest>=8"]

[project.scripts]
spamfilter = "spamfilter.cli:main"

[tool.setuptools.packages.find]
include = ["spamfilter*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
build/
dist/

# Never commit real mail or models trained on it
*.mbox
*.pst
*.eml
*.model
```

- [ ] **Step 4: Write the failing test `tests/test_config.py`** (plus the empty `tests/__init__.py` and `tests/conftest.py`)

`tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def spam_home(tmp_path, monkeypatch):
    """Every test gets its own cache root, so the real %LOCALAPPDATA% cache is never touched."""
    home = tmp_path / "spamfilter-home"
    monkeypatch.setenv("SPAMFILTER_HOME", str(home))
    return home
```

`tests/test_config.py`:

```python
from spamfilter import config


def test_home_uses_env_override(spam_home):
    assert config.home() == spam_home
    assert spam_home.is_dir()


def test_subdirs_are_created(spam_home):
    assert config.corpora_dir() == spam_home / "corpora"
    assert config.datasets_dir().is_dir()
    assert config.models_dir().is_dir()


def test_settings_roundtrip():
    assert config.load_settings() == {}
    config.save_settings({"active_model": "C:/x/y.model"})
    assert config.load_settings() == {"active_model": "C:/x/y.model"}
```

- [ ] **Step 5: Install pytest and run the test to verify it fails**

```bash
.venv/Scripts/python -m pip install "pytest>=8"
.venv/Scripts/python -m pytest tests/test_config.py -v
```
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter'`.

- [ ] **Step 6: Implement `spamfilter/__init__.py` and `spamfilter/config.py`, then install the package**

`spamfilter/__init__.py`:

```python
"""Local, personalizable spam filter."""

__version__ = "0.1.0"
```

`spamfilter/config.py`:

```python
"""Where spamfilter keeps corpora, datasets, models and settings."""

from __future__ import annotations

import json
import os
from pathlib import Path


def home() -> Path:
    env = os.environ.get("SPAMFILTER_HOME")
    if env:
        base = Path(env)
    elif os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "spamfilter"
    else:
        base = Path.home() / ".spamfilter"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _subdir(name: str) -> Path:
    path = home() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def corpora_dir() -> Path:
    return _subdir("corpora")


def datasets_dir() -> Path:
    return _subdir("datasets")


def models_dir() -> Path:
    return _subdir("models")


def _settings_path() -> Path:
    return home() / "config.json"


def load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_settings(data: dict) -> None:
    _settings_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
```

Then install the package in editable mode:

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 8: Remove the superseded prototype and stub the README**

```bash
$GIT rm -q project/full_spam_pipeline.py project/spambase.csv
```

Replace `README.md` with:

```markdown
# SpamReport

- `FinalReport/`: the original NMSU Data Mining class project (Spambase study), kept as an archive.
- `spamfilter/`: a local, personalizable spam filter that trains on public raw-email corpora and your own mail exports. Work in progress on the `spamfilter` branch; see `docs/superpowers/specs/2026-09-30-spamfilter-design.md`.
```

- [ ] **Step 9: Commit and push**

```bash
$GIT add .gitignore pyproject.toml README.md spamfilter tests
$GIT commit -m "Scaffold spamfilter package, config and test harness; retire prototype script"
$GIT push
```

---

### Task 2: Email record and parsing

**Files:**
- Create: `spamfilter/message.py`, `spamfilter/parse.py`, `tests/builders.py`, `tests/test_parse.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `message.SPAM = "spam"`, `message.HAM = "ham"`, `message.SKIP = "skip"`
  - `message.Email` (dataclass; fields `id, raw_sha256, norm_body_hash, source, folder, label, date, subject, from_addr, reply_to, to, headers, text, html_share, urls, n_attachments, attachment_types`; methods `to_dict() -> dict`, `Email.from_dict(d) -> Email`)
  - `message.ParseResult(email: Email | None, error: str | None = None)`
  - `parse.MAX_TEXT = 200_000`, `parse.URL_RE` (compiled regex)
  - `parse.parse_bytes(raw: bytes, *, source: str = "", folder: str = "", label: str | None = None) -> ParseResult`. Error reasons: `"empty"`, `"not-an-email"`, `"unparseable: <Exc>"`, `"extract-failed: <Exc>"`
  - `parse.email_from_text(subject: str, body: str, *, source="", folder="", label=None, date=None, raw_id: bytes | None = None) -> Email`
  - `parse.html_to_text(html: str) -> tuple[str, list[str]]`
  - `parse.norm_hash(subject: str, text: str, raw_sha256: str) -> str`
  - Test helpers in `tests/builders.py`: `make_raw(...) -> bytes`, `synthetic_emails(n_spam, n_ham, *, source="synthetic", seed=0, start=...) -> list[Email]`, `write_mbox(path, messages, *, crlf=False) -> Path`

- [ ] **Step 1: Write the test builders `tests/builders.py`**

```python
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
```

- [ ] **Step 2: Write the failing tests `tests/test_parse.py`**

```python
import json

import pytest

from spamfilter.message import Email
from spamfilter.parse import email_from_text, norm_hash, parse_bytes
from tests.builders import make_raw


def test_plain_message_fields():
    r = parse_bytes(make_raw("Lunch?", "Want to get lunch at noon?", reply_to="x@other.org"),
                    source="s", folder="Inbox", label="ham")
    e = r.email
    assert r.error is None
    assert e.subject == "Lunch?" and "lunch at noon" in e.text
    assert e.from_addr == "alice@example.com" and e.reply_to == "x@other.org"
    assert e.date.tzinfo is not None and e.date.year == 2024
    assert (e.source, e.folder, e.label) == ("s", "Inbox", "ham")
    assert len(e.id) == 16 and e.id == e.raw_sha256[:16]


def test_html_only_message_is_converted_to_text():
    raw = make_raw("Sale", None, html="<html><style>p{color:red}</style><p>Big <b>SALE</b> today</p>"
                                      "<a href='http://shop.example.biz/x'>go</a></html>")
    e = parse_bytes(raw).email
    assert "Big SALE today" in e.text and "color:red" not in e.text
    assert e.html_share == 1.0
    assert "http://shop.example.biz/x" in e.urls


def test_multipart_alternative_prefers_plain():
    e = parse_bytes(make_raw("Hi", "plain version", html="<p>html version</p>")).email
    assert "plain version" in e.text and "html version" not in e.text
    assert 0 < e.html_share < 1


def test_attachments_are_counted_not_read():
    e = parse_bytes(make_raw("Invoice", "see attached", attachments=[("invoice.exe", b"MZ\x00\x01")])).email
    assert e.n_attachments == 1 and e.attachment_types == ["exe"]
    assert "MZ" not in e.text


def test_encoded_subject_and_unknown_charset():
    raw = (b"From: a@example.com\nSubject: =?utf-8?b?w4ljaGFudGlsbG9u?=\n"
           b"Content-Type: text/plain; charset=x-unknown-charset\n\ncaf\xe9 menu\n")
    e = parse_bytes(raw).email
    assert e.subject == "Échantillon"
    assert "café menu" in e.text


def test_quoted_printable_and_base64_bodies():
    qp = (b"From: a@example.com\nSubject: q\nContent-Type: text/plain; charset=utf-8\n"
          b"Content-Transfer-Encoding: quoted-printable\n\nhello =E2=82=AC world\n")
    assert "hello € world" in parse_bytes(qp).email.text
    b64 = (b"From: a@example.com\nSubject: b\nContent-Type: text/plain\n"
           b"Content-Transfer-Encoding: base64\n\naGVsbG8gYmFzZTY0\n")
    assert "hello base64" in parse_bytes(b64).email.text


@pytest.mark.parametrize("raw, reason", [
    (b"", "empty"),
    (b"   \n\n", "empty"),
    (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj", "not-an-email"),
])
def test_rejects_non_mail(raw, reason):
    r = parse_bytes(raw)
    assert r.email is None and r.error == reason


# Review Focus 2: garbage or impossible dates must never break parsing.
@pytest.mark.parametrize("date_value", ["not a date", "Mon, 32 Foo 2024 99:99:99 +0000",
                                        "Tue, 01 Jan 1901 00:00:00 +0000"])
def test_bad_dates_become_none(date_value):
    raw = f"From: a@example.com\nSubject: x\nDate: {date_value}\n\nbody text here\n".encode()
    r = parse_bytes(raw)
    assert r.error is None and r.email.date is None


def test_date_without_timezone_is_utc():
    raw = b"From: a@example.com\nSubject: x\nDate: 01 Jan 2024 10:00:00\n\nbody\n"
    assert parse_bytes(raw).email.date.utcoffset().total_seconds() == 0


def test_norm_hash_ignores_numbers_urls_and_spacing():
    a = norm_hash("Order 123", "Your code is 4411 see http://a.example/x  now", "r1")
    b = norm_hash("Order 999", "Your  code is 9876 see http://b.example/y now", "r2")
    assert a == b and a not in ("r1", "r2")


def test_norm_hash_short_content_falls_back_to_raw_hash():
    assert norm_hash("", "hi", "rawhash") == "rawhash"


def test_huge_bodies_are_truncated():
    e = parse_bytes(make_raw("big", "x" * 300_000)).email
    assert len(e.text) == 200_000


def test_email_from_text():
    e = email_from_text("cheap meds", "buy now", source="base-enron", folder="spam", label="spam")
    assert (e.subject, e.text, e.from_addr, e.label) == ("cheap meds", "buy now", "", "spam")
    assert e.headers == {}


def test_email_dict_roundtrip():
    e = parse_bytes(make_raw()).email
    assert Email.from_dict(json.loads(json.dumps(e.to_dict()))) == e
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.message'`.

- [ ] **Step 4: Implement `spamfilter/message.py`**

```python
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
```

- [ ] **Step 5: Implement `spamfilter/parse.py`**

```python
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
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_parse.py -v`
Expected: 18 passed.

- [ ] **Step 7: Commit and push**

```bash
$GIT add spamfilter/message.py spamfilter/parse.py tests/builders.py tests/test_parse.py
$GIT commit -m "Add Email record and robust RFC 822 parsing"
$GIT push
```

---

### Task 3: Importer base: folder rules, overrides, label report

**Files:**
- Create: `spamfilter/importers/__init__.py` (empty for now; filled in Task 5), `spamfilter/importers/base.py`, `tests/test_importers_base.py`

**Interfaces:**
- Consumes: `parse.parse_bytes`, `message.SPAM/HAM/SKIP`, `message.ParseResult`.
- Produces:
  - `base.SENT_NAMES: tuple[str, ...]`
  - `base.norm_folder(name: str) -> str`: last path component, `.mbox`/`.sbd` suffix removed, lower-cased
  - `base.FolderRules(spam: tuple[str, ...], skip: tuple[str, ...], ham: tuple[str, ...] = (), default: str = HAM)` (frozen)
  - `base.Overrides(spam: list[str], ham: list[str], skip: list[str], include_sent: bool = False, label: str | None = None)`
  - `base.resolve_label(folder: str, rules: FolderRules, ov: Overrides) -> str`
  - `base.Importer` Protocol: attributes `name: str`, `rules: FolderRules`; method `iter_raw(path: Path) -> Iterator[tuple[str, bytes]]`
  - `base.LabelReport(rows: list[tuple[str, int, str]])` with `totals() -> dict[str, int]` and `format() -> str`
  - `base.build_label_report(importer, path: Path, ov: Overrides) -> LabelReport`
  - `base.iter_labeled(importer, path: Path, ov: Overrides, *, source: str) -> Iterator[ParseResult]`: skipped folders are not yielded

- [ ] **Step 1: Write the failing tests `tests/test_importers_base.py`**

```python
from pathlib import Path

import pytest

from spamfilter.importers.base import (FolderRules, Overrides, build_label_report, iter_labeled,
                                       norm_folder, resolve_label)
from tests.builders import make_raw

RULES = FolderRules(spam=("junk", "spam"), skip=("trash", "sent", "sent items"))


@pytest.mark.parametrize("folder, expected", [
    ("Inbox", "ham"), ("Junk", "spam"), ("SPAM", "spam"), ("Mail/Junk.mbox", "spam"),
    ("Local Folders/Inbox.sbd/Work", "ham"), ("Trash", "skip"), ("Sent", "skip"),
])
def test_default_rules(folder, expected):
    assert resolve_label(folder, RULES, Overrides()) == expected


def test_overrides_take_priority():
    ov = Overrides(spam=["Promotions"], ham=["Junk"], skip=["inbox"])
    assert resolve_label("Promotions", RULES, ov) == "spam"
    assert resolve_label("Junk", RULES, ov) == "ham"
    assert resolve_label("Inbox", RULES, ov) == "skip"


def test_include_sent_and_forced_label():
    assert resolve_label("Sent Items", RULES, Overrides(include_sent=True)) == "ham"
    assert resolve_label("Trash", RULES, Overrides(label="spam")) == "spam"


def test_norm_folder():
    assert norm_folder("Mail\\Junk.mbox") == "junk"
    assert norm_folder("Inbox.sbd") == "inbox"
    assert norm_folder("  Sent   Items ") == "sent items"


class FakeImporter:
    name = "fake"
    rules = RULES

    def __init__(self, items):
        self.items = items

    def iter_raw(self, path):
        yield from self.items


def _items():
    return [
        ("Inbox", make_raw("a", "hello there friend")),
        ("Inbox", make_raw("b", "another normal note")),
        ("Junk", make_raw("c", "win cash now")),
        ("Trash", make_raw("d", "old")),
        ("Inbox", b""),
    ]


def test_label_report():
    rep = build_label_report(FakeImporter(_items()), Path("x"), Overrides())
    assert rep.rows == [("Inbox", 3, "ham"), ("Junk", 1, "spam"), ("Trash", 1, "skip")]
    assert rep.totals() == {"ham": 3, "spam": 1, "skip": 1}
    text = rep.format()
    assert "Inbox" in text and "Total:" in text


def test_iter_labeled_skips_and_reports_errors():
    results = list(iter_labeled(FakeImporter(_items()), Path("x"), Overrides(), source="mine"))
    assert [r.email.label for r in results if r.email] == ["ham", "ham", "spam"]
    assert [r.error for r in results if r.email is None] == ["empty"]
    assert all(r.email.source == "mine" for r in results if r.email)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_importers_base.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.importers'`.

- [ ] **Step 3: Implement `spamfilter/importers/__init__.py` (empty) and `spamfilter/importers/base.py`**

`spamfilter/importers/__init__.py`: empty file for now (Task 5 fills it).

`spamfilter/importers/base.py`:

```python
"""Shared importer pieces: folder -> label rules, user overrides, the label report."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

from ..message import HAM, SKIP, SPAM, ParseResult
from ..parse import parse_bytes

SENT_NAMES = ("sent", "sent items", "sent mail", "sent messages")


def norm_folder(name: str) -> str:
    stripped = name.strip()
    last = re.split(r"[\\/]", stripped)[-1] if stripped else ""
    last = re.sub(r"\.(mbox|sbd)$", "", last, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", last).strip().lower()


@dataclass(frozen=True)
class FolderRules:
    spam: tuple[str, ...]
    skip: tuple[str, ...]
    ham: tuple[str, ...] = ()
    default: str = HAM


@dataclass
class Overrides:
    spam: list[str] = field(default_factory=list)
    ham: list[str] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)
    include_sent: bool = False
    label: str | None = None  # force one label for every message


def resolve_label(folder: str, rules: FolderRules, ov: Overrides) -> str:
    if ov.label:
        return ov.label
    full = folder.strip().lower()
    last = norm_folder(folder)

    def matches(names: list[str]) -> bool:
        return any(n.strip().lower() in (full, last) for n in names)

    if matches(ov.spam):
        return SPAM
    if matches(ov.ham):
        return HAM
    if matches(ov.skip):
        return SKIP
    if ov.include_sent and last in SENT_NAMES:
        return HAM
    if last in rules.spam:
        return SPAM
    if last in rules.skip:
        return SKIP
    if last in rules.ham:
        return HAM
    return rules.default


class Importer(Protocol):
    name: str
    rules: FolderRules

    def iter_raw(self, path: Path) -> Iterator[tuple[str, bytes]]: ...


@dataclass
class LabelReport:
    rows: list[tuple[str, int, str]]  # (folder, message count, label)

    def totals(self) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for _folder, count, label in self.rows:
            totals[label] += count
        return dict(totals)

    def format(self) -> str:
        width = max([len(folder) for folder, _, _ in self.rows] + [6])
        lines = [f"{'Folder':<{width}}  {'Count':>7}  Label"]
        lines += [f"{folder:<{width}}  {count:>7}  {label}" for folder, count, label in self.rows]
        lines.append("Total: " + ", ".join(f"{k} {v}" for k, v in sorted(self.totals().items())))
        return "\n".join(lines)


def build_label_report(importer: Importer, path: Path, ov: Overrides) -> LabelReport:
    counts: Counter[str] = Counter()
    for folder, _raw in importer.iter_raw(path):
        counts[folder] += 1
    return LabelReport([(f, n, resolve_label(f, importer.rules, ov)) for f, n in sorted(counts.items())])


def iter_labeled(importer: Importer, path: Path, ov: Overrides, *, source: str) -> Iterator[ParseResult]:
    for folder, raw in importer.iter_raw(path):
        label = resolve_label(folder, importer.rules, ov)
        if label == SKIP:
            continue
        yield parse_bytes(raw, source=source, folder=folder, label=label)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_importers_base.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/importers tests/test_importers_base.py
$GIT commit -m "Add importer base: folder label rules, overrides and label report"
$GIT push
```

---

### Task 4: mbox importer (Thunderbird, Apple Mail, Gmail Takeout)

**Files:**
- Create: `spamfilter/importers/mbox.py`, `tests/test_mbox.py`

**Interfaces:**
- Consumes: `base.FolderRules`, `base.Overrides`, `base.build_label_report`, `base.iter_labeled`.
- Produces:
  - `mbox.MBOX_RULES`, `mbox.TAKEOUT_RULES` (FolderRules)
  - `mbox.is_mbox_file(p: Path) -> bool`
  - `mbox.find_mbox_files(path: Path) -> list[tuple[str, Path]]`
  - `mbox.is_takeout(file: Path) -> bool`
  - `mbox.gmail_labels(value: str | None) -> list[str]`
  - `mbox.takeout_folder(labels: list[str]) -> str`: one of `Spam, Trash, Drafts, Chat, Sent, Inbox, Archived`
  - `mbox.MboxImporter(takeout: bool | None = None)` with `.name` (`"mbox"` or `"takeout"`), `.rules`, `.configure(path: Path) -> None`, `.iter_raw(path: Path)`

- [ ] **Step 1: Write the failing tests `tests/test_mbox.py`**

```python
from spamfilter.importers.base import Overrides, build_label_report, iter_labeled
from spamfilter.importers.mbox import MboxImporter, gmail_labels
from tests.builders import make_raw, write_mbox


def test_thunderbird_profile(tmp_path):
    root = tmp_path / "profile"
    write_mbox(root / "Inbox", [make_raw("hi", "normal message one"), make_raw("hey", "normal message two")])
    write_mbox(root / "Junk", [make_raw("WIN", "cash prize now")])
    (root / "Inbox.msf").write_text("index data")
    write_mbox(root / "Inbox.sbd" / "Work", [make_raw("proj", "project notes")])
    write_mbox(root / "Trash", [make_raw("old", "old stuff")])
    imp = MboxImporter()
    imp.configure(root)
    assert imp.name == "mbox"
    rep = build_label_report(imp, root, Overrides())
    assert {f: (n, lab) for f, n, lab in rep.rows} == {
        "Inbox": (2, "ham"), "Junk": (1, "spam"), "Inbox/Work": (1, "ham"), "Trash": (1, "skip")}


def test_apple_mail_export(tmp_path):
    export = tmp_path / "export"
    write_mbox(export / "Junk.mbox" / "mbox", [make_raw("x", "spam text here")])
    write_mbox(export / "INBOX.mbox" / "mbox", [make_raw("y", "ham text here")])
    (export / "INBOX.mbox" / "table_of_contents").write_bytes(b"\x00\x01binary")
    rep = build_label_report(MboxImporter(), export, Overrides())
    assert {(f, lab) for f, _, lab in rep.rows} == {("Junk", "spam"), ("INBOX", "ham")}


def test_single_apple_mbox_file_uses_parent_name(tmp_path):
    f = write_mbox(tmp_path / "Junk.mbox" / "mbox", [make_raw("x", "spam text here")])
    rep = build_label_report(MboxImporter(), f, Overrides())
    assert rep.rows == [("Junk", 1, "spam")]


def _gm(labels, subject):
    return make_raw(subject, f"body for {subject} message", headers={"X-Gmail-Labels": labels})


def test_gmail_takeout_labels(tmp_path):
    f = write_mbox(tmp_path / "All mail Including Spam and Trash.mbox", [
        _gm("Inbox,Category Updates", "a"), _gm("Spam", "b"), _gm("Spam,Trash", "c"), _gm("Trash", "d"),
        _gm("Sent,Inbox", "e"), _gm("Archived,Important", "f"), _gm('Inbox,"Clients, 2024"', "g"),
        _gm("Chat", "h"),
    ])
    imp = MboxImporter()
    imp.configure(f)
    assert imp.name == "takeout"
    rep = build_label_report(imp, f, Overrides())
    assert {f_: (n, lab) for f_, n, lab in rep.rows} == {
        "Inbox": (2, "ham"), "Spam": (2, "spam"), "Trash": (1, "skip"), "Sent": (1, "skip"),
        "Archived": (1, "ham"), "Chat": (1, "skip")}


# Review Focus 4: quoted commas and encoded label names.
def test_gmail_labels_parsing():
    assert gmail_labels('Inbox,"Clients, 2024",Opened') == ["Inbox", "Clients, 2024", "Opened"]
    assert gmail_labels("=?UTF-8?Q?Caf=C3=A9?=,Inbox") == ["Café", "Inbox"]
    assert gmail_labels(None) == []
    assert gmail_labels("") == []


# Review Focus 3: CRLF files and '>From ' escaped lines.
def test_crlf_mbox_and_from_escaping(tmp_path):
    msgs = [make_raw("one", "line\nFrom here we escape\nend"), make_raw("two", "second body")]
    f = write_mbox(tmp_path / "Inbox", msgs, crlf=True)
    res = list(iter_labeled(MboxImporter(), f, Overrides(), source="t"))
    assert [r.email.subject for r in res] == ["one", "two"]
    assert "we escape" in res[0].email.text
    assert all(r.email.label == "ham" for r in res)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_mbox.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.importers.mbox'`.

- [ ] **Step 3: Implement `spamfilter/importers/mbox.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_mbox.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/importers/mbox.py tests/test_mbox.py
$GIT commit -m "Add mbox importer for Thunderbird, Apple Mail and Gmail Takeout"
$GIT push
```

---

### Task 5: .eml and .pst importers, format detection

**Files:**
- Create: `spamfilter/importers/eml.py`, `spamfilter/importers/pst.py`, `tests/test_eml_pst_detect.py`
- Modify: `spamfilter/importers/__init__.py`

**Interfaces:**
- Consumes: `base.FolderRules`, `mbox.MboxImporter`, `mbox.is_mbox_file`, `mbox.find_mbox_files`.
- Produces:
  - `eml.EmlImporter(pattern: str = "*.eml")`: `name = "eml"`. Folder is the first directory under the given path, or `"."`. Defaults: `spam/`, `junk/` → spam; `ham/`, `inbox/`, `not spam/`, `legit/` → ham; anything else → skip.
  - `pst.PstImporter(opener: Callable[[Path], folder] | None = None)`: `name = "pst"`. `opener` is a test hook returning a pypff-like root folder.
  - `pst.message_to_bytes(message) -> bytes`
  - `importers.UnsupportedFormat(Exception)`
  - `importers.detect_importer(path: Path, fmt: str = "auto")`, where `fmt` is one of `auto, takeout, mbox, eml, pst`. Raises `FileNotFoundError` or `UnsupportedFormat`.

- [ ] **Step 1: Write the failing tests `tests/test_eml_pst_detect.py`**

```python
import sys
from datetime import datetime
from pathlib import Path

import pytest

from spamfilter.importers import UnsupportedFormat, detect_importer
from spamfilter.importers.base import Overrides, build_label_report, iter_labeled
from spamfilter.importers.eml import EmlImporter
from spamfilter.importers.mbox import MboxImporter
from spamfilter.importers.pst import PstImporter
from tests.builders import make_raw, write_mbox


def test_eml_tree(tmp_path):
    (tmp_path / "spam" / "2024").mkdir(parents=True)
    (tmp_path / "ham").mkdir()
    (tmp_path / "spam" / "2024" / "a.eml").write_bytes(make_raw("win", "cash cash cash"))
    (tmp_path / "ham" / "b.eml").write_bytes(make_raw("hi", "see you soon"))
    (tmp_path / "c.eml").write_bytes(make_raw("loose", "unsorted message"))
    imp = detect_importer(tmp_path)
    assert isinstance(imp, EmlImporter)
    rep = build_label_report(imp, tmp_path, Overrides())
    assert set(rep.rows) == {("spam", 1, "spam"), ("ham", 1, "ham"), (".", 1, "skip")}
    assert build_label_report(imp, tmp_path / "ham", Overrides(label="ham")).rows == [(".", 1, "ham")]


class FakeMsg:
    def __init__(self, subject, body, headers=None, html=None):
        self.subject = subject
        self.plain_text_body = body.encode()
        self.html_body = html.encode() if html else None
        self.transport_headers = headers
        self.sender_name = "Sender"
        self.delivery_time = datetime(2024, 5, 1, 12, 0)


class BrokenMsg:
    @property
    def transport_headers(self):
        raise OSError("corrupt item")


class FakeFolder:
    def __init__(self, name, messages=(), folders=()):
        self.name = name
        self._messages = list(messages)
        self._folders = list(folders)

    @property
    def number_of_sub_messages(self):
        return len(self._messages)

    def get_sub_message(self, i):
        return self._messages[i]

    @property
    def number_of_sub_folders(self):
        return len(self._folders)

    def get_sub_folder(self, i):
        return self._folders[i]


def _fake_pst():
    headers = ("From: Boss <boss@corp.example>\r\nSubject: Q3 numbers\r\n"
               "Date: Wed, 01 May 2024 09:00:00 +0000\r\nContent-Type: text/plain\r\n\r\n")
    return FakeFolder("", folders=[FakeFolder("Top of Personal Folders", folders=[
        FakeFolder("Inbox", [FakeMsg("Q3 numbers", "numbers attached", headers=headers)]),
        FakeFolder("Junk Email", [FakeMsg("You won", "claim prize", html="<p>claim <b>prize</b></p>")]),
        FakeFolder("Deleted Items", [FakeMsg("old", "old")]),
        FakeFolder("Broken", [BrokenMsg()]),
    ])])


def test_pst_walk_labels_and_reconstruction():
    root = _fake_pst()
    imp = PstImporter(opener=lambda p: root)
    rep = build_label_report(imp, Path("x.pst"), Overrides())
    assert {(f.split("/")[-1], lab) for f, _, lab in rep.rows} == {
        ("Inbox", "ham"), ("Junk Email", "spam"), ("Deleted Items", "skip"), ("Broken", "ham")}
    results = list(iter_labeled(imp, Path("x.pst"), Overrides(), source="pst"))
    good = [r.email for r in results if r.email]
    inbox = next(e for e in good if e.folder.endswith("Inbox"))
    assert inbox.folder == "Top of Personal Folders/Inbox"
    assert inbox.from_addr == "boss@corp.example" and inbox.subject == "Q3 numbers"
    assert "numbers attached" in inbox.text
    junk = next(e for e in good if e.folder.endswith("Junk Email"))
    assert junk.subject == "You won" and "claim prize" in junk.text and junk.date.year == 2024
    assert [r.error for r in results if r.email is None] == ["empty"]


def test_pst_without_library_explains_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypff", None)
    with pytest.raises(ImportError, match="pip install"):
        list(PstImporter().iter_raw(Path("x.pst")))


def test_detect_formats(tmp_path):
    write_mbox(tmp_path / "box" / "Inbox", [make_raw()])
    assert isinstance(detect_importer(tmp_path / "box"), MboxImporter)
    pst = tmp_path / "a.pst"
    pst.write_bytes(b"!BDN")
    assert isinstance(detect_importer(pst), PstImporter)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UnsupportedFormat):
        detect_importer(empty)
    with pytest.raises(FileNotFoundError):
        detect_importer(tmp_path / "missing")
    with pytest.raises(UnsupportedFormat):
        detect_importer(tmp_path / "box", fmt="zip")
    takeout = detect_importer(tmp_path / "box", fmt="takeout")
    assert takeout.name == "takeout"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_eml_pst_detect.py -v`
Expected: collection error `ImportError: cannot import name 'UnsupportedFormat' from 'spamfilter.importers'`.

- [ ] **Step 3: Implement `spamfilter/importers/eml.py`**

```python
"""Folders of .eml files: spam/ and ham/ subfolders give the label."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..message import SKIP
from .base import FolderRules

EML_RULES = FolderRules(spam=("spam", "junk"), skip=(), ham=("ham", "inbox", "not spam", "legit"), default=SKIP)


class EmlImporter:
    name = "eml"
    rules = EML_RULES

    def __init__(self, pattern: str = "*.eml") -> None:
        self.pattern = pattern

    def iter_raw(self, path: Path) -> Iterator[tuple[str, bytes]]:
        root = path if path.is_dir() else path.parent
        files = [path] if path.is_file() else sorted(p for p in root.rglob(self.pattern) if p.is_file())
        for file in files:
            rel = file.parent.relative_to(root)
            folder = rel.parts[0] if rel.parts else "."
            yield folder, file.read_bytes()
```

- [ ] **Step 4: Implement `spamfilter/importers/pst.py`**

```python
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
```

- [ ] **Step 5: Implement `spamfilter/importers/__init__.py`**

```python
"""Pick the right importer for a path."""

from __future__ import annotations

from pathlib import Path

from .eml import EmlImporter
from .mbox import MboxImporter, find_mbox_files, is_mbox_file
from .pst import PstImporter

FORMATS = ("auto", "takeout", "mbox", "eml", "pst")


class UnsupportedFormat(Exception):
    pass


def detect_importer(path: Path, fmt: str = "auto"):
    path = Path(path)
    if fmt not in FORMATS:
        raise UnsupportedFormat(f"unknown format {fmt!r}; use one of: {', '.join(FORMATS)}")
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if fmt == "pst" or (fmt == "auto" and path.is_file() and path.suffix.lower() == ".pst"):
        return PstImporter()
    if fmt == "eml":
        return EmlImporter()
    if fmt in ("mbox", "takeout"):
        importer = MboxImporter(takeout=(fmt == "takeout"))
        importer.configure(path)
        return importer
    if path.is_file():
        if path.suffix.lower() == ".eml":
            return EmlImporter()
        if is_mbox_file(path):
            importer = MboxImporter()
            importer.configure(path)
            return importer
    elif any(path.rglob("*.eml")):
        return EmlImporter()
    elif find_mbox_files(path):
        importer = MboxImporter()
        importer.configure(path)
        return importer
    raise UnsupportedFormat(
        f"no mail found in {path}. Supported: Gmail Takeout .mbox, Thunderbird/Apple mbox folders, "
        "folders of .eml files, Outlook .pst")
```

Note: `detect_importer(path, fmt="mbox")` sets `takeout=False` explicitly. That lets a user force plain folder labeling on a Takeout file.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_eml_pst_detect.py tests/test_mbox.py tests/test_importers_base.py -v`
Expected: all passed.

- [ ] **Step 7: Install the pst extra and smoke-test the real library import**

```bash
.venv/Scripts/python -m pip install -e ".[pst,dev]"
.venv/Scripts/python -c "import pypff; print(pypff.get_version())"
```
Expected: a version string such as `20260926`.

- [ ] **Step 8: Commit and push**

```bash
$GIT add spamfilter/importers tests/test_eml_pst_detect.py
$GIT commit -m "Add .eml and Outlook .pst importers and format detection"
$GIT push
```

---

### Task 6: Dataset store

**Files:**
- Create: `spamfilter/dataset.py` (store half), `tests/test_dataset_store.py`

**Interfaces:**
- Consumes: `config.datasets_dir()`, `message.Email`, `message.ParseResult`, `spamfilter.__version__`.
- Produces:
  - Exceptions: `DatasetExists(Exception)`, `DatasetNotFound(Exception)`, `InvalidDatasetName(ValueError)`
  - `dataset_path(name: str) -> Path`
  - `dataset_exists(name: str) -> bool`
  - `write_dataset(name: str, results: Iterable[ParseResult], *, manifest: dict, force: bool = False, progress: Callable[[int], None] | None = None) -> dict`: returns the manifest. The manifest has keys `name, created, count, labels, folders, skipped, version` plus the caller's keys.
  - `read_dataset(name: str) -> Iterator[Email]`
  - `load_manifest(name: str) -> dict`
  - `list_datasets() -> list[dict]`: manifests sorted by name
  - `delete_dataset(name: str) -> None`

- [ ] **Step 1: Write the failing tests `tests/test_dataset_store.py`**

```python
import pytest

from spamfilter.dataset import (DatasetExists, DatasetNotFound, InvalidDatasetName, dataset_exists,
                                delete_dataset, list_datasets, load_manifest, read_dataset, write_dataset)
from spamfilter.message import ParseResult
from tests.builders import synthetic_emails


def test_write_read_roundtrip():
    emails = synthetic_emails(3, 4)
    results = [ParseResult(e) for e in emails] + [ParseResult(None, "empty"), ParseResult(None, "not-an-email"),
                                                  ParseResult(None, "empty")]
    seen = []
    man = write_dataset("mine", results, manifest={"importer": "test"}, progress=seen.append)
    assert man["count"] == 7
    assert man["labels"] == {"spam": 3, "ham": 4}
    assert man["folders"] == {"Junk": {"spam": 3}, "Inbox": {"ham": 4}}
    assert man["skipped"] == {"empty": 2, "not-an-email": 1}
    assert list(read_dataset("mine")) == emails
    assert load_manifest("mine")["importer"] == "test"
    assert [m["name"] for m in list_datasets()] == ["mine"]


# Review Focus 5: an existing dataset survives unless --force; a crashed import leaves nothing behind.
def test_existing_name_requires_force():
    write_dataset("mine", [], manifest={})
    with pytest.raises(DatasetExists, match="--force"):
        write_dataset("mine", [], manifest={})
    write_dataset("mine", [ParseResult(synthetic_emails(1, 0)[0])], manifest={}, force=True)
    assert load_manifest("mine")["count"] == 1


def test_failed_import_leaves_no_dataset_and_keeps_old_one():
    write_dataset("keep", [ParseResult(synthetic_emails(1, 0)[0])], manifest={})

    def crashing():
        yield ParseResult(synthetic_emails(1, 0, seed=1)[0])
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        write_dataset("new", crashing(), manifest={})
    with pytest.raises(RuntimeError):
        write_dataset("keep", crashing(), manifest={}, force=True)
    assert not dataset_exists("new")
    assert load_manifest("keep")["count"] == 1
    assert [m["name"] for m in list_datasets()] == ["keep"]


def test_names_are_validated_and_missing_is_reported():
    for bad in ("../evil", "", "a b", "x" * 80):
        with pytest.raises(InvalidDatasetName):
            write_dataset(bad, [], manifest={})
    with pytest.raises(DatasetNotFound):
        list(read_dataset("nope"))
    with pytest.raises(DatasetNotFound):
        delete_dataset("nope")


def test_delete():
    write_dataset("gone", [], manifest={})
    delete_dataset("gone")
    assert not dataset_exists("gone")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dataset_store.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.dataset'`.

- [ ] **Step 3: Implement the store half of `spamfilter/dataset.py`**

```python
"""Datasets on disk (gzipped JSONL + manifest), deduplication and train/validation/test splits."""

from __future__ import annotations

import gzip
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import __version__
from .config import datasets_dir
from .message import Email, ParseResult

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_EMAILS_FILE = "emails.jsonl.gz"
_MANIFEST_FILE = "manifest.json"
_PARTIAL = ".partial"
_OLD = ".old"


class DatasetExists(Exception):
    pass


class DatasetNotFound(Exception):
    pass


class InvalidDatasetName(ValueError):
    pass


def _check_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise InvalidDatasetName(
            f"invalid dataset name {name!r}: use 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit")
    return name


def dataset_path(name: str) -> Path:
    return datasets_dir() / _check_name(name)


def dataset_exists(name: str) -> bool:
    return (dataset_path(name) / _MANIFEST_FILE).exists()


def write_dataset(name: str, results: Iterable[ParseResult], *, manifest: dict, force: bool = False,
                  progress: Callable[[int], None] | None = None) -> dict:
    final = dataset_path(name)
    if final.exists() and not force:
        raise DatasetExists(f"dataset '{name}' already exists; use --force to replace it")
    tmp = final.with_name(name + _PARTIAL)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    labels: Counter[str] = Counter()
    folders: dict[str, Counter[str]] = defaultdict(Counter)
    skipped: Counter[str] = Counter()
    count = 0
    try:
        with gzip.open(tmp / _EMAILS_FILE, "wt", encoding="utf-8") as fh:
            for result in results:
                if result.email is None:
                    skipped[result.error or "unknown"] += 1
                    continue
                email = result.email
                fh.write(json.dumps(email.to_dict(), ensure_ascii=False) + "\n")
                label = email.label or "unlabeled"
                labels[label] += 1
                folders[email.folder][label] += 1
                count += 1
                if progress and count % 1000 == 0:
                    progress(count)
        full = {
            **manifest,
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(),
            "count": count,
            "labels": dict(labels),
            "folders": {folder: dict(c) for folder, c in folders.items()},
            "skipped": dict(skipped),
            "version": __version__,
        }
        (tmp / _MANIFEST_FILE).write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    old = final.with_name(name + _OLD)
    if final.exists():  # only reached with force=True, after the new data was fully written
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    return full


def load_manifest(name: str) -> dict:
    path = dataset_path(name) / _MANIFEST_FILE
    if not path.exists():
        raise DatasetNotFound(f"no dataset named '{name}'; see `spamfilter datasets list`")
    return json.loads(path.read_text(encoding="utf-8"))


def read_dataset(name: str) -> Iterator[Email]:
    load_manifest(name)
    with gzip.open(dataset_path(name) / _EMAILS_FILE, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield Email.from_dict(json.loads(line))


def list_datasets() -> list[dict]:
    out = []
    for path in sorted(datasets_dir().iterdir()):
        if path.is_dir() and (path / _MANIFEST_FILE).exists() and not path.name.endswith((_PARTIAL, _OLD)):
            out.append(json.loads((path / _MANIFEST_FILE).read_text(encoding="utf-8")))
    return out


def delete_dataset(name: str) -> None:
    load_manifest(name)
    shutil.rmtree(dataset_path(name))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dataset_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/dataset.py tests/test_dataset_store.py
$GIT commit -m "Add crash-safe dataset store"
$GIT push
```

---

### Task 7: Deduplication and splits

**Files:**
- Modify: `spamfilter/dataset.py` (append dedupe and split code; add imports)
- Create: `tests/test_dedupe_split.py`

**Interfaces:**
- Consumes: `message.Email`, `message.SPAM`.
- Produces:
  - `DedupeStats(exact: int = 0, near: int = 0, conflict_groups: int = 0, conflict_emails: int = 0)` (dataclass)
  - `dedupe(emails: Iterable[Email]) -> tuple[list[Email], DedupeStats]`: keeps the first copy of each raw message. It drops **every** email in a `norm_body_hash` group with conflicting labels. `near` counts kept emails that share a group with another kept email.
  - `Split(train: list[Email], val: list[Email], test: list[Email], time_based: bool = False)` (dataclass)
  - `labels_of(emails: Sequence[Email]) -> np.ndarray` (1 = spam)
  - `split_base(emails, *, test_frac=0.2, val_frac=0.15, seed=42) -> Split`: grouped and stratified
  - `split_user(emails, *, test_frac=0.2, val_frac=0.15, min_dated=200, seed=42) -> Split`: time-based. Undated emails go to train. It falls back to `split_base` (with `time_based=False`) below `min_dated`.

- [ ] **Step 1: Write the failing tests `tests/test_dedupe_split.py`**

```python
import itertools

from spamfilter.dataset import dedupe, split_base, split_user
from spamfilter.parse import parse_bytes
from tests.builders import make_raw, synthetic_emails


def test_dedupe_exact_near_and_conflicts():
    emails = synthetic_emails(2, 2, seed=1)
    ham = parse_bytes(make_raw("Meeting notes", "Please review the attached meeting notes today"), label="ham").email
    spam = parse_bytes(make_raw("Meeting notes", "Please  review the attached meeting notes today",
                                sender="x@y.biz"), label="spam").email
    near_a = parse_bytes(make_raw("Invoice 1001", "Your invoice number 1001 is ready to view"), label="ham").email
    near_b = parse_bytes(make_raw("Invoice 2002", "Your invoice number 2002 is ready to view"), label="ham").email
    kept, stats = dedupe(emails + [emails[0], ham, spam, near_a, near_b])
    assert stats.exact == 1
    assert (stats.conflict_groups, stats.conflict_emails) == (1, 2)
    assert stats.near == 2
    assert len(kept) == 6 and ham not in kept and spam not in kept


def test_base_split_keeps_groups_together_and_stratifies():
    emails = synthetic_emails(60, 90, seed=2)
    for i in range(0, 40, 2):  # make 20 near-duplicate pairs
        emails[i + 1].norm_body_hash = emails[i].norm_body_hash
        emails[i + 1].label = emails[i].label
    s = split_base(emails, seed=0)
    parts = [s.train, s.val, s.test]
    assert sum(map(len, parts)) == 150
    for a, b in itertools.combinations(parts, 2):
        assert not ({e.norm_body_hash for e in a} & {e.norm_body_hash for e in b})
    assert 0.12 <= len(s.test) / 150 <= 0.28
    assert {e.label for e in s.test} == {"spam", "ham"}
    assert s.time_based is False


def test_user_split_is_time_ordered():
    emails = synthetic_emails(100, 200, seed=3)  # hourly dates, shuffled labels
    s = split_user(emails)
    assert s.time_based is True
    assert (len(s.train), len(s.val), len(s.test)) == (204, 36, 60)
    assert max(e.date for e in s.train) < min(e.date for e in s.val)
    assert max(e.date for e in s.val) < min(e.date for e in s.test)


def test_user_split_undated_to_train_and_small_fallback():
    emails = synthetic_emails(100, 200, seed=4)
    emails[0].date = None
    s = split_user(emails)
    assert emails[0] in s.train and s.time_based
    small = synthetic_emails(20, 40, seed=5)
    fallback = split_user(small)
    assert fallback.time_based is False
    assert sum(map(len, (fallback.train, fallback.val, fallback.test))) == 60
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_dedupe_split.py -v`
Expected: collection error `ImportError: cannot import name 'dedupe' from 'spamfilter.dataset'`.

- [ ] **Step 3: Append dedupe and split code to `spamfilter/dataset.py`**

Add these imports at the top of the file, next to the existing ones:

```python
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .message import SPAM
```

Append at the end of the file:

```python
@dataclass
class DedupeStats:
    exact: int = 0
    near: int = 0
    conflict_groups: int = 0
    conflict_emails: int = 0


def dedupe(emails: Iterable[Email]) -> tuple[list[Email], DedupeStats]:
    stats = DedupeStats()
    seen: set[str] = set()
    unique: list[Email] = []
    for email in emails:
        if email.raw_sha256 in seen:
            stats.exact += 1
            continue
        seen.add(email.raw_sha256)
        unique.append(email)
    group_labels: dict[str, set] = defaultdict(set)
    for email in unique:
        group_labels[email.norm_body_hash].add(email.label)
    conflicted = {h for h, labels in group_labels.items() if len(labels) > 1}
    stats.conflict_groups = len(conflicted)
    kept = []
    for email in unique:
        if email.norm_body_hash in conflicted:
            stats.conflict_emails += 1
        else:
            kept.append(email)
    sizes = Counter(e.norm_body_hash for e in kept)
    stats.near = sum(1 for e in kept if sizes[e.norm_body_hash] > 1)
    return kept, stats


@dataclass
class Split:
    train: list[Email] = field(default_factory=list)
    val: list[Email] = field(default_factory=list)
    test: list[Email] = field(default_factory=list)
    time_based: bool = False


def labels_of(emails: Sequence[Email]) -> np.ndarray:
    return np.array([1 if e.label == SPAM else 0 for e in emails], dtype=int)


def _grouped_holdout(emails: list[Email], frac: float, seed: int) -> tuple[list[Email], list[Email]]:
    """Hold out about `frac` of the emails, stratified by label, never splitting a norm_body_hash group."""
    n_splits = max(2, round(1 / frac))
    if len(emails) < n_splits:
        return list(emails), []
    y = labels_of(emails)
    groups = [e.norm_body_hash for e in emails]
    idx = np.arange(len(emails))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # small classes: sklearn warns but still splits
        if len(set(y.tolist())) < 2:
            keep, hold = next(GroupKFold(n_splits=n_splits).split(idx, y, groups))
        else:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            keep, hold = next(splitter.split(idx, y, groups))
    return [emails[i] for i in keep], [emails[i] for i in hold]


def split_base(emails: Sequence[Email], *, test_frac: float = 0.2, val_frac: float = 0.15, seed: int = 42) -> Split:
    rest, test = _grouped_holdout(list(emails), test_frac, seed)
    train, val = _grouped_holdout(rest, val_frac, seed)
    return Split(train, val, test, time_based=False)


def split_user(emails: Sequence[Email], *, test_frac: float = 0.2, val_frac: float = 0.15,
               min_dated: int = 200, seed: int = 42) -> Split:
    dated = sorted((e for e in emails if e.date), key=lambda e: e.date)
    undated = [e for e in emails if not e.date]
    if len(dated) < min_dated:
        return split_base(emails, test_frac=test_frac, val_frac=val_frac, seed=seed)
    n_test = int(round(len(dated) * test_frac))
    rest, test = dated[: len(dated) - n_test], dated[len(dated) - n_test:]
    n_val = int(round(len(rest) * val_frac))
    train, val = rest[: len(rest) - n_val], rest[len(rest) - n_val:]
    return Split(train + undated, val, test, time_based=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_dedupe_split.py tests/test_dataset_store.py -v`
Expected: all passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/dataset.py tests/test_dedupe_split.py
$GIT commit -m "Add deduplication and leak-free grouped and time-based splits"
$GIT push
```

---

### Task 8: Public corpora: pinned downloads and loaders

**Files:**
- Create: `spamfilter/corpora.py`, `tests/test_corpora.py`

**Interfaces:**
- Consumes: `config.corpora_dir()`, `dataset.write_dataset`, `parse.parse_bytes`, `parse.email_from_text`, `message.SPAM/HAM/ParseResult`.
- Produces:
  - `CorpusError(Exception)`
  - `CorpusFile(corpus: str, url: str, sha256: str)` (frozen dataclass) with `.filename`
  - `FILES: list[CorpusFile]`: 16 pinned files (9 SpamAssassin, 6 Enron, 1 Spambase)
  - `DATASET_FOR = {"spamassassin": "base-spamassassin", "enron": "base-enron"}`
  - `SPAMBASE_COLUMNS: list[str]` (58 names; last is `is_spam`)
  - `fetch(cf: CorpusFile, dest_dir: Path, *, log=print) -> Path`
  - `download(corpus: str, log=print) -> list[Path]`
  - `iter_spamassassin(paths) -> Iterator[ParseResult]`, `iter_enron(paths) -> Iterator[ParseResult]`, `parse_enron(raw, *, folder, label, name) -> ParseResult`
  - `prepare(corpus: str, log=print) -> dict` (dataset manifest)
  - `load_spambase(log=print) -> pandas.DataFrame`

- [ ] **Step 1: Write the failing tests `tests/test_corpora.py`**

```python
import hashlib
import io
import re
import tarfile
import zipfile

import pytest

from spamfilter import corpora
from spamfilter.config import corpora_dir
from spamfilter.corpora import CorpusError, CorpusFile
from spamfilter.dataset import read_dataset
from tests.builders import make_raw


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tar(path, members, mode):
    with tarfile.open(path, mode) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _quiet(*_args):
    pass


@pytest.fixture
def fake_corpora(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    sa = _tar(src / "sa.tar.bz2", {
        "spam/0001.abc": make_raw("WIN", "cash prize now claim"),
        "spam/cmds": b"mv 00001 00002",
        "easy_ham/0002.def": make_raw("lunch", "want to grab lunch today"),
    }, "w:bz2")
    en = _tar(src / "enron1.tar.gz", {
        "enron1/ham/0001.1999-12-10.farmer.ham.txt": b"Subject: christmas tree farm pictures\r\n",
        "enron1/spam/0006.2003-12-18.GP.spam.txt": b"Subject: cheap meds\r\nbuy now ! best price\r\n",
        "enron1/Summary.txt": b"stats",
    }, "w:gz")
    files = [CorpusFile("spamassassin", sa.as_uri(), _sha(sa)), CorpusFile("enron", en.as_uri(), _sha(en))]
    monkeypatch.setattr(corpora, "FILES", files)
    return files


def test_prepare_builds_base_datasets(fake_corpora):
    sa = corpora.prepare("spamassassin", log=_quiet)
    assert sa["name"] == "base-spamassassin" and sa["labels"] == {"spam": 1, "ham": 1}
    en = corpora.prepare("enron", log=_quiet)
    assert en["labels"] == {"ham": 1, "spam": 1}
    enron = {e.label: e for e in read_dataset("base-enron")}
    assert enron["spam"].subject == "cheap meds" and "best price" in enron["spam"].text
    assert enron["ham"].date.year == 1999 and enron["ham"].text.strip() == ""
    assert enron["spam"].from_addr == "" and enron["spam"].source == "base-enron"


def test_checksum_mismatch_is_reported(fake_corpora, monkeypatch):
    monkeypatch.setattr(corpora, "FILES", [CorpusFile("spamassassin", fake_corpora[0].url, "0" * 64)])
    with pytest.raises(CorpusError, match="checksum mismatch"):
        corpora.prepare("spamassassin", log=_quiet)
    assert not list(corpora_dir().glob("*.part"))


def test_download_failure_says_where_to_place_file(monkeypatch, tmp_path):
    missing = (tmp_path / "missing.tar.gz").as_uri()
    monkeypatch.setattr(corpora, "FILES", [CorpusFile("enron", missing, "0" * 64)])
    with pytest.raises(CorpusError, match="place it at"):
        corpora.prepare("enron", log=_quiet)


def test_verified_cached_file_is_not_downloaded_again(fake_corpora, monkeypatch):
    corpora.fetch(fake_corpora[0], corpora_dir(), log=_quiet)

    def no_network(*_a, **_k):
        raise AssertionError("network used")

    monkeypatch.setattr(corpora.urllib.request, "urlopen", no_network)
    corpora.fetch(fake_corpora[0], corpora_dir(), log=_quiet)


def test_unknown_corpus():
    with pytest.raises(CorpusError):
        corpora.prepare("nope", log=_quiet)


def test_real_file_table_is_pinned():
    real = corpora.FILES
    assert len(real) == 16
    assert sum(cf.corpus == "spamassassin" for cf in real) == 9
    assert sum(cf.corpus == "enron" for cf in real) == 6
    assert all(cf.url.startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", cf.sha256) for cf in real)


def test_load_spambase(monkeypatch, tmp_path):
    z = tmp_path / "spambase.zip"
    row = ",".join(["0"] * 57) + ",1\n"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("spambase.data", row * 2)
    monkeypatch.setattr(corpora, "FILES", [CorpusFile("spambase", z.as_uri(), _sha(z))])
    df = corpora.load_spambase(log=_quiet)
    assert df.shape == (2, 58) and df.columns[-1] == "is_spam" and df.columns[0] == "word_freq_make"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_corpora.py -v`
Expected: collection error `ImportError: cannot import name 'corpora' from 'spamfilter'`.

- [ ] **Step 3: Implement `spamfilter/corpora.py`**

```python
"""Public corpora: pinned, checksum-verified downloads and conversion into base datasets."""

from __future__ import annotations

import hashlib
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import corpora_dir
from .dataset import write_dataset
from .message import HAM, SPAM, ParseResult
from .parse import email_from_text, parse_bytes


class CorpusError(Exception):
    pass


@dataclass(frozen=True)
class CorpusFile:
    corpus: str
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


_SA = "https://spamassassin.apache.org/old/publiccorpus/"
_EN = "https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/"

# SHA-256 values recorded from downloads on 2026-09-30.
FILES: list[CorpusFile] = [
    CorpusFile("spamassassin", _SA + "20021010_easy_ham.tar.bz2", "f9fc56e1f68780f9afdc6d23a2e24c4584af988afc58bbfabae500e51f3a2f36"),
    CorpusFile("spamassassin", _SA + "20021010_hard_ham.tar.bz2", "7cd46378877e00caa4e943920a854e2ef95ac29e296cf2a3ce691fb4104c4fbd"),
    CorpusFile("spamassassin", _SA + "20021010_spam.tar.bz2", "048c2d2e61ff13f7cef88788c7184ee57dbc9049c58e5bd706fdb8537e9bac81"),
    CorpusFile("spamassassin", _SA + "20030228_easy_ham.tar.bz2", "2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438"),
    CorpusFile("spamassassin", _SA + "20030228_easy_ham_2.tar.bz2", "b4bd3dc5ae5b40f38e99a0e41ad7d16b428b56e818e3a6ffa07330d62004dd38"),
    CorpusFile("spamassassin", _SA + "20030228_hard_ham.tar.bz2", "ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9"),
    CorpusFile("spamassassin", _SA + "20030228_spam.tar.bz2", "c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743"),
    CorpusFile("spamassassin", _SA + "20030228_spam_2.tar.bz2", "40bc444c596ff3e71505ae334f115b16f9b946786ee68f09b921b6b99217fd00"),
    CorpusFile("spamassassin", _SA + "20050311_spam_2.tar.bz2", "44280a0e28bf7645b2279e8e42659271ad82c6f40bf8d364dbda7f1df344a765"),
    CorpusFile("enron", _EN + "enron1.tar.gz", "3d6a1fcced6c30b1701a3fd04c6b912d7046208f6ef6f4fdcf022f64bdc6c32f"),
    CorpusFile("enron", _EN + "enron2.tar.gz", "9b5bf5c4844eb4ecead52de8c676ff8761c2d7037fdc175851c88ef52c827315"),
    CorpusFile("enron", _EN + "enron3.tar.gz", "9321ca1c195b45b6f0fb795963a8020ff317e7ebac2c102205272f0b91a1ac0b"),
    CorpusFile("enron", _EN + "enron4.tar.gz", "61be71c59d7e9fdf99df63b814ddbeb18e0c9e8a69ce691d8e4e8127ed3588aa"),
    CorpusFile("enron", _EN + "enron5.tar.gz", "d7b839c2a7071b6d52f1d76be8d05e66733d872614475fad7522740d7c60540d"),
    CorpusFile("enron", _EN + "enron6.tar.gz", "283f70c45a7d819e0b988536fbb8e21cc17c17a22998403c2255d2c38cbfc5a0"),
    CorpusFile("spambase", "https://archive.ics.uci.edu/static/public/94/spambase.zip", "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636"),
]

DATASET_FOR = {"spamassassin": "base-spamassassin", "enron": "base-enron"}

SPAMBASE_COLUMNS = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
    "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
    "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
    "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
    "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_semicolon",
    "char_freq_left_paren", "char_freq_left_bracket", "char_freq_exclamation", "char_freq_dollar",
    "char_freq_pound", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "is_spam",
]

_ENRON_DATE = re.compile(r"\.(\d{4})-(\d{2})-(\d{2})\.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(cf: CorpusFile, dest_dir: Path, *, log=print) -> Path:
    dest = dest_dir / cf.filename
    if dest.exists() and _sha256(dest) == cf.sha256:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    log(f"Downloading {cf.url}")
    try:
        with urllib.request.urlopen(cf.url, timeout=60) as response, tmp.open("wb") as fh:
            shutil.copyfileobj(response, fh, 1 << 20)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise CorpusError(f"could not download {cf.url} ({exc}). Download it manually and place it at {dest}") from exc
    got = _sha256(tmp)
    if got != cf.sha256:
        tmp.unlink(missing_ok=True)
        raise CorpusError(f"checksum mismatch for {cf.filename} (expected {cf.sha256}, got {got}). "
                          f"Download a verified copy and place it at {dest}")
    tmp.replace(dest)
    return dest


def download(corpus: str, log=print) -> list[Path]:
    files = [cf for cf in FILES if cf.corpus == corpus]
    if not files:
        raise CorpusError(f"unknown corpus {corpus!r}; choose from spamassassin, enron, spambase")
    return [fetch(cf, corpora_dir(), log=log) for cf in files]


def iter_spamassassin(paths: Iterable[Path]) -> Iterator[ParseResult]:
    for path in paths:
        with tarfile.open(path, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) < 2 or parts[-1] == "cmds":
                    continue
                label = SPAM if "spam" in parts[-2].lower() else HAM
                yield parse_bytes(tar.extractfile(member).read(), source="base-spamassassin",
                                  folder=parts[-2], label=label)


def parse_enron(raw: bytes, *, folder: str, label: str, name: str) -> ParseResult:
    text = None
    for charset in ("utf-8", "cp1252"):
        try:
            text = raw.decode(charset)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1")
    text = text.replace("\r\n", "\n")
    first, _, body = text.partition("\n")
    if first.lower().startswith("subject:"):
        subject = first[len("subject:"):].strip()
    else:
        subject, body = "", text
    if not subject and not body.strip():
        return ParseResult(None, "empty")
    date = None
    match = _ENRON_DATE.search(name)
    if match:
        try:
            date = datetime(int(match[1]), int(match[2]), int(match[3]), tzinfo=timezone.utc)
        except ValueError:
            date = None
    return ParseResult(email_from_text(subject, body, source="base-enron", folder=folder, label=label,
                                       date=date, raw_id=raw))


def iter_enron(paths: Iterable[Path]) -> Iterator[ParseResult]:
    for path in paths:
        with tarfile.open(path, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) < 2 or parts[-2] not in (SPAM, HAM):
                    continue
                yield parse_enron(tar.extractfile(member).read(), folder=parts[-2], label=parts[-2], name=parts[-1])


def prepare(corpus: str, log=print) -> dict:
    if corpus not in DATASET_FOR:
        raise CorpusError(f"unknown corpus {corpus!r}; choose from {', '.join(DATASET_FOR)}")
    paths = download(corpus, log)
    name = DATASET_FOR[corpus]
    log(f"Converting {corpus} into dataset {name} ...")
    results = iter_spamassassin(paths) if corpus == "spamassassin" else iter_enron(paths)
    return write_dataset(name, results, manifest={"importer": "corpus", "corpus": corpus,
                                                   "files": [p.name for p in paths]}, force=True)


def load_spambase(log=print) -> pd.DataFrame:
    (path,) = download("spambase", log)
    with zipfile.ZipFile(path) as archive, archive.open("spambase.data") as fh:
        frame = pd.read_csv(fh, header=None)
    frame.columns = SPAMBASE_COLUMNS
    return frame
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_corpora.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/corpora.py tests/test_corpora.py
$GIT commit -m "Add pinned SpamAssassin, Enron-Spam and Spambase corpus loaders"
$GIT push
```

---

### Task 9: Features

**Files:**
- Create: `spamfilter/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `message.Email`, `parse.URL_RE`.
- Produces:
  - `CHAR_HEAD = 2000`
  - `STRUCT_NAMES: list[str]` (13 names, in order): `url_count, url_domains, html_share, n_attachments, risky_attachment, reply_to_mismatch, caps_ratio, exclaim_per_1k, dollar_per_1k, log_length, ip_url, shortener_url, has_headers`
  - `url_domain(url: str) -> str`
  - `normalize_text(s: str) -> str`: lower-case, with `__url__` + `urldom_<domain>` tokens, `__email__` and `__num__`
  - `structural_features(e: Email) -> list[float]`
  - `header_tokens(e: Email) -> str`
  - `emails_to_frame(emails: Sequence[Email]) -> pandas.DataFrame`: columns `subject, text, head, hdr` + `STRUCT_NAMES`
  - `make_features(kind: str = "full", *, min_df: int = 2, max_word_features: int = 200_000, max_char_features: int = 300_000) -> ColumnTransformer`. `kind` is `"full"` or `"words"`; `"words"` blocks are exactly `subj`, `body`.
  - `n_word_columns(fitted_ct) -> int`: the number of leading columns that come from `subj` + `body`

- [ ] **Step 1: Write the failing tests `tests/test_features.py`**

```python
import numpy as np
from scipy import sparse

from spamfilter.features import (STRUCT_NAMES, emails_to_frame, header_tokens, make_features, n_word_columns,
                                 normalize_text, structural_features, url_domain)
from spamfilter.parse import email_from_text, parse_bytes
from tests.builders import make_raw, synthetic_emails


def test_url_domain():
    assert url_domain("http://www.Deals.example.com/x?y=1") == "deals.example.com"
    assert url_domain("www.foo.org/path") == "foo.org"
    assert url_domain("http://[broken") == ""


def test_normalize_text():
    s = normalize_text("Visit http://www.Deals.example.com/x?id=5 or mail me@x.org, call 555-1234 NOW")
    assert "__url__" in s and "urldom_deals_example_com" in s
    assert "__email__" in s and "__num__" in s
    assert s == s.lower() and "555" not in s


def _suspicious():
    raw = make_raw("hi", None,
                   html="<p>FREE!!! $$$</p><a href='http://bit.ly/x'>a</a><a href='http://1.2.3.4/y'>b</a>",
                   sender="a@shop.example", reply_to="z@other.example", attachments=[("x.exe", b"MZ")])
    return parse_bytes(raw).email


def test_structural_features():
    f = dict(zip(STRUCT_NAMES, structural_features(_suspicious())))
    assert len(f) == 13
    assert f["url_count"] == 2 and f["url_domains"] == 2
    assert f["shortener_url"] == 1 and f["ip_url"] == 1
    assert f["reply_to_mismatch"] == 1 and f["risky_attachment"] == 1
    assert f["has_headers"] == 1 and f["html_share"] == 1.0
    assert f["exclaim_per_1k"] > 0 and f["dollar_per_1k"] > 0 and f["caps_ratio"] > 0.5
    headerless = dict(zip(STRUCT_NAMES, structural_features(email_from_text("x", "plain words"))))
    assert headerless["has_headers"] == 0 and headerless["url_count"] == 0


def test_header_tokens():
    toks = header_tokens(_suspicious()).split()
    assert "fromdom=shop.example" in toks and "replydom=other.example" in toks
    assert "ctype=multipart/mixed" in toks
    assert header_tokens(email_from_text("x", "y")) == "no_headers"


def test_make_features_full_and_words():
    emails = synthetic_emails(10, 10)
    frame = emails_to_frame(emails)
    assert list(frame.columns[:4]) == ["subject", "text", "head", "hdr"]
    full = make_features("full", min_df=1)
    X = full.fit_transform(frame)
    assert sparse.issparse(X) and X.shape[0] == 20
    words = make_features("words", min_df=1).fit_transform(frame)
    assert words.shape[1] == n_word_columns(full)
    assert np.allclose(sparse.csr_matrix(X)[:, : words.shape[1]].toarray(), words.toarray())
    assert (words.data >= 0).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.features'`.

- [ ] **Step 3: Implement `spamfilter/features.py`**

```python
"""Feature extraction: TF-IDF text blocks, header tokens and structural signals."""

from __future__ import annotations

import math
import re
from typing import Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .message import Email
from .parse import URL_RE

CHAR_HEAD = 2000
STRUCT_NAMES = [
    "url_count", "url_domains", "html_share", "n_attachments", "risky_attachment", "reply_to_mismatch",
    "caps_ratio", "exclaim_per_1k", "dollar_per_1k", "log_length", "ip_url", "shortener_url", "has_headers",
]
SHORTENERS = frozenset({"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
                        "cutt.ly", "tiny.cc", "rb.gy", "shorturl.at"})
RISKY_EXT = frozenset({"exe", "scr", "js", "vbs", "bat", "cmd", "com", "jar", "html", "htm", "zip", "rar", "7z",
                       "iso", "img", "docm", "xlsm", "lnk", "hta"})
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUM_RE = re.compile(r"\b\d[\d,.\-]*\b")
_IP_URL_RE = re.compile(r"^(?:https?://)?\d{1,3}(?:\.\d{1,3}){3}(?:[:/]|$)")
_MAILER_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")


def url_domain(url: str) -> str:
    candidate = url if "://" in url else "http://" + url
    try:
        host = urlsplit(candidate).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


def normalize_text(s: str) -> str:
    def url_token(match: re.Match) -> str:
        domain = re.sub(r"[^a-z0-9]", "_", url_domain(match.group(0)))
        return f" __url__ urldom_{domain} "

    s = URL_RE.sub(url_token, s)
    s = _EMAIL_RE.sub(" __email__ ", s)
    s = _NUM_RE.sub(" __num__ ", s)
    return s.lower()


def _domain_of(address: str) -> str:
    return address.rsplit("@", 1)[1] if "@" in address else ""


def structural_features(e: Email) -> list[float]:
    text = e.text or ""
    n_chars = max(len(text), 1)
    sample = text[:20_000]
    letters = sum(c.isalpha() for c in sample)
    caps = sum(c.isupper() for c in sample)
    domains = {url_domain(u) for u in e.urls} - {""}
    from_domain, reply_domain = _domain_of(e.from_addr), _domain_of(e.reply_to)
    return [
        float(len(e.urls)),
        float(len(domains)),
        float(e.html_share),
        float(e.n_attachments),
        float(any(t.lower() in RISKY_EXT for t in e.attachment_types)),
        float(bool(from_domain and reply_domain and from_domain != reply_domain)),
        caps / letters if letters else 0.0,
        1000.0 * text.count("!") / n_chars,
        1000.0 * text.count("$") / n_chars,
        math.log1p(len(text)),
        float(any(_IP_URL_RE.match(u) for u in e.urls)),
        float(any(d in SHORTENERS for d in domains)),
        float(bool(e.from_addr)),
    ]


def header_tokens(e: Email) -> str:
    tokens = []
    if _domain_of(e.from_addr):
        tokens.append("fromdom=" + _domain_of(e.from_addr))
    if _domain_of(e.reply_to):
        tokens.append("replydom=" + _domain_of(e.reply_to))
    mailer = e.headers.get("x-mailer") or e.headers.get("user-agent") or ""
    match = _MAILER_RE.match(mailer.strip())
    if match:
        tokens.append("mailer=" + match.group(0).lower())
    content_type = e.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type:
        tokens.append("ctype=" + content_type)
    if "list-unsubscribe" in e.headers:
        tokens.append("has_list_unsub")
    return " ".join(tokens) if tokens else "no_headers"


def emails_to_frame(emails: Sequence[Email]) -> pd.DataFrame:
    text_cols = {
        "subject": [e.subject or "" for e in emails],
        "text": [e.text or "" for e in emails],
        "head": [(e.subject or "") + "\n" + (e.text or "")[:CHAR_HEAD] for e in emails],
        "hdr": [header_tokens(e) for e in emails],
    }
    struct = pd.DataFrame([structural_features(e) for e in emails], columns=STRUCT_NAMES, dtype=float)
    return pd.concat([pd.DataFrame(text_cols), struct], axis=1)


def make_features(kind: str = "full", *, min_df: int = 2, max_word_features: int = 200_000,
                  max_char_features: int = 300_000) -> ColumnTransformer:
    word = dict(preprocessor=normalize_text, ngram_range=(1, 2), sublinear_tf=True, min_df=min_df, dtype=np.float32)
    blocks = [
        ("subj", TfidfVectorizer(max_features=50_000, **word), "subject"),
        ("body", TfidfVectorizer(max_features=max_word_features, **word), "text"),
    ]
    weights = {"subj": 2.0, "body": 1.0}
    if kind == "full":
        blocks += [
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=min_df,
                                     max_features=max_char_features, dtype=np.float32), "head"),
            ("hdr", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, dtype=np.float32), "hdr"),
            ("num", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                              ("scale", StandardScaler())]), STRUCT_NAMES),
        ]
        weights.update(char=1.0, hdr=1.0, num=1.0)
    elif kind != "words":
        raise ValueError(f"unknown feature kind {kind!r}")
    return ColumnTransformer(blocks, transformer_weights=weights, sparse_threshold=1.0)


def n_word_columns(fitted: ColumnTransformer) -> int:
    return sum(len(fitted.named_transformers_[name].vocabulary_) for name in ("subj", "body"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/features.py tests/test_features.py
$GIT commit -m "Add TF-IDF, header and structural spam features"
$GIT push
```

---

### Task 10: Model and bundle

**Files:**
- Create: `spamfilter/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `features.make_features`, `features.emails_to_frame`, `message.Email`, `spamfilter.__version__`.
- Produces:
  - `ALGOS = ("logreg", "cnb", "svm")`
  - `make_classifier(algo: str, *, C: float = 1.0, alpha: float = 0.3)` → LogisticRegression (liblinear) / ComplementNB / LinearSVC
  - `Reason(feature: str, weight: float)` with `to_dict()`
  - `LinearTextModel(algo="logreg", *, C=1.0, alpha=0.3, min_df=2, calibration="sigmoid", feature_kwargs=None)` with `backend = "linear"`, `fit(emails, y, sample_weight=None) -> self`, `predict_proba(emails) -> np.ndarray`, `explain(email, top_k=8) -> list[Reason]`
  - `BundleError(Exception)`
  - `current_versions() -> dict` (keys `spamfilter`, `sklearn`, `python`)
  - `Bundle(model, flag_threshold, move_threshold, report={}, manifest={}, versions=current_versions())` with `backend`, `verdict(p) -> "spam" | "suspicious" | "ham"`, `save(path)`, `Bundle.load(path) -> Bundle`

- [ ] **Step 1: Write the failing tests `tests/test_model.py`**

```python
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from spamfilter.model import Bundle, BundleError, LinearTextModel
from tests.builders import synthetic_emails


@pytest.fixture(scope="module")
def data():
    return synthetic_emails(80, 120, seed=10), synthetic_emails(20, 30, seed=11)


def _y(emails):
    return np.array([e.label == "spam" for e in emails], dtype=int)


@pytest.mark.parametrize("algo", ["logreg", "cnb", "svm"])
def test_fit_predict(algo, data):
    train, test = data
    model = LinearTextModel(algo, min_df=1).fit(train, _y(train))
    p = model.predict_proba(test)
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(_y(test), p) > 0.95


def test_sample_weight_is_accepted(data):
    train, test = data
    weights = np.where(np.arange(len(train)) % 2 == 0, 3.0, 1.0)
    model = LinearTextModel("logreg", min_df=1).fit(train, _y(train), sample_weight=weights)
    assert model.predict_proba(test).shape == (50,)


@pytest.mark.parametrize("algo", ["logreg", "cnb"])
def test_explain_returns_signed_reasons(algo, data):
    train, test = data
    model = LinearTextModel(algo, min_df=1).fit(train, _y(train))
    spam = next(e for e in test if e.label == "spam")
    reasons = model.explain(spam, top_k=5)
    assert 1 <= len(reasons) <= 5
    assert all(isinstance(r.feature, str) and r.feature for r in reasons)
    assert sum(r.weight for r in reasons) > 0
    assert reasons[0].to_dict().keys() == {"feature", "weight"}


def test_bundle_roundtrip_and_verdict(tmp_path, data):
    train, test = data
    model = LinearTextModel("logreg", min_df=1).fit(train, _y(train))
    bundle = Bundle(model, flag_threshold=0.5, move_threshold=0.9, report={"x": 1})
    path = tmp_path / "m.model"
    bundle.save(path)
    loaded = Bundle.load(path)
    np.testing.assert_allclose(loaded.model.predict_proba(test), model.predict_proba(test))
    assert loaded.report == {"x": 1} and loaded.backend == "linear"
    assert (loaded.verdict(0.95), loaded.verdict(0.6), loaded.verdict(0.1)) == ("spam", "suspicious", "ham")


def test_bundle_version_mismatch_is_refused(tmp_path, data):
    train, _ = data
    bundle = Bundle(LinearTextModel("cnb", min_df=1).fit(train, _y(train)), 0.5, 0.9)
    bundle.versions["sklearn"] = "0.1.0"
    path = tmp_path / "old.model"
    bundle.save(path)
    with pytest.raises(BundleError, match="Retrain"):
        Bundle.load(path)


def test_loading_garbage_is_refused(tmp_path):
    path = tmp_path / "junk.model"
    path.write_bytes(b"not a model")
    with pytest.raises(BundleError):
        Bundle.load(path)
    with pytest.raises(BundleError):
        Bundle.load(tmp_path / "missing.model")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.model'`.

- [ ] **Step 3: Implement `spamfilter/model.py`**

```python
"""The spam model: calibrated linear classifiers over spamfilter features, saved as one bundle file."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import __version__
from .features import emails_to_frame, make_features
from .message import Email

ALGOS = ("logreg", "cnb", "svm")


def make_classifier(algo: str, *, C: float = 1.0, alpha: float = 0.3):
    if algo == "logreg":
        return LogisticRegression(C=C, solver="liblinear", max_iter=2000)
    if algo == "svm":
        return LinearSVC(C=C, max_iter=5000)
    if algo == "cnb":
        return ComplementNB(alpha=alpha)
    raise ValueError(f"unknown algorithm {algo!r}; choose from {', '.join(ALGOS)}")


@dataclass
class Reason:
    feature: str
    weight: float  # positive pushes toward spam

    def to_dict(self) -> dict:
        return {"feature": self.feature, "weight": round(self.weight, 4)}


_BLOCK_NAMES = {"subj": "subject", "body": "word", "char": "text", "hdr": "header", "num": "signal"}
_TOKEN_NAMES = {"__url__": "<link>", "__email__": "<email>", "__num__": "<number>"}


def _pretty(name: str) -> str:
    block, _, feature = name.partition("__")
    words = []
    for token in feature.split(" "):
        if token.startswith("urldom_"):
            token = "link:" + token[len("urldom_"):].replace("_", ".")
        words.append(_TOKEN_NAMES.get(token, token))
    feature = " ".join(words)
    if block == "char":
        return f'text "{feature}"'
    return f"{_BLOCK_NAMES.get(block, block)}: {feature}"


class LinearTextModel:
    backend = "linear"

    def __init__(self, algo: str = "logreg", *, C: float = 1.0, alpha: float = 0.3, min_df: int = 2,
                 calibration: str = "sigmoid", feature_kwargs: dict | None = None) -> None:
        if algo not in ALGOS:
            raise ValueError(f"unknown algorithm {algo!r}")
        self.algo, self.C, self.alpha, self.min_df = algo, C, alpha, min_df
        self.calibration = calibration
        self.feature_kwargs = dict(feature_kwargs or {})
        self.pipeline: Pipeline | None = None

    def fit(self, emails: Sequence[Email], y, sample_weight=None) -> LinearTextModel:
        features = make_features("words" if self.algo == "cnb" else "full", min_df=self.min_df, **self.feature_kwargs)
        calibrated = CalibratedClassifierCV(make_classifier(self.algo, C=self.C, alpha=self.alpha),
                                            method=self.calibration, cv=3)
        self.pipeline = Pipeline([("feat", features), ("clf", calibrated)])
        params = {"clf__sample_weight": np.asarray(sample_weight)} if sample_weight is not None else {}
        self.pipeline.fit(emails_to_frame(emails), np.asarray(y), **params)
        return self

    def predict_proba(self, emails: Sequence[Email]) -> np.ndarray:
        return self.pipeline.predict_proba(emails_to_frame(emails))[:, 1]

    def _coef(self) -> np.ndarray:
        coefs = []
        for calibrated in self.pipeline.named_steps["clf"].calibrated_classifiers_:
            estimator = calibrated.estimator
            if hasattr(estimator, "coef_"):
                coefs.append(np.ravel(estimator.coef_))
            else:  # ComplementNB: class 1 vs class 0 log weights
                coefs.append(estimator.feature_log_prob_[1] - estimator.feature_log_prob_[0])
        return np.mean(coefs, axis=0)

    def explain(self, email: Email, top_k: int = 8) -> list[Reason]:
        features = self.pipeline.named_steps["feat"]
        row = sparse.csr_matrix(features.transform(emails_to_frame([email])))
        contributions = row.multiply(self._coef()).tocsr()
        names = features.get_feature_names_out()
        order = np.argsort(-np.abs(contributions.data))[:top_k]
        return [Reason(_pretty(names[contributions.indices[i]]), float(contributions.data[i])) for i in order]


class BundleError(Exception):
    pass


def current_versions() -> dict:
    return {"spamfilter": __version__, "sklearn": sklearn.__version__, "python": platform.python_version()}


def _major_minor(version: str) -> tuple[str, ...]:
    return tuple(version.split(".")[:2])


@dataclass
class Bundle:
    model: LinearTextModel
    flag_threshold: float
    move_threshold: float
    report: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    versions: dict = field(default_factory=current_versions)

    @property
    def backend(self) -> str:
        return self.model.backend

    def verdict(self, p_spam: float) -> str:
        if p_spam >= self.move_threshold:
            return "spam"
        if p_spam >= self.flag_threshold:
            return "suspicious"
        return "ham"

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> Bundle:
        path = Path(path)
        if not path.exists():
            raise BundleError(f"model file {path} does not exist")
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise BundleError(f"could not load model {path.name} ({type(exc).__name__}). "
                              "Retrain with `spamfilter train`.") from exc
        if not isinstance(bundle, Bundle):
            raise BundleError(f"{path.name} is not a spamfilter model bundle")
        trained = bundle.versions.get("sklearn", "?")
        if _major_minor(trained) != _major_minor(sklearn.__version__):
            raise BundleError(f"{path.name} was trained with scikit-learn {trained} but {sklearn.__version__} "
                              "is installed. Retrain with `spamfilter train`.")
        return bundle
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/model.py tests/test_model.py
$GIT commit -m "Add calibrated linear spam model with explanations and versioned bundles"
$GIT push
```

---

### Task 11: Evaluation: thresholds, metrics, reports, model selection

**Files:**
- Create: `spamfilter/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `features.emails_to_frame`, `features.make_features`, `features.n_word_columns`, `model.make_classifier`, `message.Email`.
- Produces:
  - `threshold_for_fpr(y, scores, fpr_target, mask=None) -> float`: the lowest threshold such that the share of (masked) ham with `score >= t` is ≤ target. Raises `ValueError` without ham.
  - `recall_at_fpr(y, scores, fpr_target, mask=None) -> tuple[float, float]` → `(recall, threshold)`
  - `metrics_at(y, proba, threshold) -> dict` (keys `tp fp tn fn precision recall f1 fpr accuracy`; ratios are `None` when undefined)
  - `report_for(emails, y, proba, flag, move) -> dict`: one entry per `email.source` plus `"all"`. Each entry has keys `n n_spam n_ham flag move roc_auc pr_auc pr_curve roc_curve`.
  - `Candidate(algo: str, C: float = 1.0, alpha: float = 0.3)` (frozen) with `label() -> str`
  - `DEFAULT_CANDIDATES: list[Candidate]` (4 logreg, 3 cnb, 3 svm)
  - `SelectionResult(best: Candidate, user_weight: float, table: list[dict])`
  - `select_model(emails, y, *, is_user, weights=(1.0,), candidates=None, folds=5, fp_target=0.005, min_df=2, seed=42, feature_kwargs=None, log=None) -> SelectionResult`
  - `format_source_table(per_source: dict) -> str`, `format_report(report: dict) -> str`

- [ ] **Step 1: Write the failing tests `tests/test_evaluate.py`**

```python
import numpy as np
import pytest

from spamfilter.evaluate import (Candidate, format_report, metrics_at, recall_at_fpr, report_for, select_model,
                                 threshold_for_fpr)
from tests.builders import synthetic_emails


def _y(emails):
    return np.array([e.label == "spam" for e in emails], dtype=int)


def test_threshold_for_fpr_allows_exactly_the_target():
    y = np.array([0] * 1000 + [1] * 10)
    s = np.concatenate([np.linspace(0, 1, 1000), np.full(10, 2.0)])
    t = threshold_for_fpr(y, s, 0.005)
    assert (s[:1000] >= t).sum() == 5
    assert recall_at_fpr(y, s, 0.005)[0] == 1.0


def test_threshold_zero_target_and_mask():
    y = np.array([0, 0, 0, 0, 1])
    s = np.array([0.9, 0.1, 0.2, 0.3, 0.95])
    mask = np.array([False, True, True, True, True])
    t = threshold_for_fpr(y, s, 0.0, mask)
    assert 0.3 < t <= 0.9
    assert threshold_for_fpr(y, s, 0.0) > 0.9


def test_threshold_requires_ham():
    with pytest.raises(ValueError):
        threshold_for_fpr(np.array([1, 1]), np.array([0.1, 0.2]), 0.01)


def test_metrics_at():
    m = metrics_at(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5
    empty = metrics_at(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
    assert empty["precision"] is None and empty["recall"] is None and empty["fpr"] == 0.0


def test_report_for_groups_by_source():
    emails = synthetic_emails(5, 5, source="a") + synthetic_emails(5, 5, source="b", seed=1)
    y = _y(emails)
    p = y * 0.7 + 0.1  # spam 0.8, ham 0.1
    r = report_for(emails, y, p, 0.5, 0.85)
    assert set(r) == {"a", "b", "all"}
    assert r["a"]["flag"]["recall"] == 1.0 and r["a"]["move"]["recall"] == 0.0
    assert r["all"]["n"] == 20 and r["all"]["roc_auc"] == 1.0
    assert r["all"]["pr_curve"] and r["all"]["roc_curve"]


def test_select_model_ranks_candidates():
    emails = synthetic_emails(60, 90, seed=12)
    result = select_model(emails, _y(emails), is_user=np.zeros(150, bool),
                          candidates=[Candidate("logreg"), Candidate("cnb", alpha=0.3)],
                          folds=3, fp_target=0.05, min_df=1)
    assert result.best in (Candidate("logreg"), Candidate("cnb", alpha=0.3))
    assert len(result.table) == 2 and result.user_weight == 1.0
    assert result.table[0]["recall_at_fpr"] >= result.table[1]["recall_at_fpr"]
    assert all(0 <= row["recall_at_fpr"] <= 1 for row in result.table)


def test_select_model_tunes_user_weight():
    base = synthetic_emails(40, 60, seed=13, source="base-x")
    user = synthetic_emails(30, 120, seed=14, source="me")
    emails = base + user
    is_user = np.array([e.source == "me" for e in emails])
    result = select_model(emails, _y(emails), is_user=is_user, weights=(1.0, 3.0),
                          candidates=[Candidate("logreg")], folds=3, fp_target=0.05, min_df=1)
    assert result.user_weight in (1.0, 3.0) and len(result.table) == 2


def test_format_report():
    emails = synthetic_emails(5, 5, source="a")
    y = _y(emails)
    report = {
        "best": {"label": "logreg C=1"}, "user_weight": 3.0,
        "thresholds": {"flag": 0.5, "move": 0.9, "val_fpr_flag": 0.004, "val_fpr_move": 0.001},
        "test": report_for(emails, y, y * 0.7 + 0.1, 0.5, 0.9),
        "cross_corpus": {"spamassassin->enron": {"roc_auc": 0.9, "pr_auc": 0.8}},
        "warnings": ["small validation set"],
    }
    text = format_report(report)
    assert "logreg C=1" in text and "0.40%" in text and "spamassassin->enron" in text
    assert "Warning: small validation set" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.evaluate'`.

- [ ] **Step 3: Implement `spamfilter/evaluate.py`**

```python
"""Metrics, false-positive-controlled thresholds, per-source reports and cross-validated model selection."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from .features import emails_to_frame, make_features, n_word_columns
from .message import Email
from .model import make_classifier


def threshold_for_fpr(y, scores, fpr_target: float, mask=None) -> float:
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    selected = y == 0 if mask is None else (y == 0) & np.asarray(mask, bool)
    ham = s[selected]
    if ham.size == 0:
        raise ValueError("no ham examples to set a threshold on")
    allowed = int(math.floor(fpr_target * ham.size))
    descending = np.sort(ham)[::-1]
    if allowed >= ham.size:
        return float(descending[-1])
    return float(np.nextafter(descending[allowed], np.inf))


def recall_at_fpr(y, scores, fpr_target: float, mask=None) -> tuple[float, float]:
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    threshold = threshold_for_fpr(y, s, fpr_target, mask)
    spam = s[y == 1]
    recall = float((spam >= threshold).mean()) if spam.size else float("nan")
    return recall, threshold


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def metrics_at(y, proba, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(proba, dtype=float) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "fpr": _ratio(fp, fp + tn), "accuracy": _ratio(tp + tn, len(y))}


def _curve(y: np.ndarray, p: np.ndarray, kind: str, max_points: int = 200):
    if len(set(y.tolist())) < 2:
        return None
    if kind == "pr":
        precision, recall, _ = precision_recall_curve(y, p)
        points = list(zip(recall.tolist(), precision.tolist()))
    else:
        fpr, tpr, _ = roc_curve(y, p)
        points = list(zip(fpr.tolist(), tpr.tolist()))
    step = max(1, len(points) // max_points)
    return [[round(a, 4), round(b, 4)] for a, b in points[::step]]


def _block(y: np.ndarray, p: np.ndarray, flag: float, move: float) -> dict:
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)), "n_spam": int(y.sum()), "n_ham": int((y == 0).sum()),
        "flag": metrics_at(y, p, flag), "move": metrics_at(y, p, move),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "pr_curve": _curve(y, p, "pr"), "roc_curve": _curve(y, p, "roc"),
    }


def report_for(emails: Sequence[Email], y, proba, flag: float, move: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype=float)
    sources = np.array([e.source for e in emails])
    out = {s: _block(y[sources == s], p[sources == s], flag, move) for s in dict.fromkeys(sources.tolist())}
    out["all"] = _block(y, p, flag, move)
    return out


@dataclass(frozen=True)
class Candidate:
    algo: str
    C: float = 1.0
    alpha: float = 0.3

    def label(self) -> str:
        return f"cnb alpha={self.alpha:g}" if self.algo == "cnb" else f"{self.algo} C={self.C:g}"


DEFAULT_CANDIDATES = (
    [Candidate("logreg", C=c) for c in (0.3, 1.0, 3.0, 10.0)]
    + [Candidate("cnb", alpha=a) for a in (0.1, 0.3, 1.0)]
    + [Candidate("svm", C=c) for c in (0.1, 0.3, 1.0)]
)


@dataclass
class SelectionResult:
    best: Candidate
    user_weight: float
    table: list[dict]


def _scores(clf, X) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    log_proba = clf.predict_log_proba(X)
    return log_proba[:, 1] - log_proba[:, 0]


def select_model(emails: Sequence[Email], y, *, is_user, weights=(1.0,), candidates=None, folds: int = 5,
                 fp_target: float = 0.005, min_df: int = 2, seed: int = 42, feature_kwargs: dict | None = None,
                 log=None) -> SelectionResult:
    """Stratified, grouped k-fold CV. Features are fit once per fold; every candidate x user weight shares them."""
    candidates = list(candidates or DEFAULT_CANDIDATES)
    y = np.asarray(y).astype(int)
    is_user = np.asarray(is_user, dtype=bool)
    n_splits = min(folds, int(np.bincount(y, minlength=2).min()))
    if n_splits < 2:
        raise ValueError("need at least 2 spam and 2 ham emails for cross-validation")
    frame = emails_to_frame(emails)
    groups = np.array([e.norm_body_hash for e in emails])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: dict[tuple[Candidate, float], list[tuple[float, float]]] = defaultdict(list)
    for fold, (tr, va) in enumerate(splitter.split(frame, y, groups), start=1):
        features = make_features("full", min_df=min_df, **(feature_kwargs or {}))
        X_tr = sparse.csr_matrix(features.fit_transform(frame.iloc[tr]))
        X_va = sparse.csr_matrix(features.transform(frame.iloc[va]))
        n_words = n_word_columns(features)
        user_va = is_user[va]
        mask = user_va if int(((y[va] == 0) & user_va).sum()) >= 50 else None
        for cand in candidates:
            a, b = (X_tr[:, :n_words], X_va[:, :n_words]) if cand.algo == "cnb" else (X_tr, X_va)
            for weight in weights:
                clf = make_classifier(cand.algo, C=cand.C, alpha=cand.alpha)
                clf.fit(a, y[tr], sample_weight=np.where(is_user[tr], weight, 1.0))
                s = _scores(clf, b)
                recall, _ = recall_at_fpr(y[va], s, fp_target, mask)
                scores[(cand, float(weight))].append((recall, float(average_precision_score(y[va], s))))
        if log:
            log(f"  cross-validation fold {fold}/{n_splits} done")
    table = []
    for (cand, weight), values in scores.items():
        arr = np.array(values)
        table.append({"candidate": cand.label(), "algo": cand.algo, "C": cand.C, "alpha": cand.alpha,
                      "user_weight": weight, "recall_at_fpr": float(arr[:, 0].mean()),
                      "recall_std": float(arr[:, 0].std()), "pr_auc": float(arr[:, 1].mean())})
    table.sort(key=lambda row: (row["recall_at_fpr"], row["pr_auc"]), reverse=True)
    best = next(c for c in candidates if c.label() == table[0]["candidate"])
    return SelectionResult(best=best, user_weight=table[0]["user_weight"], table=table)


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100 * value:.2f}%"


def format_source_table(per_source: dict) -> str:
    lines = [f"{'source':<22} {'n':>7} {'spam recall':>11} {'precision':>9} {'ham FPR':>8} {'PR-AUC':>7}"]
    for name, block in per_source.items():
        flag = block["flag"]
        pr_auc = f"{block['pr_auc']:.3f}" if block.get("pr_auc") is not None else "n/a"
        lines.append(f"{name:<22} {block['n']:>7} {_pct(flag['recall']):>11} {_pct(flag['precision']):>9} "
                     f"{_pct(flag['fpr']):>8} {pr_auc:>7}")
    return "\n".join(lines)


def format_report(report: dict) -> str:
    thresholds = report["thresholds"]
    lines = [
        f"Model: {report['best']['label']}  (weight on personal mail: {report['user_weight']:g})",
        f"Flag threshold: p >= {thresholds['flag']:.4f}   validation ham FPR {_pct(thresholds['val_fpr_flag'])}",
        f"Move threshold: p >= {thresholds['move']:.4f}   validation ham FPR {_pct(thresholds['val_fpr_move'])}",
        "",
        "Held-out test results at the flag threshold:",
        format_source_table(report["test"]),
    ]
    if report.get("cross_corpus"):
        lines += ["", "Cross-corpus check (ranking quality on a corpus never trained on):"]
        lines += [f"  {name}: ROC-AUC {v['roc_auc']:.3f}, PR-AUC {v['pr_auc']:.3f}"
                  for name, v in report["cross_corpus"].items()]
    lines += [f"Warning: {w}" for w in report.get("warnings", [])]
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit and push**

```bash
$GIT add spamfilter/evaluate.py tests/test_evaluate.py
$GIT commit -m "Add FPR-controlled thresholds, per-source reports and CV model selection"
$GIT push
```

---

### Task 12: Training orchestration

**Files:**
- Create: `spamfilter/training.py`, `tests/test_training.py`
- Modify: `tests/conftest.py` (add the `synthetic_stores` fixture)

**Interfaces:**
- Consumes: `dataset.read_dataset`, `dataset.dataset_exists`, `dataset.dedupe`, `dataset.split_base`, `dataset.split_user`, `dataset.labels_of`, `evaluate.select_model`, `evaluate.threshold_for_fpr`, `evaluate.report_for`, `evaluate.Candidate`, `model.LinearTextModel`, `model.Bundle`.
- Produces:
  - `BASE_DATASETS = ("base-spamassassin", "base-enron")`
  - `TrainingError(Exception)`
  - `TrainOptions(datasets=[], include_base=True, fp_target=0.005, move_fp_target=0.001, user_weight="auto", folds=5, min_df=2, seed=42, candidates=None, cross_corpus=True, feature_kwargs=None)`
  - `train(opts: TrainOptions, log=print) -> Bundle`. `bundle.report` keys: `best, user_weight, selection, thresholds (flag, move, val_fpr_flag, val_fpr_move, set_on), test, dedupe, split (base/user → train/val/test/time_based), warnings, cross_corpus (optional), train_seconds`. `bundle.manifest` keys: `datasets, counts, options, trained_at`.
  - Fixture `synthetic_stores` (in `tests/conftest.py`): writes datasets `base-spamassassin` (60 spam/90 ham), `base-enron` (60/90) and `me` (40/260, hourly dates).

- [ ] **Step 1: Add the `synthetic_stores` fixture to `tests/conftest.py`**

Append:

```python
@pytest.fixture
def synthetic_stores(spam_home):
    from spamfilter.dataset import write_dataset
    from spamfilter.message import ParseResult
    from tests.builders import synthetic_emails

    for name, (n_spam, n_ham, seed) in {"base-spamassassin": (60, 90, 20), "base-enron": (60, 90, 21),
                                        "me": (40, 260, 22)}.items():
        write_dataset(name, [ParseResult(e) for e in synthetic_emails(n_spam, n_ham, seed=seed, source=name)],
                      manifest={"importer": "test"}, force=True)
    return spam_home
```

- [ ] **Step 2: Write the failing tests `tests/test_training.py`**

```python
import pytest

from spamfilter.dataset import write_dataset
from spamfilter.evaluate import Candidate
from spamfilter.message import ParseResult
from spamfilter.training import TrainingError, TrainOptions, train
from tests.builders import synthetic_emails

FAST = dict(folds=3, min_df=1, candidates=[Candidate("logreg"), Candidate("cnb")],
            feature_kwargs={"max_char_features": 5000})


def _quiet(*_args):
    pass


def test_train_with_base_and_personal_mail(synthetic_stores):
    bundle = train(TrainOptions(datasets=["me"], fp_target=0.02, move_fp_target=0.01, **FAST), log=_quiet)
    report = bundle.report
    assert set(report["test"]) == {"base-spamassassin", "base-enron", "me", "all"}
    assert report["split"]["user"]["time_based"] is True
    assert report["thresholds"]["val_fpr_flag"] <= 0.02
    assert report["thresholds"]["val_fpr_move"] <= 0.01
    assert bundle.move_threshold >= bundle.flag_threshold
    assert report["user_weight"] in (1.0, 3.0, 10.0)
    assert {row["user_weight"] for row in report["selection"]} == {1.0, 3.0, 10.0}
    assert set(report["cross_corpus"]) == {"spamassassin->enron", "enron->spamassassin"}
    assert bundle.manifest["datasets"] == ["base-spamassassin", "base-enron", "me"]
    assert report["test"]["all"]["roc_auc"] > 0.95


def test_train_base_only(synthetic_stores):
    bundle = train(TrainOptions(cross_corpus=False, fp_target=0.02, **FAST), log=_quiet)
    assert "me" not in bundle.report["test"] and "cross_corpus" not in bundle.report
    assert bundle.report["user_weight"] == 1.0


def test_one_class_dataset_cannot_train_alone(synthetic_stores):
    write_dataset("hamonly", [ParseResult(e) for e in synthetic_emails(0, 50, seed=23, source="hamonly")],
                  manifest={})
    with pytest.raises(TrainingError, match="both spam and ham"):
        train(TrainOptions(datasets=["hamonly"], include_base=False, **FAST), log=_quiet)


def test_missing_base_corpora_are_explained(spam_home):
    with pytest.raises(TrainingError, match="corpora download"):
        train(TrainOptions(**FAST), log=_quiet)


def test_unknown_dataset(synthetic_stores):
    with pytest.raises(TrainingError, match="no dataset named 'nope'"):
        train(TrainOptions(datasets=["nope"], **FAST), log=_quiet)


def test_nothing_to_train_on(synthetic_stores):
    with pytest.raises(TrainingError, match="nothing to train on"):
        train(TrainOptions(include_base=False, **FAST), log=_quiet)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_training.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.training'`.

- [ ] **Step 4: Implement `spamfilter/training.py`**

```python
"""Train a model bundle: load, dedupe, split, select, fit, calibrate, set thresholds, test."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .dataset import dataset_exists, dedupe, labels_of, read_dataset, split_base, split_user
from .evaluate import Candidate, report_for, select_model, threshold_for_fpr
from .message import HAM, SPAM, Email
from .model import Bundle, LinearTextModel

BASE_DATASETS = ("base-spamassassin", "base-enron")


class TrainingError(Exception):
    pass


@dataclass
class TrainOptions:
    datasets: list[str] = field(default_factory=list)
    include_base: bool = True
    fp_target: float = 0.005
    move_fp_target: float = 0.001
    user_weight: str | float = "auto"
    folds: int = 5
    min_df: int = 2
    seed: int = 42
    candidates: list[Candidate] | None = None
    cross_corpus: bool = True
    feature_kwargs: dict | None = None


def _load(names, *, personal: bool) -> list[Email]:
    out: list[Email] = []
    for name in names:
        if not dataset_exists(name):
            if personal:
                raise TrainingError(f"no dataset named '{name}'; see `spamfilter datasets list`")
            raise TrainingError("the base corpora are not prepared yet; run `spamfilter corpora download` first")
        for email in read_dataset(name):
            if email.label in (SPAM, HAM):
                email.source = name
                out.append(email)
    return out


def _cross_corpus(sa: list[Email], en: list[Email], cand: Candidate, opts: TrainOptions, log) -> dict:
    out = {}
    for name, (train_set, test_set) in {"spamassassin->enron": (sa, en), "enron->spamassassin": (en, sa)}.items():
        model = LinearTextModel(cand.algo, C=cand.C, alpha=cand.alpha, min_df=opts.min_df,
                                feature_kwargs=opts.feature_kwargs)
        model.fit(train_set, labels_of(train_set))
        p, y = model.predict_proba(test_set), labels_of(test_set)
        out[name] = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}
        log(f"  cross-corpus {name}: ROC-AUC {out[name]['roc_auc']:.3f}")
    return out


def train(opts: TrainOptions, log=print) -> Bundle:
    started = time.monotonic()
    base_names = list(BASE_DATASETS) if opts.include_base else []
    if not base_names and not opts.datasets:
        raise TrainingError("nothing to train on: give --datasets or keep the base corpora")
    personal = _load(opts.datasets, personal=True)
    base = _load(base_names, personal=False)
    log(f"Loaded {len(base)} base and {len(personal)} personal emails")

    kept, dedupe_stats = dedupe(personal + base)  # personal first, so their copy of a duplicate is kept
    if len(set(labels_of(kept).tolist())) < 2:
        raise TrainingError("training data must contain both spam and ham; a one-class dataset can only be "
                            "trained together with the base corpora")
    user_names = set(opts.datasets)
    user_emails = [e for e in kept if e.source in user_names]
    base_emails = [e for e in kept if e.source not in user_names]
    splits = {}
    if base_emails:
        splits["base"] = split_base(base_emails, seed=opts.seed)
    if user_emails:
        splits["user"] = split_user(user_emails, seed=opts.seed)
    train_set = [e for s in splits.values() for e in s.train]
    val = [e for s in splits.values() for e in s.val]
    test = [e for s in splits.values() for e in s.test]
    if not val or not test:
        raise TrainingError("not enough data to hold out validation and test sets")
    y_train, y_val, y_test = labels_of(train_set), labels_of(val), labels_of(test)
    is_user_train = np.array([e.source in user_names for e in train_set])

    weights: tuple[float, ...] = (1.0,)
    if user_emails:
        if opts.user_weight == "auto":
            n_user_ham = int(((y_train == 0) & is_user_train).sum())
            n_user_spam = int(((y_train == 1) & is_user_train).sum())
            weights = (1.0, 3.0, 10.0) if (base_emails and n_user_ham >= 100 and n_user_spam >= 20) else (3.0,)
        else:
            weights = (float(opts.user_weight),)

    log(f"Selecting a model on {len(train_set)} training emails ({opts.folds}-fold cross-validation)")
    selection = select_model(train_set, y_train, is_user=is_user_train, weights=weights,
                             candidates=opts.candidates, folds=opts.folds, fp_target=opts.fp_target,
                             min_df=opts.min_df, seed=opts.seed, feature_kwargs=opts.feature_kwargs, log=log)
    best = selection.best
    log(f"Best: {best.label()} (weight on personal mail {selection.user_weight:g}); fitting the final model")
    model = LinearTextModel(best.algo, C=best.C, alpha=best.alpha, min_df=opts.min_df,
                            calibration="isotonic" if len(train_set) > 10_000 else "sigmoid",
                            feature_kwargs=opts.feature_kwargs)
    model.fit(train_set, y_train, sample_weight=np.where(is_user_train, selection.user_weight, 1.0))

    p_val = model.predict_proba(val)
    user_val = np.array([e.source in user_names for e in val])
    warnings: list[str] = []
    use_personal = int(((y_val == 0) & user_val).sum()) >= 200
    if user_emails and not use_personal:
        warnings.append("fewer than 200 personal ham emails in validation, so thresholds were set on the "
                        "combined validation set")
    mask = user_val if use_personal else None
    n_val_ham = int(((y_val == 0) & (user_val if use_personal else True)).sum())
    if n_val_ham < 1 / opts.move_fp_target:
        warnings.append(f"only {n_val_ham} ham emails in validation; the {opts.move_fp_target:.1%} move target "
                        "cannot be measured precisely")
    flag = threshold_for_fpr(y_val, p_val, opts.fp_target, mask)
    move = max(flag, threshold_for_fpr(y_val, p_val, opts.move_fp_target, mask))
    ham_sel = (y_val == 0) & (mask if mask is not None else True)
    thresholds = {"flag": flag, "move": move,
                  "val_fpr_flag": float((p_val[ham_sel] >= flag).mean()),
                  "val_fpr_move": float((p_val[ham_sel] >= move).mean()),
                  "set_on": "personal ham" if use_personal else "combined validation"}

    p_test = model.predict_proba(test)
    report = {
        "best": {"label": best.label(), "algo": best.algo, "C": best.C, "alpha": best.alpha},
        "user_weight": selection.user_weight,
        "selection": selection.table,
        "thresholds": thresholds,
        "test": report_for(test, y_test, p_test, flag, move),
        "dedupe": asdict(dedupe_stats),
        "split": {name: {"train": len(s.train), "val": len(s.val), "test": len(s.test), "time_based": s.time_based}
                  for name, s in splits.items()},
        "warnings": warnings,
    }
    if opts.cross_corpus and opts.include_base:
        sa = [e for e in base_emails if e.source == "base-spamassassin"]
        en = [e for e in base_emails if e.source == "base-enron"]
        if sa and en:
            log("Cross-corpus check")
            report["cross_corpus"] = _cross_corpus(sa, en, best, opts, log)
    report["train_seconds"] = round(time.monotonic() - started, 1)
    names = base_names + list(opts.datasets)
    manifest = {
        "datasets": names,
        "counts": {n: sum(1 for e in kept if e.source == n) for n in names},
        "options": {k: v for k, v in asdict(opts).items() if k != "candidates"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return Bundle(model=model, flag_threshold=flag, move_threshold=move, report=report, manifest=manifest)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_training.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit and push**

```bash
$GIT add spamfilter/training.py tests/test_training.py tests/conftest.py
$GIT commit -m "Add training orchestration with personal-mail weighting and FPR thresholds"
$GIT push
```

---

### Task 13: CLI

**Files:**
- Create: `spamfilter/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above. Specifically `config.*`, `corpora.prepare/download/FILES/DATASET_FOR/CorpusError`, `dataset.*` store functions and exceptions, `importers.detect_importer/UnsupportedFormat`, `importers.base.Overrides/build_label_report/iter_labeled`, `parse.parse_bytes`, `model.Bundle/BundleError`, `evaluate.DEFAULT_CANDIDATES/format_report/format_source_table/report_for`, `training.TrainOptions/TrainingError/train`.
- Produces: `cli.main(argv: list[str] | None = None) -> int` (exit codes: 0 ok, 1 known error printed as `error: …` on stderr, 2 usage, 130 interrupted); `cli.build_parser() -> argparse.ArgumentParser`; `cli.resolve_model(name: str | None) -> Path`; console script `spamfilter`.

Deviation from the spec's CLI sketch: the model is chosen with `--model NAME` on `predict` and `evaluate`, not as a positional argument. With the positional form, `predict FILE` was ambiguous with `predict MODEL`.

- [ ] **Step 1: Write the failing tests `tests/test_cli.py`**

```python
import io
import json
import sys

from spamfilter import config
from spamfilter.cli import main
from spamfilter.dataset import dataset_exists, load_manifest
from tests.builders import make_raw, write_mbox


def run(args, capsys):
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _mail(tmp_path):
    root = tmp_path / "mail"
    write_mbox(root / "Inbox", [make_raw("hi", "hello friend"), make_raw("notes", "meeting notes attached")])
    write_mbox(root / "Junk", [make_raw("WIN", "cash prize claim now")])
    return root


def test_import_dry_run_then_real_then_duplicate(tmp_path, capsys):
    root = _mail(tmp_path)
    code, out, _ = run(["import", str(root), "--name", "mine", "--dry-run"], capsys)
    assert code == 0 and "Junk" in out and "spam" in out and "Dry run" in out
    assert not dataset_exists("mine")
    code, out, _ = run(["import", str(root), "--name", "mine"], capsys)
    assert code == 0 and "Imported 3 emails" in out
    assert load_manifest("mine")["labels"] == {"ham": 2, "spam": 1}
    code, _, err = run(["import", str(root), "--name", "mine"], capsys)
    assert code == 1 and "--force" in err


def test_import_rejects_reserved_and_missing(tmp_path, capsys):
    code, _, err = run(["import", str(_mail(tmp_path)), "--name", "base-x"], capsys)
    assert code == 1 and "reserved" in err
    code, _, err = run(["import", str(tmp_path / "nope"), "--name", "x"], capsys)
    assert code == 1 and "does not exist" in err


def test_datasets_commands(tmp_path, capsys):
    run(["import", str(_mail(tmp_path)), "--name", "mine"], capsys)
    code, out, _ = run(["datasets", "list"], capsys)
    assert code == 0 and "mine" in out
    code, out, _ = run(["datasets", "show", "mine"], capsys)
    assert code == 0 and json.loads(out)["count"] == 3
    code, _, err = run(["datasets", "show"], capsys)
    assert code == 1 and "name" in err
    code, out, _ = run(["datasets", "delete", "mine"], capsys)
    assert code == 0 and not dataset_exists("mine")


def test_train_predict_evaluate_models(synthetic_stores, tmp_path, capsys, monkeypatch):
    code, out, err = run(["train", "-o", "t1", "--datasets", "me", "--folds", "3", "--min-df", "1",
                          "--algos", "logreg", "--fp-target", "0.02", "--move-fp-target", "0.01",
                          "--no-cross-corpus"], capsys)
    assert code == 0, err
    assert "Held-out test results" in out and "Saved model" in out
    assert config.load_settings()["active_model"].endswith("t1.model")

    code, out, _ = run(["models", "list"], capsys)
    assert code == 0 and "t1" in out and "(active)" in out

    spam = tmp_path / "spam.eml"
    spam.write_bytes(make_raw("free cash prize winner", "click offer guaranteed cash bonus deal",
                              sender="deals@promo.biz"))
    code, out, _ = run(["predict", str(spam)], capsys)
    assert code == 0 and "P(spam)=" in out and "Subject: free cash prize winner" in out

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(spam.read_bytes())))
    code, out, _ = run(["predict", "-", "--json"], capsys)
    payload = json.loads(out)
    assert code == 0 and payload["verdict"] in ("spam", "suspicious", "ham") and payload["reasons"]

    code, out, _ = run(["evaluate"], capsys)
    assert code == 0 and "Held-out test results" in out
    code, out, _ = run(["evaluate", "--dataset", "me"], capsys)
    assert code == 0 and "me" in out and "spam recall" in out

    notmail = tmp_path / "notes.pdf"
    notmail.write_bytes(b"%PDF-1.4\n%binary")
    code, _, err = run(["predict", str(notmail)], capsys)
    assert code == 1 and "not an email" in err


def test_predict_without_model_explains(tmp_path, capsys):
    eml = tmp_path / "a.eml"
    eml.write_bytes(make_raw())
    code, _, err = run(["predict", str(eml)], capsys)
    assert code == 1 and "spamfilter train" in err


def test_corpora_list(capsys):
    code, out, _ = run(["corpora", "list"], capsys)
    assert code == 0 and "20021010_spam.tar.bz2" in out and "missing" in out and "not prepared" in out


def test_no_command_prints_help(capsys):
    code, out, _ = run([], capsys)
    assert code == 2 and "usage" in out.lower()


# Review Focus 1: a cp1252 console must not crash on non-ASCII folder names or subjects.
def test_output_survives_cp1252_console(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", console)
    root = tmp_path / "mail"
    write_mbox(root / "Входящие", [make_raw("привет 🎉", "текст письма здесь")])
    assert main(["import", str(root), "--name", "ru", "--dry-run"]) == 0
    console.flush()
    assert b"?" in buffer.getvalue()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'spamfilter.cli'`.

- [ ] **Step 3: Implement `spamfilter/cli.py`**

```python
"""The `spamfilter` command line."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__, config, corpora
from .corpora import CorpusError
from .dataset import (DatasetExists, DatasetNotFound, InvalidDatasetName, dataset_exists, delete_dataset,
                      labels_of, list_datasets, load_manifest, read_dataset, write_dataset)
from .evaluate import DEFAULT_CANDIDATES, format_report, format_source_table, report_for
from .importers import UnsupportedFormat, detect_importer
from .importers.base import Overrides, build_label_report, iter_labeled
from .message import HAM, SPAM
from .model import Bundle, BundleError
from .parse import parse_bytes
from .training import TrainingError, TrainOptions, train


class CliError(Exception):
    pass


KNOWN_ERRORS = (CliError, DatasetExists, DatasetNotFound, InvalidDatasetName, UnsupportedFormat, CorpusError,
                TrainingError, BundleError, FileNotFoundError, ImportError)


def _out(text: str = "") -> None:
    print(text, flush=True)


def _counts(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "none"


def _model_file(name: str) -> Path:
    path = Path(name)
    if path.suffix == ".model" and (path.is_absolute() or path.parent != Path(".")):
        return path
    return config.models_dir() / (name if name.endswith(".model") else name + ".model")


def resolve_model(name: str | None) -> Path:
    if name is None:
        active = config.load_settings().get("active_model")
        if not active:
            raise BundleError("no model selected. Train one with `spamfilter train -o NAME` "
                              "(run `spamfilter corpora download` first).")
        return Path(active)
    path = _model_file(name)
    if not path.exists():
        raise BundleError(f"no model named '{name}'; see `spamfilter models list`")
    return path


def cmd_corpora_download(a) -> int:
    for corpus in [a.only] if a.only else ["spamassassin", "enron", "spambase"]:
        if corpus == "spambase":
            corpora.download("spambase", log=_out)
            _out("spambase: downloaded (used by the benchmark notebook)")
            continue
        manifest = corpora.prepare(corpus, log=_out)
        _out(f"{manifest['name']}: {manifest['count']} emails ({_counts(manifest['labels'])}); "
             f"skipped {sum(manifest['skipped'].values())}")
    return 0


def cmd_corpora_list(a) -> int:
    for cf in corpora.FILES:
        present = (config.corpora_dir() / cf.filename).exists()
        _out(f"{cf.corpus:<13} {cf.filename:<32} {'present' if present else 'missing'}")
    for name in corpora.DATASET_FOR.values():
        _out(f"{name}: " + (f"{load_manifest(name)['count']} emails" if dataset_exists(name) else "not prepared"))
    return 0


def cmd_import(a) -> int:
    if a.name.startswith("base-"):
        raise InvalidDatasetName("dataset names starting with 'base-' are reserved for the public corpora")
    if dataset_exists(a.name) and not a.force:
        raise DatasetExists(f"dataset '{a.name}' already exists; use --force to replace it")
    path = Path(a.path)
    importer = detect_importer(path, a.format)
    ov = Overrides(spam=a.spam_folder or [], ham=a.ham_folder or [], skip=a.skip_folder or [],
                   include_sent=a.include_sent, label=a.label)
    _out(f"Format: {importer.name}")
    _out(build_label_report(importer, path, ov).format())
    if a.dry_run:
        _out("Dry run: nothing imported.")
        return 0
    started = time.monotonic()
    manifest = write_dataset(a.name, iter_labeled(importer, path, ov, source=a.name),
                             manifest={"importer": importer.name, "source_path": str(path.resolve()),
                                       "overrides": asdict(ov)},
                             force=a.force, progress=lambda n: _out(f"  {n} messages..."))
    _out(f"Imported {manifest['count']} emails into '{a.name}' ({_counts(manifest['labels'])}) "
         f"in {time.monotonic() - started:.0f}s")
    if manifest["skipped"]:
        _out("Skipped: " + _counts(manifest["skipped"]))
    if not manifest["labels"].get(SPAM) or not manifest["labels"].get(HAM):
        _out("Note: this dataset has only one class. Train it together with the base corpora, not alone.")
    return 0


def cmd_datasets(a) -> int:
    if a.action == "list":
        rows = list_datasets()
        if not rows:
            _out("No datasets. Run `spamfilter corpora download` or import mail with `spamfilter import`.")
        for m in rows:
            _out(f"{m['name']:<24} {m['count']:>8}  {_counts(m.get('labels', {}))}")
        return 0
    if not a.name:
        raise CliError(f"`datasets {a.action}` needs a dataset name")
    if a.action == "show":
        _out(json.dumps(load_manifest(a.name), indent=2, ensure_ascii=False))
    else:
        delete_dataset(a.name)
        _out(f"Deleted dataset '{a.name}'.")
    return 0


def cmd_train(a) -> int:
    candidates = None
    if a.algos:
        wanted = {s.strip() for s in a.algos.split(",") if s.strip()}
        candidates = [c for c in DEFAULT_CANDIDATES if c.algo in wanted]
        if not candidates:
            raise CliError(f"unknown --algos {a.algos!r}; choose from logreg, cnb, svm")
    user_weight = a.user_weight if a.user_weight == "auto" else float(a.user_weight)
    opts = TrainOptions(datasets=[s for s in (a.datasets or "").split(",") if s], include_base=not a.no_base,
                        fp_target=a.fp_target, move_fp_target=a.move_fp_target, user_weight=user_weight,
                        folds=a.folds, min_df=a.min_df, candidates=candidates, cross_corpus=not a.no_cross_corpus)
    bundle = train(opts, log=_out)
    out_path = _model_file(a.output)
    bundle.save(out_path)
    settings = config.load_settings()
    if a.activate or not settings.get("active_model"):
        settings["active_model"] = str(out_path)
        config.save_settings(settings)
    _out("")
    _out(format_report(bundle.report))
    _out(f"Saved model to {out_path}")
    return 0


def cmd_models(a) -> int:
    active = config.load_settings().get("active_model")
    if a.action == "list":
        files = sorted(config.models_dir().glob("*.model"))
        if not files:
            _out("No models yet. Train one with `spamfilter train -o NAME`.")
        for f in files:
            mark = "  (active)" if active and Path(active) == f else ""
            _out(f"{f.stem:<24} {f.stat().st_size // 1024:>8} KB{mark}")
        return 0
    if not a.name:
        raise CliError("`models activate` needs a model name")
    path = resolve_model(a.name)
    Bundle.load(path)
    settings = config.load_settings()
    settings["active_model"] = str(path)
    config.save_settings(settings)
    _out(f"Active model: {path.stem}")
    return 0


def cmd_evaluate(a) -> int:
    bundle = Bundle.load(resolve_model(a.model))
    if not a.dataset:
        _out(format_report(bundle.report))
        return 0
    emails = [e for e in read_dataset(a.dataset) if e.label in (SPAM, HAM)]
    if not emails:
        raise CliError(f"dataset '{a.dataset}' has no labeled emails")
    for e in emails:
        e.source = a.dataset
    y = labels_of(emails)
    report = report_for(emails, y, bundle.model.predict_proba(emails), bundle.flag_threshold, bundle.move_threshold)
    _out(format_source_table({a.dataset: report[a.dataset]}))
    return 0


def cmd_predict(a) -> int:
    bundle = Bundle.load(resolve_model(a.model))
    raw = sys.stdin.buffer.read() if a.file == "-" else Path(a.file).read_bytes()
    result = parse_bytes(raw, source="predict")
    if result.email is None:
        raise CliError(f"{a.file} is not an email ({result.error})")
    p_spam = float(bundle.model.predict_proba([result.email])[0])
    verdict = bundle.verdict(p_spam)
    reasons = bundle.model.explain(result.email, top_k=a.top)
    if a.json:
        _out(json.dumps({"verdict": verdict, "p_spam": round(p_spam, 4),
                         "flag_threshold": round(bundle.flag_threshold, 4),
                         "move_threshold": round(bundle.move_threshold, 4),
                         "subject": result.email.subject,
                         "reasons": [r.to_dict() for r in reasons]}, ensure_ascii=False))
        return 0
    _out(f"{verdict.upper()}  P(spam)={p_spam:.3f}  (flag at >= {bundle.flag_threshold:.3f}, "
         f"move at >= {bundle.move_threshold:.3f})")
    _out(f"Subject: {result.email.subject}")
    for r in reasons:
        _out(f"  {'+' if r.weight > 0 else '-'} {r.feature}  ({r.weight:+.3f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spamfilter", description="Local, personalizable spam filter.")
    parser.add_argument("--version", action="version", version=f"spamfilter {__version__}")
    sub = parser.add_subparsers(dest="command")

    corp = sub.add_parser("corpora", help="download and prepare the public training corpora")
    corp_sub = corp.add_subparsers(dest="action", required=True)
    dl = corp_sub.add_parser("download", help="download, verify and convert the corpora")
    dl.add_argument("--only", choices=["spamassassin", "enron", "spambase"])
    dl.set_defaults(func=cmd_corpora_download)
    corp_sub.add_parser("list", help="show corpus files and base datasets").set_defaults(func=cmd_corpora_list)

    imp = sub.add_parser("import", help="import an email export as a labeled dataset")
    imp.add_argument("path")
    imp.add_argument("--name", required=True)
    imp.add_argument("--format", default="auto", choices=["auto", "takeout", "mbox", "eml", "pst"])
    imp.add_argument("--spam-folder", action="append", help="treat this folder as spam (repeatable)")
    imp.add_argument("--ham-folder", action="append", help="treat this folder as ham (repeatable)")
    imp.add_argument("--skip-folder", action="append", help="ignore this folder (repeatable)")
    imp.add_argument("--include-sent", action="store_true", help="count Sent mail as ham")
    imp.add_argument("--label", choices=[SPAM, HAM], help="give every message this label")
    imp.add_argument("--dry-run", action="store_true", help="only print the label report")
    imp.add_argument("--force", action="store_true", help="replace an existing dataset")
    imp.set_defaults(func=cmd_import)

    ds = sub.add_parser("datasets", help="list, show or delete datasets")
    ds.add_argument("action", choices=["list", "show", "delete"])
    ds.add_argument("name", nargs="?")
    ds.set_defaults(func=cmd_datasets)

    tr = sub.add_parser("train", help="train a model bundle")
    tr.add_argument("-o", "--output", required=True, help="model name or path to a .model file")
    tr.add_argument("--datasets", help="comma-separated personal datasets")
    tr.add_argument("--no-base", action="store_true", help="do not use the public base corpora")
    tr.add_argument("--fp-target", type=float, default=0.005, help="max share of real mail flagged (default 0.005)")
    tr.add_argument("--move-fp-target", type=float, default=0.001,
                    help="max share of real mail moved to Junk (default 0.001)")
    tr.add_argument("--user-weight", default="auto", help="'auto' or a number (weight on personal mail)")
    tr.add_argument("--folds", type=int, default=5)
    tr.add_argument("--min-df", type=int, default=2)
    tr.add_argument("--algos", help="restrict candidates, e.g. logreg,cnb")
    tr.add_argument("--no-cross-corpus", action="store_true")
    tr.add_argument("--activate", action="store_true", help="make this the active model")
    tr.set_defaults(func=cmd_train)

    mo = sub.add_parser("models", help="list models or choose the active one")
    mo.add_argument("action", choices=["list", "activate"])
    mo.add_argument("name", nargs="?")
    mo.set_defaults(func=cmd_models)

    ev = sub.add_parser("evaluate", help="show a model's test report or score a dataset")
    ev.add_argument("--model")
    ev.add_argument("--dataset")
    ev.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("predict", help="classify one email (.eml file or - for stdin)")
    pr.add_argument("file")
    pr.add_argument("--model")
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--top", type=int, default=8, help="number of reasons to show")
    pr.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return args.func(args) or 0
    except KNOWN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all tests pass, no errors.

- [ ] **Step 6: Commit and push**

```bash
$GIT add spamfilter/cli.py tests/test_cli.py
$GIT commit -m "Add spamfilter CLI: corpora, import, datasets, train, models, evaluate, predict"
$GIT push
```

---

### Task 14: Real-corpus benchmark and README

**Files:**
- Create: `docs/benchmarks/2026-09-30-base-model.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the installed `spamfilter` console script.
- Produces: measured base-model results committed to the repo, and a README quick start.

- [ ] **Step 1: Download and prepare the real corpora**

Run: `.venv/Scripts/spamfilter corpora download`
Expected: `base-spamassassin: N emails (ham …, spam …)` and `base-enron: N emails (…)`. SpamAssassin is about 9.3k files before dedupe (with overlaps); Enron is 33,716 (16,545 ham / 17,171 spam). Then `spambase: downloaded`. Files already in `%LOCALAPPDATA%\spamfilter\corpora` are verified by checksum, not downloaded again.

- [ ] **Step 2: Train the base model and time it**

Run: `time .venv/Scripts/spamfilter train -o base --activate`
Expected: it completes. Record the full printed report and the wall-clock time. If it takes longer than 60 minutes, stop and report the timing to your human partner instead of changing defaults silently.

- [ ] **Step 3: Sanity-check predictions on two hand-written messages**

```bash
.venv/Scripts/python - <<'EOF'
from tests.builders import make_raw
open("spam_check.eml", "wb").write(make_raw("URGENT: claim your $1,000,000 prize",
    "Congratulations winner! Click http://bit.ly/claim-now to receive your cash prize today!!!",
    sender="Prize Dept <winner@lucky-prize.biz>", reply_to="claims@another.biz"))
open("ham_check.eml", "wb").write(make_raw("Notes from Tuesday's project meeting",
    "Hi team, attached are the notes from Tuesday. Please review the budget section before Friday. Thanks, Sam",
    sender="Sam <sam@example.org>"))
EOF
.venv/Scripts/spamfilter predict spam_check.eml
.venv/Scripts/spamfilter predict ham_check.eml
rm spam_check.eml ham_check.eml
```
Expected: the first prints `SPAM` or `SUSPICIOUS` with spam reasons, and the second prints `HAM`. Record both outputs. If either is wrong, report it; don't tune thresholds by hand.

- [ ] **Step 4: Write `docs/benchmarks/2026-09-30-base-model.md`**

Paste the **actual** numbers from Steps 1–3 into this structure. Every value comes from the command output; nothing is estimated:

```markdown
# Base model benchmark: 2026-09-30

Machine: <CPU model from `wmic cpu get name`>, <RAM>, Windows 11, Python 3.14.7, scikit-learn 1.9.1

## Data (after deduplication)
<paste the "Loaded … base emails" line and the dedupe numbers from `spamfilter evaluate` / report>

## Selected model
<paste the "Model:", "Flag threshold:" and "Move threshold:" lines>

## Held-out test results
<paste the source table>

## Cross-corpus check
<paste the two cross-corpus lines>

## Timing
Corpus download + conversion: <m:ss>. Training (5-fold selection + final fit + cross-corpus): <m:ss>.

## Spot checks
<paste the two predict outputs>
```

- [ ] **Step 5: Update `README.md` with a quick start**

Replace `README.md` with the content between the `~~~~` fences below (the fences themselves are not part of the file):

~~~~markdown
# SpamReport

- `FinalReport/`: the original NMSU Data Mining class project (Spambase study), kept as an archive.
- `spamfilter/`: a local, personalizable spam filter. It trains on public raw-email corpora (SpamAssassin + Enron-Spam) and on your own mail exports. Everything runs on your machine; no mail leaves it.

## Quick start (Windows, Python 3.12+)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[pst]"
.venv/Scripts/spamfilter corpora download          # ~40 MB, checksum-verified
.venv/Scripts/spamfilter train -o base --activate  # base model
.venv/Scripts/spamfilter predict some_message.eml
```

## Train on your own mail

```bash
# Gmail Takeout, Thunderbird/Apple mbox folders, a folder of .eml files (spam/ and ham/), or an Outlook .pst
.venv/Scripts/spamfilter import "C:/path/to/export" --name mymail --dry-run   # check the spam/ham/skip mapping
.venv/Scripts/spamfilter import "C:/path/to/export" --name mymail
.venv/Scripts/spamfilter train -o personal --datasets mymail --activate
.venv/Scripts/spamfilter evaluate
```

The label report shows how each folder is treated. Adjust it with `--spam-folder`, `--ham-folder`, `--skip-folder` and `--include-sent`.

## Results

See `docs/benchmarks/2026-09-30-base-model.md` for the measured base-model results.

## Privacy and safety

- A model trained on your mail contains words from your mail. Keep `.model` files private; the base-only model is safe to share.
- Only load `.model` files you created. They are Python pickles and can run code when loaded.
- `.gitignore` blocks `*.mbox`, `*.pst`, `*.eml` and `*.model` so real mail can't be committed by accident.

Design: `docs/superpowers/specs/2026-09-30-spamfilter-design.md`.
~~~~

- [ ] **Step 6: Commit and push**

```bash
$GIT add README.md docs/benchmarks/2026-09-30-base-model.md
$GIT commit -m "Record measured base-model benchmark and add README quick start"
$GIT push
```

---

## Plans 2–4 (written after Plan 1 lands)

- **Plan 2, live IMAP (spec §5.8):** `spamfilter/imap.py` + `imap setup|test|import|scan|undo`, keyring, in-process fake IMAP server tests.
- **Plan 3, web UI (spec §5.9):** FastAPI app with a token plus CSRF, datasets/train/models/classify/IMAP pages, confirm-guarded move and undo.
- **Plan 4, benchmark notebook + final README (spec §9, milestone 8).**
