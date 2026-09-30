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
