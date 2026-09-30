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
