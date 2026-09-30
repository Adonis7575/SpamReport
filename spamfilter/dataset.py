"""Datasets on disk (gzipped JSONL + manifest), deduplication and train/validation/test splits."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from . import __version__
from .config import datasets_dir
from .message import SPAM, Email, ParseResult

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
