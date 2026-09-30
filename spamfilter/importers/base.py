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
