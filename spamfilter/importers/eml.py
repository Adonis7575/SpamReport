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
