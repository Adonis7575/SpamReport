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
