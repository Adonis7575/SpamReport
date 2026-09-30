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
