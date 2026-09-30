import pytest


@pytest.fixture(autouse=True)
def spam_home(tmp_path, monkeypatch):
    """Every test gets its own cache root, so the real %LOCALAPPDATA% cache is never touched."""
    home = tmp_path / "spamfilter-home"
    monkeypatch.setenv("SPAMFILTER_HOME", str(home))
    return home
