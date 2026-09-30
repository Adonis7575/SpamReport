import pytest


@pytest.fixture(autouse=True)
def spam_home(tmp_path, monkeypatch):
    """Every test gets its own cache root, so the real %LOCALAPPDATA% cache is never touched."""
    home = tmp_path / "spamfilter-home"
    monkeypatch.setenv("SPAMFILTER_HOME", str(home))
    return home


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
