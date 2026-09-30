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
