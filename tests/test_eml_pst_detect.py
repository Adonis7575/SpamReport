import sys
from datetime import datetime
from pathlib import Path

import pytest

from spamfilter.importers import UnsupportedFormat, detect_importer
from spamfilter.importers.base import Overrides, build_label_report, iter_labeled
from spamfilter.importers.eml import EmlImporter
from spamfilter.importers.mbox import MboxImporter
from spamfilter.importers.pst import PstImporter
from tests.builders import make_raw, write_mbox


def test_eml_tree(tmp_path):
    (tmp_path / "spam" / "2024").mkdir(parents=True)
    (tmp_path / "ham").mkdir()
    (tmp_path / "spam" / "2024" / "a.eml").write_bytes(make_raw("win", "cash cash cash"))
    (tmp_path / "ham" / "b.eml").write_bytes(make_raw("hi", "see you soon"))
    (tmp_path / "c.eml").write_bytes(make_raw("loose", "unsorted message"))
    imp = detect_importer(tmp_path)
    assert isinstance(imp, EmlImporter)
    rep = build_label_report(imp, tmp_path, Overrides())
    assert set(rep.rows) == {("spam", 1, "spam"), ("ham", 1, "ham"), (".", 1, "skip")}
    assert build_label_report(imp, tmp_path / "ham", Overrides(label="ham")).rows == [(".", 1, "ham")]


class FakeMsg:
    def __init__(self, subject, body, headers=None, html=None):
        self.subject = subject
        self.plain_text_body = body.encode()
        self.html_body = html.encode() if html else None
        self.transport_headers = headers
        self.sender_name = "Sender"
        self.delivery_time = datetime(2024, 5, 1, 12, 0)


class BrokenMsg:
    @property
    def transport_headers(self):
        raise OSError("corrupt item")


class FakeFolder:
    def __init__(self, name, messages=(), folders=()):
        self.name = name
        self._messages = list(messages)
        self._folders = list(folders)

    @property
    def number_of_sub_messages(self):
        return len(self._messages)

    def get_sub_message(self, i):
        return self._messages[i]

    @property
    def number_of_sub_folders(self):
        return len(self._folders)

    def get_sub_folder(self, i):
        return self._folders[i]


def _fake_pst():
    headers = ("From: Boss <boss@corp.example>\r\nSubject: Q3 numbers\r\n"
               "Date: Wed, 01 May 2024 09:00:00 +0000\r\nContent-Type: text/plain\r\n\r\n")
    return FakeFolder("", folders=[FakeFolder("Top of Personal Folders", folders=[
        FakeFolder("Inbox", [FakeMsg("Q3 numbers", "numbers attached", headers=headers)]),
        FakeFolder("Junk Email", [FakeMsg("You won", "claim prize", html="<p>claim <b>prize</b></p>")]),
        FakeFolder("Deleted Items", [FakeMsg("old", "old")]),
        FakeFolder("Broken", [BrokenMsg()]),
    ])])


def test_pst_walk_labels_and_reconstruction():
    root = _fake_pst()
    imp = PstImporter(opener=lambda p: root)
    rep = build_label_report(imp, Path("x.pst"), Overrides())
    assert {(f.split("/")[-1], lab) for f, _, lab in rep.rows} == {
        ("Inbox", "ham"), ("Junk Email", "spam"), ("Deleted Items", "skip"), ("Broken", "ham")}
    results = list(iter_labeled(imp, Path("x.pst"), Overrides(), source="pst"))
    good = [r.email for r in results if r.email]
    inbox = next(e for e in good if e.folder.endswith("Inbox"))
    assert inbox.folder == "Top of Personal Folders/Inbox"
    assert inbox.from_addr == "boss@corp.example" and inbox.subject == "Q3 numbers"
    assert "numbers attached" in inbox.text
    junk = next(e for e in good if e.folder.endswith("Junk Email"))
    assert junk.subject == "You won" and "claim prize" in junk.text and junk.date.year == 2024
    assert [r.error for r in results if r.email is None] == ["empty"]


def test_pst_without_library_explains_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypff", None)
    with pytest.raises(ImportError, match="pip install"):
        list(PstImporter().iter_raw(Path("x.pst")))


def test_detect_formats(tmp_path):
    write_mbox(tmp_path / "box" / "Inbox", [make_raw()])
    assert isinstance(detect_importer(tmp_path / "box"), MboxImporter)
    pst = tmp_path / "a.pst"
    pst.write_bytes(b"!BDN")
    assert isinstance(detect_importer(pst), PstImporter)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UnsupportedFormat):
        detect_importer(empty)
    with pytest.raises(FileNotFoundError):
        detect_importer(tmp_path / "missing")
    with pytest.raises(UnsupportedFormat):
        detect_importer(tmp_path / "box", fmt="zip")
    takeout = detect_importer(tmp_path / "box", fmt="takeout")
    assert takeout.name == "takeout"
