from spamfilter.importers.base import Overrides, build_label_report, iter_labeled
from spamfilter.importers.mbox import MboxImporter, gmail_labels
from tests.builders import make_raw, write_mbox


def test_thunderbird_profile(tmp_path):
    root = tmp_path / "profile"
    write_mbox(root / "Inbox", [make_raw("hi", "normal message one"), make_raw("hey", "normal message two")])
    write_mbox(root / "Junk", [make_raw("WIN", "cash prize now")])
    (root / "Inbox.msf").write_text("index data")
    write_mbox(root / "Inbox.sbd" / "Work", [make_raw("proj", "project notes")])
    write_mbox(root / "Trash", [make_raw("old", "old stuff")])
    imp = MboxImporter()
    imp.configure(root)
    assert imp.name == "mbox"
    rep = build_label_report(imp, root, Overrides())
    assert {f: (n, lab) for f, n, lab in rep.rows} == {
        "Inbox": (2, "ham"), "Junk": (1, "spam"), "Inbox/Work": (1, "ham"), "Trash": (1, "skip")}


def test_apple_mail_export(tmp_path):
    export = tmp_path / "export"
    write_mbox(export / "Junk.mbox" / "mbox", [make_raw("x", "spam text here")])
    write_mbox(export / "INBOX.mbox" / "mbox", [make_raw("y", "ham text here")])
    (export / "INBOX.mbox" / "table_of_contents").write_bytes(b"\x00\x01binary")
    rep = build_label_report(MboxImporter(), export, Overrides())
    assert {(f, lab) for f, _, lab in rep.rows} == {("Junk", "spam"), ("INBOX", "ham")}


def test_single_apple_mbox_file_uses_parent_name(tmp_path):
    f = write_mbox(tmp_path / "Junk.mbox" / "mbox", [make_raw("x", "spam text here")])
    rep = build_label_report(MboxImporter(), f, Overrides())
    assert rep.rows == [("Junk", 1, "spam")]


def _gm(labels, subject):
    return make_raw(subject, f"body for {subject} message", headers={"X-Gmail-Labels": labels})


def test_gmail_takeout_labels(tmp_path):
    f = write_mbox(tmp_path / "All mail Including Spam and Trash.mbox", [
        _gm("Inbox,Category Updates", "a"), _gm("Spam", "b"), _gm("Spam,Trash", "c"), _gm("Trash", "d"),
        _gm("Sent,Inbox", "e"), _gm("Archived,Important", "f"), _gm('Inbox,"Clients, 2024"', "g"),
        _gm("Chat", "h"),
    ])
    imp = MboxImporter()
    imp.configure(f)
    assert imp.name == "takeout"
    rep = build_label_report(imp, f, Overrides())
    assert {f_: (n, lab) for f_, n, lab in rep.rows} == {
        "Inbox": (2, "ham"), "Spam": (2, "spam"), "Trash": (1, "skip"), "Sent": (1, "skip"),
        "Archived": (1, "ham"), "Chat": (1, "skip")}


# Review Focus 4: quoted commas and encoded label names.
def test_gmail_labels_parsing():
    assert gmail_labels('Inbox,"Clients, 2024",Opened') == ["Inbox", "Clients, 2024", "Opened"]
    assert gmail_labels("=?UTF-8?Q?Caf=C3=A9?=,Inbox") == ["Café", "Inbox"]
    assert gmail_labels(None) == []
    assert gmail_labels("") == []


# Review Focus 3: CRLF files and '>From ' escaped lines.
def test_crlf_mbox_and_from_escaping(tmp_path):
    msgs = [make_raw("one", "line\nFrom here we escape\nend"), make_raw("two", "second body")]
    f = write_mbox(tmp_path / "Inbox", msgs, crlf=True)
    res = list(iter_labeled(MboxImporter(), f, Overrides(), source="t"))
    assert [r.email.subject for r in res] == ["one", "two"]
    assert "we escape" in res[0].email.text
    assert all(r.email.label == "ham" for r in res)
