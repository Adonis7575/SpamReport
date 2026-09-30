from pathlib import Path

import pytest

from spamfilter.importers.base import (FolderRules, Overrides, build_label_report, iter_labeled,
                                       norm_folder, resolve_label)
from tests.builders import make_raw

RULES = FolderRules(spam=("junk", "spam"), skip=("trash", "sent", "sent items"))


@pytest.mark.parametrize("folder, expected", [
    ("Inbox", "ham"), ("Junk", "spam"), ("SPAM", "spam"), ("Mail/Junk.mbox", "spam"),
    ("Local Folders/Inbox.sbd/Work", "ham"), ("Trash", "skip"), ("Sent", "skip"),
])
def test_default_rules(folder, expected):
    assert resolve_label(folder, RULES, Overrides()) == expected


def test_overrides_take_priority():
    ov = Overrides(spam=["Promotions"], ham=["Junk"], skip=["inbox"])
    assert resolve_label("Promotions", RULES, ov) == "spam"
    assert resolve_label("Junk", RULES, ov) == "ham"
    assert resolve_label("Inbox", RULES, ov) == "skip"


def test_include_sent_and_forced_label():
    assert resolve_label("Sent Items", RULES, Overrides(include_sent=True)) == "ham"
    assert resolve_label("Trash", RULES, Overrides(label="spam")) == "spam"


def test_norm_folder():
    assert norm_folder("Mail\\Junk.mbox") == "junk"
    assert norm_folder("Inbox.sbd") == "inbox"
    assert norm_folder("  Sent   Items ") == "sent items"


class FakeImporter:
    name = "fake"
    rules = RULES

    def __init__(self, items):
        self.items = items

    def iter_raw(self, path):
        yield from self.items


def _items():
    return [
        ("Inbox", make_raw("a", "hello there friend")),
        ("Inbox", make_raw("b", "another normal note")),
        ("Junk", make_raw("c", "win cash now")),
        ("Trash", make_raw("d", "old")),
        ("Inbox", b""),
    ]


def test_label_report():
    rep = build_label_report(FakeImporter(_items()), Path("x"), Overrides())
    assert rep.rows == [("Inbox", 3, "ham"), ("Junk", 1, "spam"), ("Trash", 1, "skip")]
    assert rep.totals() == {"ham": 3, "spam": 1, "skip": 1}
    text = rep.format()
    assert "Inbox" in text and "Total:" in text


def test_iter_labeled_skips_and_reports_errors():
    results = list(iter_labeled(FakeImporter(_items()), Path("x"), Overrides(), source="mine"))
    assert [r.email.label for r in results if r.email] == ["ham", "ham", "spam"]
    assert [r.error for r in results if r.email is None] == ["empty"]
    assert all(r.email.source == "mine" for r in results if r.email)
