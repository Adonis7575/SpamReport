import pytest

from spamfilter.dataset import (DatasetExists, DatasetNotFound, InvalidDatasetName, dataset_exists,
                                delete_dataset, list_datasets, load_manifest, read_dataset, write_dataset)
from spamfilter.message import ParseResult
from tests.builders import synthetic_emails


def test_write_read_roundtrip():
    emails = synthetic_emails(3, 4)
    results = [ParseResult(e) for e in emails] + [ParseResult(None, "empty"), ParseResult(None, "not-an-email"),
                                                  ParseResult(None, "empty")]
    seen = []
    man = write_dataset("mine", results, manifest={"importer": "test"}, progress=seen.append)
    assert man["count"] == 7
    assert man["labels"] == {"spam": 3, "ham": 4}
    assert man["folders"] == {"Junk": {"spam": 3}, "Inbox": {"ham": 4}}
    assert man["skipped"] == {"empty": 2, "not-an-email": 1}
    assert list(read_dataset("mine")) == emails
    assert load_manifest("mine")["importer"] == "test"
    assert [m["name"] for m in list_datasets()] == ["mine"]


# Review Focus 5: an existing dataset survives unless --force; a crashed import leaves nothing behind.
def test_existing_name_requires_force():
    write_dataset("mine", [], manifest={})
    with pytest.raises(DatasetExists, match="--force"):
        write_dataset("mine", [], manifest={})
    write_dataset("mine", [ParseResult(synthetic_emails(1, 0)[0])], manifest={}, force=True)
    assert load_manifest("mine")["count"] == 1


def test_failed_import_leaves_no_dataset_and_keeps_old_one():
    write_dataset("keep", [ParseResult(synthetic_emails(1, 0)[0])], manifest={})

    def crashing():
        yield ParseResult(synthetic_emails(1, 0, seed=1)[0])
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        write_dataset("new", crashing(), manifest={})
    with pytest.raises(RuntimeError):
        write_dataset("keep", crashing(), manifest={}, force=True)
    assert not dataset_exists("new")
    assert load_manifest("keep")["count"] == 1
    assert [m["name"] for m in list_datasets()] == ["keep"]


def test_names_are_validated_and_missing_is_reported():
    for bad in ("../evil", "", "a b", "x" * 80):
        with pytest.raises(InvalidDatasetName):
            write_dataset(bad, [], manifest={})
    with pytest.raises(DatasetNotFound):
        list(read_dataset("nope"))
    with pytest.raises(DatasetNotFound):
        delete_dataset("nope")


def test_delete():
    write_dataset("gone", [], manifest={})
    delete_dataset("gone")
    assert not dataset_exists("gone")
