import io
import json
import sys

from spamfilter import config
from spamfilter.cli import main
from spamfilter.dataset import dataset_exists, load_manifest
from tests.builders import make_raw, write_mbox


def run(args, capsys):
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _mail(tmp_path):
    root = tmp_path / "mail"
    write_mbox(root / "Inbox", [make_raw("hi", "hello friend"), make_raw("notes", "meeting notes attached")])
    write_mbox(root / "Junk", [make_raw("WIN", "cash prize claim now")])
    return root


def test_import_dry_run_then_real_then_duplicate(tmp_path, capsys):
    root = _mail(tmp_path)
    code, out, _ = run(["import", str(root), "--name", "mine", "--dry-run"], capsys)
    assert code == 0 and "Junk" in out and "spam" in out and "Dry run" in out
    assert not dataset_exists("mine")
    code, out, _ = run(["import", str(root), "--name", "mine"], capsys)
    assert code == 0 and "Imported 3 emails" in out
    assert load_manifest("mine")["labels"] == {"ham": 2, "spam": 1}
    code, _, err = run(["import", str(root), "--name", "mine"], capsys)
    assert code == 1 and "--force" in err


def test_import_rejects_reserved_and_missing(tmp_path, capsys):
    code, _, err = run(["import", str(_mail(tmp_path)), "--name", "base-x"], capsys)
    assert code == 1 and "reserved" in err
    code, _, err = run(["import", str(tmp_path / "nope"), "--name", "x"], capsys)
    assert code == 1 and "does not exist" in err


def test_datasets_commands(tmp_path, capsys):
    run(["import", str(_mail(tmp_path)), "--name", "mine"], capsys)
    code, out, _ = run(["datasets", "list"], capsys)
    assert code == 0 and "mine" in out
    code, out, _ = run(["datasets", "show", "mine"], capsys)
    assert code == 0 and json.loads(out)["count"] == 3
    code, _, err = run(["datasets", "show"], capsys)
    assert code == 1 and "name" in err
    code, out, _ = run(["datasets", "delete", "mine"], capsys)
    assert code == 0 and not dataset_exists("mine")


def test_train_predict_evaluate_models(synthetic_stores, tmp_path, capsys, monkeypatch):
    code, out, err = run(["train", "-o", "t1", "--datasets", "me", "--folds", "3", "--min-df", "1",
                          "--algos", "logreg", "--fp-target", "0.02", "--move-fp-target", "0.01",
                          "--no-cross-corpus"], capsys)
    assert code == 0, err
    assert "Held-out test results" in out and "Saved model" in out
    assert config.load_settings()["active_model"].endswith("t1.model")

    code, out, _ = run(["models", "list"], capsys)
    assert code == 0 and "t1" in out and "(active)" in out

    spam = tmp_path / "spam.eml"
    spam.write_bytes(make_raw("free cash prize winner", "click offer guaranteed cash bonus deal",
                              sender="deals@promo.biz"))
    code, out, _ = run(["predict", str(spam)], capsys)
    assert code == 0 and "P(spam)=" in out and "Subject: free cash prize winner" in out

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(spam.read_bytes())))
    code, out, _ = run(["predict", "-", "--json"], capsys)
    payload = json.loads(out)
    assert code == 0 and payload["verdict"] in ("spam", "suspicious", "ham") and payload["reasons"]

    code, out, _ = run(["evaluate"], capsys)
    assert code == 0 and "Held-out test results" in out
    code, out, _ = run(["evaluate", "--dataset", "me"], capsys)
    assert code == 0 and "me" in out and "spam recall" in out

    notmail = tmp_path / "notes.pdf"
    notmail.write_bytes(b"%PDF-1.4\n%binary")
    code, _, err = run(["predict", str(notmail)], capsys)
    assert code == 1 and "not an email" in err


def test_predict_without_model_explains(tmp_path, capsys):
    eml = tmp_path / "a.eml"
    eml.write_bytes(make_raw())
    code, _, err = run(["predict", str(eml)], capsys)
    assert code == 1 and "spamfilter train" in err


def test_corpora_list(capsys):
    code, out, _ = run(["corpora", "list"], capsys)
    assert code == 0 and "20021010_spam.tar.bz2" in out and "missing" in out and "not prepared" in out


def test_no_command_prints_help(capsys):
    code, out, _ = run([], capsys)
    assert code == 2 and "usage" in out.lower()


# Review Focus 1: a cp1252 console must not crash on non-ASCII folder names or subjects.
def test_output_survives_cp1252_console(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", console)
    root = tmp_path / "mail"
    write_mbox(root / "Входящие", [make_raw("привет 🎉", "текст письма здесь")])
    assert main(["import", str(root), "--name", "ru", "--dry-run"]) == 0
    console.flush()
    assert b"?" in buffer.getvalue()
