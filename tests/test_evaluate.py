import numpy as np
import pytest

from spamfilter.evaluate import (Candidate, format_report, metrics_at, recall_at_fpr, report_for, select_model,
                                 threshold_for_fpr)
from tests.builders import synthetic_emails


def _y(emails):
    return np.array([e.label == "spam" for e in emails], dtype=int)


def test_threshold_for_fpr_allows_exactly_the_target():
    y = np.array([0] * 1000 + [1] * 10)
    s = np.concatenate([np.linspace(0, 1, 1000), np.full(10, 2.0)])
    t = threshold_for_fpr(y, s, 0.005)
    assert (s[:1000] >= t).sum() == 5
    assert recall_at_fpr(y, s, 0.005)[0] == 1.0


def test_threshold_zero_target_and_mask():
    y = np.array([0, 0, 0, 0, 1])
    s = np.array([0.9, 0.1, 0.2, 0.3, 0.95])
    mask = np.array([False, True, True, True, True])
    t = threshold_for_fpr(y, s, 0.0, mask)
    assert 0.3 < t <= 0.9
    assert threshold_for_fpr(y, s, 0.0) > 0.9


def test_threshold_requires_ham():
    with pytest.raises(ValueError):
        threshold_for_fpr(np.array([1, 1]), np.array([0.1, 0.2]), 0.01)


def test_metrics_at():
    m = metrics_at(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5
    empty = metrics_at(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
    assert empty["precision"] is None and empty["recall"] is None and empty["fpr"] == 0.0


def test_report_for_groups_by_source():
    emails = synthetic_emails(5, 5, source="a") + synthetic_emails(5, 5, source="b", seed=1)
    y = _y(emails)
    p = y * 0.7 + 0.1  # spam 0.8, ham 0.1
    r = report_for(emails, y, p, 0.5, 0.85)
    assert set(r) == {"a", "b", "all"}
    assert r["a"]["flag"]["recall"] == 1.0 and r["a"]["move"]["recall"] == 0.0
    assert r["all"]["n"] == 20 and r["all"]["roc_auc"] == 1.0
    assert r["all"]["pr_curve"] and r["all"]["roc_curve"]


def test_select_model_ranks_candidates():
    emails = synthetic_emails(60, 90, seed=12)
    result = select_model(emails, _y(emails), is_user=np.zeros(150, bool),
                          candidates=[Candidate("logreg"), Candidate("cnb", alpha=0.3)],
                          folds=3, fp_target=0.05, min_df=1)
    assert result.best in (Candidate("logreg"), Candidate("cnb", alpha=0.3))
    assert len(result.table) == 2 and result.user_weight == 1.0
    assert result.table[0]["recall_at_fpr"] >= result.table[1]["recall_at_fpr"]
    assert all(0 <= row["recall_at_fpr"] <= 1 for row in result.table)


def test_select_model_tunes_user_weight():
    base = synthetic_emails(40, 60, seed=13, source="base-x")
    user = synthetic_emails(30, 120, seed=14, source="me")
    emails = base + user
    is_user = np.array([e.source == "me" for e in emails])
    result = select_model(emails, _y(emails), is_user=is_user, weights=(1.0, 3.0),
                          candidates=[Candidate("logreg")], folds=3, fp_target=0.05, min_df=1)
    assert result.user_weight in (1.0, 3.0) and len(result.table) == 2


def test_format_report():
    emails = synthetic_emails(5, 5, source="a")
    y = _y(emails)
    report = {
        "best": {"label": "logreg C=1"}, "user_weight": 3.0,
        "thresholds": {"flag": 0.5, "move": 0.9, "val_fpr_flag": 0.004, "val_fpr_move": 0.001},
        "test": report_for(emails, y, y * 0.7 + 0.1, 0.5, 0.9),
        "cross_corpus": {"spamassassin->enron": {"roc_auc": 0.9, "pr_auc": 0.8}},
        "warnings": ["small validation set"],
    }
    text = format_report(report)
    assert "logreg C=1" in text and "0.40%" in text and "spamassassin->enron" in text
    assert "Warning: small validation set" in text
