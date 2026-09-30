import pytest

from spamfilter.dataset import write_dataset
from spamfilter.evaluate import Candidate
from spamfilter.message import ParseResult
from spamfilter.training import TrainingError, TrainOptions, train
from tests.builders import synthetic_emails

FAST = dict(folds=3, min_df=1, candidates=[Candidate("logreg"), Candidate("cnb")],
            feature_kwargs={"max_char_features": 5000})


def _quiet(*_args):
    pass


def test_train_with_base_and_personal_mail(synthetic_stores):
    bundle = train(TrainOptions(datasets=["me"], fp_target=0.02, move_fp_target=0.01, **FAST), log=_quiet)
    report = bundle.report
    assert set(report["test"]) == {"base-spamassassin", "base-enron", "me", "all"}
    assert report["split"]["user"]["time_based"] is True
    assert report["thresholds"]["val_fpr_flag"] <= 0.02
    assert report["thresholds"]["val_fpr_move"] <= 0.01
    assert bundle.move_threshold >= bundle.flag_threshold
    assert report["user_weight"] in (1.0, 3.0, 10.0)
    assert {row["user_weight"] for row in report["selection"]} == {1.0, 3.0, 10.0}
    assert set(report["cross_corpus"]) == {"spamassassin->enron", "enron->spamassassin"}
    assert bundle.manifest["datasets"] == ["base-spamassassin", "base-enron", "me"]
    assert report["test"]["all"]["roc_auc"] > 0.95


def test_train_base_only(synthetic_stores):
    bundle = train(TrainOptions(cross_corpus=False, fp_target=0.02, **FAST), log=_quiet)
    assert "me" not in bundle.report["test"] and "cross_corpus" not in bundle.report
    assert bundle.report["user_weight"] == 1.0


def test_one_class_dataset_cannot_train_alone(synthetic_stores):
    write_dataset("hamonly", [ParseResult(e) for e in synthetic_emails(0, 50, seed=23, source="hamonly")],
                  manifest={})
    with pytest.raises(TrainingError, match="both spam and ham"):
        train(TrainOptions(datasets=["hamonly"], include_base=False, **FAST), log=_quiet)


def test_missing_base_corpora_are_explained(spam_home):
    with pytest.raises(TrainingError, match="corpora download"):
        train(TrainOptions(**FAST), log=_quiet)


def test_unknown_dataset(synthetic_stores):
    with pytest.raises(TrainingError, match="no dataset named 'nope'"):
        train(TrainOptions(datasets=["nope"], **FAST), log=_quiet)


def test_nothing_to_train_on(synthetic_stores):
    with pytest.raises(TrainingError, match="nothing to train on"):
        train(TrainOptions(include_base=False, **FAST), log=_quiet)
