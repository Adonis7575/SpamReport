import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from spamfilter.model import Bundle, BundleError, LinearTextModel
from tests.builders import synthetic_emails


@pytest.fixture(scope="module")
def data():
    return synthetic_emails(80, 120, seed=10), synthetic_emails(20, 30, seed=11)


def _y(emails):
    return np.array([e.label == "spam" for e in emails], dtype=int)


@pytest.mark.parametrize("algo", ["logreg", "cnb", "svm"])
def test_fit_predict(algo, data):
    train, test = data
    model = LinearTextModel(algo, min_df=1).fit(train, _y(train))
    p = model.predict_proba(test)
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(_y(test), p) > 0.95


def test_sample_weight_is_accepted(data):
    train, test = data
    weights = np.where(np.arange(len(train)) % 2 == 0, 3.0, 1.0)
    model = LinearTextModel("logreg", min_df=1).fit(train, _y(train), sample_weight=weights)
    assert model.predict_proba(test).shape == (50,)


@pytest.mark.parametrize("algo", ["logreg", "cnb"])
def test_explain_returns_signed_reasons(algo, data):
    train, test = data
    model = LinearTextModel(algo, min_df=1).fit(train, _y(train))
    spam = next(e for e in test if e.label == "spam")
    reasons = model.explain(spam, top_k=5)
    assert 1 <= len(reasons) <= 5
    assert all(isinstance(r.feature, str) and r.feature for r in reasons)
    assert sum(r.weight for r in reasons) > 0
    assert reasons[0].to_dict().keys() == {"feature", "weight"}


def test_bundle_roundtrip_and_verdict(tmp_path, data):
    train, test = data
    model = LinearTextModel("logreg", min_df=1).fit(train, _y(train))
    bundle = Bundle(model, flag_threshold=0.5, move_threshold=0.9, report={"x": 1})
    path = tmp_path / "m.model"
    bundle.save(path)
    loaded = Bundle.load(path)
    np.testing.assert_allclose(loaded.model.predict_proba(test), model.predict_proba(test))
    assert loaded.report == {"x": 1} and loaded.backend == "linear"
    assert (loaded.verdict(0.95), loaded.verdict(0.6), loaded.verdict(0.1)) == ("spam", "suspicious", "ham")


def test_bundle_version_mismatch_is_refused(tmp_path, data):
    train, _ = data
    bundle = Bundle(LinearTextModel("cnb", min_df=1).fit(train, _y(train)), 0.5, 0.9)
    bundle.versions["sklearn"] = "0.1.0"
    path = tmp_path / "old.model"
    bundle.save(path)
    with pytest.raises(BundleError, match="Retrain"):
        Bundle.load(path)


def test_loading_garbage_is_refused(tmp_path):
    path = tmp_path / "junk.model"
    path.write_bytes(b"not a model")
    with pytest.raises(BundleError):
        Bundle.load(path)
    with pytest.raises(BundleError):
        Bundle.load(tmp_path / "missing.model")
