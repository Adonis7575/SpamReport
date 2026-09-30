"""The spam model: calibrated linear classifiers over spamfilter features, saved as one bundle file."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import __version__
from .features import emails_to_frame, make_features
from .message import Email

ALGOS = ("logreg", "cnb", "svm")


def make_classifier(algo: str, *, C: float = 1.0, alpha: float = 0.3):
    if algo == "logreg":
        return LogisticRegression(C=C, solver="liblinear", max_iter=2000)
    if algo == "svm":
        return LinearSVC(C=C, max_iter=5000)
    if algo == "cnb":
        return ComplementNB(alpha=alpha)
    raise ValueError(f"unknown algorithm {algo!r}; choose from {', '.join(ALGOS)}")


@dataclass
class Reason:
    feature: str
    weight: float  # positive pushes toward spam

    def to_dict(self) -> dict:
        return {"feature": self.feature, "weight": round(self.weight, 4)}


_BLOCK_NAMES = {"subj": "subject", "body": "word", "char": "text", "hdr": "header", "num": "signal"}
_TOKEN_NAMES = {"__url__": "<link>", "__email__": "<email>", "__num__": "<number>"}


def _pretty(name: str) -> str:
    block, _, feature = name.partition("__")
    words = []
    for token in feature.split(" "):
        if token.startswith("urldom_"):
            token = "link:" + token[len("urldom_"):].replace("_", ".")
        words.append(_TOKEN_NAMES.get(token, token))
    feature = " ".join(words)
    if block == "char":
        return f'text "{feature}"'
    return f"{_BLOCK_NAMES.get(block, block)}: {feature}"


class LinearTextModel:
    backend = "linear"

    def __init__(self, algo: str = "logreg", *, C: float = 1.0, alpha: float = 0.3, min_df: int = 2,
                 calibration: str = "sigmoid", feature_kwargs: dict | None = None) -> None:
        if algo not in ALGOS:
            raise ValueError(f"unknown algorithm {algo!r}")
        self.algo, self.C, self.alpha, self.min_df = algo, C, alpha, min_df
        self.calibration = calibration
        self.feature_kwargs = dict(feature_kwargs or {})
        self.pipeline: Pipeline | None = None

    def fit(self, emails: Sequence[Email], y, sample_weight=None) -> LinearTextModel:
        features = make_features("words" if self.algo == "cnb" else "full", min_df=self.min_df, **self.feature_kwargs)
        calibrated = CalibratedClassifierCV(make_classifier(self.algo, C=self.C, alpha=self.alpha),
                                            method=self.calibration, cv=3)
        self.pipeline = Pipeline([("feat", features), ("clf", calibrated)])
        params = {"clf__sample_weight": np.asarray(sample_weight)} if sample_weight is not None else {}
        self.pipeline.fit(emails_to_frame(emails), np.asarray(y), **params)
        return self

    def predict_proba(self, emails: Sequence[Email]) -> np.ndarray:
        return self.pipeline.predict_proba(emails_to_frame(emails))[:, 1]

    def _coef(self) -> np.ndarray:
        coefs = []
        for calibrated in self.pipeline.named_steps["clf"].calibrated_classifiers_:
            estimator = calibrated.estimator
            if hasattr(estimator, "coef_"):
                coefs.append(np.ravel(estimator.coef_))
            else:  # ComplementNB: class 1 vs class 0 log weights
                coefs.append(estimator.feature_log_prob_[1] - estimator.feature_log_prob_[0])
        return np.mean(coefs, axis=0)

    def explain(self, email: Email, top_k: int = 8) -> list[Reason]:
        features = self.pipeline.named_steps["feat"]
        row = sparse.csr_matrix(features.transform(emails_to_frame([email])))
        contributions = row.multiply(self._coef()).tocsr()
        names = features.get_feature_names_out()
        order = np.argsort(-np.abs(contributions.data))[:top_k]
        return [Reason(_pretty(names[contributions.indices[i]]), float(contributions.data[i])) for i in order]


class BundleError(Exception):
    pass


def current_versions() -> dict:
    return {"spamfilter": __version__, "sklearn": sklearn.__version__, "python": platform.python_version()}


def _major_minor(version: str) -> tuple[str, ...]:
    return tuple(version.split(".")[:2])


@dataclass
class Bundle:
    model: LinearTextModel
    flag_threshold: float
    move_threshold: float
    report: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    versions: dict = field(default_factory=current_versions)

    @property
    def backend(self) -> str:
        return self.model.backend

    def verdict(self, p_spam: float) -> str:
        if p_spam >= self.move_threshold:
            return "spam"
        if p_spam >= self.flag_threshold:
            return "suspicious"
        return "ham"

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: Path) -> Bundle:
        path = Path(path)
        if not path.exists():
            raise BundleError(f"model file {path} does not exist")
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            raise BundleError(f"could not load model {path.name} ({type(exc).__name__}). "
                              "Retrain with `spamfilter train`.") from exc
        if not isinstance(bundle, Bundle):
            raise BundleError(f"{path.name} is not a spamfilter model bundle")
        trained = bundle.versions.get("sklearn", "?")
        if _major_minor(trained) != _major_minor(sklearn.__version__):
            raise BundleError(f"{path.name} was trained with scikit-learn {trained} but {sklearn.__version__} "
                              "is installed. Retrain with `spamfilter train`.")
        return bundle
