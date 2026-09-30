"""Metrics, false-positive-controlled thresholds, per-source reports and cross-validated model selection."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from .features import emails_to_frame, make_features, n_word_columns
from .message import Email
from .model import make_classifier


def threshold_for_fpr(y, scores, fpr_target: float, mask=None) -> float:
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    selected = y == 0 if mask is None else (y == 0) & np.asarray(mask, bool)
    ham = s[selected]
    if ham.size == 0:
        raise ValueError("no ham examples to set a threshold on")
    allowed = int(math.floor(fpr_target * ham.size))
    descending = np.sort(ham)[::-1]
    if allowed >= ham.size:
        return float(descending[-1])
    return float(np.nextafter(descending[allowed], np.inf))


def recall_at_fpr(y, scores, fpr_target: float, mask=None) -> tuple[float, float]:
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    threshold = threshold_for_fpr(y, s, fpr_target, mask)
    spam = s[y == 1]
    recall = float((spam >= threshold).mean()) if spam.size else float("nan")
    return recall, threshold


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def metrics_at(y, proba, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(proba, dtype=float) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "fpr": _ratio(fp, fp + tn), "accuracy": _ratio(tp + tn, len(y))}


def _curve(y: np.ndarray, p: np.ndarray, kind: str, max_points: int = 200):
    if len(set(y.tolist())) < 2:
        return None
    if kind == "pr":
        precision, recall, _ = precision_recall_curve(y, p)
        points = list(zip(recall.tolist(), precision.tolist()))
    else:
        fpr, tpr, _ = roc_curve(y, p)
        points = list(zip(fpr.tolist(), tpr.tolist()))
    step = max(1, len(points) // max_points)
    return [[round(a, 4), round(b, 4)] for a, b in points[::step]]


def _block(y: np.ndarray, p: np.ndarray, flag: float, move: float) -> dict:
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)), "n_spam": int(y.sum()), "n_ham": int((y == 0).sum()),
        "flag": metrics_at(y, p, flag), "move": metrics_at(y, p, move),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "pr_curve": _curve(y, p, "pr"), "roc_curve": _curve(y, p, "roc"),
    }


def report_for(emails: Sequence[Email], y, proba, flag: float, move: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype=float)
    sources = np.array([e.source for e in emails])
    out = {s: _block(y[sources == s], p[sources == s], flag, move) for s in dict.fromkeys(sources.tolist())}
    out["all"] = _block(y, p, flag, move)
    return out


@dataclass(frozen=True)
class Candidate:
    algo: str
    C: float = 1.0
    alpha: float = 0.3

    def label(self) -> str:
        return f"cnb alpha={self.alpha:g}" if self.algo == "cnb" else f"{self.algo} C={self.C:g}"


DEFAULT_CANDIDATES = (
    [Candidate("logreg", C=c) for c in (0.3, 1.0, 3.0, 10.0)]
    + [Candidate("cnb", alpha=a) for a in (0.1, 0.3, 1.0)]
    + [Candidate("svm", C=c) for c in (0.1, 0.3, 1.0)]
)


@dataclass
class SelectionResult:
    best: Candidate
    user_weight: float
    table: list[dict]


def _scores(clf, X) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    log_proba = clf.predict_log_proba(X)
    return log_proba[:, 1] - log_proba[:, 0]


def select_model(emails: Sequence[Email], y, *, is_user, weights=(1.0,), candidates=None, folds: int = 5,
                 fp_target: float = 0.005, min_df: int = 2, seed: int = 42, feature_kwargs: dict | None = None,
                 log=None) -> SelectionResult:
    """Stratified, grouped k-fold CV. Features are fit once per fold; every candidate x user weight shares them."""
    candidates = list(candidates or DEFAULT_CANDIDATES)
    y = np.asarray(y).astype(int)
    is_user = np.asarray(is_user, dtype=bool)
    n_splits = min(folds, int(np.bincount(y, minlength=2).min()))
    if n_splits < 2:
        raise ValueError("need at least 2 spam and 2 ham emails for cross-validation")
    frame = emails_to_frame(emails)
    groups = np.array([e.norm_body_hash for e in emails])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: dict[tuple[Candidate, float], list[tuple[float, float]]] = defaultdict(list)
    for fold, (tr, va) in enumerate(splitter.split(frame, y, groups), start=1):
        features = make_features("full", min_df=min_df, **(feature_kwargs or {}))
        X_tr = sparse.csr_matrix(features.fit_transform(frame.iloc[tr]))
        X_va = sparse.csr_matrix(features.transform(frame.iloc[va]))
        n_words = n_word_columns(features)
        user_va = is_user[va]
        mask = user_va if int(((y[va] == 0) & user_va).sum()) >= 50 else None
        for cand in candidates:
            a, b = (X_tr[:, :n_words], X_va[:, :n_words]) if cand.algo == "cnb" else (X_tr, X_va)
            for weight in weights:
                clf = make_classifier(cand.algo, C=cand.C, alpha=cand.alpha)
                clf.fit(a, y[tr], sample_weight=np.where(is_user[tr], weight, 1.0))
                s = _scores(clf, b)
                recall, _ = recall_at_fpr(y[va], s, fp_target, mask)
                scores[(cand, float(weight))].append((recall, float(average_precision_score(y[va], s))))
        if log:
            log(f"  cross-validation fold {fold}/{n_splits} done")
    table = []
    for (cand, weight), values in scores.items():
        arr = np.array(values)
        table.append({"candidate": cand.label(), "algo": cand.algo, "C": cand.C, "alpha": cand.alpha,
                      "user_weight": weight, "recall_at_fpr": float(arr[:, 0].mean()),
                      "recall_std": float(arr[:, 0].std()), "pr_auc": float(arr[:, 1].mean())})
    table.sort(key=lambda row: (row["recall_at_fpr"], row["pr_auc"]), reverse=True)
    best = next(c for c in candidates if c.label() == table[0]["candidate"])
    return SelectionResult(best=best, user_weight=table[0]["user_weight"], table=table)


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100 * value:.2f}%"


def format_source_table(per_source: dict) -> str:
    lines = [f"{'source':<22} {'n':>7} {'spam recall':>11} {'precision':>9} {'ham FPR':>8} {'PR-AUC':>7}"]
    for name, block in per_source.items():
        flag = block["flag"]
        pr_auc = f"{block['pr_auc']:.3f}" if block.get("pr_auc") is not None else "n/a"
        lines.append(f"{name:<22} {block['n']:>7} {_pct(flag['recall']):>11} {_pct(flag['precision']):>9} "
                     f"{_pct(flag['fpr']):>8} {pr_auc:>7}")
    return "\n".join(lines)


def format_report(report: dict) -> str:
    thresholds = report["thresholds"]
    lines = [
        f"Model: {report['best']['label']}  (weight on personal mail: {report['user_weight']:g})",
        f"Flag threshold: p >= {thresholds['flag']:.4f}   validation ham FPR {_pct(thresholds['val_fpr_flag'])}",
        f"Move threshold: p >= {thresholds['move']:.4f}   validation ham FPR {_pct(thresholds['val_fpr_move'])}",
        "",
        "Held-out test results at the flag threshold:",
        format_source_table(report["test"]),
    ]
    if report.get("cross_corpus"):
        lines += ["", "Cross-corpus check (ranking quality on a corpus never trained on):"]
        lines += [f"  {name}: ROC-AUC {v['roc_auc']:.3f}, PR-AUC {v['pr_auc']:.3f}"
                  for name, v in report["cross_corpus"].items()]
    lines += [f"Warning: {w}" for w in report.get("warnings", [])]
    return "\n".join(lines)
