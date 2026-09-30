"""Train a model bundle: load, dedupe, split, select, fit, calibrate, set thresholds, test."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .dataset import dataset_exists, dedupe, labels_of, read_dataset, split_base, split_user
from .evaluate import Candidate, report_for, select_model, threshold_for_fpr
from .message import HAM, SPAM, Email
from .model import Bundle, LinearTextModel

BASE_DATASETS = ("base-spamassassin", "base-enron")


class TrainingError(Exception):
    pass


@dataclass
class TrainOptions:
    datasets: list[str] = field(default_factory=list)
    include_base: bool = True
    fp_target: float = 0.005
    move_fp_target: float = 0.001
    user_weight: str | float = "auto"
    folds: int = 5
    min_df: int = 2
    seed: int = 42
    candidates: list[Candidate] | None = None
    cross_corpus: bool = True
    feature_kwargs: dict | None = None


def _load(names, *, personal: bool) -> list[Email]:
    out: list[Email] = []
    for name in names:
        if not dataset_exists(name):
            if personal:
                raise TrainingError(f"no dataset named '{name}'; see `spamfilter datasets list`")
            raise TrainingError("the base corpora are not prepared yet; run `spamfilter corpora download` first")
        for email in read_dataset(name):
            if email.label in (SPAM, HAM):
                email.source = name
                out.append(email)
    return out


def _cross_corpus(sa: list[Email], en: list[Email], cand: Candidate, opts: TrainOptions, log) -> dict:
    out = {}
    for name, (train_set, test_set) in {"spamassassin->enron": (sa, en), "enron->spamassassin": (en, sa)}.items():
        model = LinearTextModel(cand.algo, C=cand.C, alpha=cand.alpha, min_df=opts.min_df,
                                feature_kwargs=opts.feature_kwargs)
        model.fit(train_set, labels_of(train_set))
        p, y = model.predict_proba(test_set), labels_of(test_set)
        out[name] = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}
        log(f"  cross-corpus {name}: ROC-AUC {out[name]['roc_auc']:.3f}")
    return out


def train(opts: TrainOptions, log=print) -> Bundle:
    started = time.monotonic()
    base_names = list(BASE_DATASETS) if opts.include_base else []
    if not base_names and not opts.datasets:
        raise TrainingError("nothing to train on: give --datasets or keep the base corpora")
    personal = _load(opts.datasets, personal=True)
    base = _load(base_names, personal=False)
    log(f"Loaded {len(base)} base and {len(personal)} personal emails")

    kept, dedupe_stats = dedupe(personal + base)  # personal first, so their copy of a duplicate is kept
    if len(set(labels_of(kept).tolist())) < 2:
        raise TrainingError("training data must contain both spam and ham; a one-class dataset can only be "
                            "trained together with the base corpora")
    user_names = set(opts.datasets)
    user_emails = [e for e in kept if e.source in user_names]
    base_emails = [e for e in kept if e.source not in user_names]
    splits = {}
    if base_emails:
        splits["base"] = split_base(base_emails, seed=opts.seed)
    if user_emails:
        splits["user"] = split_user(user_emails, seed=opts.seed)
    train_set = [e for s in splits.values() for e in s.train]
    val = [e for s in splits.values() for e in s.val]
    test = [e for s in splits.values() for e in s.test]
    if not val or not test:
        raise TrainingError("not enough data to hold out validation and test sets")
    y_train, y_val, y_test = labels_of(train_set), labels_of(val), labels_of(test)
    is_user_train = np.array([e.source in user_names for e in train_set])

    weights: tuple[float, ...] = (1.0,)
    if user_emails:
        if opts.user_weight == "auto":
            n_user_ham = int(((y_train == 0) & is_user_train).sum())
            n_user_spam = int(((y_train == 1) & is_user_train).sum())
            weights = (1.0, 3.0, 10.0) if (base_emails and n_user_ham >= 100 and n_user_spam >= 20) else (3.0,)
        else:
            weights = (float(opts.user_weight),)

    log(f"Selecting a model on {len(train_set)} training emails ({opts.folds}-fold cross-validation)")
    selection = select_model(train_set, y_train, is_user=is_user_train, weights=weights,
                             candidates=opts.candidates, folds=opts.folds, fp_target=opts.fp_target,
                             min_df=opts.min_df, seed=opts.seed, feature_kwargs=opts.feature_kwargs, log=log)
    best = selection.best
    log(f"Best: {best.label()} (weight on personal mail {selection.user_weight:g}); fitting the final model")
    model = LinearTextModel(best.algo, C=best.C, alpha=best.alpha, min_df=opts.min_df,
                            calibration="isotonic" if len(train_set) > 10_000 else "sigmoid",
                            feature_kwargs=opts.feature_kwargs)
    model.fit(train_set, y_train, sample_weight=np.where(is_user_train, selection.user_weight, 1.0))

    p_val = model.predict_proba(val)
    user_val = np.array([e.source in user_names for e in val])
    warnings: list[str] = []
    use_personal = int(((y_val == 0) & user_val).sum()) >= 200
    if user_emails and not use_personal:
        warnings.append("fewer than 200 personal ham emails in validation, so thresholds were set on the "
                        "combined validation set")
    mask = user_val if use_personal else None
    n_val_ham = int(((y_val == 0) & (user_val if use_personal else True)).sum())
    if n_val_ham < 1 / opts.move_fp_target:
        warnings.append(f"only {n_val_ham} ham emails in validation; the {opts.move_fp_target:.1%} move target "
                        "cannot be measured precisely")
    flag = threshold_for_fpr(y_val, p_val, opts.fp_target, mask)
    move = max(flag, threshold_for_fpr(y_val, p_val, opts.move_fp_target, mask))
    ham_sel = (y_val == 0) & (mask if mask is not None else True)
    thresholds = {"flag": flag, "move": move,
                  "val_fpr_flag": float((p_val[ham_sel] >= flag).mean()),
                  "val_fpr_move": float((p_val[ham_sel] >= move).mean()),
                  "set_on": "personal ham" if use_personal else "combined validation"}

    p_test = model.predict_proba(test)
    report = {
        "best": {"label": best.label(), "algo": best.algo, "C": best.C, "alpha": best.alpha},
        "user_weight": selection.user_weight,
        "selection": selection.table,
        "thresholds": thresholds,
        "test": report_for(test, y_test, p_test, flag, move),
        "dedupe": asdict(dedupe_stats),
        "split": {name: {"train": len(s.train), "val": len(s.val), "test": len(s.test), "time_based": s.time_based}
                  for name, s in splits.items()},
        "warnings": warnings,
    }
    if opts.cross_corpus and opts.include_base:
        sa = [e for e in base_emails if e.source == "base-spamassassin"]
        en = [e for e in base_emails if e.source == "base-enron"]
        if sa and en:
            log("Cross-corpus check")
            report["cross_corpus"] = _cross_corpus(sa, en, best, opts, log)
    report["train_seconds"] = round(time.monotonic() - started, 1)
    names = base_names + list(opts.datasets)
    manifest = {
        "datasets": names,
        "counts": {n: sum(1 for e in kept if e.source == n) for n in names},
        "options": {k: v for k, v in asdict(opts).items() if k != "candidates"},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return Bundle(model=model, flag_threshold=flag, move_threshold=move, report=report, manifest=manifest)
