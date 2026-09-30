"""The `spamfilter` command line."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__, config, corpora
from .corpora import CorpusError
from .dataset import (DatasetExists, DatasetNotFound, InvalidDatasetName, dataset_exists, delete_dataset,
                      labels_of, list_datasets, load_manifest, read_dataset, write_dataset)
from .evaluate import DEFAULT_CANDIDATES, format_report, format_source_table, report_for
from .importers import UnsupportedFormat, detect_importer
from .importers.base import Overrides, build_label_report, iter_labeled
from .message import HAM, SPAM
from .model import Bundle, BundleError
from .parse import parse_bytes
from .training import TrainingError, TrainOptions, train


class CliError(Exception):
    pass


KNOWN_ERRORS = (CliError, DatasetExists, DatasetNotFound, InvalidDatasetName, UnsupportedFormat, CorpusError,
                TrainingError, BundleError, FileNotFoundError, ImportError)


def _out(text: str = "") -> None:
    print(text, flush=True)


def _counts(d: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items())) or "none"


def _model_file(name: str) -> Path:
    path = Path(name)
    if path.suffix == ".model" and (path.is_absolute() or path.parent != Path(".")):
        return path
    return config.models_dir() / (name if name.endswith(".model") else name + ".model")


def resolve_model(name: str | None) -> Path:
    if name is None:
        active = config.load_settings().get("active_model")
        if not active:
            raise BundleError("no model selected. Train one with `spamfilter train -o NAME` "
                              "(run `spamfilter corpora download` first).")
        return Path(active)
    path = _model_file(name)
    if not path.exists():
        raise BundleError(f"no model named '{name}'; see `spamfilter models list`")
    return path


def cmd_corpora_download(a) -> int:
    for corpus in [a.only] if a.only else ["spamassassin", "enron", "spambase"]:
        if corpus == "spambase":
            corpora.download("spambase", log=_out)
            _out("spambase: downloaded (used by the benchmark notebook)")
            continue
        manifest = corpora.prepare(corpus, log=_out)
        _out(f"{manifest['name']}: {manifest['count']} emails ({_counts(manifest['labels'])}); "
             f"skipped {sum(manifest['skipped'].values())}")
    return 0


def cmd_corpora_list(a) -> int:
    for cf in corpora.FILES:
        present = (config.corpora_dir() / cf.filename).exists()
        _out(f"{cf.corpus:<13} {cf.filename:<32} {'present' if present else 'missing'}")
    for name in corpora.DATASET_FOR.values():
        _out(f"{name}: " + (f"{load_manifest(name)['count']} emails" if dataset_exists(name) else "not prepared"))
    return 0


def cmd_import(a) -> int:
    if a.name.startswith("base-"):
        raise InvalidDatasetName("dataset names starting with 'base-' are reserved for the public corpora")
    if dataset_exists(a.name) and not a.force:
        raise DatasetExists(f"dataset '{a.name}' already exists; use --force to replace it")
    path = Path(a.path)
    importer = detect_importer(path, a.format)
    ov = Overrides(spam=a.spam_folder or [], ham=a.ham_folder or [], skip=a.skip_folder or [],
                   include_sent=a.include_sent, label=a.label)
    _out(f"Format: {importer.name}")
    _out(build_label_report(importer, path, ov).format())
    if a.dry_run:
        _out("Dry run: nothing imported.")
        return 0
    started = time.monotonic()
    manifest = write_dataset(a.name, iter_labeled(importer, path, ov, source=a.name),
                             manifest={"importer": importer.name, "source_path": str(path.resolve()),
                                       "overrides": asdict(ov)},
                             force=a.force, progress=lambda n: _out(f"  {n} messages..."))
    _out(f"Imported {manifest['count']} emails into '{a.name}' ({_counts(manifest['labels'])}) "
         f"in {time.monotonic() - started:.0f}s")
    if manifest["skipped"]:
        _out("Skipped: " + _counts(manifest["skipped"]))
    if not manifest["labels"].get(SPAM) or not manifest["labels"].get(HAM):
        _out("Note: this dataset has only one class. Train it together with the base corpora, not alone.")
    return 0


def cmd_datasets(a) -> int:
    if a.action == "list":
        rows = list_datasets()
        if not rows:
            _out("No datasets. Run `spamfilter corpora download` or import mail with `spamfilter import`.")
        for m in rows:
            _out(f"{m['name']:<24} {m['count']:>8}  {_counts(m.get('labels', {}))}")
        return 0
    if not a.name:
        raise CliError(f"`datasets {a.action}` needs a dataset name")
    if a.action == "show":
        _out(json.dumps(load_manifest(a.name), indent=2, ensure_ascii=False))
    else:
        delete_dataset(a.name)
        _out(f"Deleted dataset '{a.name}'.")
    return 0


def cmd_train(a) -> int:
    candidates = None
    if a.algos:
        wanted = {s.strip() for s in a.algos.split(",") if s.strip()}
        candidates = [c for c in DEFAULT_CANDIDATES if c.algo in wanted]
        if not candidates:
            raise CliError(f"unknown --algos {a.algos!r}; choose from logreg, cnb, svm")
    user_weight = a.user_weight if a.user_weight == "auto" else float(a.user_weight)
    opts = TrainOptions(datasets=[s for s in (a.datasets or "").split(",") if s], include_base=not a.no_base,
                        fp_target=a.fp_target, move_fp_target=a.move_fp_target, user_weight=user_weight,
                        folds=a.folds, min_df=a.min_df, candidates=candidates, cross_corpus=not a.no_cross_corpus)
    bundle = train(opts, log=_out)
    out_path = _model_file(a.output)
    bundle.save(out_path)
    settings = config.load_settings()
    if a.activate or not settings.get("active_model"):
        settings["active_model"] = str(out_path)
        config.save_settings(settings)
    _out("")
    _out(format_report(bundle.report))
    _out(f"Saved model to {out_path}")
    return 0


def cmd_models(a) -> int:
    active = config.load_settings().get("active_model")
    if a.action == "list":
        files = sorted(config.models_dir().glob("*.model"))
        if not files:
            _out("No models yet. Train one with `spamfilter train -o NAME`.")
        for f in files:
            mark = "  (active)" if active and Path(active) == f else ""
            _out(f"{f.stem:<24} {f.stat().st_size // 1024:>8} KB{mark}")
        return 0
    if not a.name:
        raise CliError("`models activate` needs a model name")
    path = resolve_model(a.name)
    Bundle.load(path)
    settings = config.load_settings()
    settings["active_model"] = str(path)
    config.save_settings(settings)
    _out(f"Active model: {path.stem}")
    return 0


def cmd_evaluate(a) -> int:
    bundle = Bundle.load(resolve_model(a.model))
    if not a.dataset:
        _out(format_report(bundle.report))
        return 0
    emails = [e for e in read_dataset(a.dataset) if e.label in (SPAM, HAM)]
    if not emails:
        raise CliError(f"dataset '{a.dataset}' has no labeled emails")
    for e in emails:
        e.source = a.dataset
    y = labels_of(emails)
    report = report_for(emails, y, bundle.model.predict_proba(emails), bundle.flag_threshold, bundle.move_threshold)
    _out(format_source_table({a.dataset: report[a.dataset]}))
    return 0


def cmd_predict(a) -> int:
    bundle = Bundle.load(resolve_model(a.model))
    raw = sys.stdin.buffer.read() if a.file == "-" else Path(a.file).read_bytes()
    result = parse_bytes(raw, source="predict")
    if result.email is None:
        raise CliError(f"{a.file} is not an email ({result.error})")
    p_spam = float(bundle.model.predict_proba([result.email])[0])
    verdict = bundle.verdict(p_spam)
    reasons = bundle.model.explain(result.email, top_k=a.top)
    if a.json:
        _out(json.dumps({"verdict": verdict, "p_spam": round(p_spam, 4),
                         "flag_threshold": round(bundle.flag_threshold, 4),
                         "move_threshold": round(bundle.move_threshold, 4),
                         "subject": result.email.subject,
                         "reasons": [r.to_dict() for r in reasons]}, ensure_ascii=False))
        return 0
    _out(f"{verdict.upper()}  P(spam)={p_spam:.3f}  (flag at >= {bundle.flag_threshold:.3f}, "
         f"move at >= {bundle.move_threshold:.3f})")
    _out(f"Subject: {result.email.subject}")
    for r in reasons:
        _out(f"  {'+' if r.weight > 0 else '-'} {r.feature}  ({r.weight:+.3f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spamfilter", description="Local, personalizable spam filter.")
    parser.add_argument("--version", action="version", version=f"spamfilter {__version__}")
    sub = parser.add_subparsers(dest="command")

    corp = sub.add_parser("corpora", help="download and prepare the public training corpora")
    corp_sub = corp.add_subparsers(dest="action", required=True)
    dl = corp_sub.add_parser("download", help="download, verify and convert the corpora")
    dl.add_argument("--only", choices=["spamassassin", "enron", "spambase"])
    dl.set_defaults(func=cmd_corpora_download)
    corp_sub.add_parser("list", help="show corpus files and base datasets").set_defaults(func=cmd_corpora_list)

    imp = sub.add_parser("import", help="import an email export as a labeled dataset")
    imp.add_argument("path")
    imp.add_argument("--name", required=True)
    imp.add_argument("--format", default="auto", choices=["auto", "takeout", "mbox", "eml", "pst"])
    imp.add_argument("--spam-folder", action="append", help="treat this folder as spam (repeatable)")
    imp.add_argument("--ham-folder", action="append", help="treat this folder as ham (repeatable)")
    imp.add_argument("--skip-folder", action="append", help="ignore this folder (repeatable)")
    imp.add_argument("--include-sent", action="store_true", help="count Sent mail as ham")
    imp.add_argument("--label", choices=[SPAM, HAM], help="give every message this label")
    imp.add_argument("--dry-run", action="store_true", help="only print the label report")
    imp.add_argument("--force", action="store_true", help="replace an existing dataset")
    imp.set_defaults(func=cmd_import)

    ds = sub.add_parser("datasets", help="list, show or delete datasets")
    ds.add_argument("action", choices=["list", "show", "delete"])
    ds.add_argument("name", nargs="?")
    ds.set_defaults(func=cmd_datasets)

    tr = sub.add_parser("train", help="train a model bundle")
    tr.add_argument("-o", "--output", required=True, help="model name or path to a .model file")
    tr.add_argument("--datasets", help="comma-separated personal datasets")
    tr.add_argument("--no-base", action="store_true", help="do not use the public base corpora")
    tr.add_argument("--fp-target", type=float, default=0.005, help="max share of real mail flagged (default 0.005)")
    tr.add_argument("--move-fp-target", type=float, default=0.001,
                    help="max share of real mail moved to Junk (default 0.001)")
    tr.add_argument("--user-weight", default="auto", help="'auto' or a number (weight on personal mail)")
    tr.add_argument("--folds", type=int, default=5)
    tr.add_argument("--min-df", type=int, default=2)
    tr.add_argument("--algos", help="restrict candidates, e.g. logreg,cnb")
    tr.add_argument("--no-cross-corpus", action="store_true")
    tr.add_argument("--activate", action="store_true", help="make this the active model")
    tr.set_defaults(func=cmd_train)

    mo = sub.add_parser("models", help="list models or choose the active one")
    mo.add_argument("action", choices=["list", "activate"])
    mo.add_argument("name", nargs="?")
    mo.set_defaults(func=cmd_models)

    ev = sub.add_parser("evaluate", help="show a model's test report or score a dataset")
    ev.add_argument("--model")
    ev.add_argument("--dataset")
    ev.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("predict", help="classify one email (.eml file or - for stdin)")
    pr.add_argument("file")
    pr.add_argument("--model")
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--top", type=int, default=8, help="number of reasons to show")
    pr.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return args.func(args) or 0
    except KNOWN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
