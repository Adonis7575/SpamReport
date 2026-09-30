# spamfilter — Design Spec

**Date:** 2026-09-30
**Status:** Draft for review
**Repo:** `Adonis7575/SpamReport`, branch `spamfilter`

## 1. Purpose

Turn the NMSU Data Mining class project (a Spambase notebook study) into a
spam filter that works on real mail. Anyone should be able to:

1. start from a base model trained on public raw-email corpora,
2. personalize it by training on their own exported mail, and
3. use it from a CLI, a local web UI, or against a live IMAP mailbox.

The class project is finished; there is no report deliverable. `FinalReport/`
stays untouched as the archive of that work.

### Success criteria

- A measured low false-positive rate (legitimate mail flagged as spam),
  because that is the costly error. The flag threshold meets a configurable
  false-positive target (default 0.5%) on held-out validation data.
- Honest evaluation: duplicates removed before splitting, personal mail tested
  on messages newer than anything trained on, and cross-corpus checks.
- Runs locally on an ordinary Windows laptop, CPU only. Training on tens of
  thousands of messages takes minutes, not hours.
- No email content leaves the machine: no cloud APIs, no telemetry.

### Non-goals (v1)

- Outlook.com / Microsoft 365 over IMAP (needs OAuth2 and an Azure app
  registration). App-password providers only.
- Transformer models. The model interface leaves room for one (§5.4), but no
  transformer code is written in v1.
- Server-side or multi-user deployment. The web UI is single-user, localhost.

## 2. Background: why the old approach is replaced

`project/full_spam_pipeline.py` trained Logistic Regression on Spambase and
scored a Gmail Takeout export by recomputing Spambase's 57 features. Problems:

- Spambase comes from one HP Labs mailbox in 1999. Features such as
  `word_freq_george`, `word_freq_hp`, `word_freq_650` and `word_freq_1999`
  are ham signals only for that person, so the model does not transfer.
- Labels came from the mbox file name. A Takeout export is normally a single
  `All mail Including Spam and Trash.mbox`, so every message was labeled ham;
  Gmail's real labels are in the `X-Gmail-Labels` header.
- Only the subject plus the first 500 body characters were scored, and
  HTML-only messages produced empty text.
- The model never learned from the user's mail, and no model was saved.

The audit of the class notebooks also found: 391 exact duplicate rows in
Spambase that leak across the train/test split, model selection on the test
set, unscaled models in the comparison, and no cross-validation. Those fixes
go into the benchmark notebook (§9).

## 3. Decisions

| Topic | Decision |
|---|---|
| Location | New package `spamfilter/` in the SpamReport repo |
| Input formats | Gmail Takeout mbox, Thunderbird/Apple mbox, `.eml` folders, Outlook `.pst` |
| Base corpora | SpamAssassin public corpus + Enron-Spam; Spambase only as a legacy benchmark |
| Model | TF-IDF + structural features, calibrated linear models; pluggable model interface |
| Interfaces | CLI, local web UI, live IMAP |
| IMAP action | Dry-run report by default; opt-in move to Junk above a strict threshold; never delete |
| IMAP auth | App passwords stored in the OS keyring (Windows Credential Manager) |
| Web move | Allowed, behind a confirmation screen, with the same move log and undo |
| Python | 3.14; `libpff-python` has a cp314 Windows wheel (checked 2026-09-30) |

## 4. Architecture

```
SpamReport/
├── FinalReport/                 # untouched class archive
├── spamfilter/
│   ├── message.py      # Email record
│   ├── parse.py        # raw RFC822 bytes -> Email
│   ├── importers/
│   │   ├── base.py     # Importer protocol + folder-to-label mapping + label report
│   │   ├── mbox.py     # Gmail Takeout + Thunderbird/Apple mbox
│   │   ├── eml.py      # .eml folder trees
│   │   └── pst.py      # Outlook .pst via libpff-python
│   ├── corpora.py      # download/verify/cache SpamAssassin, Enron-Spam, Spambase
│   ├── dataset.py      # dataset storage, dedupe, merge, splits
│   ├── features.py     # text + structural feature extraction
│   ├── model.py        # SpamModel interface + linear implementation + bundle I/O
│   ├── evaluate.py     # CV model selection, calibration, thresholds, reports
│   ├── imap.py         # live scanning, move, undo, IMAP import
│   ├── config.py       # paths (%LOCALAPPDATA%\spamfilter), settings
│   ├── cli.py          # `spamfilter` entry point
│   └── web/            # FastAPI app, templates, static assets
├── notebooks/spambase_benchmark.ipynb
├── tests/
├── pyproject.toml      # extras: [web], [pst], [dev]
└── README.md
```

`project/` is removed; it stays recoverable from git history.

**Data flow:** importer or corpus loader -> `Email` records -> `dataset`
(dedupe, merge, split) -> `features` + `model` (fit) -> `evaluate`
(selection, calibration, thresholds, test report) -> **model bundle** file.
The CLI, web UI and IMAP code only load a bundle and call `predict_proba` /
`explain`; they never import training code.

**Storage:**
- Cache root `%LOCALAPPDATA%\spamfilter\` (override: `SPAMFILTER_HOME`):
  `corpora/`, `datasets/<name>/`, `models/`, `imap/` (state, reports, move logs).
- `.gitignore` blocks `*.mbox`, `*.pst`, `*.eml`, `*.model`, `.venv/` and the
  cache directory, so personal mail cannot be committed by accident.

## 5. Components

### 5.1 `Email` record and parsing

`Email` fields: `id` (stable hash), `source` (dataset name), `folder`,
`label` (`spam` / `ham` / `None`), `date` (UTC, may be `None`), `subject`,
`from_addr`, `reply_to`, `to`, `headers` (selected, lower-cased names),
`text` (plain body plus text extracted from HTML), `html_share` (fraction of
the body that was HTML), `urls`, `n_attachments`, `attachment_types`,
`raw_sha256`, `norm_body_hash`.

`parse.py`:
- Uses the `email` package with `policy.default`. Walks all MIME parts,
  prefers `text/plain`, and always also converts `text/html` to text
  (stdlib `html.parser`: strips tags, `<script>`/`<style>`; keeps link targets).
- Handles unknown or mislabeled charsets with a fallback chain
  (declared -> utf-8 -> cp1252 -> latin-1 with replacement).
- Truncates body text to 200,000 characters before feature extraction.
- Never raises on bad input. It returns `ParseResult(email | None, error_reason)`
  so importers can count and log skips.

### 5.2 Importers

Common protocol: `iter_emails(path, mapping) -> Iterator[ParseResult]` plus
`list_folders(path) -> dict[folder, count]` for the label report. Formats are
auto-detected: a directory of `.eml` files; an `.mbox` file or a directory of
them; a Takeout mbox, detected by `X-Gmail-Labels` headers; a `.pst` file.

Default label mapping (overridable with `--spam-folder` / `--ham-folder` /
`--skip-folder`, all repeatable, matched case-insensitively):

| Source | Spam | Ham | Skipped by default |
|---|---|---|---|
| Gmail Takeout | `X-Gmail-Labels` contains `Spam` | Inbox, Archived, Important, `Category *` | Trash, Drafts, Chats, Sent |
| Thunderbird/Apple mbox | file/folder named Junk, Spam, Bulk | all other folders | Trash, Deleted, Drafts, Sent, Templates |
| `.eml` tree | `spam/` subfolder | `ham/` subfolder | everything else (or `--label spam|ham` for the whole tree) |
| Outlook `.pst` | Junk Email | all other folders | Deleted Items, Drafts, Outbox, Sent Items, Sync Issues |

`--include-sent` moves Sent into ham. For Gmail Takeout, a message labeled
both Spam and Trash counts as spam; Trash alone is skipped.

Before anything is stored, every import prints a **label report** (folder ->
count -> label), then parse-skip counts by reason. `--dry-run` stops there.

The PST importer (`[pst]` extra) walks the folder tree with `pypff`,
reconstructs RFC822 from the transport headers plus plain/HTML bodies, and
uses the same parse path. If `pypff` is missing, it errors with the install
command.

### 5.3 Corpora and datasets

`corpora.py` downloads into `corpora/`, verifying a SHA-256 pinned in code
for every archive:
- **SpamAssassin public corpus**: all nine tarballs at
  `https://spamassassin.apache.org/old/publiccorpus/`:
  `20021010_{easy_ham,hard_ham,spam}`,
  `20030228_{easy_ham,easy_ham_2,hard_ham,spam,spam_2}`,
  `20050311_spam_2` (`.tar.bz2`). Full RFC822 messages; the folder name gives
  the label. The sets overlap, and dedupe removes the repeats.
- **Enron-Spam** (Metsis et al., 2006), `enron1`–`enron6` from
  `https://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enronN.tar.gz`.
  These contain subject + body only; missing headers are empty, and the
  header-based features are zero for these rows (see §5.5).
- **Spambase**, `https://archive.ics.uci.edu/static/public/94/spambase.zip`,
  for the benchmark notebook only.

(All URLs returned HTTP 200 on 2026-09-30.) SHA-256 values are recorded on
first successful download during implementation and pinned in code. If a
download fails or a checksum mismatches, the command stops with the URL and
tells you where to place a manually downloaded file.

A **dataset** is stored as `datasets/<name>/emails.jsonl.gz` plus
`manifest.json`: source path, importer, mapping used, counts, skip reasons,
created time, package version. Imports stream, so memory stays bounded.

`dataset.py`:
- **Dedupe:** drop exact duplicates by `raw_sha256`. Group near-duplicates by
  `norm_body_hash` (lower-case, collapse whitespace, mask URLs, emails and
  digits, then SHA-256). Groups with conflicting labels are dropped and
  listed in the training report.
- **Splits:**
  - Base corpora: stratified split, 80% train+validation / 20% test, grouped
    by `norm_body_hash` (`StratifiedGroupKFold`), so no near-duplicate group
    spans both sides.
  - User datasets: split by time. The newest 20% by `date` is the test set.
    Messages without a date go to training. With fewer than 200 dated
    messages, a grouped stratified split is used instead and the report says so.
    The time split is deliberately *not* grouped: a newsletter that arrives
    weekly really will reappear in future mail, so seeing earlier copies in
    training is realistic, not leakage. Exact duplicates are still removed.
  - Validation for thresholds: the newest 15% of user training data plus a
    grouped 15% of base training data.

### 5.4 Model interface

```python
class SpamModel(Protocol):
    def fit(self, emails: Sequence[Email], labels: np.ndarray, sample_weight: np.ndarray | None) -> None: ...
    def predict_proba(self, emails: Sequence[Email]) -> np.ndarray: ...   # P(spam)
    def explain(self, email: Email, top_k: int = 8) -> list[Reason]: ...   # signed contributions
```

v1 implementation: `LinearTextModel`, a scikit-learn pipeline built on the
features in §5.5, a linear classifier and `CalibratedClassifierCV`. A future
transformer backend implements the same protocol and registers under a name;
bundles record which backend they use.

**Model bundle** (single file, `joblib`, extension `.model`): backend name +
fitted pipeline, thresholds (`flag`, `move`), metrics report (JSON), training
manifest (datasets, counts, dedupe stats, weights, CV results), and the
`spamfilter`, scikit-learn and Python versions. Loading refuses a bundle with
a different major/minor scikit-learn version and says to retrain.

Bundles trained on personal mail contain vocabulary from that mail. They are
stored locally, and the README warns not to share them. Base-only bundles are
safe to share.

### 5.5 Features

`ColumnTransformer` over `Email` records:
- **Word TF-IDF** on `subject` (weighted ×2 via a separate block) and `text`:
  1–2-grams, sublinear TF, `min_df=2`, `max_features` tuned (default 200k).
  URLs, email addresses and numbers are normalized to tokens (`__url__`,
  `__email__`, `__num__`); URL domains are kept as tokens.
- **Char TF-IDF** (3–5 char_wb n-grams) on `subject` + first 2,000 characters
  of `text`, for obfuscations like `V1agra` and `fr€e`. (Reduced from 5,000
  during planning: per-fold char n-grams over ~39k emails were too slow for
  the "minutes, not hours" requirement on a laptop CPU.)
- **Header tokens** (when present): From domain, Reply-To domain, mailer/
  X-Mailer family, Content-Type, as categorical tokens.
- **Structural numerics** (scaled): URL count, distinct URL domains,
  `html_share`, attachment count and risky types (exe/js/html/zip),
  Reply-To ≠ From domain, capital-letter ratio, `!` / `$` density, body length
  (log), IP-literal URLs, URL-shortener domains, and a `has_headers` flag so
  header-less rows (Enron) are distinguishable.

Personal information such as your own address is not removed. The model runs
locally and learns from it like any other token.

### 5.6 Training and evaluation

`train`:
1. Load the base corpora (unless `--no-base`) and the chosen user datasets,
   then dedupe and split them (§5.3).
2. **Model selection** on training data only, with stratified-grouped
   5-fold CV over: Logistic Regression (`C` in {0.3, 1, 3, 10}),
   Complement Naive Bayes (word features only, `alpha` in {0.1, 0.3, 1}),
   linear SVM (`C` in {0.1, 0.3, 1}). Primary metric: **spam recall at the
   false-positive target**; tie-break on PR-AUC.
3. **User weight:** user-mail sample weight in {1, 3, 10}. `--user-weight auto`
   (the default) chooses it by CV when the user training set has at least 100
   ham and 20 spam; otherwise it uses 3.
4. Refit the winner on train, then calibrate with `CalibratedClassifierCV`
   (sigmoid; isotonic when there are more than 10k training rows).
5. **Thresholds** chosen on the validation set: `flag` is the lowest score
   where ham FPR ≤ `--fp-target` (default 0.005); `move` is the lowest score
   where ham FPR ≤ `--move-fp-target` (default 0.001). When the user
   validation set has at least 200 ham, the thresholds must meet the targets
   on user ham specifically; otherwise on the combined validation set, with a
   warning.
6. **Final test** on untouched test sets. The report per source (each base
   corpus, each user dataset) includes precision, recall, F1, FPR, PR-AUC,
   ROC-AUC and a confusion matrix at both thresholds, plus PR/ROC curve data.
7. **Cross-corpus check** (reported, not used for selection): train on
   SpamAssassin -> test on Enron, and the reverse, with the chosen
   configuration.

`evaluate <bundle>` prints the stored report. `evaluate <bundle> --dataset X`
scores another dataset.

### 5.7 CLI

Entry point `spamfilter` (argparse, subcommands):

```
spamfilter corpora download [--only spamassassin|enron|spambase]
spamfilter corpora list
spamfilter import <path> --name NAME [--format auto|takeout|mbox|eml|pst]
                  [--spam-folder X]... [--ham-folder Y]... [--skip-folder Z]...
                  [--include-sent] [--label spam|ham] [--dry-run]
spamfilter datasets list | show NAME | delete NAME
spamfilter train -o NAME.model [--datasets A,B] [--no-base]
                 [--fp-target 0.005] [--move-fp-target 0.001] [--user-weight auto|N]
spamfilter models list | activate NAME
spamfilter evaluate [MODEL] [--dataset NAME]
spamfilter predict [MODEL] FILE.eml|- [--json]
spamfilter imap setup | test
spamfilter imap import --name NAME [--ham-folder INBOX]... [--spam-folder Junk]... [--limit N]
spamfilter imap scan [--folder INBOX]... [--since 7d] [--model M] [--move] [--watch 5m]
spamfilter imap undo MOVE_LOG
spamfilter web [--port 8765] [--no-browser]
```

`MODEL` defaults to the active model. `predict` prints the verdict
(`spam` / `suspicious` / `ham`, from the two thresholds), P(spam) and reasons.

### 5.8 Live IMAP

- `imap setup` asks for host, port, username and SSL settings and stores them
  in `imap/account.json`. The app password goes to `keyring` only. `imap test`
  logs in, lists folders, identifies Junk and prints provider-specific
  app-password help on failure.
- **Scan** (`IMAP4_SSL`):
  - `SELECT` the folder read-only unless `--move` is set.
  - Track `UIDVALIDITY` + last processed UID per folder in `imap/state.json`.
    On the first run, or with `--since`, `UID SEARCH SINCE`.
  - Fetch `BODY.PEEK[]` in batches of 50, so read state never changes.
  - Score each message and write `imap/reports/<timestamp>.csv` and `.html`
    (date, from, subject, P(spam), verdict, top reasons, action taken).
  - Per-message errors are logged and skipped.
- **Move** (`--move` only): only messages with P(spam) ≥ the `move`
  threshold.
  - Junk folder: `LIST` with the `\Junk` special-use attribute, then the
    fallbacks `[Gmail]/Spam`, `Junk`, `Junk Email`, `Spam`, `Bulk Mail`. If
    none is found, it stops without moving anything.
  - `UID MOVE` if the server advertises `MOVE`. Otherwise `UID COPY` +
    `UID STORE +FLAGS \Deleted` + `UID EXPUNGE` of that UID (requires
    `UIDPLUS`). Without `UIDPLUS` it refuses, because a plain `EXPUNGE` could
    remove other deleted messages.
  - Every move is appended to `imap/moves/<timestamp>.jsonl` *before* the
    command is sent, then marked done: Message-ID, source folder, target
    folder, subject, P(spam).
- **Undo:** reads a move log, finds each message in the Junk folder by
  `HEADER Message-ID`, and moves it back. The result is reported per message.
- **Watch:** scan loop every N minutes. Reconnects with exponential backoff
  (max 15 min). Ctrl+C stops cleanly after the current batch.
- **IMAP import:** builds a dataset straight from the mailbox using the §5.2
  mapping (default: INBOX = ham, Junk = spam), through the same dataset writer.
- Nothing is ever deleted: no `\Deleted` flag except as part of the
  COPY+EXPUNGE move fallback described above.

### 5.9 Local web UI (`[web]` extra)

FastAPI + Jinja2 templates + htmx (vendored, no CDN) for progress polling.
- Binds `127.0.0.1` only. A random token generated at startup is required on
  every request (query param on the first visit, then an HttpOnly SameSite
  cookie). State-changing requests are POST-only with a CSRF token.
- **Datasets:** enter a *local path* (no browser upload of large exports),
  preview the label report, adjust the mapping, then import as a background job
  with progress.
- **Train:** choose datasets and FP targets, then train as a background job
  with a progress/log view. One training job runs at a time.
- **Models:** list and activate models; show the metrics report, PR/ROC
  curves (server-rendered SVG), and confusion matrices per source.
- **Classify:** paste raw email or choose an `.eml` file (small upload is
  fine), then see the verdict, P(spam) and reasons.
- **IMAP:** test the connection; run a dry-run scan; browse reports. **Move to
  Junk:** from a scan report, a button opens a confirmation page listing
  exactly the messages at or above the move threshold. Confirming runs the same
  move routine and move log as the CLI. Each move log has an **Undo** button
  (also behind confirmation).

## 6. Error handling summary

| Situation | Behavior |
|---|---|
| Malformed / undecodable message | Skipped, counted by reason, logged; the run continues |
| Huge body | Truncated to 200k chars before features |
| Corpus download fails / checksum mismatch | Stop with URL and manual-placement path |
| Unknown import format | Error listing supported formats and detection rules |
| Dataset with 0 spam or 0 ham after mapping | Refuse to train on it alone; allowed when combined with base |
| IMAP auth failure | Explain app passwords with provider links |
| IMAP disconnect in watch mode | Backoff reconnect |
| No Junk folder / no MOVE and no UIDPLUS | Refuse to move, explain why; scanning still works |
| Bundle version mismatch | Refuse to load, instruct to retrain |
| `pypff` / web extras missing | Error with the exact `pip install` command |

## 7. Testing

`pytest`, all fixtures synthetic and generated in `tests/fixtures/` builders.
No real mail in the repo.

- **parse:** multipart/alternative, HTML-only, nested multipart, bad and
  unknown charsets, base64/quoted-printable, attachments, empty, truncated,
  non-mail bytes.
- **importers:** mini Takeout mbox with `X-Gmail-Labels` (Spam, Trash,
  Spam+Trash, Inbox, Sent); Thunderbird Inbox/Junk pair; `.eml` spam/ham
  tree; mapping overrides; label report counts. PST: fake `pypff`-like folder
  tree object; an optional real-file test runs if `SPAMFILTER_TEST_PST` is set.
- **dataset:** exact and near-dup removal, conflicting-label drop, grouped
  splits have no group overlap, time split ordering, small-set fallback.
- **features/model:** fit on a synthetic set; thresholds meet FPR targets on
  validation; save/load round trip; version-mismatch refusal; `explain`
  returns signed reasons.
- **imap:** in-process fake IMAP server (asyncio, implementing the subset
  used). Tests cover: PEEK-only fetch, UID state, UIDVALIDITY change, Junk
  detection, MOVE path, COPY+EXPUNGE fallback, refusal without UIDPLUS,
  move-log-before-send, undo, and that dry-run never issues a write command.
  An optional live test runs when `SPAMFILTER_TEST_IMAP_*` env vars are set.
- **web:** FastAPI TestClient checks: requests without the token are
  rejected, GET never mutates, CSRF enforced, move requires the confirm step,
  and it writes the same log format as the CLI.
- **End-to-end benchmark** (not part of the unit run): download both corpora,
  train, and record the real metrics in the README.

## 8. Delivery

- Branch `spamfilter`, one commit per milestone; nothing pushed without asking.
- Git ownership warning handled with per-command `-c safe.directory=...`.
- Repo-local `.venv` (Python 3.14); `pip install -e .[web,pst,dev]`.
- Milestones:
  1. package skeleton, parsing, importers
  2. corpora + datasets
  3. features, model, evaluation, `train`
  4. `predict` / `evaluate` / model management
  5. IMAP
  6. web UI
  7. Spambase benchmark notebook
  8. README with measured results

## 9. Spambase benchmark notebook

`notebooks/spambase_benchmark.ipynb`, clearly labeled as a legacy benchmark:
- Loads Spambase through `corpora.py` (no additional CSV copy in the repo).
- Removes the 391 exact duplicates, and drops the 3 conflicting-label groups
  before splitting.
- One stratified 80/20 split; the test set is touched once at the end.
- Each model is a `Pipeline` (scaler where relevant + classifier): Gaussian NB,
  Multinomial NB, Logistic Regression, Decision Tree, Random Forest, SVM-RBF,
  kNN. Stratified 5-fold CV on train for selection, with a small grid for
  RF/SVM/LR.
- Reports CV mean ± std; the final model's test precision, recall, F1,
  ROC-AUC and PR-AUC; a confusion matrix; and FPR at the chosen threshold.
  Includes a short discussion of why false positives are the costly error and
  why Spambase does not transfer to real mailboxes (the `george`/`hp`
  features).
- Runs top to bottom from a fresh kernel (executed before commit).
