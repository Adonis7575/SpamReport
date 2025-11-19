import os
import glob
import re
import csv
import mailbox
import numpy as np
import pandas as pd

from email import policy
from email.parser import BytesParser

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# =========================
# CONFIGURATION
# =========================

# 1) Path to UCI Spambase CSV (downloaded from UCI or your course)
SPAMBASE_CSV = "spambase.csv"

# 2) Path to the "Mail" folder from your Google Takeout export
#    Example: "takeout/Mail"
TAKEOUT_MAIL_DIR = "Mail"

# 3) Where to write outputs
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SPAMBASE COLUMN NAMES
# (standard UCI Spambase names)
# =========================

SPAMBASE_COLS = [
    # 48 word_freq_* features
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',

    # 6 char_freq_* features
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',

    # 3 capital_run_length_* features
    'capital_run_length_average',
    'capital_run_length_longest',
    'capital_run_length_total',

    # label
    'is_spam'
]

FEATURE_COLS = SPAMBASE_COLS[:-1]  # all except is_spam


# =========================
# HELPER: LOAD & TRAIN MODEL ON SPAMBASE
# =========================

def train_spam_model(spambase_csv):
    print(f"[+] Loading Spambase from {spambase_csv}")
    df = pd.read_csv(spambase_csv, header=None)
    if df.shape[1] != len(SPAMBASE_COLS):
        raise ValueError(
            f"Expected {len(SPAMBASE_COLS)} columns in Spambase, got {df.shape[1]}"
        )

    df.columns = SPAMBASE_COLS

    X = df[FEATURE_COLS]
    y = df["is_spam"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("[+] Training Logistic Regression spam classifier...")
    model = LogisticRegression(max_iter=5000, solver="lbfgs", n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Spambase Performance (hold-out test set) ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("==============================================\n")

    return model, X.columns.tolist()


# =========================
# HELPER: EMAIL BODY EXTRACTION
# =========================

def extract_email_body(msg):
    """
    Extract a readable text body from an email.message.Message (mbox item).
    Tries text/plain parts first.
    """
    # Use modern policy if possible
    if not isinstance(msg, bytes) and hasattr(msg, 'get_content_type'):
        # message object as is
        pass

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "")).lower()

            # skip attachments
            if "attachment" in disp:
                continue

            if ctype == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    try:
                        return part.get_payload()
                    except Exception:
                        return ""
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain" or ctype == "text/html":
            try:
                return msg.get_payload(decode=True).decode(errors="ignore")
            except Exception:
                try:
                    return msg.get_payload()
                except Exception:
                    return ""

    return ""


def detect_spam_from_filename(mbox_filename):
    """
    Simple heuristic: if the mbox file name contains 'spam',
    mark all messages in it as spam (label 1), else 0.
    """
    name = os.path.basename(mbox_filename).lower()
    return 1 if "spam" in name else 0


# =========================
# HELPER: FEATURE EXTRACTION FROM TEXT
# (minimal approximation of Spambase-style features)
# =========================

# subset of words that often appear in spam (from Spambase + intuition)
WORDS_FOR_FREQ = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet",
    "order", "mail", "receive", "will", "people", "report", "addresses", "free",
    "business", "email", "you", "credit", "your", "font", "000", "money",
    "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857",
    "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference"
]

# maps char to the corresponding Spambase column
CHAR_TO_COL = {
    ';': 'char_freq_;',
    '(': 'char_freq_(',
    '[': 'char_freq_[',
    '!': 'char_freq_!',
    '$': 'char_freq_$',
    '#': 'char_freq_#'
}


def extract_features_from_text(text):
    """
    Approximate Spambase-like features from raw text.
    Returns a dict with keys in FEATURE_COLS (missing ones defaulted later).
    """
    if text is None:
        text = ""

    raw_text = text
    lower = raw_text.lower()

    # words
    tokens = re.findall(r"[a-zA-Z0-9]+", lower)
    n_words = len(tokens) if len(tokens) > 0 else 1  # avoid div by zero

    features = {}

    # word frequencies (percentage of words)
    for w in WORDS_FOR_FREQ:
        count = sum(1 for t in tokens if t == w)
        features[f"word_freq_{w}"] = 100.0 * count / n_words

    # char frequencies (percentage of characters)
    n_chars = len(raw_text) if len(raw_text) > 0 else 1
    for ch, colname in CHAR_TO_COL.items():
        count = raw_text.count(ch)
        features[colname] = 100.0 * count / n_chars

    # capital run stats (runs of >= 2 uppercase letters)
    caps_runs = re.findall(r"[A-Z]{2,}", raw_text)
    if caps_runs:
        lengths = [len(run) for run in caps_runs]
        features["capital_run_length_average"] = float(np.mean(lengths))
        features["capital_run_length_longest"] = float(np.max(lengths))
        features["capital_run_length_total"] = float(np.sum(lengths))
    else:
        features["capital_run_length_average"] = 0.0
        features["capital_run_length_longest"] = 0.0
        features["capital_run_length_total"] = 0.0

    # Any missing word_freq_* or char_freq_* columns will be filled with 0
    return features


# =========================
# STEP 1: TRAIN MODEL ON SPAMBASE
# =========================

best_model, trained_feature_cols = train_spam_model(SPAMBASE_CSV)

# Sanity: ensure we know which columns to use
trained_feature_cols = list(trained_feature_cols)


# =========================
# STEP 2: WALK ALL .mbox FILES FROM TAKEOUT
# =========================

mbox_files = glob.glob(os.path.join(TAKEOUT_MAIL_DIR, "*.mbox"))

if not mbox_files:
    print(f"[!] No .mbox files found in {TAKEOUT_MAIL_DIR}.")
    print("    Make sure you point TAKEOUT_MAIL_DIR to your Takeout 'Mail' folder.")
    exit(1)

print(f"[+] Found {len(mbox_files)} .mbox files in {TAKEOUT_MAIL_DIR}")

gmail_rows = []  # raw meta/text rows
row_id = 0

for mbox_path in mbox_files:
    print(f"[+] Reading mbox: {mbox_path}")
    spam_label_for_file = detect_spam_from_filename(mbox_path)

    mbox = mailbox.mbox(mbox_path)

    for msg in mbox:
        try:
            # normalize message as needed
            msg_from = msg.get("From", "")
            msg_to = msg.get("To", "")
            subject = msg.get("Subject", "")
            date = msg.get("Date", "")

            body = extract_email_body(msg)
            snippet = (body[:500] or "").replace("\n", " ").replace("\r", " ")

            gmail_rows.append({
                "id": row_id,
                "mbox_file": os.path.basename(mbox_path),
                "from": msg_from,
                "to": msg_to,
                "subject": subject,
                "date": date,
                "body_snippet": snippet,
                "is_spam_file_label": spam_label_for_file
            })
            row_id += 1
        except Exception as e:
            print(f"    [!] Error reading message: {e}")

# Create raw Gmail DataFrame
gmail_raw_df = pd.DataFrame(gmail_rows)
raw_csv_path = os.path.join(OUTPUT_DIR, "gmail_raw_all.csv")
gmail_raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8")
print(f"[+] Wrote raw Gmail CSV: {raw_csv_path}")


# =========================
# STEP 3: FEATURE EXTRACTION FOR ALL GMAIL MESSAGES
# =========================

print("[+] Extracting features from Gmail messages...")

feature_dicts = []
for _, row in gmail_raw_df.iterrows():
    text = (str(row.get("subject", "")) + " " +
            str(row.get("body_snippet", "")))
    feats = extract_features_from_text(text)
    feature_dicts.append(feats)

X_gmail = pd.DataFrame(feature_dicts)

# Align columns with Spambase feature columns used for training
X_gmail = X_gmail.reindex(columns=trained_feature_cols, fill_value=0.0)

# =========================
# STEP 4: PREDICT SPAM ON GMAIL
# =========================

print("[+] Running spam predictions on Gmail messages...")
spam_pred = best_model.predict(X_gmail)

if hasattr(best_model, "predict_proba"):
    spam_prob = best_model.predict_proba(X_gmail)[:, 1]
else:
    spam_prob = np.zeros(len(spam_pred))

gmail_scored_df = gmail_raw_df.copy()
gmail_scored_df["spam_pred"] = spam_pred
gmail_scored_df["spam_prob"] = spam_prob

scored_csv_path = os.path.join(OUTPUT_DIR, "gmail_scored_all.csv")
gmail_scored_df.to_csv(scored_csv_path, index=False, encoding="utf-8")

print(f"[+] Wrote scored Gmail CSV (with spam_pred & spam_prob): {scored_csv_path}")
print("[+] Done.")
