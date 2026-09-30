"""Feature extraction: TF-IDF text blocks, header tokens and structural signals."""

from __future__ import annotations

import math
import re
from typing import Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .message import Email
from .parse import URL_RE

CHAR_HEAD = 2000
STRUCT_NAMES = [
    "url_count", "url_domains", "html_share", "n_attachments", "risky_attachment", "reply_to_mismatch",
    "caps_ratio", "exclaim_per_1k", "dollar_per_1k", "log_length", "ip_url", "shortener_url", "has_headers",
]
SHORTENERS = frozenset({"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
                        "cutt.ly", "tiny.cc", "rb.gy", "shorturl.at"})
RISKY_EXT = frozenset({"exe", "scr", "js", "vbs", "bat", "cmd", "com", "jar", "html", "htm", "zip", "rar", "7z",
                       "iso", "img", "docm", "xlsm", "lnk", "hta"})
# Lookbehind keeps this linear on long unbroken lines (see parse._EMAIL_RE).
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUM_RE = re.compile(r"\b\d[\d,.\-]*\b")
_IP_URL_RE = re.compile(r"^(?:https?://)?\d{1,3}(?:\.\d{1,3}){3}(?:[:/]|$)")
_MAILER_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")


def url_domain(url: str) -> str:
    candidate = url if "://" in url else "http://" + url
    try:
        host = urlsplit(candidate).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


def normalize_text(s: str) -> str:
    def url_token(match: re.Match) -> str:
        domain = re.sub(r"[^a-z0-9]", "_", url_domain(match.group(0)))
        return f" __url__ urldom_{domain} "

    s = URL_RE.sub(url_token, s)
    s = _EMAIL_RE.sub(" __email__ ", s)
    s = _NUM_RE.sub(" __num__ ", s)
    return s.lower()


def _domain_of(address: str) -> str:
    return address.rsplit("@", 1)[1] if "@" in address else ""


def structural_features(e: Email) -> list[float]:
    text = e.text or ""
    n_chars = max(len(text), 1)
    sample = text[:20_000]
    letters = sum(c.isalpha() for c in sample)
    caps = sum(c.isupper() for c in sample)
    domains = {url_domain(u) for u in e.urls} - {""}
    from_domain, reply_domain = _domain_of(e.from_addr), _domain_of(e.reply_to)
    return [
        float(len(e.urls)),
        float(len(domains)),
        float(e.html_share),
        float(e.n_attachments),
        float(any(t.lower() in RISKY_EXT for t in e.attachment_types)),
        float(bool(from_domain and reply_domain and from_domain != reply_domain)),
        caps / letters if letters else 0.0,
        1000.0 * text.count("!") / n_chars,
        1000.0 * text.count("$") / n_chars,
        math.log1p(len(text)),
        float(any(_IP_URL_RE.match(u) for u in e.urls)),
        float(any(d in SHORTENERS for d in domains)),
        float(bool(e.from_addr)),
    ]


def header_tokens(e: Email) -> str:
    tokens = []
    if _domain_of(e.from_addr):
        tokens.append("fromdom=" + _domain_of(e.from_addr))
    if _domain_of(e.reply_to):
        tokens.append("replydom=" + _domain_of(e.reply_to))
    mailer = e.headers.get("x-mailer") or e.headers.get("user-agent") or ""
    match = _MAILER_RE.match(mailer.strip())
    if match:
        tokens.append("mailer=" + match.group(0).lower())
    content_type = e.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type:
        tokens.append("ctype=" + content_type)
    if "list-unsubscribe" in e.headers:
        tokens.append("has_list_unsub")
    return " ".join(tokens) if tokens else "no_headers"


def emails_to_frame(emails: Sequence[Email]) -> pd.DataFrame:
    text_cols = {
        "subject": [e.subject or "" for e in emails],
        "text": [e.text or "" for e in emails],
        "head": [(e.subject or "") + "\n" + (e.text or "")[:CHAR_HEAD] for e in emails],
        "hdr": [header_tokens(e) for e in emails],
    }
    struct = pd.DataFrame([structural_features(e) for e in emails], columns=STRUCT_NAMES, dtype=float)
    return pd.concat([pd.DataFrame(text_cols), struct], axis=1)


def make_features(kind: str = "full", *, min_df: int = 2, max_word_features: int = 200_000,
                  max_char_features: int = 300_000) -> ColumnTransformer:
    word = dict(preprocessor=normalize_text, ngram_range=(1, 2), sublinear_tf=True, min_df=min_df, dtype=np.float32)
    blocks = [
        ("subj", TfidfVectorizer(max_features=50_000, **word), "subject"),
        ("body", TfidfVectorizer(max_features=max_word_features, **word), "text"),
    ]
    weights = {"subj": 2.0, "body": 1.0}
    if kind == "full":
        blocks += [
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=min_df,
                                     max_features=max_char_features, dtype=np.float32), "head"),
            ("hdr", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, binary=True, dtype=np.float32), "hdr"),
            ("num", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                              ("scale", StandardScaler())]), STRUCT_NAMES),
        ]
        weights.update(char=1.0, hdr=1.0, num=1.0)
    elif kind != "words":
        raise ValueError(f"unknown feature kind {kind!r}")
    return ColumnTransformer(blocks, transformer_weights=weights, sparse_threshold=1.0)


def n_word_columns(fitted: ColumnTransformer) -> int:
    return sum(len(fitted.named_transformers_[name].vocabulary_) for name in ("subj", "body"))
