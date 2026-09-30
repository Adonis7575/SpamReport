import numpy as np
from scipy import sparse

from spamfilter.features import (STRUCT_NAMES, emails_to_frame, header_tokens, make_features, n_word_columns,
                                 normalize_text, structural_features, url_domain)
from spamfilter.parse import email_from_text, parse_bytes
from tests.builders import make_raw, synthetic_emails


def test_url_domain():
    assert url_domain("http://www.Deals.example.com/x?y=1") == "deals.example.com"
    assert url_domain("www.foo.org/path") == "foo.org"
    assert url_domain("http://[broken") == ""


def test_normalize_text():
    s = normalize_text("Visit http://www.Deals.example.com/x?id=5 or mail me@x.org, call 555-1234 NOW")
    assert "__url__" in s and "urldom_deals_example_com" in s
    assert "__email__" in s and "__num__" in s
    assert s == s.lower() and "555" not in s


def _suspicious():
    raw = make_raw("hi", None,
                   html="<p>FREE!!! $$$</p><a href='http://bit.ly/x'>a</a><a href='http://1.2.3.4/y'>b</a>",
                   sender="a@shop.example", reply_to="z@other.example", attachments=[("x.exe", b"MZ")])
    return parse_bytes(raw).email


def test_structural_features():
    f = dict(zip(STRUCT_NAMES, structural_features(_suspicious())))
    assert len(f) == 13
    assert f["url_count"] == 2 and f["url_domains"] == 2
    assert f["shortener_url"] == 1 and f["ip_url"] == 1
    assert f["reply_to_mismatch"] == 1 and f["risky_attachment"] == 1
    assert f["has_headers"] == 1 and f["html_share"] == 1.0
    assert f["exclaim_per_1k"] > 0 and f["dollar_per_1k"] > 0 and f["caps_ratio"] > 0.5
    headerless = dict(zip(STRUCT_NAMES, structural_features(email_from_text("x", "plain words"))))
    assert headerless["has_headers"] == 0 and headerless["url_count"] == 0


def test_header_tokens():
    toks = header_tokens(_suspicious()).split()
    assert "fromdom=shop.example" in toks and "replydom=other.example" in toks
    assert "ctype=multipart/mixed" in toks
    assert header_tokens(email_from_text("x", "y")) == "no_headers"


def test_make_features_full_and_words():
    emails = synthetic_emails(10, 10)
    frame = emails_to_frame(emails)
    assert list(frame.columns[:4]) == ["subject", "text", "head", "hdr"]
    full = make_features("full", min_df=1)
    X = full.fit_transform(frame)
    assert sparse.issparse(X) and X.shape[0] == 20
    words = make_features("words", min_df=1).fit_transform(frame)
    assert words.shape[1] == n_word_columns(full)
    assert np.allclose(sparse.csr_matrix(X)[:, : words.shape[1]].toarray(), words.toarray())
    assert (words.data >= 0).all()
