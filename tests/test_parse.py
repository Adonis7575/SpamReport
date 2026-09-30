import json

import pytest

from spamfilter.message import Email
from spamfilter.parse import email_from_text, norm_hash, parse_bytes
from tests.builders import make_raw


def test_plain_message_fields():
    r = parse_bytes(make_raw("Lunch?", "Want to get lunch at noon?", reply_to="x@other.org"),
                    source="s", folder="Inbox", label="ham")
    e = r.email
    assert r.error is None
    assert e.subject == "Lunch?" and "lunch at noon" in e.text
    assert e.from_addr == "alice@example.com" and e.reply_to == "x@other.org"
    assert e.date.tzinfo is not None and e.date.year == 2024
    assert (e.source, e.folder, e.label) == ("s", "Inbox", "ham")
    assert len(e.id) == 16 and e.id == e.raw_sha256[:16]


def test_html_only_message_is_converted_to_text():
    raw = make_raw("Sale", None, html="<html><style>p{color:red}</style><p>Big <b>SALE</b> today</p>"
                                      "<a href='http://shop.example.biz/x'>go</a></html>")
    e = parse_bytes(raw).email
    assert "Big SALE today" in e.text and "color:red" not in e.text
    assert e.html_share == 1.0
    assert "http://shop.example.biz/x" in e.urls


def test_multipart_alternative_prefers_plain():
    e = parse_bytes(make_raw("Hi", "plain version", html="<p>html version</p>")).email
    assert "plain version" in e.text and "html version" not in e.text
    assert 0 < e.html_share < 1


def test_attachments_are_counted_not_read():
    e = parse_bytes(make_raw("Invoice", "see attached", attachments=[("invoice.exe", b"MZ\x00\x01")])).email
    assert e.n_attachments == 1 and e.attachment_types == ["exe"]
    assert "MZ" not in e.text


def test_encoded_subject_and_unknown_charset():
    raw = (b"From: a@example.com\nSubject: =?utf-8?b?w4ljaGFudGlsbG9u?=\n"
           b"Content-Type: text/plain; charset=x-unknown-charset\n\ncaf\xe9 menu\n")
    e = parse_bytes(raw).email
    assert e.subject == "Échantillon"
    assert "café menu" in e.text


def test_quoted_printable_and_base64_bodies():
    qp = (b"From: a@example.com\nSubject: q\nContent-Type: text/plain; charset=utf-8\n"
          b"Content-Transfer-Encoding: quoted-printable\n\nhello =E2=82=AC world\n")
    assert "hello € world" in parse_bytes(qp).email.text
    b64 = (b"From: a@example.com\nSubject: b\nContent-Type: text/plain\n"
           b"Content-Transfer-Encoding: base64\n\naGVsbG8gYmFzZTY0\n")
    assert "hello base64" in parse_bytes(b64).email.text


@pytest.mark.parametrize("raw, reason", [
    (b"", "empty"),
    (b"   \n\n", "empty"),
    (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj", "not-an-email"),
])
def test_rejects_non_mail(raw, reason):
    r = parse_bytes(raw)
    assert r.email is None and r.error == reason


# Review Focus 2: garbage or impossible dates must never break parsing.
@pytest.mark.parametrize("date_value", ["not a date", "Mon, 32 Foo 2024 99:99:99 +0000",
                                        "Tue, 01 Jan 1901 00:00:00 +0000"])
def test_bad_dates_become_none(date_value):
    raw = f"From: a@example.com\nSubject: x\nDate: {date_value}\n\nbody text here\n".encode()
    r = parse_bytes(raw)
    assert r.error is None and r.email.date is None


def test_date_without_timezone_is_utc():
    raw = b"From: a@example.com\nSubject: x\nDate: 01 Jan 2024 10:00:00\n\nbody\n"
    assert parse_bytes(raw).email.date.utcoffset().total_seconds() == 0


def test_norm_hash_ignores_numbers_urls_and_spacing():
    a = norm_hash("Order 123", "Your code is 4411 see http://a.example/x  now", "r1")
    b = norm_hash("Order 999", "Your  code is 9876 see http://b.example/y now", "r2")
    assert a == b and a not in ("r1", "r2")


def test_norm_hash_short_content_falls_back_to_raw_hash():
    assert norm_hash("", "hi", "rawhash") == "rawhash"


def test_huge_bodies_are_truncated():
    e = parse_bytes(make_raw("big", "x" * 300_000)).email
    assert len(e.text) == 200_000


def test_email_from_text():
    e = email_from_text("cheap meds", "buy now", source="base-enron", folder="spam", label="spam")
    assert (e.subject, e.text, e.from_addr, e.label) == ("cheap meds", "buy now", "", "spam")
    assert e.headers == {}


def test_email_dict_roundtrip():
    e = parse_bytes(make_raw()).email
    assert Email.from_dict(json.loads(json.dumps(e.to_dict()))) == e
