import itertools

from spamfilter.dataset import dedupe, split_base, split_user
from spamfilter.parse import parse_bytes
from tests.builders import make_raw, synthetic_emails


def test_dedupe_exact_near_and_conflicts():
    emails = synthetic_emails(2, 2, seed=1)
    ham = parse_bytes(make_raw("Meeting notes", "Please review the attached meeting notes today"), label="ham").email
    spam = parse_bytes(make_raw("Meeting notes", "Please  review the attached meeting notes today",
                                sender="x@y.biz"), label="spam").email
    near_a = parse_bytes(make_raw("Invoice 1001", "Your invoice number 1001 is ready to view"), label="ham").email
    near_b = parse_bytes(make_raw("Invoice 2002", "Your invoice number 2002 is ready to view"), label="ham").email
    kept, stats = dedupe(emails + [emails[0], ham, spam, near_a, near_b])
    assert stats.exact == 1
    assert (stats.conflict_groups, stats.conflict_emails) == (1, 2)
    assert stats.near == 2
    assert len(kept) == 6 and ham not in kept and spam not in kept


def test_base_split_keeps_groups_together_and_stratifies():
    emails = synthetic_emails(60, 90, seed=2)
    for i in range(0, 40, 2):  # make 20 near-duplicate pairs
        emails[i + 1].norm_body_hash = emails[i].norm_body_hash
        emails[i + 1].label = emails[i].label
    s = split_base(emails, seed=0)
    parts = [s.train, s.val, s.test]
    assert sum(map(len, parts)) == 150
    for a, b in itertools.combinations(parts, 2):
        assert not ({e.norm_body_hash for e in a} & {e.norm_body_hash for e in b})
    assert 0.12 <= len(s.test) / 150 <= 0.28
    assert {e.label for e in s.test} == {"spam", "ham"}
    assert s.time_based is False


def test_user_split_is_time_ordered():
    emails = synthetic_emails(100, 200, seed=3)  # hourly dates, shuffled labels
    s = split_user(emails)
    assert s.time_based is True
    assert (len(s.train), len(s.val), len(s.test)) == (204, 36, 60)
    assert max(e.date for e in s.train) < min(e.date for e in s.val)
    assert max(e.date for e in s.val) < min(e.date for e in s.test)


def test_user_split_undated_to_train_and_small_fallback():
    emails = synthetic_emails(100, 200, seed=4)
    emails[0].date = None
    s = split_user(emails)
    assert emails[0] in s.train and s.time_based
    small = synthetic_emails(20, 40, seed=5)
    fallback = split_user(small)
    assert fallback.time_based is False
    assert sum(map(len, (fallback.train, fallback.val, fallback.test))) == 60
