"""Tests for ops/failure_census.py -- the failure/retry accounting no receipt can show."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import failure_census as F


def test_parse_reads_completed_lines(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("13:21 alpha exit=0\n13:25 beta exit=1\nnot a line\n13:29 beta exit=0\n")
    ev = F.parse(str(p))
    assert ev == [(13 * 60 + 21, "alpha", 0), (13 * 60 + 25, "beta", 1), (13 * 60 + 29, "beta", 0)]


def test_retry_pairs_matches_a_failure_to_its_next_same_name_run():
    ev = [(100, "beta", 1), (104, "beta", 0)]
    assert F.retry_pairs(ev) == [("beta", 4, 0)]


def test_retry_pairs_ignores_a_different_script_in_between():
    """The next run of the SAME name is the retry; an unrelated script between them is not."""
    ev = [(100, "beta", 1), (101, "gamma", 0), (105, "beta", 0)]
    assert F.retry_pairs(ev) == [("beta", 5, 0)]


def test_retry_pairs_respects_the_window():
    ev = [(100, "beta", 1), (200, "beta", 0)]
    assert F.retry_pairs(ev, window_minutes=15) == []


def test_retry_pairs_records_a_retry_that_also_failed():
    ev = [(100, "beta", 1), (103, "beta", 1)]
    assert F.retry_pairs(ev) == [("beta", 3, 1)]
