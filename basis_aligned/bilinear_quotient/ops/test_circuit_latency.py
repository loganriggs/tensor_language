"""Tests for ops/circuit_latency.py."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuit_latency as C


def test_slugs_cover_the_known_id_to_filename_mismatches():
    assert "task14_agreement" in C._slugs("subject_verb.number_agreement")
    assert "pronoun" in C._slugs("pronoun_antecedent.gender_reference")
    assert "quote_parity" in C._slugs("quote_parity.pending_close")


def test_rows_parse_the_ledger_and_carry_a_terminal_time():
    rs = C.rows()
    if not rs:
        return
    assert all(isinstance(r["terminal"], datetime.datetime) for r in rs)
    assert all(r["candidate_id"] for r in rs)


def test_target_is_ten_minutes():
    assert C.TARGET_MIN == 10.0


def test_mtime_returns_none_for_a_missing_path():
    assert C._mtime("/definitely/not/here.py") is None


def test_first_existing_picks_the_earliest_match():
    hit = C._first_existing([os.path.join(C.BQ, "ops", "circuit_latency.py")])
    assert hit and hit[1].endswith("circuit_latency.py")
