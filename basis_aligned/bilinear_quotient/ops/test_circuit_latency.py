"""Tests for ops/circuit_latency.py."""
import datetime
import os
import sys
import tempfile

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


def test_completed_events_advance_the_date_at_midnight():
    with tempfile.NamedTemporaryFile("w", delete=False) as stream:
        stream.write("23:58 run_before exit=0\n")
        stream.write("00:04 run_after exit=0\n")
        stream.write("00:04 bilin18_canary2 exit=0\n")
        path = stream.name
    try:
        events = list(C._completed_events(path, datetime.date(2026, 9, 5)))
    finally:
        os.unlink(path)
    assert events[0][0] == datetime.datetime(2026, 9, 4, 23, 58)
    assert events[1][0] == datetime.datetime(2026, 9, 5, 0, 4)
    assert events[2][0] == datetime.datetime(2026, 9, 5, 0, 4)


def test_runner_rows_keep_post_midnight_executions(monkeypatch, tmp_path):
    completed = tmp_path / "_completed.txt"
    completed.write_text("23:58 run_before exit=0\n00:04 run_after exit=0\n")
    monkeypatch.setattr(C, "BQ", str(tmp_path.parent))
    runlogs = tmp_path.parent / "runlogs"
    runlogs.mkdir(exist_ok=True)
    (runlogs / "_completed.txt").write_text(completed.read_text())
    known = [
        {"terminal": datetime.datetime(2026, 9, 4, 23, 50)},
        {"terminal": datetime.datetime(2026, 9, 5, 0, 10)},
    ]
    rows = C.runner_rows(known)
    assert [row["terminal"] for row in rows] == [
        datetime.datetime(2026, 9, 4, 23, 58),
        datetime.datetime(2026, 9, 5, 0, 4),
    ]


def test_merge_keeps_a_second_runner_in_the_same_minute():
    minute = datetime.datetime(2026, 9, 5, 7, 34)
    receipt = {"candidate_id": "quote.receipt", "terminal": minute.replace(second=31)}
    first = {"candidate_id": "[runner] run_quote", "terminal": minute}
    second = {"candidate_id": "[runner] run_bracket", "terminal": minute}
    merged = C.merge_receipts_and_runner_rows([receipt, first, second])
    assert {row["candidate_id"] for row in merged} == {
        "quote.receipt", "[runner] run_bracket",
    }
