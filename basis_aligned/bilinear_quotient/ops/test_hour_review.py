"""Tests for ops/hour_review.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hour_review as H


def test_category_reads_the_commit_prefix():
    assert H.category("circuits: retract the null") == "circuits"
    assert H.category("ops: add a tool") == "ops"
    assert H.category("board: coordination") == "board"
    assert H.category("Record Task14 result") == "other"


def test_screens_excludes_canaries_and_respects_the_day_boundary():
    """Regression: _completed.txt has no dates, so a naive filter reported 2825 screens in two hours."""
    runs = H.screens(2)
    assert len(runs) < 50, f"day-boundary leak: {len(runs)} screens in 2h"
    assert all("canary" not in name for _t, name, _c in runs)


def test_commits_returns_sorted_timestamped_subjects():
    rows = H.commits(6)
    assert rows == sorted(rows)
    assert all(isinstance(ts, int) and subject for ts, subject in rows)
