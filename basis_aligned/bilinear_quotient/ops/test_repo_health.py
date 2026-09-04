"""Tests for the lane-liveness check in ops/repo_health.py."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repo_health as H


def test_only_agent_lanes_gate_the_check():
    assert H.AGENT_LANES == ("Claude", "Codex")


def test_header_regex_matches_a_real_board_entry():
    m = H.LANE_HEADER.match("### 2026-09-04T21:44Z — Claude (ops lane) — something happened")
    assert m and m.group(1) == "2026-09-04T21:44" and m.group(2) == "Claude"


def test_header_regex_matches_a_codex_entry():
    m = H.LANE_HEADER.match("### 2026-09-04T18:18Z — Codex — Task14 Phase 0 design audited")
    assert m and m.group(2) == "Codex"


def test_non_agent_headers_are_parsed_but_filtered_by_agent_lanes():
    """USER DIRECTIVE headers parse as a 'lane' and must not be able to fail the check."""
    m = H.LANE_HEADER.match("### 2026-09-04T16:22Z — USER DIRECTIVE refreshed for Claude")
    assert m and m.group(2) == "USER"
    assert m.group(2) not in H.AGENT_LANES


def test_check_lanes_returns_a_bool_and_a_note():
    ok, note = H.check_lanes()
    assert isinstance(ok, bool) and isinstance(note, str) and note
