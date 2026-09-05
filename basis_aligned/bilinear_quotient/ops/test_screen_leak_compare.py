"""Tests for ops/screen_leak_compare.py."""
import os
import subprocess
import sys

OPS = os.path.dirname(os.path.abspath(__file__))
BQ = os.path.dirname(OPS)
TOOL = os.path.join(OPS, "screen_leak_compare.py")
A = os.path.join(BQ, "circuits/fast_screens/numbered_list_cached_value_sufficiency_v3_result.json")
B = os.path.join(BQ, "circuits/fast_screens/numeric_sequence_cross_construction_v1_result.json")
sys.path.insert(0, OPS)
import screen_leak_compare as C


def test_load_parses_sites_and_leaks():
    d = C.load(A)
    assert len(d["sites"]) == 55
    assert d["reason"] == "no_selective_causal_site"
    assert d["leaks"] and d["leaks"] <= set(d["sites"])


def test_recovery_is_present_for_every_ranked_site():
    d = C.load(B)
    assert all(s in d["recovery"] for s in d["sites"])


def test_the_two_live_receipts_share_their_site_set():
    assert set(C.load(A)["sites"]) == set(C.load(B)["sites"])


def test_report_names_the_non_residual_shared_leaker():
    r = subprocess.run([sys.executable, TOOL, A, B], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "attn:08" in r.stdout, r.stdout
    assert "overlap 11" in r.stdout


def test_residual_sites_are_excluded_from_the_informative_list():
    """resid:* leak in both, but patching near the output moves any endpoint -- they must not be reported."""
    r = subprocess.run([sys.executable, TOOL, A, B], capture_output=True, text=True)
    tail = r.stdout.split("NON-residual sites")[1]
    assert "resid:" not in tail, tail
