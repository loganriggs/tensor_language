import sys
from pathlib import Path

import torch


OPS = Path(__file__).resolve().parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import equality_matcher_causal_action_quotient_rung498 as rung


def test_fit_report_exact_and_wrong_sign():
    native = torch.tensor([1.0, 2.0, -1.0])
    exact = rung._fit_report(native, 2 * native)
    assert exact["cosine"] > .999999
    assert exact["positive_fit_scale"] > 0
    assert exact["scaled_residual"] < 1e-12
    wrong = rung._fit_report(native, -native)
    assert wrong["cosine"] < -.999999
    assert wrong["positive_fit_scale"] < 0
    assert wrong["scaled_residual"] == 1.0


def test_discovery_price_and_action_identity_are_frozen():
    assert rung.DONORS == ("L5H5", "L7H3")
    assert rung.PAIRS == ((0, 3), (1, 3))
    assert rung.BACKGROUNDS == ("early_present", "early_absent")
    assert rung.STATES == (
        "late_native", "late_absent", "score_donor", "payload_donor", "whole_donor")
    assert (1 + 19) * ((rung.DISCOVERY[1] - rung.DISCOVERY[0]) // rung.BATCH) == 2500


def test_positive_scale_residual_is_scale_invariant():
    native = torch.tensor([1.0, -2.0, 4.0, .5])
    report = rung._fit_report(native, native / 7)
    assert abs(report["positive_fit_scale"] - 7) < 1e-6
    assert report["scaled_residual"] < 1e-7
