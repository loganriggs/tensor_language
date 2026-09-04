#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free adversarial checks for the immutable R592 preregistration review."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT.parent / "polynomial_causal" / (
    "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION.md"
)
PREREG_SHA256 = "870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a"
NAMESPACE = "a8-r585-replacement-group-bootstrap-v1"
TOLERANCE = 1e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_index(cell_id: str, replicate: int, draw: int, groups: int) -> int:
    payload = f"{NAMESPACE}:{cell_id}:{replicate}:{draw}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % groups


def _bilinear(edges: tuple[float, float], values: tuple[float, float]) -> float:
    return sum(edge * value for edge, value in zip(edges, values))


def test_exact_prereg_blob_and_reconstructed_authority_price() -> None:
    assert _sha256(PREREG) == PREREG_SHA256
    phases = {
        "FIT": {"rows": 1_872, "endpoints": 1_728, "directions": 3_744},
        "SELECT": {"rows": 936, "endpoints": 864, "directions": 1_872},
    }
    assert phases["FIT"]["endpoints"] * 4 * 2 == 13_824
    assert phases["SELECT"]["endpoints"] * 4 * 2 == 6_912
    fit = math.ceil(1_728 / 32) + 5 * math.ceil(3_744 / 32)
    select = math.ceil(864 / 32) + 5 * math.ceil(1_872 / 32)
    assert (fit, select, fit + select) == (639, 322, 961)
    assert 3_744 % 32 == 0
    assert 1_872 % 32 == 16


def test_renaming_arm_changes_frozen_bootstrap_identity() -> None:
    prefix = (
        "FIT|two_valid_sources_selector_swap|payload_assignment_0|"
        "s0p0|base_to_donor"
    )
    legacy = prefix + "|score|numerator_mean"
    renamed = prefix + "|coefficient|numerator_mean"
    assert legacy != renamed
    legacy_draws = [_draw_index(legacy, b, k, 72) for b in range(8) for k in range(8)]
    renamed_draws = [_draw_index(renamed, b, k, 72) for b in range(8) for k in range(8)]
    assert legacy_draws != renamed_draws


def test_self_factor_match_does_not_certify_hybrid_factor_transport() -> None:
    # Capture and directed geometries can agree on B(E_x,U_x) while disagreeing
    # on both hybrid terms used to name coefficient and projected-content arms.
    cached_e, cached_u = (1.0, 0.0), (1.0, 0.0)
    live_e, live_u = (0.0, 1.0), (0.0, 1.0)
    donor_e, donor_u = (2.0, 3.0), (4.0, 5.0)
    assert _bilinear(cached_e, cached_u) == _bilinear(live_e, live_u) == 1.0
    assert _bilinear(donor_e, cached_u) != _bilinear(donor_e, live_u)
    assert _bilinear(cached_e, donor_u) != _bilinear(live_e, donor_u)


def test_rms_sufficient_statistic_cannot_certify_elementwise_tolerance() -> None:
    # Both vectors have the same sum of squares and RMS, but only one breaches
    # the registered elementwise 1e-5 threshold.
    safe = (0.8 * TOLERANCE, 0.8 * TOLERANCE)
    unsafe = (math.sqrt(2.0) * 0.8 * TOLERANCE, 0.0)
    safe_sum_sq = sum(value * value for value in safe)
    unsafe_sum_sq = sum(value * value for value in unsafe)
    assert math.isclose(safe_sum_sq, unsafe_sum_sq, rel_tol=0.0, abs_tol=1e-24)
    assert max(abs(value) for value in safe) < TOLERANCE
    assert max(abs(value) for value in unsafe) > TOLERANCE


def test_centered_activity_is_not_legacy_inserted_minus_live_removed() -> None:
    recipient_factor = 1.0
    native_equality_term = 1.00002
    inserted_factor = 1.000005
    centered_delta = inserted_factor - recipient_factor
    legacy_delta = inserted_factor - native_equality_term
    assert abs(centered_delta) < TOLERANCE
    assert abs(legacy_delta) > TOLERANCE
