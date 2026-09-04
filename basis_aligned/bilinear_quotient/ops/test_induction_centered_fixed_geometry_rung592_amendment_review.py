#!/usr/bin/env python3
# BQLANE: cpu
"""Independent model-free attacks on the exact R592 amendment."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"
AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_AMENDMENT.md"
PREREG = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION.md"
BLOCK_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_REVIEW.md"
BLOCK_TEST = OPS / "test_induction_centered_fixed_geometry_rung592_prereg_review.py"
R585_AMENDMENT = POLY / "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_REPLACEMENT_AMENDMENT.md"
R585_MANIFEST = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
HANDOFF_V7 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v7_addendum.json"

EXPECTED_HASHES = {
    AMENDMENT: "5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094",
    PREREG: "870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a",
    BLOCK_REVIEW: "9b76b91995374697b8a828ce042e59d81bfddcbaa5f6e843cb0f32f6b01e57f7",
    BLOCK_TEST: "7356aebd017ba6c6c5ce92176ff95fbffd01d5924b5b7d4cc91dd90e0618b07c",
    R585_AMENDMENT: "98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf",
    R585_MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    HANDOFF_V7: "595b43156117e0ba2e568972f76af81ac4e716ed5537861ac48f13b23d4ed9fd",
}

MACHINE_ARMS = ("replay", "score", "payload", "joint")
INVALID_PREDICATES = {
    "native_full_write_reconstruction_failed",
    "native_equality_remainder_reconstruction_failed",
    "centered_hook_delta_failed",
    "structural_output_identity_failed",
    "nonfinite_observation",
    "fixed_width_token_manifest_failed",
    "directed_native_zero_replay_failed",
    "factor_transport_failed",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest():
    spec = importlib.util.spec_from_file_location("r592_review_r585_manifest", R585_MANIFEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bilinear(edges: tuple[float, float], values: tuple[float, float]) -> float:
    return sum(edge * value for edge, value in zip(edges, values))


def centered_terms(ex, ux, ey, uy):
    baseline = bilinear(ex, ux)
    return {
        "replay": 0.0,
        "score": bilinear(ey, ux) - baseline,
        "payload": bilinear(ex, uy) - baseline,
        "joint": bilinear(ey, uy) - baseline,
    }


def test_exact_amendment_and_frozen_authorities() -> None:
    assert {path: sha256(path) for path in EXPECTED_HASHES} == EXPECTED_HASHES


def test_legacy_manifest_bootstrap_and_cell_censuses_are_unchanged() -> None:
    manifest = load_manifest()
    authority = manifest.build_authority_manifest()
    cells = manifest.build_cell_manifests(authority)
    bootstrap = manifest.expected_bootstrap_cells(cells)
    assert manifest.ARMS == ("score", "payload", "joint")
    assert manifest.BOOTSTRAP_NAMESPACE == "a8-r585-replacement-group-bootstrap-v1"
    for split, expected in {
        "FIT": (1_872, 1_728, 3_744, 124),
        "SELECT": (936, 864, 1_872, 124),
    }.items():
        observed = (
            sum(row["split"] == split for row in authority["rows"]),
            sum(row["split"] == split for row in authority["endpoints"]),
            sum(row["split"] == split for row in authority["directions"]),
            sum(row["cell_id"].startswith(split + "|") for row in bootstrap),
        )
        assert observed == expected
    assert all("|coefficient|" not in row["cell_id"] for row in bootstrap)
    assert all("|projected_content|" not in row["cell_id"] for row in bootstrap)


def test_gate_supersession_is_exact_and_old_failures_cannot_enter_terminal() -> None:
    text = AMENDMENT.read_text()
    for predicate in INVALID_PREDICATES:
        assert predicate in text
    for forbidden in ("canonical_term_failure", "factor_mismatch", "padding_failure"):
        assert f"`{forbidden}`" in text
    assert "**not** an R592 validity gate" in text
    assert "no natural-length or length-sorted comparator is permitted" in text


def test_frozen_centered_terms_and_all_hybrids_detect_self_preserving_drift() -> None:
    cached_e, cached_u = (1.0, 0.0), (1.0, 0.0)
    live_e, live_u = (0.0, 1.0), (0.0, 1.0)
    donor_e, donor_u = (2.0, 3.0), (4.0, 5.0)
    assert bilinear(cached_e, cached_u) == bilinear(live_e, live_u) == 1.0
    # The componentwise/hybrid transport repair catches the ambiguity that a
    # self-product-only check missed.
    assert bilinear(donor_e, cached_u) != bilinear(donor_e, live_u)
    assert bilinear(cached_e, donor_u) != bilinear(live_e, donor_u)
    terms = centered_terms(cached_e, cached_u, donor_e, donor_u)
    assert tuple(terms) == MACHINE_ARMS
    assert terms["replay"] == 0.0
    assert math.isclose(
        terms["joint"] - terms["score"] - terms["payload"],
        bilinear(
            tuple(y - x for x, y in zip(cached_e, donor_e)),
            tuple(y - x for x, y in zip(cached_u, donor_u)),
        ),
    )


def test_centered_activity_is_median_of_actual_site_delta_norms() -> None:
    norms = [0.2, 4.0, 0.6, 1.0]
    centered_activity = (0.6 + 1.0) / 2
    assert centered_activity == 0.8
    native_contraction_errors = [5.0, 5.0, 5.0, 5.0]
    legacy_inserted_minus_removed = sorted(
        abs(delta - error) for delta, error in zip(norms, native_contraction_errors)
    )
    legacy_activity = sum(legacy_inserted_minus_removed[1:3]) / 2
    assert legacy_activity != centered_activity


def test_complete_call_and_raw_byte_arithmetic() -> None:
    fit = math.ceil(1_728 / 32) + 5 * math.ceil(3_744 / 32)
    select = math.ceil(864 / 32) + 5 * math.ceil(1_872 / 32)
    assert (fit, select, fit + select) == (639, 322, 961)
    assert 1_872 // 32 == 58 and 1_872 % 32 == 16

    fit_logit = 3_744 * 4 * 50_257 * 4
    select_logit = 1_872 * 4 * 50_257 * 4
    fit_hook = 3_744 * 4 * 4 * 1_152 * 4
    select_hook = 1_872 * 4 * 4 * 1_152 * 4
    fit_live_u = 3_744 * 4 * 2 * 1_152 * 4
    select_live_u = 1_872 * 4 * 2 * 1_152 * 4
    assert (fit_logit, select_logit) == (3_010_595_328, 1_505_297_664)
    assert (fit_hook, select_hook) == (276_037_632, 138_018_816)
    assert (fit_live_u, select_live_u) == (138_018_816, 69_009_408)
    assert sum((fit_logit, select_logit, fit_hook, select_hook,
                fit_live_u, select_live_u)) == 5_136_977_664


def test_complete_artifact_requires_all_native_relative_logit_differences() -> None:
    required_axes = (
        "native_minus_replay", "score_minus_replay",
        "payload_minus_replay", "joint_minus_replay",
    )
    planted = {axis: [0.0, 0.0] for axis in required_axes}
    assert tuple(planted) == required_axes
    del planted["native_minus_replay"]
    with pytest.raises(AssertionError, match="difference axis"):
        if tuple(planted) != required_axes:
            raise AssertionError("difference axis is missing")


def test_fake_complete_partial_array_is_rejected_by_literal_byte_count() -> None:
    complete_shape = (3_744, 4, 50_257)
    complete_bytes = math.prod(complete_shape) * 4
    partial_shape = (32, 4, 50_257)
    partial_bytes = math.prod(partial_shape) * 4
    assert complete_bytes == 3_010_595_328
    assert partial_bytes != complete_bytes
    with pytest.raises(AssertionError, match="complete FIT logit array byte count"):
        declared_complete = True
        if declared_complete and partial_bytes != complete_bytes:
            raise AssertionError("complete FIT logit array byte count is false")


def test_select_cannot_open_before_complete_valid_held_fit() -> None:
    def may_open_select(fit_calls: int, instrument_valid: bool, scientific_held: bool) -> bool:
        return fit_calls == 639 and instrument_valid and scientific_held

    assert may_open_select(639, True, True)
    assert not may_open_select(638, True, True)
    assert not may_open_select(639, False, True)
    assert not may_open_select(639, True, False)


def test_mixed_machine_namespace_and_live_removal_are_rejected() -> None:
    labels = {
        "replay": "literal_zero_centered_replay",
        "score": "registered_equality_factor_coefficient_swap",
        "payload": "registered_projected_content_swap",
        "joint": "registered_joint_output_factor_swap",
    }
    assert tuple(labels) == MACHINE_ARMS
    assert "coefficient" not in labels and "projected_content" not in labels
    evidence_fields = {
        "recipient_factor_baseline", "planned_centered_delta", "actual_hook_delta"
    }
    assert "live_removed" not in evidence_fields


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the amendment permits call- or chunk-granularity failure but gives only "
        "rectangular complete-arm arrays for prefix diagnostics"
    ),
)
def test_mid_chunk_invalid_diagnostic_has_one_exact_unpadded_array_schema() -> None:
    text = AMENDMENT.read_text()
    # After native, replay, and score have completed for a directed chunk,
    # payload and joint do not exist.  A [directions,4,...] array would have to
    # pad those arms, while a [directions,2,...] array changes the frozen arm
    # axis.  The specification must choose an exact ragged/prefix encoding or
    # require chunk completion before diagnostic publication.
    assert "stop at the first failing completed call or chunk" not in text
    assert "diagnostic_completed_arm_prefix" in text
