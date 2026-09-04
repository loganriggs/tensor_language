#!/usr/bin/env python3
# BQLANE: cpu
"""Owner tests for the prospective R592 producer core."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).with_name("induction_centered_fixed_geometry_rung592.py")
SPEC = importlib.util.spec_from_file_location("r592_owner", MODULE)
assert SPEC and SPEC.loader
r592 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r592)


@pytest.fixture(scope="module")
def authority():
    return r592.load_authority()[1]


def test_exact_authority_schedule_and_price(authority) -> None:
    fit = r592.build_phase_manifest(authority, "FIT")
    select = r592.build_phase_manifest(authority, "SELECT")
    assert len(fit["calls"]) == 639
    assert len(select["calls"]) == 322
    assert all(call["physical_width"] == 30 for call in fit["calls"] + select["calls"])
    assert [call["batch_size"] for call in select["calls"][-5:]] == [16] * 5
    assert [call["call_kind"] for call in select["calls"][-5:]] == list(r592.DIRECTED_KINDS)


def test_paired_calls_share_literal_token_record(authority) -> None:
    fit = r592.build_phase_manifest(authority, "FIT")
    for chunk in range(117):
        calls = [row for row in fit["calls"] if row["call_id"].startswith(f"FIT:directed:{chunk:04d}:")]
        assert len(calls) == 5
        assert len({row["token_record_id"] for row in calls}) == 1
        assert len({row["token_sha256"] for row in calls}) == 1
        token = fit["token_arrays"][calls[0]["token_record_id"]]
        assert token.dtype == np.dtype("<i8") and token.shape == (32, 30)


def test_centered_formula_and_replay_are_exact() -> None:
    rng = np.random.default_rng(44)
    ex = rng.normal(size=(5, 4, 2)).astype("<f4")
    ux = rng.normal(size=(5, 4, 2, 1152)).astype("<f4")
    ey = rng.normal(size=(5, 4, 2)).astype("<f4")
    uy = rng.normal(size=(5, 4, 2, 1152)).astype("<f4")
    delta = r592.centered_deltas(ex, ux, ey, uy)
    xx = r592.bilinear(ex, ux)
    assert delta.shape == (5, 4, 4, 1152)
    assert np.array_equal(delta[:, 0], np.zeros_like(xx))
    assert np.array_equal(delta[:, 1], r592.bilinear(ey, ux) - xx)
    assert np.array_equal(delta[:, 2], r592.bilinear(ex, uy) - xx)
    assert np.array_equal(delta[:, 3], r592.bilinear(ey, uy) - xx)
    assert r592.mixed_identity_error(delta, ex, ux, ey, uy) <= 1e-5


def test_transport_checks_components_and_all_hybrids() -> None:
    e = np.ones((2, 4, 2), dtype="<f4")
    u = np.ones((2, 4, 2, 1152), dtype="<f4")
    maxima = r592.transport_maxima(e, u, 2 * e, 3 * u, e.copy(), u.copy())
    assert maxima == {"e": 0.0, "u": 0.0, "xx": 0.0, "yx": 0.0, "xy": 0.0, "yy": 0.0}
    live = u.copy(); live[0, 0, 0, 0] += 2e-5
    maxima = r592.transport_maxima(e, u, 2 * e, 3 * u, e, live)
    assert maxima["u"] > 1e-5 and maxima["yx"] > 1e-5


def test_activity_is_actual_centered_delta_not_removed_term() -> None:
    delta = np.zeros((2, 4, 4, 1152), dtype="<f4")
    delta[:, 2, :, 0] = np.asarray([1, 2, 3, 100], dtype="<f4")
    observed = r592.activity(delta)
    assert np.array_equal(observed[:, 0], np.zeros(2))
    assert np.array_equal(observed[:, 2], np.full(2, 2.5))


def test_call_array_contract_rejects_wrong_width_and_partial_placeholders() -> None:
    call = {"batch_size": 16, "call_kind": "score"}
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (dtype, shape) in r592.mandatory_call_shapes(call).items()
    }
    r592.validate_call_arrays(call, arrays)
    bad = dict(arrays)
    bad["tokens.npy"] = np.zeros((16, 29), dtype="<i8")
    with pytest.raises(ValueError, match="dtype/shape"):
        r592.validate_call_arrays(call, bad)
    bad = dict(arrays); bad["payload_placeholder.npy"] = np.zeros((16,), dtype="<f4")
    with pytest.raises(ValueError, match="missing or extra"):
        r592.validate_call_arrays(call, bad)


def test_prefix_and_failure_precedence() -> None:
    manifest = [{"call_id": str(i)} for i in range(5)]
    r592.validate_prefix(manifest, manifest[:3])
    with pytest.raises(ValueError, match="prefix"):
        r592.validate_prefix(manifest, [manifest[0], manifest[2]])
    assert r592.first_failure([
        "structural_output_identity_failed", "nonfinite_observation"
    ]) == "nonfinite_observation"
    with pytest.raises(ValueError, match="forbidden"):
        r592.first_failure(["factor_mismatch"])


def test_nonfinite_masks_are_one_to_one_and_exact() -> None:
    arrays = {
        "logits.npy": np.array([[np.nan, 1]], dtype="<f4"),
        "hook_deltas.npy": np.array([[np.inf, 0]], dtype="<f4"),
        "tokens.npy": np.zeros((1, 2), dtype="<i8"),
    }
    masks, index = r592.nonfinite_mask_records(arrays)
    assert set(masks) == {
        "nonfinite_masks/logits.mask.npy",
        "nonfinite_masks/hook_deltas.mask.npy",
    }
    assert [row["raw_filename"] for row in index] == ["hook_deltas.npy", "logits.npy"]
    assert all(row["nonfinite_count"] == 1 for row in index)
    assert r592.canonical_mask_name("logits.npy") == "nonfinite_masks/logits.mask.npy"
    for unsafe in ("../logits.npy", "/logits.npy", "nested/logits.npy"):
        with pytest.raises(ValueError, match="unsafe"):
            r592.canonical_mask_name(unsafe)


def test_complete_evidence_and_terminal_shapes() -> None:
    fit = r592.phase_evidence_schema("FIT")
    select = r592.phase_evidence_schema("SELECT")
    assert fit["hook_deltas.npy"]["shape"] == [3744, 4, 4, 1152]
    assert fit["logit_differences.npy"]["shape"] == [3744, 4, 50257]
    assert select["directed_live_u.npy"]["shape"] == [1872, 4, 2, 1152]
    assert r592.terminal_contract("fit_scientific_null", 639)["namespace"] == "normal"
    assert r592.terminal_contract("select_runtime_invalid", 639, 1)["namespace"] == "invalid"
    with pytest.raises(ValueError, match="envelope"):
        r592.terminal_contract("held", 639, 321)


def test_dryrun_is_model_free_and_preserves_legacy_ids() -> None:
    result = r592.build_dryrun()
    assert result["model_forwards"] == result["model_backwards"] == 0
    assert result["model_weights_updated"] is False
    assert result["phase_counts"]["FIT"]["calls"] == 639
    assert result["phase_counts"]["SELECT"]["calls"] == 322
    assert result["bootstrap_cell_count_by_split"] == {"FIT": 124, "SELECT": 124}
    assert result["machine_arm_order"] == ["replay", "score", "payload", "joint"]
    assert result["legacy_machine_ids_observed"] == ["joint", "payload", "score"]
    assert result["select_opened"] is result["final_opened"] is result["ood_opened"] is False
