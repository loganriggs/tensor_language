#!/usr/bin/env python3
"""RUNG478 -- sparse mixed product gates for a shared downstream response.

Registered before fitting any gate to the corrected discovery response tensor:
  pred_a: lawful corrected input, deterministic solver, and reserved outcomes closed.
  pred_b: one pair/arm sparsely reaches its parent-defined common fit target.
  pred_c: the shared response transfers across source and document views.
  pred_d: mixing beats complete parents and circuit-alignment controls.
  pred_e: the response is task-selective and not one-family driven.
Strong null: invalid input, no two-endpoint fit, or no above-control transfer.
Literal deployed price: zero parameters saved and zero added; CPU discovery only.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import nnls
import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import equality_product_circuit_response_graph_rung477b as parent
import equality_product_circuit_response_graph_rung477 as graph_parent


PREREG = POLY / "EQUALITY_MIXED_PRODUCT_SHARED_RESPONSE_RUNG478_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_product_circuit_response_graph_rung477b_results.json"
PARENT_BUNDLE = ROOT / "equality_product_circuit_response_graph_rung477b_bundle.pt"
PARENT_SOURCE = ROOT / "ops/equality_product_circuit_response_graph_rung477b.py"
OUT = ROOT / "equality_mixed_product_shared_response_rung478_results.json"
SOURCES = parent.SOURCES
SITES = parent.SITES
PAIRS = graph_parent.PAIRS
PAIR_NAMES = graph_parent.PAIR_NAMES
ARMS = ("nonnegative", "signed")
FIT_HALF = 0
FIT_SOURCE = 0
MAX_STEPS = 32
PERMUTATION_SEEDS = tuple(range(2026090200, 2026090216))
HASHES = {
    PREREG: "2aff13348b8487996180ba0496fd8883a9e1416bb4ffea64be007e2504685f5a",
    PARENT_RESULT: "38349612eb9ca8cf480afe63a1c9cad8c258948ed64383680f42dcf7876a2191",
    PARENT_BUNDLE: "c7d976945d1a0fdce627408e2b3dcb8e126c5f6b07e3a50442f0797decb7dd26",
    PARENT_SOURCE: "ebf9c91e0a823cd263ec997ff185822323d41aadb5f53cdee031bfc8c908cd6b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-30)


def _cosine(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-30))


def _scaled_error(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale = float(np.dot(prediction, target) / max(float(np.dot(prediction, prediction)), 1e-30))
    return float(np.linalg.norm(scale * prediction - target) / max(float(np.linalg.norm(target)), 1e-30))


def common_target(left_matrix, right_matrix):
    left_parent = left_matrix.sum(1)
    right_parent = right_matrix.sum(1)
    combined = _unit(left_parent) + _unit(right_parent)
    if np.linalg.norm(combined) <= 1e-12:
        raise ValueError("complete-parent common direction is numerically undefined")
    return _unit(combined), left_parent, right_parent


def matching_pursuit(matrix, target, arm):
    if arm not in ARMS:
        raise ValueError("unknown sparse gate arm")
    matrix = np.asarray(matrix, dtype=np.float64)
    target = _unit(target)
    column_norms = np.linalg.norm(matrix, axis=0)
    chosen, active, coefficients = set(), [], np.empty(0, dtype=np.float64)
    prediction = np.zeros_like(target)
    history = []
    for step in range(MAX_STEPS):
        residual = target - prediction
        correlations = matrix.T @ residual / np.maximum(column_norms, 1e-30)
        if arm == "nonnegative":
            score = correlations.copy()
        else:
            score = np.abs(correlations)
        if chosen:
            score[np.fromiter(chosen, dtype=np.int64)] = -np.inf
        index = int(np.argmax(score))
        if not np.isfinite(score[index]) or score[index] <= 1e-12 or column_norms[index] <= 1e-30:
            break
        chosen.add(index)
        active.append(index)
        submatrix = matrix[:, active]
        if arm == "nonnegative":
            coefficients, _ = nnls(submatrix, target, maxiter=10_000)
        else:
            coefficients = np.linalg.lstsq(submatrix, target, rcond=None)[0]
        prediction = submatrix @ coefficients
        cosine = _cosine(prediction, target)
        error = _scaled_error(prediction, target)
        history.append({"step": step + 1, "cosine": cosine, "scaled_relative_l2": error})
        if cosine >= .95 and error <= .20:
            break
    if coefficients.size:
        keep = np.abs(coefficients) > max(float(np.max(np.abs(coefficients))) * 1e-10, 1e-14)
        active = [active[index] for index in np.flatnonzero(keep)]
        coefficients = coefficients[keep]
    if not active:
        gate = np.zeros(matrix.shape[1], dtype=np.float64)
        prediction = np.zeros_like(target)
    else:
        scale = max(float(np.max(np.abs(coefficients))), 1e-30)
        coefficients = coefficients / scale
        gate = np.zeros(matrix.shape[1], dtype=np.float64)
        gate[np.asarray(active)] = coefficients
        prediction = matrix @ gate
    cosine = _cosine(prediction, target)
    error = _scaled_error(prediction, target)
    return {
        "arm": arm, "indices": active, "coefficients": coefficients,
        "gate": gate, "fit_response": prediction, "fit_cosine": cosine,
        "fit_scaled_relative_l2": error, "steps_attempted": len(chosen),
        "history": history,
        "fit_passes": bool(2 <= len(active) <= MAX_STEPS and cosine >= .95 and error <= .20),
    }


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != "477b" or not all(result.get(key) is True for key in (
        "pred_a_corrected_instrument", "pred_b_native_instability_robust",
        "pred_c_graph_remains_empty", "pred_d_only_half_allocation_changed",
        "pred_e_reserved_outcomes_closed",
    )) or result.get("strong_null_native_coordinate_basis_remains") is not True:
        raise RuntimeError("rung477b full repair receipt changed")
    bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=False)
    sums, counts = bundle.get("response_sums"), bundle.get("response_counts")
    expected_shape = [2, 2, 2, 3, 4608, 32]
    if bundle.get("schema") != "rung477b_split_aware_discovery_response_v1" \
            or list(sums.shape) != expected_shape or list(counts.shape) != [2, 2, 32] \
            or bundle.get("validation_tags_or_responses_included") is not False \
            or bundle.get("raw_tokens_logits_or_hidden_states_included") is not False:
        raise RuntimeError("corrected response bundle contract changed")
    means = sums / counts[:, None, :, None, None, :].clamp_min(1)
    contrast = means[:, :, 0] - means[:, :, 1]
    contrast = contrast - contrast.mean(-1, keepdim=True)
    metadata = {
        "rung477b_result_sha256": sha256(PARENT_RESULT),
        "rung477b_bundle_sha256": sha256(PARENT_BUNDLE),
        "shape": expected_shape, "fit_half": FIT_HALF, "fit_source": SOURCES[FIT_SOURCE],
        "discovery_tags": bundle["discovery_tags"],
        "validation_tags_or_responses_opened": False,
    }
    return means.numpy(), contrast.numpy(), metadata


def fit_candidate(contrast, pair_index, arm, *, fit_permutation=None):
    left, right = PAIRS[pair_index]
    left_matrix = contrast[FIT_HALF, FIT_SOURCE, left].T
    right_matrix = contrast[FIT_HALF, FIT_SOURCE, right].T
    if fit_permutation is not None:
        right_matrix = right_matrix[np.asarray(fit_permutation)]
    try:
        target, left_parent, right_parent = common_target(left_matrix, right_matrix)
    except ValueError:
        return None
    left_fit = matching_pursuit(left_matrix, target, arm)
    right_fit = matching_pursuit(right_matrix, target, arm)
    view_rows = []
    parent_cosines = []
    for hi in range(2):
        for si in range(2):
            left_view = contrast[hi, si, left].T
            right_view = contrast[hi, si, right].T
            left_response = left_view @ left_fit["gate"]
            right_response = right_view @ right_fit["gate"]
            view_rows.append({
                "half": hi, "source": SOURCES[si],
                "cross_mlp_cosine": _cosine(left_response, right_response),
                "left_target_cosine": _cosine(left_response, target),
                "right_target_cosine": _cosine(right_response, target),
                "left_scaled_relative_l2": _scaled_error(left_response, target),
                "right_scaled_relative_l2": _scaled_error(right_response, target),
            })
            parent_cosines.append(_cosine(left_view.sum(1), right_view.sum(1)))
    nonfit = [row for row in view_rows if not (row["half"] == FIT_HALF
                                               and row["source"] == SOURCES[FIT_SOURCE])]
    return {
        "pair": PAIR_NAMES[pair_index], "pair_index": pair_index, "arm": arm,
        "target": target, "left_parent_fit": left_parent, "right_parent_fit": right_parent,
        "left_fit": left_fit, "right_fit": right_fit, "views": view_rows,
        "minimum_nonfit_cross_cosine": min(row["cross_mlp_cosine"] for row in nonfit),
        "minimum_nonfit_endpoint_target_cosine": min(
            min(row["left_target_cosine"], row["right_target_cosine"]) for row in nonfit),
        "minimum_nonfit_parent_cosine": min(parent_cosines[index] for index in (1, 2, 3)),
    }


def _fit_summary(fit):
    return {
        "indices": fit["indices"], "coefficients": fit["coefficients"].tolist(),
        "fit_cosine": fit["fit_cosine"],
        "fit_scaled_relative_l2": fit["fit_scaled_relative_l2"],
        "steps_attempted": fit["steps_attempted"], "history": fit["history"],
        "fit_passes": fit["fit_passes"],
    }


def _candidate_summary(candidate):
    return {
        "pair": candidate["pair"], "pair_index": candidate["pair_index"],
        "arm": candidate["arm"], "left_fit": _fit_summary(candidate["left_fit"]),
        "right_fit": _fit_summary(candidate["right_fit"]), "views": candidate["views"],
        "minimum_nonfit_cross_cosine": candidate["minimum_nonfit_cross_cosine"],
        "minimum_nonfit_endpoint_target_cosine": candidate["minimum_nonfit_endpoint_target_cosine"],
        "minimum_nonfit_parent_cosine": candidate["minimum_nonfit_parent_cosine"],
    }


def controls_for_candidate(contrast, candidate):
    scores = []
    for seed in PERMUTATION_SEEDS:
        permutation = np.random.default_rng(seed).permutation(contrast.shape[-1])
        control = fit_candidate(
            contrast, candidate["pair_index"], candidate["arm"], fit_permutation=permutation,
        )
        score = -1.0 if control is None or not (
            control["left_fit"]["fit_passes"] and control["right_fit"]["fit_passes"]
        ) else control["minimum_nonfit_cross_cosine"]
        scores.append(float(score))
    return scores, float(np.quantile(scores, .95, method="higher"))


def selectivity_report(means, contrast, candidate, tags):
    left, right = PAIRS[candidate["pair_index"]]
    reports = {}
    for mi, fit in ((left, candidate["left_fit"]), (right, candidate["right_fit"])):
        ratios = []
        for hi in range(2):
            for si in range(2):
                member = means[hi, si, 0, mi].T @ fit["gate"]
                control = means[hi, si, 1, mi].T @ fit["gate"]
                ratios.append(float(np.linalg.norm(member) / max(float(np.linalg.norm(control)), 1e-30)))
        reports[SITES[mi]] = {"member_control_norm_ratios": ratios, "minimum_ratio": min(ratios)}
    leave_one = []
    roots = sorted(set(int(tag.split(".")[1]) for tag in tags))
    for root in roots:
        keep = np.asarray([int(tag.split(".")[1]) != root for tag in tags])
        cosines = []
        for hi in range(2):
            for si in range(2):
                left_response = contrast[hi, si, left][:, keep].T @ candidate["left_fit"]["gate"]
                right_response = contrast[hi, si, right][:, keep].T @ candidate["right_fit"]["gate"]
                left_response -= left_response.mean()
                right_response -= right_response.mean()
                cosines.append(_cosine(left_response, right_response))
        leave_one.append({"omitted_root": root, "minimum_view_cosine": min(cosines)})
    return reports, leave_one


def main():
    started = time.time()
    means, contrast, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 478, "model_loaded": False,
            "mixed_gate_outcomes_opened": False, "validation_family_outcomes_opened": False,
            "sealed_opened": False, "pairs": len(PAIRS), "arms": list(ARMS),
            "permutation_controls": len(PERMUTATION_SEEDS),
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung478 output namespace already exists")
    candidates = []
    for pair_index in range(len(PAIRS)):
        for arm in ARMS:
            candidate = fit_candidate(contrast, pair_index, arm)
            if candidate is not None:
                candidates.append(candidate)
    arm_priority = {arm: index for index, arm in enumerate(ARMS)}
    candidates.sort(key=lambda row: (-row["minimum_nonfit_cross_cosine"],
                                     arm_priority[row["arm"]], row["pair_index"]))
    fitted_candidates = [row for row in candidates
                         if row["left_fit"]["fit_passes"] and row["right_fit"]["fit_passes"]]
    selected = fitted_candidates[0] if fitted_candidates else candidates[0]
    control_scores, control_q95 = controls_for_candidate(contrast, selected)
    selectivity, leave_one = selectivity_report(
        means, contrast, selected, metadata["discovery_tags"],
    )
    repeated = fit_candidate(contrast, selected["pair_index"], selected["arm"])
    repeat_exact = bool(
        repeated is not None
        and selected["left_fit"]["indices"] == repeated["left_fit"]["indices"]
        and selected["right_fit"]["indices"] == repeated["right_fit"]["indices"]
        and np.array_equal(selected["left_fit"]["coefficients"],
                           repeated["left_fit"]["coefficients"])
        and np.array_equal(selected["right_fit"]["coefficients"],
                           repeated["right_fit"]["coefficients"])
    )
    pred_a = bool(repeat_exact and np.isfinite(contrast).all())
    pred_b = any(candidate["left_fit"]["fit_passes"]
                 and candidate["right_fit"]["fit_passes"] for candidate in candidates)
    nonfit_rows = [row for row in selected["views"]
                   if not (row["half"] == FIT_HALF and row["source"] == SOURCES[FIT_SOURCE])]
    pred_c = bool(selected["minimum_nonfit_cross_cosine"] >= .80
                  and selected["minimum_nonfit_endpoint_target_cosine"] >= .70)
    pred_d = bool(
        selected["minimum_nonfit_cross_cosine"]
        >= selected["minimum_nonfit_parent_cosine"] + .15
        and selected["minimum_nonfit_cross_cosine"] >= control_q95 + .15
    )
    pred_e = bool(min(row["minimum_ratio"] for row in selectivity.values()) >= 1.5
                  and sum(row["minimum_view_cosine"] >= .70 for row in leave_one) >= 5)
    any_fit = pred_b
    any_transfer = any(candidate["minimum_nonfit_cross_cosine"] > .50
                       for candidate in candidates
                       if candidate["left_fit"]["fit_passes"] and candidate["right_fit"]["fit_passes"])
    strong_null = bool(not pred_a or not any_fit or not any_transfer
                       or selected["minimum_nonfit_cross_cosine"] <= control_q95)
    result = {
        "status": "complete", "rung": 478,
        "claim_level": "discovery_only_sparse_mixed_response_gate",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "sealed_attention0_confirmation_opened": False,
        "validation_family_product_responses_opened": False,
        "candidates": [_candidate_summary(candidate) for candidate in candidates],
        "selected_candidate": _candidate_summary(selected),
        "alignment_destroyed_control_scores": control_scores,
        "alignment_destroyed_control_95pct": control_q95,
        "selectivity": selectivity, "leave_one_family": leave_one,
        "solver_repeat_exact": repeat_exact,
        "execution_price": {"model_forwards": 0, "model_backwards": 0,
                            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
                            "left_active_terms": len(selected["left_fit"]["indices"]),
                            "right_active_terms": len(selected["right_fit"]["indices"])},
        'pred_a_lawful_deterministic': pred_a,
        'pred_b_sparse_fit': pred_b,
        'pred_c_cross_view_transfer': pred_c,
        'pred_d_beats_parent_and_control': pred_d,
        'pred_e_task_selective': pred_e,
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": ("heldout_family_exact_weighted_intervention" if all(
            (pred_a, pred_b, pred_c, pred_d, pred_e)) else "gauge_aware_block_bilinear_response_factors"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 478,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "selected": result["selected_candidate"],
        "control_scores": control_scores, "control_95pct": control_q95,
        "selectivity": selectivity, "leave_one_family": leave_one,
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
