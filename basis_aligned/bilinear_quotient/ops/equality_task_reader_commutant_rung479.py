#!/usr/bin/env python3
"""RUNG479 -- gauge-aware blocks of circuit-specific equality readers.

Registered before new reader/state collection:
  pred_a_lawful_collection: exact frozen collection and projected algebra.
  pred_b_nontrivial_shared_algebra: a non-scalar direction beats conjugated controls.
  pred_c_fit_block_structure: its two blocks fit the discovery reader family.
  pred_d_cross_view_blocks: the same projectors transfer across source and half.
  pred_e_circuit_labelled_block: one block preserves circuit-labelled equivalence.
Strong null: invalid collection, no above-control algebra, or no transferable profile.
Discovery-only screen. A survivor is not a circuit until a later paired-response
and exact held-out projector intervention passes.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh
import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_mixed_product_shared_response_rung478 as route_parent
import equality_product_circuit_response_graph_rung477b as parent
import equality_product_circuit_response_graph_rung477 as graph_parent
import equality_mlp_product_term_group_rung467 as product_parent
import equality_mlp_response_form_rung469 as form_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent


PREREG = POLY / "EQUALITY_TASK_READER_COMMUTANT_RUNG479_PREREGISTRATION.md"
ROUTE_RESULT = ROOT / "equality_mixed_product_shared_response_rung478_results.json"
ROUTE_SOURCE = ROOT / "ops/equality_mixed_product_shared_response_rung478.py"
PARENT_RESULT = ROOT / "equality_product_circuit_response_graph_rung477b_results.json"
PARENT_BUNDLE = ROOT / "equality_product_circuit_response_graph_rung477b_bundle.pt"
PARENT_SOURCE = ROOT / "ops/equality_product_circuit_response_graph_rung477b.py"
OUT = ROOT / "equality_task_reader_commutant_rung479_results.json"
BUNDLE = ROOT / "equality_task_reader_commutant_rung479_bundle.pt"
SOURCES = parent.SOURCES
SITES = parent.SITES
MODULES = parent.MODULES
PAIRS = graph_parent.PAIRS
PAIR_NAMES = graph_parent.PAIR_NAMES
MASK_TYPES = parent.MASK_TYPES
HALVES = parent.HALVES
DISCOVERY_STOP = parent.DISCOVERY_STOP
HALF_STOP = parent.HALF_STOP
BATCH = parent.BATCH
STATE_DIM = 1152
OBSERVATION_DIM = 32
CHECK_INDICES = (29, 307, 614, 849, 1106, 1379, 1597, 1843,
                 2115, 2468, 2785, 3091, 3401, 3720, 4087, 4345)
CONTROL_SEEDS = tuple(range(2026090240, 2026090256))
EXPECTED_FORWARDS = parent.EXPECTED_FORWARDS
HASHES = {
    PREREG: "1cd18f1fe75a13995ff746302da708b7ef23855071e4c1919859deb067954bce",
    ROUTE_RESULT: "9a60806575a47765396e919ed6b221b27756fd209a0ae01366172039aae5d9e3",
    ROUTE_SOURCE: "fec94670dc4c744a58c4ded9f3e3ddbba38398e98c65ec9dafc85ecded393c8f",
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


def _cosine(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.dot(left, right) /
                 max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-30))


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    route = json.loads(ROUTE_RESULT.read_text())
    if route.get("rung") != 478 or route.get("pred_a_lawful_deterministic") is not True \
            or route.get("pred_b_sparse_fit") is not True \
            or any(route.get(key) is not False for key in (
                "pred_c_cross_view_transfer", "pred_d_beats_parent_and_control",
                "pred_e_task_selective",
            )) or route.get("strong_null") is not True:
        raise RuntimeError("rung478 frozen route changed")
    rows, positive, circuit_masks, scale, tags, validation_tags, metadata, old_bundle = \
        parent.validate_inputs()
    if len(tags) != OBSERVATION_DIM or len(validation_tags) != 30:
        raise RuntimeError("discovery/validation family split changed")
    expected_backwards = parent.expected_backwards(circuit_masks, tags)
    metadata = {
        **metadata, "rung478_result_sha256": sha256(ROUTE_RESULT),
        "rung477b_result_sha256": sha256(PARENT_RESULT),
        "rung477b_bundle_sha256": sha256(PARENT_BUNDLE),
        "expected_backwards": expected_backwards,
        "observation_subspace_dimension": OBSERVATION_DIM,
        "checksum_product_indices": list(CHECK_INDICES),
    }
    return (rows, positive, circuit_masks, scale, tags, validation_tags,
            metadata, old_bundle)


def collect_statistics(model, rows, positive, circuit_masks, scale, tags,
                       audit_totals, replay):
    gradients = torch.zeros(
        2, len(SOURCES), len(MASK_TYPES), len(SITES), len(tags), STATE_DIM,
        dtype=torch.float64,
    )
    checksum = torch.zeros(
        2, len(SOURCES), len(MASK_TYPES), len(SITES), len(CHECK_INDICES), len(tags),
        dtype=torch.float64,
    )
    counts = torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    covariance = torch.zeros(
        2, len(SOURCES), len(SITES), STATE_DIM, STATE_DIM,
        dtype=torch.float64, device="cuda",
    )
    state_counts = torch.zeros(2, len(SOURCES), len(SITES), dtype=torch.float64)
    check_index = torch.tensor(CHECK_INDICES, device="cuda")
    reconstruction, backwards = 0.0, 0
    device = next(model.parameters()).device
    for start in range(0, DISCOVERY_STOP, BATCH):
        stop = min(start + BATCH, DISCOVERY_STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = source_parent.run_forward(model, tokens, arm="native")
        audit_parent._record_audit(
            audit_totals, "rung479:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = source_parent.run_forward(model, tokens, arm="replay")
        audit_parent._record_audit(
            audit_totals, "rung479:replay", audit, analytical=True, captures=0, patches=0,
        )
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        with torch.no_grad():
            _, absent_products, _, audit, error, absent_states = form_parent._capture_states(
                model, tokens, arm="base", capture_products=True,
            )
        graph_parent._record(audit_totals, "rung479:absent", audit)
        reconstruction = max(reconstruction, error)
        active = []
        positive_masks = []
        for hi, (half_start, half_stop) in enumerate(HALVES):
            positive_selected = parent._half_batch_mask(
                positive, start, stop, half_start, half_stop,
            ).to(device)
            positive_masks.append(positive_selected)
            for ci, tag in enumerate(tags):
                for ki, mask_type in enumerate(MASK_TYPES):
                    selected = parent._half_batch_mask(
                        circuit_masks[tag][mask_type], start, stop, half_start, half_stop,
                    ).to(device)
                    observed = int(selected.sum())
                    counts[hi, ki, ci] += observed
                    if observed:
                        active.append((hi, ki, ci, selected))
        for si, source in enumerate(SOURCES):
            with torch.enable_grad():
                logits, products, writes, audit, error, states = form_parent._capture_states(
                    model, tokens, arm=source_parent.SOURCE_ARMS[source], scale=scale,
                    capture_products=True, gradient_writes=True,
                )
                graph_parent._record(audit_totals, f"rung479:source:{source}", audit)
                reconstruction = max(reconstruction, error)
                nll = graph_parent._nll(logits, batch_rows)
                for hi, selected in enumerate(positive_masks):
                    for mi, site in enumerate(SITES):
                        delta_state = (states[site] - absent_states[site]).float().detach()
                        chosen = delta_state[selected]
                        if chosen.numel():
                            covariance[hi, si, mi] += (chosen.T @ chosen).double()
                            state_counts[hi, si, mi] += len(chosen)
                for ai, (hi, ki, ci, selected) in enumerate(active):
                    site_gradients = torch.autograd.grad(
                        nll[selected].sum(), tuple(writes[site] for site in SITES),
                        retain_graph=ai + 1 < len(active), allow_unused=False,
                    )
                    backwards += 1
                    for mi, (site, gradient) in enumerate(zip(SITES, site_gradients)):
                        gradients[hi, si, ki, mi, ci] += \
                            gradient.float().sum((0, 1)).double().cpu()
                        module = model.transformer.h[MODULES[mi]].mlp
                        reader = gradient.float() @ module.Down.weight.float()
                        delta_product = (products[site] - absent_products[site]).float()
                        local = -(reader[..., check_index] *
                                  delta_product[..., check_index]).sum((0, 1))
                        checksum[hi, si, ki, mi, :, ci] += local.double().cpu()
                del logits, products, writes, states, nll
        del absent_products, absent_states
    return {
        "gradient_sums": gradients, "response_counts": counts,
        "state_covariance_sums": covariance.cpu(), "state_counts": state_counts,
        "checksum_response_sums": checksum, "reconstruction": reconstruction,
        "backwards": backwards,
    }


def _top_subspace(covariance, left, right):
    pieces = []
    for mi in (left, right):
        matrix = covariance[mi].float().cuda()
        pieces.append(matrix / torch.trace(matrix).clamp_min(1e-30))
    combined = (pieces[0] + pieces[1] + pieces[0].T + pieces[1].T) / 2
    values, vectors = torch.linalg.eigh(combined)
    return vectors[:, -OBSERVATION_DIM:], values


def projected_reader_family(model, statistics, pair_index):
    left, right = PAIRS[pair_index]
    covariance = statistics["state_covariance_sums"]
    counts = statistics["response_counts"]
    gradient_sums = statistics["gradient_sums"]
    fit_u, spectrum = _top_subspace(covariance[0, 0], left, right)
    view_overlap, captured = [], []
    for hi in range(2):
        for si in range(2):
            view_u, _ = _top_subspace(covariance[hi, si], left, right)
            overlap = float(torch.linalg.matrix_norm(fit_u.T @ view_u).square()
                            / OBSERVATION_DIM)
            view_overlap.append(overlap)
            endpoint_capture = []
            for mi in (left, right):
                local = covariance[hi, si, mi].float().cuda()
                endpoint_capture.append(float(torch.trace(fit_u.T @ local @ fit_u)
                                              / torch.trace(local).clamp_min(1e-30)))
            captured.append(endpoint_capture)
    projected = np.zeros((2, 2, 2, len(tags := range(OBSERVATION_DIM)),
                          OBSERVATION_DIM, OBSERVATION_DIM), dtype=np.float64)
    identity_error = 0.0
    generator = torch.Generator(device="cuda").manual_seed(47900 + pair_index)
    for endpoint, mi in enumerate((left, right)):
        module = model.transformer.h[MODULES[mi]].mlp
        left_u = module.Left.weight.float() @ fit_u
        right_u = module.Right.weight.float() @ fit_u
        down = module.Down.weight.float()
        for hi in range(2):
            for si in range(2):
                member = gradient_sums[hi, si, 0, mi] / counts[hi, 0, :, None].clamp_min(1)
                control = gradient_sums[hi, si, 1, mi] / counts[hi, 1, :, None].clamp_min(1)
                readers = (member - control).float().cuda()
                coefficients = readers @ down
                raw = torch.einsum("pa,cp,pb->cab", left_u, coefficients, right_u)
                matrices = (raw + raw.transpose(-1, -2)) / 2
                projected[hi, si, endpoint] = matrices.double().cpu().numpy()
                for check in range(4):
                    ci = (check * 7 + hi * 3 + si + endpoint) % OBSERVATION_DIM
                    vector = torch.randn(OBSERVATION_DIM, generator=generator, device="cuda")
                    lhs = torch.sum(coefficients[ci] * ((left_u @ vector) * (right_u @ vector)))
                    rhs = vector @ matrices[ci] @ vector
                    identity_error = max(
                        identity_error,
                        float(torch.abs(lhs - rhs) / torch.abs(lhs).clamp_min(1e-6)),
                    )
    return {
        "matrices": projected, "fit_u": fit_u.double().cpu(),
        "fit_subspace_spectrum_tail": spectrum[-40:].double().cpu().tolist(),
        "view_subspace_overlap": view_overlap,
        "view_captured_state_variance": captured,
        "projected_identity_relative_max": identity_error,
    }


def _symmetric_basis(n):
    columns = []
    for i in range(n):
        vector = np.zeros((n, n)); vector[i, i] = 1
        columns.append(vector.reshape(-1))
    for i in range(n):
        for j in range(i + 1, n):
            vector = np.zeros((n, n))
            vector[i, j] = vector[j, i] = 1 / math.sqrt(2)
            columns.append(vector.reshape(-1))
    return np.stack(columns, axis=1)


SYMMETRIC_BASIS = _symmetric_basis(OBSERVATION_DIM)


def approximate_commutant(matrices):
    matrices = np.asarray(matrices, dtype=np.float64)
    norms = np.linalg.norm(matrices.reshape(len(matrices), -1), axis=1)
    normalized = matrices / np.maximum(norms[:, None, None], 1e-30)
    n = normalized.shape[-1]
    eye = np.eye(n)
    square_mean = np.mean(normalized @ normalized, axis=0)
    operator = np.kron(eye, square_mean) + np.kron(square_mean, eye)
    operator -= 2 * np.mean([np.kron(matrix, matrix) for matrix in normalized], axis=0)
    operator = (operator + operator.T) / 2
    symmetric_operator = SYMMETRIC_BASIS.T @ operator @ SYMMETRIC_BASIS
    values, vectors = eigh(symmetric_operator, subset_by_index=(0, 3), driver="evr")
    direction = (SYMMETRIC_BASIS @ vectors[:, 1]).reshape(n, n)
    direction = (direction + direction.T) / 2
    direction -= np.trace(direction) / n * eye
    direction /= max(np.linalg.norm(direction), 1e-30)
    direction_values, direction_vectors = np.linalg.eigh(direction)
    gaps = np.diff(direction_values)
    eligible = np.arange(1, n - 1)
    cut = int(eligible[np.argmax(gaps[eligible - 1])])
    projector_left = direction_vectors[:, :cut] @ direction_vectors[:, :cut].T
    projector_right = eye - projector_left
    return {
        "lambda0": float(values[0]), "lambda2": float(values[1]),
        "lambda3": float(values[2]), "direction": direction,
        "direction_eigenvalues": direction_values,
        "cut": cut, "block_sizes": [cut, n - cut],
        "projectors": (projector_left, projector_right),
        "symmetric_error": float(np.linalg.norm(direction - direction.T)),
        "scalar_residual": float(abs(np.trace(direction))),
    }


def offblock_summary(matrices, projectors):
    first, second = projectors
    fractions = []
    for matrix in np.asarray(matrices):
        diagonal = first @ matrix @ first + second @ matrix @ second
        fractions.append(float(np.linalg.norm(matrix - diagonal) /
                               max(np.linalg.norm(matrix), 1e-30)))
    return {
        "median": float(np.median(fractions)),
        "p90": float(np.quantile(fractions, .90, method="higher")),
        "maximum": max(fractions),
    }


def profile_report(matrices, projectors, tags):
    blocks = []
    roots = sorted(set(int(tag.split(".")[1]) for tag in tags))
    for bi, projector in enumerate(projectors):
        views, cosines = [], []
        for hi in range(2):
            for si in range(2):
                profiles = []
                for endpoint in range(2):
                    profile = np.einsum("ab,cba->c", projector,
                                        matrices[hi, si, endpoint])
                    profile -= profile.mean()
                    profiles.append(profile)
                cosine = _cosine(profiles[0], profiles[1])
                cosines.append(cosine)
                views.append({"half": hi, "source": SOURCES[si], "cosine": cosine})
        leave_one = []
        for root in roots:
            keep = np.asarray([int(tag.split(".")[1]) != root for tag in tags])
            local = []
            for hi in range(2):
                for si in range(2):
                    profiles = []
                    for endpoint in range(2):
                        profile = np.einsum("ab,cba->c", projector,
                                            matrices[hi, si, endpoint])[keep]
                        profile -= profile.mean()
                        profiles.append(profile)
                    local.append(_cosine(profiles[0], profiles[1]))
            leave_one.append({"omitted_root": root, "minimum_view_cosine": min(local)})
        blocks.append({
            "block": bi, "views": views, "minimum_view_cosine": min(cosines),
            "leave_one_family": leave_one,
        })
    blocks.sort(key=lambda row: -row["minimum_view_cosine"])
    return blocks


def analyze_pair(projected, pair_index, tags):
    matrices = projected["matrices"]
    fit = matrices[0, 0].reshape(2 * OBSERVATION_DIM,
                                 OBSERVATION_DIM, OBSERVATION_DIM)
    real = approximate_commutant(fit)
    fit_offblock = offblock_summary(fit, real["projectors"])
    view_offblock = []
    for hi in range(2):
        for si in range(2):
            local = matrices[hi, si].reshape(2 * OBSERVATION_DIM,
                                             OBSERVATION_DIM, OBSERVATION_DIM)
            view_offblock.append({"half": hi, "source": SOURCES[si],
                                  **offblock_summary(local, real["projectors"])})
    profiles = profile_report(matrices, real["projectors"], tags)
    controls = []
    for seed in CONTROL_SEEDS:
        rng = np.random.default_rng(seed + pair_index * 100)
        rotation = np.linalg.qr(rng.standard_normal((OBSERVATION_DIM,
                                                     OBSERVATION_DIM)))[0]
        altered = matrices.copy()
        altered[:, :, 1] = np.einsum("ab,hscbd,de->hscae",
                                     rotation.T, altered[:, :, 1], rotation)
        control_fit = altered[0, 0].reshape(2 * OBSERVATION_DIM,
                                            OBSERVATION_DIM, OBSERVATION_DIM)
        algebra = approximate_commutant(control_fit)
        control_profiles = profile_report(altered, algebra["projectors"], tags)
        controls.append({
            "seed": seed, "lambda2": algebra["lambda2"],
            "best_minimum_view_profile_cosine": control_profiles[0]["minimum_view_cosine"],
        })
    control_lambdas = [row["lambda2"] for row in controls]
    control_profiles = [row["best_minimum_view_profile_cosine"] for row in controls]
    lambda_q05 = float(np.quantile(control_lambdas, .05, method="lower"))
    profile_q95 = float(np.quantile(control_profiles, .95, method="higher"))
    return {
        "pair": PAIR_NAMES[pair_index], "pair_index": pair_index,
        "lambda0": real["lambda0"], "lambda2": real["lambda2"],
        "lambda3": real["lambda3"], "block_sizes": real["block_sizes"],
        "symmetric_error": real["symmetric_error"],
        "scalar_residual": real["scalar_residual"],
        "fit_offblock": fit_offblock, "view_offblock": view_offblock,
        "profile_blocks": profiles, "controls": controls,
        "control_lambda2_5pct": lambda_q05,
        "control_profile_cosine_95pct": profile_q95,
        "selection_ratio": lambda_q05 / max(real["lambda2"], 1e-30),
        "view_subspace_overlap": projected["view_subspace_overlap"],
        "view_captured_state_variance": projected["view_captured_state_variance"],
        "fit_subspace_spectrum_tail": projected["fit_subspace_spectrum_tail"],
        "projected_identity_relative_max": projected["projected_identity_relative_max"],
    }


def main():
    started = time.time()
    (rows, positive, circuit_masks, scale, tags, validation_tags,
     metadata, old_bundle) = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 479, "model_loaded": False,
            "reader_or_state_outcomes_opened": False,
            "validation_family_outcomes_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_backwards": metadata["expected_backwards"],
            "pairs": len(PAIRS), "controls_per_pair": len(CONTROL_SEEDS),
            "observation_dimension": OBSERVATION_DIM,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung479 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    statistics = collect_statistics(
        model, rows, positive, circuit_masks, scale, tags, audit_totals, replay,
    )
    old_check = old_bundle["response_sums"][:, :, :, :, CHECK_INDICES]
    check_difference = statistics["checksum_response_sums"] - old_check
    checksum_relative = float(check_difference.square().sum() /
                              old_check.square().sum().clamp_min(1e-30))
    covariance = statistics["state_covariance_sums"]
    symmetry_error = float((covariance - covariance.transpose(-1, -2)).abs().max())
    finite = bool(torch.isfinite(covariance).all() and
                  torch.isfinite(statistics["gradient_sums"]).all())
    pair_reports = []
    projected_identity = 0.0
    for pair_index in range(len(PAIRS)):
        projected = projected_reader_family(model, statistics, pair_index)
        projected_identity = max(projected_identity,
                                 projected["projected_identity_relative_max"])
        pair_reports.append(analyze_pair(projected, pair_index, tags))
    pair_reports.sort(key=lambda row: (-row["selection_ratio"], row["pair_index"]))
    selected = pair_reports[0]
    selected_block = selected["profile_blocks"][0]
    nonfit_offblock = [row for row in selected["view_offblock"]
                       if not (row["half"] == 0 and row["source"] == SOURCES[0])]
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12
        and statistics["reconstruction"] <= 1e-10
        and checksum_relative <= 1e-10 and symmetry_error <= 1e-8 and finite
        and projected_identity <= 1e-4
        and int(statistics["response_counts"][:, 0].min()) >= 39
        and int(statistics["response_counts"][:, 1].min()) >= 439
        and sum(row["forwards"] for row in audit_totals.values()) == EXPECTED_FORWARDS
        and statistics["backwards"] == metadata["expected_backwards"]
        and len(validation_tags) == 30
    )
    pred_b = bool(
        selected["lambda2"] <= .25 * selected["control_lambda2_5pct"]
        and selected["symmetric_error"] <= 1e-10
        and selected["scalar_residual"] <= 1e-8
    )
    pred_c = bool(
        min(selected["block_sizes"]) >= 2
        and selected["fit_offblock"]["median"] <= .20
        and selected["fit_offblock"]["p90"] <= .35
    )
    pred_d = bool(
        all(row["median"] <= .30 and row["p90"] <= .50 for row in nonfit_offblock)
        and min(selected["view_subspace_overlap"][1:]) >= .70
    )
    leave_passes = sum(row["minimum_view_cosine"] >= .60
                       for row in selected_block["leave_one_family"])
    pred_e = bool(
        selected_block["minimum_view_cosine"] >= .70
        and selected_block["minimum_view_cosine"]
        >= selected["control_profile_cosine_95pct"] + .15
        and leave_passes >= 5
    )
    strong_null = bool(
        not pred_a or not pred_b
        or all(row["profile_blocks"][0]["minimum_view_cosine"] <= .30
               or row["profile_blocks"][0]["minimum_view_cosine"]
               <= row["control_profile_cosine_95pct"] for row in pair_reports)
    )
    torch.save({
        "schema": "rung479_discovery_reader_state_statistics_v1",
        "gradient_sums": statistics["gradient_sums"],
        "response_counts": statistics["response_counts"],
        "state_covariance_sums": covariance,
        "state_counts": statistics["state_counts"],
        "checksum_response_sums": statistics["checksum_response_sums"],
        "sources": list(SOURCES), "mask_types": list(MASK_TYPES),
        "sites": list(SITES), "discovery_tags": tags,
        "validation_tags_or_responses_included": False,
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 479,
        "claim_level": "discovery_only_gauge_aware_reader_block_screen",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "native_replay": replay,
        "factor_reconstruction_relative_squared_max": statistics["reconstruction"],
        "checksum_vs_rung477b_relative_squared": checksum_relative,
        "covariance_symmetry_max_abs": symmetry_error,
        "projected_quadratic_identity_relative_max": projected_identity,
        "pair_reports": pair_reports, "selected_pair": selected,
        "sealed_attention0_confirmation_opened": False,
        "validation_family_outcomes_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        "audit_totals": audit_totals,
        "execution_price": {
            "outer_forwards": sum(row["forwards"] for row in audit_totals.values()),
            "backwards": statistics["backwards"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
            "covariance_values_saved": int(covariance.numel()),
            "gradient_values_saved": int(statistics["gradient_sums"].numel()),
        },
        "pred_a_lawful_collection": pred_a,
        "pred_b_nontrivial_shared_algebra": pred_b,
        "pred_c_fit_block_structure": pred_c,
        "pred_d_cross_view_blocks": pred_d,
        "pred_e_circuit_labelled_block": pred_e,
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": ("paired_response_and_exact_odd_family_projector_intervention"
                      if all((pred_a, pred_b, pred_c, pred_d, pred_e))
                      else "attention_qk_output_tensor_downstream_decomposition"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 479,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "selected_pair": selected,
        "instrument": {"replay": replay, "reconstruction": statistics["reconstruction"],
                       "checksum_relative_squared": checksum_relative,
                       "projected_identity_relative_max": projected_identity},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
