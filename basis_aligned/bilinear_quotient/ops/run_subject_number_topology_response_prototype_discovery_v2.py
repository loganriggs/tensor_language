#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_site_instrument pred_b_topology_response_transfer pred_c_frozen_law_and_null
"""Corrected grouped-MLP6/7 topology-conditioned response prototype discovery."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_managed_runner as managed
import run_subject_number_topology_response_prototype_discovery_v1 as base


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
CORRECTION = POLY / "SUBJECT_NUMBER_TOPOLOGY_RESPONSE_PROTOTYPE_DISCOVERY_V2_CORRECTION.md"
INVALID = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_topology_response_prototype_discovery_v1_result.json"
BINDING = POLY / "SUBJECT_NUMBER_TOPOLOGY_RESPONSE_PROTOTYPE_DISCOVERY_V2_BINDING.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_topology_response_prototype_discovery_v2_result.json"
PREDICTION_REGISTRY = dict(base.PREDICTION_REGISTRY)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"rows": base.ROWS, "artifact": base.ARTIFACT,
             "fixed_binding": base.FIXED_BINDING, "parent": base.PARENT,
             "audit": base.AUDIT, "v1_preregistration": base.PREREG,
             "v1_runner": Path(base.__file__), "invalid_v1": INVALID,
             "correction": CORRECTION}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["site_positions"] != list(base.SITE_POSITIONS) \
            or binding["rank"] != base.RANK or binding["ridge_fraction"] != base.RIDGE_FRACTION \
            or binding["nulls"] != base.NULLS or binding["seed"] != base.SEED \
            or binding["price"] != base.PRICE:
        raise ValueError("binding changed")
    frozen = json.loads(base.ROWS.read_text())
    artifact = json.loads(base.ARTIFACT.read_text())
    fixed = json.loads(base.FIXED_BINDING.read_text())
    invalid = json.loads(INVALID.read_text())
    if invalid["terminal"] != "invalid" or frozen["row_count"] != 16 \
            or artifact["terminal"] != "response_weighted_prototypes_frozen_opened_only":
        raise ValueError("parent status changed")
    return binding, frozen, artifact, fixed


def plan():
    binding, frozen, _, fixed = load_bound()
    return {"schema": "subject_number_topology_response_prototype_discovery_v2_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "opened_authority": True, "correction": "grouped_YZ_hybrid_not_full_opposite_state",
            "rows": frozen["row_count"], "templates": frozen["templates"],
            "site_positions": list(base.SITE_POSITIONS),
            "fixed_coefficient": fixed["coefficients"]["singular_to_plural.cardinality_4"],
            "rank": base.RANK, "ridge_fraction": base.RIDGE_FRACTION,
            "nulls": base.NULLS, "price": base.PRICE,
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, frozen, artifact, fixed = load_bound()
    torch, F, facade = base.prior_runner.tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    device = next(model.parameters()).device
    rows = frozen["rows"]; n = len(rows)
    native_tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    plurals = base.plural_token_ids(rows)
    axis = torch.tensor(artifact["native_axis"], dtype=torch.float32, device=device)
    old_prototype = torch.tensor(artifact["prototypes"]["singular_to_plural"],
                                 dtype=torch.float32, device=device)
    beta = torch.tensor(artifact["interaction_beta"], dtype=torch.float64, device=device)
    fixed_alpha = float(fixed["coefficients"]["singular_to_plural.cardinality_4"])
    attention = model.transformer.h[base.prior_runner.tangent.parent.LAYER].attn
    templates = np.asarray([row["template_id"] for row in rows])
    rng = np.random.default_rng(base.SEED)
    evidence = []; fit_audit = {}
    exact_s_all, mean_s_all, candidate_s_all, old_s_all = [], [], [], []
    exact_alpha_all, mean_alpha_all, candidate_alpha_all, old_alpha_all = [], [], [], []
    target_alpha_all = []; null_alpha_all = [[] for _ in range(base.NULLS)]
    head_errors = []; hybrid_errors = []; closure_errors = []
    gradient_rows = 0; sequence_count = 0

    for site_index, position in enumerate(base.SITE_POSITIONS):
        opposite_tokens = native_tokens.clone()
        for row_index, row in enumerate(rows):
            opposite_tokens[row_index, position] = plurals[row["sites"][site_index]["subject"]]
        tokens = torch.cat([native_tokens, opposite_tokens])
        finals = torch.full((2 * n,), position, dtype=torch.long, device=device)
        with torch.no_grad():
            _, captured, projection, closure, inputs = base.prior_runner.tangent.parent._decomposed_forward(
                model, tokens, finals, torch, F, facade)
        sequence_count += len(tokens)
        native_captured = base.slice_values(captured, slice(0, n))
        native_inputs = base.slice_values(inputs, slice(0, n))
        opposite_inputs = base.slice_values(inputs, slice(n, 2 * n))
        _, hybrid_raw, hybrid_error = base.prior_runner.tangent.parent._hybrid_input(
            native_inputs, opposite_inputs, "YZ", F)
        x = native_inputs["raw_state"][:, position]
        delta = hybrid_raw[:, position] - x
        function = base.prior_runner.head_function_at(model, native_captured, projection,
                                                      position, attention, torch, F)
        with torch.no_grad():
            h0 = function(x)
        head_errors.append(float((h0 - native_captured["head"]).abs().max()))
        hybrid_errors.append(float(hybrid_error))
        closure_errors.extend([closure["input_state_closure_max_absolute_error"],
                               closure["input_normalized_closure_max_absolute_error"]])

        for held_out in sorted(set(templates)):
            train_ids = np.flatnonzero(templates != held_out)
            test_ids = np.flatnonzero(templates == held_out)
            train = torch.tensor(train_ids, dtype=torch.long, device=device)
            test = torch.tensor(test_ids, dtype=torch.long, device=device)
            train_function = base.prior_runner.head_function_at(
                model, base.slice_values(native_captured, train), projection,
                position, attention, torch, F)
            test_function = base.prior_runner.head_function_at(
                model, base.slice_values(native_captured, test), projection,
                position, attention, torch, F)
            train_x, train_delta = x[train], delta[train]
            test_x, test_delta = x[test], delta[test]
            mean_p = train_delta.mean(0)
            centered = train_delta - mean_p
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            modes = vh[:base.RANK].T.contiguous()
            with torch.no_grad():
                train_h0 = train_function(train_x)
                train_h1 = train_function(train_x + train_delta)
                train_hm = train_function(train_x + mean_p)
            point = (train_x + mean_p).detach().requires_grad_(True)
            scalar = (train_function(point) @ axis).sum()
            gradient = torch.autograd.grad(scalar, point, create_graph=False)[0]
            design = (gradient @ modes).detach().double().cpu().numpy()
            exact_train_s = ((train_h1 - train_h0) @ axis).detach().double().cpu().numpy()
            mean_train_s = ((train_hm - train_h0) @ axis).detach().double().cpu().numpy()
            residual = exact_train_s - mean_train_s
            coefficients, ridge = base.ridge_coefficients(design, residual)
            candidate = mean_p + modes @ torch.tensor(coefficients, dtype=modes.dtype, device=device)
            nulls = []
            for _ in range(base.NULLS):
                null_coefficients, _ = base.ridge_coefficients(
                    design, residual[rng.permutation(len(residual))])
                nulls.append(mean_p + modes @ torch.tensor(
                    null_coefficients, dtype=modes.dtype, device=device))
            gradient_rows += len(train)
            fit_audit[f"site{site_index + 1}|{held_out}"] = {
                "training_rows": len(train_ids), "held_out_rows": len(test_ids),
                "rank": base.RANK, "ridge": ridge, "mean_norm": float(mean_p.norm()),
                "candidate_norm": float(candidate.norm()),
                "norm_ratio": float(candidate.norm() / mean_p.norm().clamp_min(1e-30)),
                "span_energy_fraction": float(singular[:base.RANK].square().sum()
                                              / singular.square().sum().clamp_min(1e-30)),
                "candidate_float32_sha256": hashlib.sha256(
                    candidate.detach().float().cpu().numpy().tobytes()).hexdigest()}
            with torch.no_grad():
                test_h0 = test_function(test_x)
                variants = [test_function(test_x + value) for value in
                            [test_delta, mean_p, candidate, old_prototype, *nulls]]
            z = (test_h0 @ axis).double()
            responses = [((value - test_h0) @ axis).double() for value in variants]
            exact_s, mean_s, candidate_s, old_s, *null_s = responses
            def program(s):
                return torch.stack([torch.ones_like(z), z, s, z * s], dim=1) @ beta
            exact_alpha, mean_alpha, candidate_alpha, old_alpha = map(
                program, (exact_s, mean_s, candidate_s, old_s))
            null_alpha = [program(value) for value in null_s]
            for local, row_index in enumerate(test_ids):
                exact_s_all.append(float(exact_s[local])); mean_s_all.append(float(mean_s[local]))
                candidate_s_all.append(float(candidate_s[local])); old_s_all.append(float(old_s[local]))
                exact_alpha_all.append(float(exact_alpha[local])); mean_alpha_all.append(float(mean_alpha[local]))
                candidate_alpha_all.append(float(candidate_alpha[local])); old_alpha_all.append(float(old_alpha[local]))
                target_alpha_all.append(fixed_alpha)
                for null_index in range(base.NULLS):
                    null_alpha_all[null_index].append(float(null_alpha[null_index][local]))
                evidence.append({"row_id": rows[row_index]["row_id"], "site": site_index + 1,
                                 "position": position, "held_out_template": held_out,
                                 "z": float(z[local]), "exact_s": float(exact_s[local]),
                                 "mean_s": float(mean_s[local]), "candidate_s": float(candidate_s[local]),
                                 "old_s": float(old_s[local]), "fixed_alpha": fixed_alpha,
                                 "exact_alpha": float(exact_alpha[local]),
                                 "mean_alpha": float(mean_alpha[local]),
                                 "candidate_alpha": float(candidate_alpha[local]),
                                 "old_alpha": float(old_alpha[local])})

    response = {"old_single_clause": base.stats(exact_s_all, old_s_all),
                "site_mean": base.stats(exact_s_all, mean_s_all),
                "topology_response_weighted": base.stats(exact_s_all, candidate_s_all)}
    exact_program = base.stats(exact_alpha_all, candidate_alpha_all)
    fixed_program = {"old_single_clause": base.stats(target_alpha_all, old_alpha_all),
                     "site_mean": base.stats(target_alpha_all, mean_alpha_all),
                     "topology_response_weighted": base.stats(target_alpha_all, candidate_alpha_all)}
    null_programs = [base.stats(target_alpha_all, values) for values in null_alpha_all]
    null_median = float(np.median([value["relative_l2_error"] for value in null_programs]))
    null_advantage = null_median - fixed_program["topology_response_weighted"]["relative_l2_error"]
    response_improvement = response["old_single_clause"]["relative_l2_error"] \
        - response["topology_response_weighted"]["relative_l2_error"]
    instrument = (len(evidence) == 32 and sequence_count == base.PRICE["sequences"]
                  and gradient_rows == base.PRICE["input_gradient_rows"] and len(fit_audit) == 4
                  and max(head_errors) <= 5e-5 and max(hybrid_errors) <= 5e-5
                  and max(closure_errors) <= 5e-5
                  and max(value["norm_ratio"] for value in fit_audit.values()) <= 2
                  and np.isfinite(np.asarray([*exact_s_all, *candidate_s_all,
                                             *exact_alpha_all, *candidate_alpha_all])).all())
    response_pass = (response["topology_response_weighted"]["cosine"] >= .90
                     and response["topology_response_weighted"]["relative_l2_error"] <= .40
                     and response_improvement >= .30)
    law_pass = (exact_program["relative_l2_error"] <= .40
                and fixed_program["topology_response_weighted"]["relative_l2_error"] <= .50
                and fixed_program["topology_response_weighted"]["sign_agreement"] >= .90
                and null_advantage >= .05)
    predictions = {"pred_a_exact_site_instrument": bool(instrument),
                   "pred_b_topology_response_transfer": bool(instrument and response_pass),
                   "pred_c_frozen_law_and_null": bool(instrument and response_pass and law_pass)}
    terminal = "invalid" if not instrument else "topology_response_prototype_held" \
        if all(predictions.values()) else "topology_response_prototype_null"
    result = {"schema": "subject_number_topology_response_prototype_discovery_v2_result",
              "terminal": terminal, "predictions": predictions,
              "response_metrics": response, "response_error_improvement_over_old": response_improvement,
              "exact_response_program_metrics": exact_program,
              "fixed_coefficient_program_metrics": fixed_program,
              "permutation_fixed_coefficient_metrics": null_programs,
              "permutation_median_fixed_coefficient_error": null_median,
              "permutation_median_advantage": null_advantage,
              "fit_audit": fit_audit, "evidence": evidence,
              "instrument": {"examples": len(evidence),
                             "native_head_replay_max_absolute_error": max(head_errors),
                             "hybrid_input_closure_max_absolute_error": max(hybrid_errors),
                             "input_closure_max_absolute_error": max(closure_errors),
                             "counts": {**base.PRICE, "observed_sequences": sequence_count,
                                        "observed_input_gradient_rows": gradient_rows}},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False,
                                 "downstream_causal_outcomes": False, "new_text": False,
                                 "opened_grouped_mlp6_7_source_states": True},
              "price": base.PRICE, "binding_sha256": sha(BINDING),
              "runner_sha256": sha(RUNNER), "invalid_v1_sha256": sha(INVALID),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Corrected opened two-clause leave-one-template-out discovery of one grouped-MLP6/7 response-weighted prototype per structural subject site; no behavioral or fresh-OOD claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "response_metrics",
                                                    "response_error_improvement_over_old",
                                                    "exact_response_program_metrics",
                                                    "fixed_coefficient_program_metrics",
                                                    "permutation_median_fixed_coefficient_error",
                                                    "permutation_median_advantage", "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
