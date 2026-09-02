#!/usr/bin/env python3
"""RUNG470 -- continuous context law for exact equality-MLP causal effects.

Registered before opening per-token effects:
  pred_a: exact replay, factor, patch, call, and parent reaggregation checks.
  pred_b: fixed continuous context features predict held-out code union effects.
  pred_c: the code-frozen rule predicts both natural-text waves.
  pred_d: one normalized context law is shared by at least two MLPs.
  pred_e: the rule predicts the non-additive cross-MLP interaction.
Strong null: invalid, code-unpredictive, or no natural improvement over four cells.
Literal deployed price: zero parameters saved and zero added.
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

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_mlp_response_form_rung469 as parent
import equality_score_downstream_gate_rung462 as audit_parent


PREREG = POLY / "EQUALITY_CONTEXT_CAUSAL_STATE_RUNG470_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_mlp_response_form_rung469_results.json"
PARENT_SOURCE = ROOT / "ops/equality_mlp_response_form_rung469.py"
CODE_RESULT = parent.CODE_RESULT
NATURAL_RESULT = parent.NATURAL_RESULT
OUT = ROOT / "equality_context_causal_state_rung470_results.json"
BUNDLE = ROOT / "equality_context_causal_state_rung470_per_token.pt"
SOURCES = parent.SOURCES
MODULES = parent.MODULES
SITES = parent.SITES
CONTEXT_CELLS = parent.CONTEXT_CELLS
TARGETS = (*SITES, "union", "interaction")
BATCH = parent.BATCH
RIDGE = 1e-3
WINDOWS = parent.WINDOWS
FORWARDS_PER_BATCH = 2 + 1 + len(SOURCES) * (2 + 4)
EXPECTED_FORWARDS = sum((stop - start) // BATCH for _, _, start, stop in WINDOWS) \
    * FORWARDS_PER_BATCH
HASHES = {
    PREREG: "c5419a1042b5703a477e3b65ab7d048bb1dc0732365c0f8db7d77b49d4ddd5ca",
    PARENT_RESULT: "ca84e37595a1d2db31d6e09b1e91e639e70f45610fd2942f463020b1d68d1f4e",
    PARENT_SOURCE: "a6a7273bba3219fe76ce34cedb37efd5f00747ebb8d1f6ce9e4786cf13793a88",
    CODE_RESULT: "cc0480fc260c81b0fe512ec694413178de181b767f1dbfec43c56804b1ee5015",
    NATURAL_RESULT: "115024d73722b2906eed0ce0739012a874c845664ba4be3e2dd9690980a37c6c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 469 or result.get("pred_a_instrument") is not True \
            or any(result.get(key) is not False for key in (
                "pred_b_reader_stability", "pred_c_failure_localization",
                "pred_d_local_response_prediction", "pred_e_exact_causal_prediction",
            )) or result.get("strong_null") is not True:
        raise RuntimeError("rung469 registered null identity changed")
    roles, scale, metadata = parent.validate_inputs()
    metadata = {
        **metadata, "rung469_result_sha256": sha256(PARENT_RESULT),
        "rung469_source_sha256": sha256(PARENT_SOURCE),
        "features": [
            "intercept", "log1p_distance", "log1p_distance_squared",
            "log1p_predecessor_count", "log1p_predecessor_count_squared",
            "log_distance_times_log_count", "query_over_256", "query_over_256_squared",
        ],
        "ridge_nonintercept": RIDGE,
        "targets": list(TARGETS),
    }
    return roles, scale, metadata


def context_features(rows, all_positive, start, stop):
    features, quadrants, coordinates = [], [], []
    cells = {cell: [] for cell in CONTEXT_CELLS}
    mask_slice = all_positive[start:stop]
    for local_doc, query in torch.nonzero(mask_slice, as_tuple=False).tolist():
        global_doc = start + local_doc
        row = rows[global_doc]
        predecessors = torch.nonzero(row[:query] == row[query], as_tuple=False).flatten()
        if not len(predecessors):
            raise RuntimeError("positive token has no predecessor")
        distance = query - int(predecessors[-1])
        count = len(predecessors)
        ld, ln, qp = math.log1p(distance), math.log1p(count), query / 256
        features.append([1.0, ld, ld * ld, ln, ln * ln, ld * ln, qp, qp * qp])
        quadrants.append(2 * int(distance > 16) + int(count > 1))
        coordinates.append([global_doc, query, distance, count])
    return (
        torch.tensor(features, dtype=torch.float64),
        torch.tensor(quadrants, dtype=torch.long),
        torch.tensor(coordinates, dtype=torch.long),
    )


def standardize_fit(features):
    mean = features[:, 1:].mean(0)
    std = features[:, 1:].std(0, unbiased=False).clamp_min(1e-8)
    return mean, std


def standardize_apply(features, mean, std):
    output = features.clone()
    output[:, 1:] = (output[:, 1:] - mean) / std
    return output


def fit_ridge(design, target, penalty=RIDGE):
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * penalty
    regularizer[0, 0] = 0
    return torch.linalg.solve(design.T @ design + regularizer, design.T @ target)


def fit_quadrant(quadrants, target):
    overall = float(target.mean())
    return torch.tensor([
        float(target[quadrants == cell].mean()) if bool((quadrants == cell).any()) else overall
        for cell in range(4)
    ], dtype=torch.float64)


def pearson(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    return parent._cosine(left - left.mean(), right - right.mean())


def vector_metrics(target, prediction):
    return parent._metrics(target, prediction)


def prediction_metrics(target, prediction, baseline):
    target = torch.as_tensor(target, dtype=torch.float64)
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    baseline = torch.as_tensor(baseline, dtype=torch.float64)
    rmse = float(torch.sqrt(torch.mean((prediction - target) ** 2)))
    baseline_rmse = float(torch.sqrt(torch.mean((baseline - target) ** 2)))
    return {
        "pearson": pearson(target, prediction), "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "rmse_improvement": 1 - rmse / max(baseline_rmse, 1e-30),
    }


def aggregate_cells(values, memberships):
    return torch.tensor([
        float(values[memberships[:, ci]].mean()) for ci in range(len(CONTEXT_CELLS))
    ], dtype=torch.float64)


def _nll(logits, rows):
    targets = rows[:, 1:].to(logits.device)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1)


def _record(audit_totals, key, audit):
    row = audit_totals.setdefault(key, {"forwards": 0, "product_captures": 0,
                                        "product_patches": 0})
    row["forwards"] += 1
    row["product_captures"] += audit["product_captures"]
    row["product_patches"] += audit["product_patches"]


@torch.no_grad()
def collect_window(model, payload, masks, scale, start, stop, audit_totals, replay):
    rows = payload["rows"]
    features, quadrants, coordinates = context_features(rows, masks["all_positive"], start, stop)
    memberships = torch.stack([
        masks[cell][start:stop][masks["all_positive"][start:stop]] for cell in CONTEXT_CELLS
    ], dim=1)
    effects = torch.zeros(len(SOURCES), len(TARGETS), len(features), dtype=torch.float64)
    cursor = 0
    device = next(model.parameters()).device
    reconstruction = 0.0
    empty_error = 0.0
    for global_start in range(start, stop, BATCH):
        batch_rows = rows[global_start:global_start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        positive = masks["all_positive"][global_start:global_start + BATCH].to(device)
        count = int(positive.sum())
        native, _, audit, _ = parent.code_parent.source_parent.run_forward(
            model, tokens, arm="native",
        )
        audit_parent._record_audit(
            audit_totals, "rung470:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = parent.code_parent.source_parent.run_forward(
            model, tokens, arm="replay",
        )
        audit_parent._record_audit(
            audit_totals, "rung470:replay", audit, analytical=True, captures=0, patches=0,
        )
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        _, absent_products, _, audit, error = parent.code_parent.run_term_forward(
            model, tokens, arm="base", capture_products=True,
        )
        _record(audit_totals, "rung470:absent", audit)
        reconstruction = max(reconstruction, error)
        for si, source in enumerate(SOURCES):
            arm = parent.code_parent.source_parent.SOURCE_ARMS[source]
            logits, _, _, audit, error = parent.code_parent.run_term_forward(
                model, tokens, arm=arm, scale=scale,
            )
            _record(audit_totals, f"rung470:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = _nll(logits, batch_rows)
            empty_logits, _, _, audit, error = parent.code_parent.run_term_forward(
                model, tokens, arm=arm, scale=scale, baseline_products={}, term_groups={},
            )
            _record(audit_totals, f"rung470:empty:{source}", audit)
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - logits).abs().max()))
            singles = []
            for ti, site in enumerate(SITES):
                patched, _, _, audit, error = parent.code_parent.run_term_forward(
                    model, tokens, arm=arm, scale=scale,
                    baseline_products={site: absent_products[site]},
                    term_groups={site: range(parent.HIDDEN)},
                )
                _record(audit_totals, f"rung470:patch:{source}:{site}", audit)
                reconstruction = max(reconstruction, error)
                value = (_nll(patched, batch_rows) - source_nll)[positive].double().cpu()
                effects[si, ti, cursor:cursor + count] = value
                singles.append(value)
            patched, _, _, audit, error = parent.code_parent.run_term_forward(
                model, tokens, arm=arm, scale=scale,
                baseline_products={site: absent_products[site] for site in SITES},
                term_groups={site: range(parent.HIDDEN) for site in SITES},
            )
            _record(audit_totals, f"rung470:patch:{source}:union", audit)
            reconstruction = max(reconstruction, error)
            union = (_nll(patched, batch_rows) - source_nll)[positive].double().cpu()
            effects[si, 3, cursor:cursor + count] = union
            effects[si, 4, cursor:cursor + count] = union - sum(singles)
        cursor += count
    if cursor != len(features):
        raise RuntimeError("per-token cursor did not cover all positives")
    return {
        "features": features, "quadrants": quadrants, "coordinates": coordinates,
        "memberships": memberships, "effects": effects,
        "empty_patch_max_abs": empty_error,
    }, reconstruction


def _parent_reaggregation_error(windows):
    code_result = json.loads(CODE_RESULT.read_text())
    natural_result = json.loads(NATURAL_RESULT.read_text())
    maximum = 0.0
    code = windows["code_validation"]
    mask_by_target = {"m8": "1", "m9": "2", "m12": "4", "union": "7"}
    for si, source in enumerate(SOURCES):
        for ti, target in enumerate(TARGETS[:4]):
            observed = aggregate_cells(code["effects"][si, ti], code["memberships"])
            expected = torch.tensor(
                code_result["analysis"]["pooled"]["complete_vectors"][source][mask_by_target[target]],
                dtype=torch.float64,
            )
            maximum = max(maximum, float((observed - expected).abs().max()))
    natural_parts = [windows["natural_wave0"], windows["natural_wave1"]]
    for si, source in enumerate(SOURCES):
        for ti, target in enumerate(TARGETS[:4]):
            sums, counts = torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
            for part in natural_parts:
                for ci in range(4):
                    selected = part["memberships"][:, ci]
                    sums[ci] += part["effects"][si, ti, selected].sum()
                    counts[ci] += int(selected.sum())
            observed = sums / counts.clamp_min(1)
            expected = torch.tensor(
                natural_result["analysis"]["pooled"]["complete_vectors"][source][mask_by_target[target]],
                dtype=torch.float64,
            )
            maximum = max(maximum, float((observed - expected).abs().max()))
    return maximum


def analyze(windows):
    fit = windows["code_discovery"]
    mean, std = standardize_fit(fit["features"])
    design_fit = standardize_apply(fit["features"], mean, std)
    models, baselines, constants = {}, {}, {}
    shared = {}
    for si, source in enumerate(SOURCES):
        models[source], baselines[source], constants[source] = {}, {}, {}
        for ti, target in enumerate(TARGETS):
            y = fit["effects"][si, ti]
            models[source][target] = fit_ridge(design_fit, y)
            baselines[source][target] = fit_quadrant(fit["quadrants"], y)
            constants[source][target] = float(y.mean())
        rms = torch.sqrt(torch.mean(fit["effects"][si, :3] ** 2, dim=1)).clamp_min(1e-12)
        pooled_design = design_fit.repeat(3, 1)
        pooled_target = torch.cat([fit["effects"][si, mi] / rms[mi] for mi in range(3)])
        shared[source] = {"coefficient": fit_ridge(pooled_design, pooled_target),
                          "rms": rms}

    reports = {}
    b_flags, c_flags = [], []
    shared_qualified = {source: set(SITES) for source in SOURCES}
    interaction_flags = []
    for name in ("code_validation", "natural_wave0", "natural_wave1"):
        window = windows[name]
        design = standardize_apply(window["features"], mean, std)
        reports[name] = {}
        for si, source in enumerate(SOURCES):
            reports[name][source] = {}
            for ti, target in enumerate(TARGETS):
                y = window["effects"][si, ti]
                prediction = design @ models[source][target]
                baseline = baselines[source][target][window["quadrants"]]
                constant = torch.full_like(y, constants[source][target])
                metrics = prediction_metrics(y, prediction, baseline)
                exact_cells = aggregate_cells(y, window["memberships"])
                predicted_cells = aggregate_cells(prediction, window["memberships"])
                baseline_cells = aggregate_cells(baseline, window["memberships"])
                reports[name][source][target] = {
                    "metrics": metrics, "constant_metrics": prediction_metrics(y, prediction, constant),
                    "exact_context_vector": exact_cells.tolist(),
                    "predicted_context_vector": predicted_cells.tolist(),
                    "baseline_context_vector": baseline_cells.tolist(),
                    "context_metrics": vector_metrics(exact_cells, predicted_cells),
                }
            union = reports[name][source]["union"]
            if name == "code_validation":
                b_flags.append(bool(
                    union["metrics"]["pearson"] >= .30
                    and union["metrics"]["rmse_improvement"] >= .15
                    and union["context_metrics"]["cosine"] >= .85
                    and .50 <= union["context_metrics"]["projection_on_target"] <= 1.50
                ))
            else:
                c_flags.append(bool(
                    union["metrics"]["pearson"] >= .20
                    and union["metrics"]["rmse_improvement"] >= .10
                    and union["context_metrics"]["cosine"] >= .80
                    and .25 <= union["context_metrics"]["projection_on_target"] <= 1.75
                ))
            for mi, site in enumerate(SITES):
                y = window["effects"][si, mi]
                shared_prediction = design @ shared[source]["coefficient"] * shared[source]["rms"][mi]
                separate_prediction = design @ models[source][site]
                baseline = baselines[source][site][window["quadrants"]]
                shared_metrics = prediction_metrics(y, shared_prediction, baseline)
                separate_rmse = float(torch.sqrt(torch.mean((separate_prediction - y) ** 2)))
                qualifies = bool(
                    shared_metrics["pearson"] >= .20
                    and shared_metrics["rmse"] <= 1.15 * separate_rmse
                    and shared_metrics["rmse_improvement"] > 0
                )
                if not qualifies:
                    shared_qualified[source].discard(site)
                reports[name][source][site]["shared_rule"] = {
                    **shared_metrics, "separate_rmse": separate_rmse,
                    "qualifies": qualifies,
                }
            interaction = reports[name][source]["interaction"]
            norm = float(torch.linalg.vector_norm(torch.tensor(
                interaction["exact_context_vector"], dtype=torch.float64,
            )))
            prediction_norm = float(torch.linalg.vector_norm(torch.tensor(
                interaction["predicted_context_vector"], dtype=torch.float64,
            )))
            if name == "code_validation":
                okay = interaction["context_metrics"]["cosine"] >= .75
            else:
                okay = bool(
                    (interaction["context_metrics"]["cosine"] >= .65
                     and interaction["context_metrics"]["projection_on_target"] > 0)
                    or (norm < .003 and prediction_norm < .003)
                )
            interaction["registered_norm"] = norm
            interaction["predicted_norm"] = prediction_norm
            interaction["registered_clause_holds"] = bool(okay)
            interaction_flags.append(bool(okay))
    shared_pair = set(SITES)
    for source in SOURCES:
        shared_pair &= shared_qualified[source]
    pred_b = all(b_flags)
    pred_c = all(c_flags)
    pred_d = len(shared_pair) >= 2
    pred_e = all(interaction_flags)
    any_natural_improvement = any(
        reports[name][source]["union"]["metrics"]["rmse_improvement"] > 0
        for name in ("natural_wave0", "natural_wave1") for source in SOURCES
    )
    return {
        "standardization": {"mean": mean.tolist(), "std": std.tolist()},
        "coefficients": {
            source: {target: models[source][target].tolist() for target in TARGETS}
            for source in SOURCES
        },
        "quadrant_baselines": {
            source: {target: baselines[source][target].tolist() for target in TARGETS}
            for source in SOURCES
        },
        "shared_rule": {
            source: {"coefficient": shared[source]["coefficient"].tolist(),
                     "rms": shared[source]["rms"].tolist(),
                     "qualifying_modules": sorted(shared_qualified[source])}
            for source in SOURCES
        },
        "shared_module_intersection": sorted(shared_pair), "reports": reports,
        "pred_b_heldout_code": bool(pred_b), "pred_c_natural": bool(pred_c),
        "pred_d_shared_mlp_law": bool(pred_d), "pred_e_interaction": bool(pred_e),
        "any_natural_improvement": bool(any_natural_improvement),
    }


def main():
    started = time.time()
    roles, scale, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 470, "model_loaded": False,
            "per_token_effects_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS, "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung470 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    windows = {}
    reconstruction = 0.0
    for name, role, start, stop in WINDOWS:
        payload, masks = roles[role]
        windows[name], error = collect_window(
            model, payload, masks, scale, start, stop, audit_totals, replay,
        )
        reconstruction = max(reconstruction, error)
    parent_error = _parent_reaggregation_error(windows)
    analysis = analyze(windows)
    forwards = sum(row["forwards"] for row in audit_totals.values())
    empty_error = max(window["empty_patch_max_abs"] for window in windows.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and empty_error == 0 and forwards == EXPECTED_FORWARDS and parent_error <= 1e-9
    )
    strong_null = bool(
        not pred_a or not analysis["pred_b_heldout_code"]
        or not analysis["any_natural_improvement"]
    )
    bundle = {
        "schema": "rung470_per_token_causal_effects_v1",
        "windows": windows, "context_cells": list(CONTEXT_CELLS),
        "targets": list(TARGETS), "sources": list(SOURCES),
        "raw_tokens_or_logits_included": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 470,
        "claim_level": "continuous_context_exact_causal_effect_transfer_test",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "raw_tokens_or_logits_included": False},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "empty_patch_max_abs": empty_error,
        "parent_reaggregation_max_abs_error_nat": parent_error,
        "audit_totals": audit_totals,
        "execution_price": {
            "outer_forwards": forwards,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_heldout_code': analysis["pred_b_heldout_code"],
        'pred_c_natural': analysis["pred_c_natural"],
        'pred_d_shared_mlp_law': analysis["pred_d_shared_mlp_law"],
        'pred_e_interaction': analysis["pred_e_interaction"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "extract_context_law_intervention_and_test_composition"
            if pred_a and all(analysis[key] for key in (
                "pred_b_heldout_code", "pred_c_natural",
                "pred_d_shared_mlp_law", "pred_e_interaction",
            )) else "add_measured_downstream_use_or_paired_response_kernel_no_rank_tuning"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 470,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "union_reports": {
            name: {source: analysis["reports"][name][source]["union"] for source in SOURCES}
            for name in analysis["reports"]
        },
        "shared_module_intersection": analysis["shared_module_intersection"],
        "interaction_reports": {
            name: {source: analysis["reports"][name][source]["interaction"] for source in SOURCES}
            for name in analysis["reports"]
        },
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "empty_error": empty_error, "parent_error": parent_error},
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
