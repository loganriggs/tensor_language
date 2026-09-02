#!/usr/bin/env python3
"""RUNG471 -- target-specific paired downstream-response kernel.

Registered before target-specific gradients:
  pred_a: frozen inputs, exact replay/factors/regions/calls, no future leakage.
  pred_b: calibrated paired kernel predicts held-out code better than context.
  pred_c: calibrated paired kernel predicts both natural waves better than context.
  pred_d: at least two MLPs have a stable spatial response profile.
  pred_e: response kernels agree across matcher sources and group MLPs causally.
Strong null: invalid, code failure, or no natural improvement over context.
Literal deployed price: zero parameters saved and zero added.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
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
import equality_context_causal_state_rung470 as parent
import equality_mlp_product_term_group_rung467 as code_parent
import equality_mlp_response_form_rung469 as form_parent
import equality_score_correction_interchange_rung464 as source_parent
import equality_score_downstream_gate_rung462 as audit_parent


PREREG = POLY / "EQUALITY_PAIRED_RESPONSE_KERNEL_RUNG471_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_context_causal_state_rung470_results.json"
PARENT_BUNDLE = ROOT / "equality_context_causal_state_rung470_per_token.pt"
PARENT_SOURCE = ROOT / "ops/equality_context_causal_state_rung470.py"
OUT = ROOT / "equality_paired_response_kernel_rung471_results.json"
BUNDLE = ROOT / "equality_paired_response_kernel_rung471.pt"
SOURCES = parent.SOURCES
MODULES = parent.MODULES
SITES = parent.SITES
CONTEXT_CELLS = parent.CONTEXT_CELLS
REGIONS = ("query", "latest_predecessor", "between", "earlier")
WINDOWS = parent.WINDOWS
BATCH = parent.BATCH
EXPECTED_FORWARDS = sum((stop - start) // BATCH for _, _, start, stop in WINDOWS) * 5
HASHES = {
    PREREG: "2ccd60727c583304801631dcb905022968301fde41abc1a372b3697c7319e16a",
    PARENT_RESULT: "1b44c25baeda0e4e18d0be3e6d9cfe95fd7b1e2d47fc7fba85b9a0262f131b3f",
    PARENT_BUNDLE: "227bb79fb60bfdec232d51f0862dbe44073887853e7afee1ab2cc517a4a94118",
    PARENT_SOURCE: "4c1b20dd795ae1028e4f580a80d9bba8318d85961752c7a3e19d3081517e4ed0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_coordinates(rows, all_positive, start, stop):
    selected = []
    for global_doc in range(start, stop):
        queries = torch.nonzero(all_positive[global_doc], as_tuple=False).flatten().tolist()
        for query in queries[:2]:
            row = rows[global_doc]
            predecessors = torch.nonzero(row[:query] == row[query], as_tuple=False).flatten()
            selected.append((global_doc, query, int(predecessors[-1])))
    return selected


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 470 or result.get("pred_a_instrument") is not True \
            or any(result.get(key) is not False for key in (
                "pred_b_heldout_code", "pred_c_natural",
                "pred_d_shared_mlp_law", "pred_e_interaction",
            )) or result.get("strong_null") is not True:
        raise RuntimeError("rung470 registered null identity changed")
    roles, scale, metadata = parent.validate_inputs()
    old = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=True)
    if old.get("schema") != "rung470_per_token_causal_effects_v1":
        raise RuntimeError("rung470 bundle schema changed")
    selections, expected_backwards = {}, 0
    for name, role, start, stop in WINDOWS:
        payload, masks = roles[role]
        coordinates = select_coordinates(payload["rows"], masks["all_positive"], start, stop)
        old_coordinates = old["windows"][name]["coordinates"][:, :2]
        lookup = {(int(row[0]), int(row[1])): i for i, row in enumerate(old_coordinates)}
        indices = []
        for doc, query, _ in coordinates:
            if (doc, query) not in lookup:
                raise RuntimeError("selected coordinate missing from rung470")
            indices.append(lookup[(doc, query)])
        selections[name] = {"coordinates": coordinates, "rung470_indices": indices}
        expected_backwards += len(coordinates) * len(SOURCES)
    metadata = {
        **metadata, "rung470_result_sha256": sha256(PARENT_RESULT),
        "rung470_bundle_sha256": sha256(PARENT_BUNDLE),
        "regions": list(REGIONS), "targets_per_document_max": 2,
        "selection_counts": {name: len(row["coordinates"]) for name, row in selections.items()},
        "expected_backwards": expected_backwards,
    }
    return roles, scale, old, selections, metadata


def region_sums(contribution, query, predecessor):
    signed = torch.stack((
        contribution[query], contribution[predecessor],
        contribution[predecessor + 1:query].sum(), contribution[:predecessor].sum(),
    ))
    absolute = torch.stack((
        contribution[query].abs(), contribution[predecessor].abs(),
        contribution[predecessor + 1:query].abs().sum(),
        contribution[:predecessor].abs().sum(),
    ))
    future = contribution[query + 1:].sum()
    return signed, absolute, future


def _record(audit_totals, key, audit):
    row = audit_totals.setdefault(key, {"forwards": 0, "product_captures": 0,
                                        "product_patches": 0})
    row["forwards"] += 1
    row["product_captures"] += audit["product_captures"]
    row["product_patches"] += audit["product_patches"]


def _nll(logits, rows):
    targets = rows[:, 1:].to(logits.device)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1)


def collect_window(model, payload, scale, selection, audit_totals, replay):
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, predecessor) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query, predecessor))
    signed = torch.zeros(len(SOURCES), len(MODULES), len(coordinates), len(REGIONS),
                         dtype=torch.float64)
    absolute = torch.zeros_like(signed)
    future = torch.zeros(len(SOURCES), len(MODULES), len(coordinates), dtype=torch.float64)
    local = torch.zeros(len(SOURCES), len(MODULES), len(coordinates), dtype=torch.float64)
    rows = payload["rows"]
    first_doc = min(doc for doc, _, _ in coordinates)
    last_doc = max(doc for doc, _, _ in coordinates) + 1
    device = next(model.parameters()).device
    reconstruction = 0.0
    backwards = 0
    for start in range(first_doc, last_doc, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        with torch.no_grad():
            native, _, audit, _ = source_parent.run_forward(
                model, tokens, arm="native",
            )
            audit_parent._record_audit(
                audit_totals, "rung471:native", audit, analytical=False, captures=0, patches=0,
            )
            replay_logits, _, audit, error = source_parent.run_forward(
                model, tokens, arm="replay",
            )
            audit_parent._record_audit(
                audit_totals, "rung471:replay", audit, analytical=True, captures=0, patches=0,
            )
            difference = replay_logits - native
            replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
            replay["relative_squared"] = max(
                replay["relative_squared"],
                float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
            )
            reconstruction = max(reconstruction, error)
            _, absent_products, _, audit, error = code_parent.run_term_forward(
                model, tokens, arm="base", capture_products=True,
            )
            _record(audit_totals, "rung471:absent", audit)
            reconstruction = max(reconstruction, error)
        batch_targets = []
        for doc in range(start, min(start + BATCH, last_doc)):
            for output_index, query, predecessor in by_doc.get(doc, []):
                batch_targets.append((output_index, doc - start, query, predecessor))
        for si, source in enumerate(SOURCES):
            arm = source_parent.SOURCE_ARMS[source]
            with torch.enable_grad():
                logits, products, writes, audit, error = code_parent.run_term_forward(
                    model, tokens, arm=arm, scale=scale, capture_products=True,
                    gradient_writes=True,
                )
                _record(audit_totals, f"rung471:source:{source}", audit)
                reconstruction = max(reconstruction, error)
                losses = _nll(logits, batch_rows)
                delta_writes = {}
                for mi, site in enumerate(SITES):
                    module = model.transformer.h[MODULES[mi]].mlp
                    delta_product = (products[site] - absent_products[site]).float()
                    delta_writes[site] = F.linear(delta_product, module.Down.weight.float())
                for target_i, (output_index, local_doc, query, predecessor) in enumerate(batch_targets):
                    gradients = torch.autograd.grad(
                        losses[local_doc, query], tuple(writes[site] for site in SITES),
                        retain_graph=target_i + 1 < len(batch_targets), allow_unused=False,
                    )
                    backwards += 1
                    for mi, (site, gradient) in enumerate(zip(SITES, gradients)):
                        contribution = (-gradient[local_doc].float()
                                        * delta_writes[site][local_doc]).sum(-1)
                        row, magnitude, later = region_sums(contribution, query, predecessor)
                        signed[si, mi, output_index] = row.double().cpu()
                        absolute[si, mi, output_index] = magnitude.double().cpu()
                        future[si, mi, output_index] = later.double().cpu()
                        local[si, mi, output_index] = contribution[:query + 1].sum().double().cpu()
            del logits, products, writes, losses, delta_writes
        del absent_products
    return {"signed": signed, "absolute": absolute, "future": future,
            "local": local, "backwards": backwards}, reconstruction


def _fit_scale(local, exact):
    return form_parent._fit_scale(local, exact)


def _metrics(target, prediction, control):
    target = torch.as_tensor(target, dtype=torch.float64)
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    control = torch.as_tensor(control, dtype=torch.float64)
    rmse = float(torch.sqrt(torch.mean((prediction - target) ** 2)))
    control_rmse = float(torch.sqrt(torch.mean((control - target) ** 2)))
    return {
        "pearson": parent.pearson(target, prediction),
        "rmse": rmse, "context_control_rmse": control_rmse,
        "rmse_improvement": 1 - rmse / max(control_rmse, 1e-30),
    }


def _context_predictions(result, bundle, name, indices, source, target):
    mean = torch.tensor(result["analysis"]["standardization"]["mean"], dtype=torch.float64)
    std = torch.tensor(result["analysis"]["standardization"]["std"], dtype=torch.float64)
    coefficient = torch.tensor(
        result["analysis"]["coefficients"][source][target], dtype=torch.float64,
    )
    features = bundle["windows"][name]["features"][indices]
    return parent.standardize_apply(features, mean, std) @ coefficient


def _context_vector(values, bundle_window, indices):
    memberships = bundle_window["memberships"][indices]
    return parent.aggregate_cells(values, memberships)


def analyze(kernels, old_bundle, selections, old_result):
    fit_name = "code_discovery"
    fit_indices = selections[fit_name]["rung470_indices"]
    fit_exact_all = old_bundle["windows"][fit_name]["effects"][:, :4, fit_indices]
    scales = {source: {} for source in SOURCES}
    for si, source in enumerate(SOURCES):
        for mi, site in enumerate(SITES):
            scales[source][site] = _fit_scale(
                kernels[fit_name]["local"][si, mi], fit_exact_all[si, mi],
            )
        scales[source]["union"] = _fit_scale(
            kernels[fit_name]["local"][si].sum(0), fit_exact_all[si, 3],
        )
    reports = {}
    b_flags, c_flags = [], []
    individual_good = {source: set(SITES) for source in SOURCES}
    spatial_good = {source: set(SITES) for source in SOURCES}
    fit_profiles = {}
    for si, source in enumerate(SOURCES):
        fit_profiles[source] = {}
        for mi, site in enumerate(SITES):
            fit_profiles[source][site] = {
                "signed": kernels[fit_name]["signed"][si, mi].mean(0),
                "absolute": kernels[fit_name]["absolute"][si, mi].mean(0),
            }
    for name in ("code_validation", "natural_wave0", "natural_wave1"):
        indices = selections[name]["rung470_indices"]
        exact_all = old_bundle["windows"][name]["effects"][:, :4, indices]
        reports[name] = {}
        for si, source in enumerate(SOURCES):
            reports[name][source] = {}
            local_union = kernels[name]["local"][si].sum(0)
            prediction = local_union * scales[source]["union"]
            exact = exact_all[si, 3]
            context = _context_predictions(old_result, old_bundle, name, indices, source, "union")
            metrics = _metrics(exact, prediction, context)
            exact_cells = _context_vector(exact, old_bundle["windows"][name], indices)
            prediction_cells = _context_vector(prediction, old_bundle["windows"][name], indices)
            context_metrics = form_parent._metrics(exact_cells, prediction_cells)
            reports[name][source]["union"] = {
                "metrics": metrics, "exact_context_vector": exact_cells.tolist(),
                "predicted_context_vector": prediction_cells.tolist(),
                "context_metrics": context_metrics,
            }
            if name == "code_validation":
                b_flags.append(bool(
                    metrics["pearson"] >= .55 and metrics["rmse_improvement"] >= .15
                    and context_metrics["cosine"] >= .90
                    and .50 <= context_metrics["projection_on_target"] <= 1.50
                ))
            else:
                c_flags.append(bool(
                    metrics["pearson"] >= .30 and metrics["rmse_improvement"] >= .15
                    and context_metrics["cosine"] >= .80
                    and .25 <= context_metrics["projection_on_target"] <= 1.75
                ))
            for mi, site in enumerate(SITES):
                exact_site = exact_all[si, mi]
                prediction_site = kernels[name]["local"][si, mi] * scales[source][site]
                context_site = _context_predictions(
                    old_result, old_bundle, name, indices, source, site,
                )
                site_metrics = _metrics(exact_site, prediction_site, context_site)
                if site_metrics["rmse_improvement"] <= 0:
                    individual_good[source].discard(site)
                signed_profile = kernels[name]["signed"][si, mi].mean(0)
                absolute_profile = kernels[name]["absolute"][si, mi].mean(0)
                signed_cos = form_parent._cosine(
                    fit_profiles[source][site]["signed"], signed_profile,
                )
                absolute_cos = form_parent._cosine(
                    fit_profiles[source][site]["absolute"], absolute_profile,
                )
                dominant_same = bool(
                    int(torch.argmax(fit_profiles[source][site]["absolute"]))
                    == int(torch.argmax(absolute_profile))
                )
                threshold = .80 if name == "code_validation" else .70
                if signed_cos < threshold or absolute_cos < threshold or not dominant_same:
                    spatial_good[source].discard(site)
                reports[name][source][site] = {
                    "metrics": site_metrics,
                    "signed_region_profile": signed_profile.tolist(),
                    "absolute_region_profile": absolute_profile.tolist(),
                    "signed_profile_cosine": signed_cos,
                    "absolute_profile_cosine": absolute_cos,
                    "dominant_region": REGIONS[int(torch.argmax(absolute_profile))],
                    "dominant_same_as_fit": dominant_same,
                }
            n_kernel = kernels[name]["signed"][0].sum(0)
            h_kernel = kernels[name]["signed"][1].sum(0)
            # Filled identically in each source row for easy receipt reading.
            reports[name][source]["cross_source_kernel_cosine"] = \
                form_parent._cosine(n_kernel, h_kernel)
    spatial_intersection = set(SITES)
    individual_intersection = set(SITES)
    for source in SOURCES:
        spatial_intersection &= spatial_good[source]
        individual_intersection &= individual_good[source]
    cross_source_ok = all(
        reports[name]["N"]["cross_source_kernel_cosine"] >= .85
        for name in reports
    )
    pred_b = all(b_flags)
    pred_c = all(c_flags)
    pred_d = len(spatial_intersection) >= 2
    pred_e = cross_source_ok and len(individual_intersection) >= 2
    any_natural_improvement = any(
        reports[name][source]["union"]["metrics"]["rmse_improvement"] > 0
        for name in ("natural_wave0", "natural_wave1") for source in SOURCES
    )
    return {
        "frozen_scales": scales, "reports": reports,
        "spatial_module_intersection": sorted(spatial_intersection),
        "individual_prediction_intersection": sorted(individual_intersection),
        "cross_source_ok": bool(cross_source_ok),
        "pred_b_heldout_code": bool(pred_b), "pred_c_natural": bool(pred_c),
        "pred_d_spatial_computation": bool(pred_d),
        "pred_e_shared_downstream_use": bool(pred_e),
        "any_natural_improvement": bool(any_natural_improvement),
    }


def main():
    started = time.time()
    roles, scale, old_bundle, selections, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 471, "model_loaded": False,
            "target_gradients_opened": False, "sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_backwards": metadata["expected_backwards"],
            "selection_counts": metadata["selection_counts"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung471 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    kernels = {}
    reconstruction = 0.0
    for name, role, _, _ in WINDOWS:
        payload, _ = roles[role]
        kernels[name], error = collect_window(
            model, payload, scale, selections[name], audit_totals, replay,
        )
        reconstruction = max(reconstruction, error)
    analysis = analyze(kernels, old_bundle, selections, json.loads(PARENT_RESULT.read_text()))
    forwards = sum(row["forwards"] for row in audit_totals.values())
    backwards = sum(row["backwards"] for row in kernels.values())
    region_error = max(float((row["signed"].sum(-1) - row["local"]).abs().max())
                       for row in kernels.values())
    future_abs = max(float(row["future"].abs().max()) for row in kernels.values())
    total_norm = max(float(torch.linalg.vector_norm(row["local"])) for row in kernels.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and region_error <= 1e-9 and future_abs <= 1e-9 * total_norm + 1e-12
        and forwards == EXPECTED_FORWARDS and backwards == metadata["expected_backwards"]
    )
    strong_null = bool(
        not pred_a or not analysis["pred_b_heldout_code"]
        or not analysis["any_natural_improvement"]
    )
    torch.save({
        "schema": "rung471_paired_response_kernel_v1", "regions": list(REGIONS),
        "selections": selections, "kernels": kernels,
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 471,
        "claim_level": "paired_downstream_response_kernel_transfer_test",
        "input_identity": metadata,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "region_sum_max_abs_error": region_error,
        "future_position_max_abs": future_abs,
        "audit_totals": audit_totals,
        "execution_price": {
            "outer_forwards": forwards, "backwards": backwards,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_heldout_code': analysis["pred_b_heldout_code"],
        'pred_c_natural': analysis["pred_c_natural"],
        'pred_d_spatial_computation': analysis["pred_d_spatial_computation"],
        'pred_e_shared_downstream_use': analysis["pred_e_shared_downstream_use"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "exact_position_region_interventions"
            if pred_a and all(analysis[key] for key in (
                "pred_b_heldout_code", "pred_c_natural",
                "pred_d_spatial_computation", "pred_e_shared_downstream_use",
            )) else "exact_target_region_intervention_or_nonlinear_state_use_variable_no_rank_tuning"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 471,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "analysis": analysis,
        "instrument": {"replay": replay, "factor_error": reconstruction,
                       "region_error": region_error, "future_abs": future_abs},
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
