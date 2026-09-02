#!/usr/bin/env python3
"""Quadratic response falsifier -- is y(lambda)=a*lambda+b*lambda^2 the causal law?

# BQGATE: EXPERIMENT
# pred_a_exact_live_three_scale_instrument
# pred_b_prospective_two_point_quadratic_extrapolation_at_two
# pred_c_in_run_two_sided_quadratic_coherence
# pred_d_document_half_stability_at_two

Parallel-lane rung (Claude). Zero fitted parameters: solve a,b per occurrence from
the in-run (0.5, y_half) and (1, y_1) points, predict the never-measured lambda=2.0
outcome as 6*y1-8*y_half, against additive (2*y1), the rung494 isotonic incumbent,
and 16 permuted-donor controls. Imports the frozen rung494 module as a library;
modifies no registered file. Preregistration:
polynomial_causal/QUADRATIC_RESPONSE_FALSIFIER_PREREGISTRATION.md
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_query_scaled_single_index_causal_rung494 as r494

parent = r494.parent

PREREG = POLY / "QUADRATIC_RESPONSE_FALSIFIER_PREREGISTRATION.md"
REVIEW = POLY / "MATHEMATICAL_REVIEW_2026-09-02_1607.md"
R494_SOURCE = ROOT / "ops/equality_query_scaled_single_index_causal_rung494.py"
R494_RESULT = ROOT / "equality_query_scaled_single_index_causal_rung494_results.json"
R494_BUNDLE = ROOT / "equality_query_scaled_single_index_causal_rung494_per_token.pt"
OUT = ROOT / "equality_query_quadratic_response_falsifier_results.json"
BUNDLE = ROOT / "equality_query_quadratic_response_falsifier_per_token.pt"
HASHES = {
    PREREG: "6f19b6be4f225a832a0dc98e2a97859972a1d6de205b344d9831c95f2a7a98b9",
    REVIEW: "524886252e46846f235c08d6df22494951492eab8c0074f8320e5405db4a11e3",
    R494_SOURCE: "e452b27b18d3f803ee6b24d518f7f7e9a2ec048f84fae5d125b3380a51787832",
    R494_RESULT: "8b384663af5fe6b9291c4180f1ea6147a40835cc5e64a172a72f73087ddad261",
    R494_BUNDLE: "3304751e965987b89f49073ed03c713ee4ffcbd0b79cc501247d1e9608870939",
}

SOURCES = parent.SOURCES
SITES = parent.SITES
WINDOWS = parent.WINDOWS
SUBSETS = parent.SUBSETS
SUBSET_NAMES = parent.SUBSET_NAMES
BATCH = parent.BATCH
MY_SCALES = (0.5, 1.5, 2.0)
SCALE_KEYS = ("0.5", "1.5", "2.0")
POSITION_SHIFTS = r494.POSITION_SHIFTS
EXPECTED_BATCHES = parent.EXPECTED_BATCHES
FORWARDS_PER_BATCH = 3 + len(SOURCES) * (
    2 + 2 * (len(SUBSETS) + len(MY_SCALES) * len(SITES)))
BRIDGE_FORWARDS = len(WINDOWS) * len(SOURCES) * len(SITES)
EXPECTED_FORWARDS = EXPECTED_BATCHES * FORWARDS_PER_BATCH + BRIDGE_FORWARDS
PATCH_CALLS_PER_BATCH = len(SOURCES) * (
    len(SITES) + 2 * (
        sum(len(indices) for indices in SUBSETS) + len(MY_SCALES) * len(SITES)))
EXPECTED_PATCH_CALLS = EXPECTED_BATCHES * PATCH_CALLS_PER_BATCH + BRIDGE_FORWARDS


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or r494.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R494_RESULT.read_text())
    required = {
        "rung": 494,
        "pred_a_exact_live_scaled_intervention": True,
        "pred_c_one_and_half_strength_causal_transfer": True,
        "strong_null": True,
        "validation_or_sealed_opened": False,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise RuntimeError("rung494 verdicts do not license this falsifier")
    roles, scale, selections, metadata = r494.validate_inputs()
    return roles, scale, selections, {
        **metadata,
        "falsifier_scales": list(MY_SCALES),
        "rung494_result_sha256": HASHES[R494_RESULT],
    }


@torch.no_grad()
def collect_window(model, payload, scale, selection, audit_totals, replay):
    """Rung494's collection loop with three scaled strengths per site."""
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, _) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query))
    effects = torch.zeros(len(SOURCES), len(SUBSETS), len(coordinates), dtype=torch.float64)
    scaled = torch.zeros(
        len(SOURCES), len(MY_SCALES), len(SITES), len(coordinates), dtype=torch.float64)
    rows = payload["rows"]
    first_doc, last_doc = min(by_doc), max(by_doc) + 1
    device = next(model.parameters()).device
    reconstruction = empty_error = bridge_error = 0.0
    patch_calls = 0
    first_batch = True
    for start in range(first_doc, last_doc, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = parent.source_parent.run_forward(model, tokens, arm="native")
        parent.audit_parent._record_audit(
            audit_totals, "rqrf:native", audit,
            analytical=False, captures=0, patches=0)
        replay_logits, _, audit, error = parent.source_parent.run_forward(
            model, tokens, arm="replay")
        parent.audit_parent._record_audit(
            audit_totals, "rqrf:replay", audit,
            analytical=True, captures=0, patches=0)
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30))
        reconstruction = max(reconstruction, error)
        _, absent_products, _, audit, error = parent.product_parent.run_term_forward(
            model, tokens, arm="base", capture_products=True)
        parent._record(audit_totals, "rqrf:absent", audit)
        reconstruction = max(reconstruction, error)
        slots = []
        for slot in range(2):
            chosen = []
            for doc in range(start, min(start + BATCH, last_doc)):
                if len(by_doc.get(doc, [])) > slot:
                    output_index, query = by_doc[doc][slot]
                    chosen.append((output_index, doc - start, query))
            slots.append(chosen)
        for si, source in enumerate(SOURCES):
            arm = parent.source_parent.SOURCE_ARMS[source]
            source_logits, source_products, _, audit, error = \
                parent.product_parent.run_term_forward(
                    model, tokens, arm=arm, scale=scale, capture_products=True)
            parent._record(audit_totals, f"rqrf:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = parent.position_parent._nll(source_logits, batch_rows)
            deltas = {
                site: source_products[site].float() - absent_products[site].float()
                for site in SITES
            }
            false_mask = torch.zeros_like(tokens, dtype=torch.bool)
            empty_logits, calls, audit, error = r494.run_scaled_patch(
                model, tokens, arm=arm, scale=scale, deltas=deltas,
                sites=SITES, position_mask=false_mask, delta_scale=1.0)
            parent._record(audit_totals, f"rqrf:empty:{source}", audit, sum(calls.values()))
            patch_calls += sum(calls.values())
            reconstruction = max(reconstruction, error)
            empty_error = max(empty_error, float((empty_logits - source_logits).abs().max()))
            bridge_logits = {}
            first_query_mask = None
            for slot, chosen in enumerate(slots):
                targets = [(local_doc, query) for _, local_doc, query in chosen]
                query_mask, _, _, _ = parent.position_parent.position_masks(
                    len(batch_rows), tokens.shape[1], targets, device)
                if first_batch and slot == 0:
                    first_query_mask = query_mask
                for subset_index, indices in enumerate(SUBSETS):
                    sites = tuple(SITES[index] for index in indices)
                    patched, calls, audit, error = r494.run_scaled_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site] for site in sites}, sites=sites,
                        position_mask=query_mask, delta_scale=1.0)
                    parent._record(
                        audit_totals,
                        f"rqrf:unit:{source}:slot{slot}:{SUBSET_NAMES[subset_index]}",
                        audit, sum(calls.values()))
                    patch_calls += sum(calls.values())
                    reconstruction = max(reconstruction, error)
                    damage = parent.position_parent._nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        effects[si, subset_index, output_index] = float(damage[local_doc, query])
                    if first_batch and slot == 0 and len(indices) == 1:
                        bridge_logits[indices[0]] = patched.detach().clone()
                for scale_index, delta_scale in enumerate(MY_SCALES):
                    for site_index, site in enumerate(SITES):
                        patched, calls, audit, error = r494.run_scaled_patch(
                            model, tokens, arm=arm, scale=scale,
                            deltas={site: deltas[site]}, sites=(site,),
                            position_mask=query_mask, delta_scale=delta_scale)
                        parent._record(
                            audit_totals,
                            f"rqrf:scaled:{source}:slot{slot}:{site}:{delta_scale}",
                            audit, sum(calls.values()))
                        patch_calls += sum(calls.values())
                        reconstruction = max(reconstruction, error)
                        damage = parent.position_parent._nll(patched, batch_rows) - source_nll
                        for output_index, local_doc, query in chosen:
                            scaled[si, scale_index, site_index, output_index] = \
                                float(damage[local_doc, query])
            if first_batch:
                if first_query_mask is None or len(bridge_logits) != len(SITES):
                    raise RuntimeError("missing first-batch unit bridge state")
                for site_index, site in enumerate(SITES):
                    ordinary, calls, audit, error = parent.run_subtractive_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site]}, sites=(site,),
                        position_mask=first_query_mask)
                    parent._record(
                        audit_totals, f"rqrf:bridge:{source}:{site}",
                        audit, sum(calls.values()))
                    patch_calls += sum(calls.values())
                    reconstruction = max(reconstruction, error)
                    scaled_nll = parent.position_parent._nll(
                        bridge_logits[site_index], batch_rows)
                    ordinary_nll = parent.position_parent._nll(ordinary, batch_rows)
                    bridge_error = max(
                        bridge_error, float((scaled_nll - ordinary_nll).abs().max()))
            del source_products, deltas, bridge_logits
        first_batch = False
        del absent_products
    return {
        "effects": effects,
        "scaled": scaled,
        "coordinates": coordinates,
        "empty_patch_max_abs": empty_error,
        "unit_bridge_max_abs_nat": bridge_error,
        "patch_calls": patch_calls,
    }, reconstruction


def _median_abs(actual, prediction):
    return float(np.median(np.abs(np.asarray(actual) - np.asarray(prediction))))


def _quad_prediction(key, y_half, y_one, y_mid):
    if key == "2.0":
        return 6.0 * y_one - 8.0 * y_half
    if key == "1.5":
        return 3.0 * (y_one - y_half)
    if key == "0.5":
        return y_one - y_mid / 3.0
    raise ValueError(key)


def analyze(windows):
    reports = {}
    all_live = True
    min_adjacent_diff = float("inf")
    b_flags, c_flags, d_flags = [], [], []
    for name, _, window_start, _ in WINDOWS:
        window = windows[name]
        docs = np.asarray([row[0] for row in window["coordinates"]])
        half_masks = (docs < window_start + 48, docs >= window_start + 48)
        reports[name] = {}
        for si, source in enumerate(SOURCES):
            mains, knots = r494._fit_curves(window["effects"][si])
            scaled = window["scaled"][si].detach().cpu().double().numpy()
            y_half, y_mid, y_two = scaled[0], scaled[1], scaled[2]
            count = mains.shape[1]
            donors_by_shift = [np.roll(np.arange(count), shift)
                               for shift in POSITION_SHIFTS]
            for site_index in range(len(SITES)):
                for scale_index in range(len(MY_SCALES)):
                    all_live &= bool(
                        np.sqrt(np.mean(scaled[scale_index, site_index] ** 2)) > 0)
                for low, high in ((0, 1), (1, 2)):
                    diff = float(np.sqrt(np.mean(
                        (scaled[high, site_index] - scaled[low, site_index]) ** 2)))
                    min_adjacent_diff = min(min_adjacent_diff, diff)
            coefficient_a = 4.0 * y_half - mains
            coefficient_b = 2.0 * mains - 4.0 * y_half
            magnitude = np.abs(coefficient_a)
            ratio = np.abs(coefficient_b) / np.maximum(magnitude, 1e-12)
            actual = {"0.5": y_half, "1.5": y_mid, "2.0": y_two}
            cell = {
                "occurrences": int(count),
                "sign_opposition_fraction": float(
                    np.mean(coefficient_a * coefficient_b < 0)),
                "abs_b_over_a_quantiles": [
                    float(np.quantile(ratio, q)) for q in (.25, .5, .75)],
            }
            for key in SCALE_KEYS:
                lam = float(key)
                quad = _quad_prediction(key, y_half, mains, y_mid)
                additive = lam * mains
                x = lam * mains
                iso = np.stack([r494._predict(knots, x[s]) for s in range(len(SITES))])
                in_range = np.zeros_like(x, dtype=bool)
                for position, (xk, _) in enumerate(knots):
                    in_range[:, position] = (
                        (x[:, position] >= xk[0]) & (x[:, position] <= xk[-1]))
                act = actual[key].reshape(-1)
                quad_med = _median_abs(act, quad.reshape(-1))
                add_med = _median_abs(act, additive.reshape(-1))
                iso_med = _median_abs(act, iso.reshape(-1))
                permuted_meds = []
                for donors in donors_by_shift:
                    rolled = np.concatenate([
                        _quad_prediction(
                            key, y_half[s][donors], mains[s][donors],
                            y_mid[s][donors])
                        for s in range(len(SITES))])
                    permuted_meds.append(_median_abs(act, rolled))
                q05 = float(np.quantile(permuted_meds, .05, method="lower"))
                halves = []
                for mask in half_masks:
                    repeated = np.tile(mask, len(SITES))
                    halves.append({
                        "quadratic_median_absolute_error_nat": _median_abs(
                            act[repeated], quad.reshape(-1)[repeated]),
                        "additive_median_absolute_error_nat": _median_abs(
                            act[repeated], additive.reshape(-1)[repeated]),
                    })
                cell[key] = {
                    "quadratic_median_absolute_error_nat": quad_med,
                    "additive_median_absolute_error_nat": add_med,
                    "isotonic_median_absolute_error_nat": iso_med,
                    "quadratic_over_additive": r494._safe_ratio(quad_med, add_med),
                    "quadratic_over_isotonic": r494._safe_ratio(quad_med, iso_med),
                    "permuted_donor_median_errors_nat": permuted_meds,
                    "permuted_error_q05_nat": q05,
                    "quadratic_over_permuted_q05": r494._safe_ratio(quad_med, q05),
                    "pearson": r494._pearson(quad.reshape(-1), act),
                    "actual_rms_nat": float(np.sqrt(np.mean(act ** 2))),
                    "isotonic_in_range_fraction": float(np.mean(in_range)),
                    "halves": halves,
                }
            two = cell["2.0"]
            b_flags.append(bool(
                two["additive_median_absolute_error_nat"] >= 1e-4
                and two["quadratic_median_absolute_error_nat"]
                    <= .80 * two["additive_median_absolute_error_nat"]
                and two["quadratic_median_absolute_error_nat"]
                    <= .90 * two["permuted_error_q05_nat"]
                and two["quadratic_median_absolute_error_nat"]
                    <= two["isotonic_median_absolute_error_nat"]))
            c_flags.append(bool(
                cell["1.5"]["quadratic_median_absolute_error_nat"]
                    <= .85 * cell["1.5"]["additive_median_absolute_error_nat"]
                and cell["0.5"]["quadratic_median_absolute_error_nat"]
                    <= 1.05 * cell["0.5"]["additive_median_absolute_error_nat"]
                and cell["0.5"]["quadratic_median_absolute_error_nat"]
                    <= cell["0.5"]["isotonic_median_absolute_error_nat"]))
            for half in two["halves"]:
                d_flags.append(bool(
                    half["additive_median_absolute_error_nat"] >= 1e-4
                    and half["quadratic_median_absolute_error_nat"]
                        <= .90 * half["additive_median_absolute_error_nat"]))
            reports[name][source] = cell
    return {
        "reports": reports,
        "all_scaled_interventions_live": bool(all_live),
        "minimum_adjacent_scale_difference_rms_nat": min_adjacent_diff,
        "pred_b_flags_per_cell": b_flags,
        "pred_c_flags_per_cell": c_flags,
        "pred_d_flags_per_half": d_flags,
        "pred_b_prospective_two_point_quadratic_extrapolation_at_two": bool(all(b_flags)),
        "pred_c_in_run_two_sided_quadratic_coherence": bool(all(c_flags)),
        "pred_d_document_half_stability_at_two": bool(all(d_flags)),
    }


def _synthetic_windows():
    rng = np.random.default_rng(0)
    windows = {}
    for name, _, window_start, _ in WINDOWS:
        count = 8
        effects = torch.zeros(len(SOURCES), len(SUBSETS), count, dtype=torch.float64)
        scaled = torch.zeros(
            len(SOURCES), len(MY_SCALES), len(SITES), count, dtype=torch.float64)
        for si in range(len(SOURCES)):
            base = rng.normal(size=(len(SITES), count)) * .01
            for subset_index, indices in enumerate(SUBSETS):
                effects[si, subset_index] = torch.from_numpy(
                    np.sum(base[list(indices)], axis=0))
            for scale_index, lam in enumerate(MY_SCALES):
                scaled[si, scale_index] = torch.from_numpy(
                    lam * base - .2 * lam * lam * base)
        windows[name] = {
            "effects": effects,
            "scaled": scaled,
            "coordinates": [
                (window_start + (i * 13) % 96, 10 + i, 0) for i in range(count)],
            "empty_patch_max_abs": 0.0,
            "unit_bridge_max_abs_nat": 0.0,
            "patch_calls": 0,
        }
    return windows


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EXPECTED_BATCHES == 72
        assert FORWARDS_PER_BATCH == 71
        assert EXPECTED_FORWARDS == 5130
        assert PATCH_CALLS_PER_BATCH == 90
        assert EXPECTED_PATCH_CALLS == 6498
        assert len(POSITION_SHIFTS) == 16
        analysis = analyze(_synthetic_windows())
        assert isinstance(
            analysis["pred_b_prospective_two_point_quadratic_extrapolation_at_two"], bool)
        assert len(analysis["pred_b_flags_per_cell"]) == len(WINDOWS) * len(SOURCES)
        assert len(analysis["pred_d_flags_per_half"]) == 2 * len(WINDOWS) * len(SOURCES)
        for path, expected in HASHES.items():
            if not path.is_file() or r494.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed", "rung": "quadratic_response_falsifier",
            "model_loaded": False, "outcomes_opened": False,
            "validation_or_sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_patch_calls": EXPECTED_PATCH_CALLS,
            "scales": list(MY_SCALES),
            "synthetic_analysis_exercised": True,
        }, indent=2, sort_keys=True))
        return
    roles, scale, selections, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("quadratic response falsifier output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    audit_totals, replay = {}, {"max_abs": 0.0, "relative_squared": 0.0}
    windows, reconstruction = {}, 0.0
    for name, role, _, _ in WINDOWS:
        payload, _ = roles[role]
        windows[name], error = collect_window(
            model, payload, scale, selections[name], audit_totals, replay)
        reconstruction = max(reconstruction, error)
    analysis = analyze(windows)
    forwards = sum(row.get("forwards", 0) for row in audit_totals.values())
    patch_calls = sum(row.get("subtractive_patch_calls", 0)
                      for row in audit_totals.values())
    empty_error = max(row["empty_patch_max_abs"] for row in windows.values())
    bridge_error = max(row["unit_bridge_max_abs_nat"] for row in windows.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12
        and reconstruction <= 1e-10
        and empty_error == 0.0
        and bridge_error <= 3e-5
        and analysis["all_scaled_interventions_live"]
        and analysis["minimum_adjacent_scale_difference_rms_nat"] >= 1e-4
        and forwards == EXPECTED_FORWARDS
        and patch_calls == EXPECTED_PATCH_CALLS)
    pred_b = analysis["pred_b_prospective_two_point_quadratic_extrapolation_at_two"]
    pred_c = analysis["pred_c_in_run_two_sided_quadratic_coherence"]
    pred_d = analysis["pred_d_document_half_stability_at_two"]
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d)
    torch.save({
        "schema": "quadratic_response_falsifier_v1",
        "windows": windows,
        "scales": list(MY_SCALES), "sites": list(SITES),
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": "quadratic_response_falsifier",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "architecture_derived_local_causal_response_law_test",
        "source_hashes": {str(path): r494.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "coordinate": "subtract_scaled_intact_source_minus_absent_product_at_query",
        "sites": list(SITES), "scales": list(MY_SCALES),
        "position_permutation_offsets": list(POSITION_SHIFTS),
        "analysis": analysis,
        "instrument": {
            "native_replay": replay,
            "factor_reconstruction_relative_squared_max": reconstruction,
            "empty_query_mask_max_abs": empty_error,
            "unit_strength_bridge_max_abs_nat": bridge_error,
            "forwards": forwards, "expected_forwards": EXPECTED_FORWARDS,
            "patch_calls": patch_calls,
            "expected_patch_calls": EXPECTED_PATCH_CALLS,
        },
        "bundle": {"path": str(BUNDLE), "sha256": r494.sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        'pred_a_exact_live_three_scale_instrument': pred_a,
        'pred_b_prospective_two_point_quadratic_extrapolation_at_two': pred_b,
        'pred_c_in_run_two_sided_quadratic_coherence': pred_c,
        'pred_d_document_half_stability_at_two': pred_d,
        "validation_or_sealed_opened": False,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": forwards,
            "product_hook_calls": patch_calls,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "per_occurrence_a_b_response_chart_and_regime_certificate"
            if not strong_null else
            "regime_map_stands_as_rung494_left_it"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": result["rung"],
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "instrument": result["instrument"],
        "analysis": {key: value for key, value in analysis.items()
                     if key != "reports"},
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
