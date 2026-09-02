#!/usr/bin/env python3
"""RUNG494 -- causal scaled-intervention test of the equality-query single-index law."""

# BQGATE: EXPERIMENT
# pred_a exact/live in-process scaled intervention instrument
# pred_b a per-occurrence monotone single-index predicts half-strength interventions
# pred_c the same frozen readout predicts one-and-a-half-strength interventions
# pred_d improvements are stable in both document halves
# pred_e A-D jointly license only a local causal-composition interpretation

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_query_subtractive_factorial_rung474 as parent
import mlp0_immediate_consumer_quotient_rung483 as shift_parent


PREREG = POLY / "EQUALITY_QUERY_SCALED_SINGLE_INDEX_CAUSAL_RUNG494_PREREGISTRATION.md"
REVIEW = POLY / "HOURLY_STRATEGIC_REVIEW_2026-09-02_1430.md"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
OUT = ROOT / "equality_query_scaled_single_index_causal_rung494_results.json"
BUNDLE = ROOT / "equality_query_scaled_single_index_causal_rung494_per_token.pt"
HASHES = {
    PREREG: "e2abeafce0acc23deda9c73cb4f87d09676707c4d14512fde03dee518fd8eda2",
    REVIEW: "4b300059ffce55e62d1383bc8fdee624b6d3c6ac191d46feb76cdedb32b5c4e8",
    ROOT / "ops/equality_query_subtractive_factorial_rung474.py":
        "3089bbb3703fa2d11b563d0ec04761f7c198422ba6fb695cbd477bd7c45cc13a",
    ROOT / "equality_query_subtractive_factorial_rung474_results.json":
        "17235cf0131d356332738dd6551df4ee60219836fbedbfd01ca27e9750998fb7",
    ROOT / "equality_query_subtractive_factorial_rung474_per_token.pt":
        "c5d2b38a1631df1c4aacf3fd5bf583e91b6c71edaf2afd32881b576025e32647",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
}
SOURCES = parent.SOURCES
SITES = parent.SITES
WINDOWS = parent.WINDOWS
SUBSETS = parent.SUBSETS
SUBSET_NAMES = parent.SUBSET_NAMES
SINGLE_INDICES = parent.SINGLE_INDICES
BATCH = parent.BATCH
SCALES = (0.5, 1.5)
POSITION_SHIFTS = shift_parent.POSITION_SHIFTS
EXPECTED_BATCHES = parent.EXPECTED_BATCHES
FORWARDS_PER_BATCH = 3 + len(SOURCES) * (
    2 + 2 * (len(SUBSETS) + len(SCALES) * len(SITES)))
BRIDGE_FORWARDS = len(WINDOWS) * len(SOURCES) * len(SITES)
EXPECTED_FORWARDS = EXPECTED_BATCHES * FORWARDS_PER_BATCH + BRIDGE_FORWARDS
PATCH_CALLS_PER_BATCH = len(SOURCES) * (
    len(SITES) + 2 * (
        sum(len(indices) for indices in SUBSETS) + len(SCALES) * len(SITES)))
EXPECTED_PATCH_CALLS = EXPECTED_BATCHES * PATCH_CALLS_PER_BATCH + BRIDGE_FORWARDS


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
    r493 = json.loads(R493_RESULT.read_text())
    expected = {
        "pred_a_exact_lawful_live_merge_instrument": True,
        "pred_b_attention1_merge_removes_T_I_contrast": False,
        "pred_c_progressive_T_I_merge": False,
        "pred_d_T_I_specific_depth_gradient": False,
        "pred_e_prospective_intervention_outcome_validation": False,
        "validation_licensed_and_opened": False,
        "strong_null": True,
    }
    if r493.get("rung") != 493 or any(r493.get(key) is not value
                                       for key, value in expected.items()):
        raise RuntimeError("rung493 did not license its frozen independent-route fork")
    values = parent.validate_inputs()
    roles, scale, old_effects, selections, old_position, old_factorial, metadata = values
    return roles, scale, selections, {
        **metadata,
        "rung493_result_sha256": sha256(R493_RESULT),
        "rung494_scales": list(SCALES),
        "position_permutation_offsets": list(POSITION_SHIFTS),
        "sklearn_version": __import__("sklearn").__version__,
    }


@torch.no_grad()
def run_scaled_patch(model, tokens, *, arm, scale, deltas, sites,
                     position_mask, delta_scale):
    sites = tuple(sites)
    if set(sites) - set(SITES) or set(deltas) != set(sites):
        raise ValueError("malformed scaled subtractive patch sites")
    if not math.isfinite(delta_scale) or delta_scale < 0:
        raise ValueError("delta_scale must be finite and nonnegative")
    handles, calls = [], {site: 0 for site in sites}
    for layer, site in zip(parent.MODULES, SITES):
        if site not in sites:
            continue
        delta = deltas[site]
        down = model.transformer.h[layer].mlp.Down

        def hook(_module, inputs, name=site, frozen_delta=delta):
            if calls[name] != 0:
                raise RuntimeError(f"duplicate scaled product patch at {name}")
            product = inputs[0]
            if frozen_delta.shape != product.shape or frozen_delta.device != product.device \
                    or frozen_delta.dtype != torch.float32:
                raise RuntimeError(f"scaled delta mismatch at {name}")
            updated = product.clone()
            current = product[position_mask].float()
            updated[position_mask] = (
                current - float(delta_scale) * frozen_delta[position_mask]
            ).to(product.dtype)
            calls[name] += 1
            return (updated,)

        handles.append(down.register_forward_pre_hook(hook))
    try:
        logits, _, audit, error = parent.source_parent.run_forward(
            model, tokens, arm=arm, scale=scale)
    finally:
        for handle in handles:
            handle.remove()
    if any(value != 1 for value in calls.values()):
        raise RuntimeError("not every scaled product patch fired exactly once")
    return logits, calls, audit, error


@torch.no_grad()
def collect_window(model, payload, scale, selection, audit_totals, replay):
    coordinates = selection["coordinates"]
    by_doc = {}
    for output_index, (doc, query, _) in enumerate(coordinates):
        by_doc.setdefault(doc, []).append((output_index, query))
    effects = torch.zeros(len(SOURCES), len(SUBSETS), len(coordinates), dtype=torch.float64)
    scaled = torch.zeros(
        len(SOURCES), len(SCALES), len(SITES), len(coordinates), dtype=torch.float64)
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
            audit_totals, "r494:native", audit,
            analytical=False, captures=0, patches=0)
        replay_logits, _, audit, error = parent.source_parent.run_forward(
            model, tokens, arm="replay")
        parent.audit_parent._record_audit(
            audit_totals, "r494:replay", audit,
            analytical=True, captures=0, patches=0)
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30))
        reconstruction = max(reconstruction, error)
        _, absent_products, _, audit, error = parent.product_parent.run_term_forward(
            model, tokens, arm="base", capture_products=True)
        parent._record(audit_totals, "r494:absent", audit)
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
            parent._record(audit_totals, f"r494:source:{source}", audit)
            reconstruction = max(reconstruction, error)
            source_nll = parent.position_parent._nll(source_logits, batch_rows)
            deltas = {
                site: source_products[site].float() - absent_products[site].float()
                for site in SITES
            }
            false_mask = torch.zeros_like(tokens, dtype=torch.bool)
            empty_logits, calls, audit, error = run_scaled_patch(
                model, tokens, arm=arm, scale=scale, deltas=deltas,
                sites=SITES, position_mask=false_mask, delta_scale=1.0)
            parent._record(audit_totals, f"r494:empty:{source}", audit, sum(calls.values()))
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
                    patched, calls, audit, error = run_scaled_patch(
                        model, tokens, arm=arm, scale=scale,
                        deltas={site: deltas[site] for site in sites}, sites=sites,
                        position_mask=query_mask, delta_scale=1.0)
                    parent._record(
                        audit_totals,
                        f"r494:unit:{source}:slot{slot}:{SUBSET_NAMES[subset_index]}",
                        audit, sum(calls.values()))
                    patch_calls += sum(calls.values())
                    reconstruction = max(reconstruction, error)
                    damage = parent.position_parent._nll(patched, batch_rows) - source_nll
                    for output_index, local_doc, query in chosen:
                        effects[si, subset_index, output_index] = float(damage[local_doc, query])
                    if first_batch and slot == 0 and len(indices) == 1:
                        bridge_logits[indices[0]] = patched.detach().clone()
                for scale_index, delta_scale in enumerate(SCALES):
                    for site_index, site in enumerate(SITES):
                        patched, calls, audit, error = run_scaled_patch(
                            model, tokens, arm=arm, scale=scale,
                            deltas={site: deltas[site]}, sites=(site,),
                            position_mask=query_mask, delta_scale=delta_scale)
                        parent._record(
                            audit_totals,
                            f"r494:scaled:{source}:slot{slot}:{site}:{delta_scale}",
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
                        audit_totals, f"r494:bridge:{source}:{site}",
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


def _safe_ratio(numerator, denominator):
    return float(numerator / max(denominator, 1e-30))


def _pearson(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    left, right = left - left.mean(), right - right.mean()
    return float(torch.dot(left, right) /
                 (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(1e-30))


def _cosine(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    return float(torch.dot(left, right) /
                 (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(1e-30))


def _fit_curves(effects):
    """Fit one frozen eight-point monotone curve per selected occurrence."""
    effects = effects.detach().cpu().double().numpy()
    mains = effects[list(SINGLE_INDICES)]
    count = effects.shape[1]
    knots = []
    for position in range(count):
        x = [0.0]
        y = [0.0]
        for subset_index, indices in enumerate(SUBSETS):
            x.append(float(sum(mains[index, position] for index in indices)))
            y.append(float(effects[subset_index, position]))
        fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
        knots.append((fit.X_thresholds_.copy(), fit.y_thresholds_.copy()))
    return mains, knots


def _predict(knots, x_values, donors=None):
    output = np.empty_like(x_values, dtype=np.float64)
    if donors is None:
        donors = np.arange(len(knots))
    for position, donor in enumerate(donors):
        xk, yk = knots[int(donor)]
        output[position] = np.interp(
            x_values[position], xk, yk, left=yk[0], right=yk[-1])
    return output


def _report(actual, prediction, additive, permuted, half_masks, in_range):
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    additive = np.asarray(additive, dtype=np.float64)
    single_error = float(np.median(np.abs(actual - prediction)))
    additive_error = float(np.median(np.abs(actual - additive)))
    permuted_errors = [float(np.median(np.abs(actual - row))) for row in permuted]
    q05 = float(np.quantile(permuted_errors, .05, method="lower"))
    halves = []
    for mask in half_masks:
        repeated = np.tile(mask, len(SITES))
        se = float(np.median(np.abs(actual[repeated] - prediction[repeated])))
        ae = float(np.median(np.abs(actual[repeated] - additive[repeated])))
        halves.append({
            "single_index_median_absolute_error_nat": se,
            "additive_median_absolute_error_nat": ae,
            "single_over_additive": _safe_ratio(se, ae),
            "holds": bool(se <= .90 * ae),
        })
    return {
        "single_index_median_absolute_error_nat": single_error,
        "additive_median_absolute_error_nat": additive_error,
        "single_over_additive": _safe_ratio(single_error, additive_error),
        "improvement_fraction": 1.0 - _safe_ratio(single_error, additive_error),
        "position_permuted_median_errors_nat": permuted_errors,
        "position_permuted_error_q05_nat": q05,
        "single_over_permuted_q05": _safe_ratio(single_error, q05),
        "pearson": _pearson(prediction, actual),
        "cosine": _cosine(prediction, actual),
        "actual_rms_nat": float(np.sqrt(np.mean(actual ** 2))),
        "prediction_rms_nat": float(np.sqrt(np.mean(prediction ** 2))),
        "in_fitted_index_range_fraction": float(np.mean(in_range)),
        "halves": halves,
        "primary_holds": bool(
            additive_error >= 1e-4
            and single_error <= .85 * additive_error
            and single_error <= .90 * q05),
    }


def analyze(windows):
    reports = {}
    b_flags, c_flags, d_flags = [], [], []
    all_scaled_live = True
    minimum_scale_difference_rms = float("inf")
    for name, _, window_start, _ in WINDOWS:
        window = windows[name]
        docs = np.asarray([row[0] for row in window["coordinates"]])
        half_masks = (docs < window_start + 48, docs >= window_start + 48)
        reports[name] = {}
        for source_index, source in enumerate(SOURCES):
            mains, knots = _fit_curves(window["effects"][source_index])
            reports[name][source] = {}
            count = mains.shape[1]
            donors_by_shift = [np.roll(np.arange(count), shift) for shift in POSITION_SHIFTS]
            scaled_values = window["scaled"][source_index].detach().cpu().double().numpy()
            for site_index, site in enumerate(SITES):
                for scale_index, delta_scale in enumerate(SCALES):
                    values = scaled_values[scale_index, site_index]
                    all_scaled_live &= bool(np.sqrt(np.mean(values ** 2)) > 0)
                difference_rms = float(np.sqrt(np.mean(
                    (scaled_values[1, site_index] - scaled_values[0, site_index]) ** 2)))
                minimum_scale_difference_rms = min(minimum_scale_difference_rms, difference_rms)
                all_scaled_live &= difference_rms >= 1e-4
            for scale_index, delta_scale in enumerate(SCALES):
                x = delta_scale * mains
                actual_matrix = scaled_values[scale_index]
                prediction_matrix = np.stack([
                    _predict(knots, x[site]) for site in range(len(SITES))])
                permuted = []
                for donors in donors_by_shift:
                    permuted.append(np.concatenate([
                        _predict(knots, x[site], donors=donors)
                        for site in range(len(SITES))]))
                in_range_matrix = np.zeros_like(x, dtype=bool)
                for position, (xk, _) in enumerate(knots):
                    in_range_matrix[:, position] = (
                        (x[:, position] >= xk[0]) & (x[:, position] <= xk[-1]))
                report = _report(
                    actual_matrix.reshape(-1), prediction_matrix.reshape(-1), x.reshape(-1),
                    permuted, half_masks, in_range_matrix.reshape(-1))
                reports[name][source][str(delta_scale)] = report
                (b_flags if delta_scale == .5 else c_flags).append(report["primary_holds"])
                d_flags.extend(row["holds"] for row in report["halves"])
    return {
        "reports": reports,
        "all_scaled_interventions_live": bool(all_scaled_live),
        "minimum_half_vs_one_and_half_effect_difference_rms_nat": minimum_scale_difference_rms,
        "pred_b_half_strength_causal_interpolation": bool(all(b_flags)),
        "pred_c_one_and_half_strength_causal_transfer": bool(all(c_flags)),
        "pred_d_document_half_stability": bool(all(d_flags)),
    }


def main():
    started = time.time()
    roles, scale, selections, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EXPECTED_BATCHES == 72
        assert FORWARDS_PER_BATCH == 59
        assert EXPECTED_FORWARDS == 4266
        assert EXPECTED_PATCH_CALLS == 5634
        assert len(POSITION_SHIFTS) == 16
        print(json.dumps({
            "status": "dry_run_passed", "rung": 494,
            "model_loaded": False, "outcomes_opened": False,
            "validation_or_sealed_opened": False,
            "expected_forwards": EXPECTED_FORWARDS,
            "expected_patch_calls": EXPECTED_PATCH_CALLS,
            "scales": list(SCALES), "sites": list(SITES),
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung494 output namespace already exists")
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
        and analysis["minimum_half_vs_one_and_half_effect_difference_rms_nat"] >= 1e-4
        and forwards == EXPECTED_FORWARDS
        and patch_calls == EXPECTED_PATCH_CALLS)
    pred_b = analysis["pred_b_half_strength_causal_interpolation"]
    pred_c = analysis["pred_c_one_and_half_strength_causal_transfer"]
    pred_d = analysis["pred_d_document_half_stability"]
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d)
    torch.save({
        "schema": "rung494_scaled_single_index_causal_v1",
        "windows": windows,
        "scales": list(SCALES), "sites": list(SITES),
        "raw_tokens_logits_or_hidden_states_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 494,
        "claim_level": "local_causal_single_index_composition_test",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "coordinate": "subtract_scaled_intact_source_minus_absent_product_at_query",
        "sites": list(SITES), "scales": list(SCALES),
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
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                   "raw_tokens_logits_or_hidden_states_included": False},
        'pred_a_exact_live_scaled_intervention': pred_a,
        'pred_b_half_strength_causal_interpolation': pred_b,
        'pred_c_one_and_half_strength_causal_transfer': pred_c,
        'pred_d_document_half_stability': pred_d,
        'pred_e_local_causal_single_index_interpretation': pred_e,
        "validation_or_sealed_opened": False,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": forwards,
            "product_hook_calls": patch_calls,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "shared_readout_cross_corpus_and_62_circuit_test" if pred_e else
            "attention1_exact_QK1_QK2_OV_downstream_use_decomposition"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": 494,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "instrument": result["instrument"],
        "analysis": analysis,
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
