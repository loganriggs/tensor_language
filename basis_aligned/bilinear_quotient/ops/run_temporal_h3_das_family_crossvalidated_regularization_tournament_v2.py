#!/usr/bin/env python3
"""Fully instrumented deterministic replay of the complete-family DAS tournament."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_instrument_finiteness_and_price pred_b_regularization_wins_both_family_folds pred_c_regularized_axes_are_cross_family_stable pred_d_regularized_refit_beats_pooled_on_sealed_v12 pred_e_improvement_is_not_target_sacrifice
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_h3_das_family_crossvalidated_regularization_tournament_v1 as base

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_h3_das_family_crossvalidated_regularization_tournament_v2.json"
BASE = ROOT / "ops/run_temporal_h3_das_family_crossvalidated_regularization_tournament_v1.py"
OUT = ROOT / "circuits/followups/temporal_h3_das_family_crossvalidated_regularization_tournament_v2_result.json"
EXPECTED = {
    "prior": "9900d53f6abca2a6d9a91b582661898676d11ba97e89a647b6c55ee1e1c74f15",
    "base_v1": "5c6ca7a519b0e6aa123aa8a03e93fa87fec1ebf467d3e1bb5d2a05aad2f05808",
    **{k: v for k, v in base.EXPECTED.items() if k != "prior"},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_numbers(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def main():
    candidate_id = "temporal_auxiliary.will_vs_had.h3_das_family_crossvalidated_regularization_tournament_v2"
    dry = {
        "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "arms": list(base.ARMS),
        "folds": ["v8_to_v10", "v10_to_v8"], "sealed": "v12",
        "steps": base.STEPS, "checkpoints": base.CHECKPOINTS,
        "model_forwards_max": base.MAX_FORWARDS, "model_updates_max": base.MAX_UPDATES,
        "fit_parameters": 128,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)

    paths = {
        "prior": PRIOR, "base_v1": BASE, "redteam": base.REDTEAM,
        "hard_feasible": base.HARD, "fit_library": base.hard.FITLIB,
        "v8_builder": base.hard.V8, "v8_capability": base.CAP8,
        "v10_builder": base.hard.V10, "v10_capability": base.CAP10,
        "v12_builder": ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v12.py",
        "v12_capability": base.CAP12,
    }
    observed = {k: sha(v) for k, v in paths.items()}
    started, tic = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    native_original = backend.native
    manual_original = base.fitlib.manual_readers
    native_count = 0
    manual_count = 0

    def counted_native(*args, **kwargs):
        nonlocal native_count
        native_count += 1
        return native_original(*args, **kwargs)

    def counted_manual(*args, **kwargs):
        nonlocal manual_count
        manual_count += 1
        return manual_original(*args, **kwargs)

    backend.native = counted_native
    base.fitlib.manual_readers = counted_manual
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)

    authority = json.loads(base.HARD.read_text())
    selection = authority["selection"]
    pooled = base.unit(torch.tensor(selection["selected_axis"], device=backend.device).float().unsqueeze(1))
    caps = [json.loads(path.read_text()) for path in (base.CAP8, base.CAP10, base.CAP12)]
    rows8, contexts8 = base.contexts(backend, base.v8, caps[0])
    rows10, contexts10 = base.contexts(backend, base.v10, caps[1])
    rows12, contexts12 = base.contexts(backend, base.v12, caps[2])
    all_contexts = contexts8 + contexts10 + contexts12
    closure = max(context["manual_base_margin_max_abs"] for context in all_contexts)
    limits8 = base.hard.limits_for(backend, contexts8, pooled)
    limits10 = base.hard.limits_for(backend, contexts10, pooled)
    limits12 = base.hard.limits_for(backend, contexts12, pooled)

    folds = [
        ("v8_to_v10", contexts8, limits8, contexts10, limits10),
        ("v10_to_v8", contexts10, limits10, contexts8, limits8),
    ]
    fits = []
    for name, train, train_limits, valid, valid_limits in folds:
        for arm in base.ARMS:
            fits.append(base.fit_arm(backend, train, train_limits, valid, valid_limits, pooled, arm, name))
    by_arm = {arm: [fit for fit in fits if fit["arm"] == arm] for arm in base.ARMS}
    arm_scores = {
        arm: {
            "worst": max(fit["best_report"]["secondary_worst"] for fit in arm_fits),
            "mean": sum(fit["best_report"]["secondary_mean"] for fit in arm_fits) / 2,
        }
        for arm, arm_fits in by_arm.items()
    }
    selected_arm = min(base.ARMS, key=lambda arm: (arm_scores[arm]["worst"], arm_scores[arm]["mean"], list(base.ARMS).index(arm)))
    selected_folds = by_arm[selected_arm]
    selected_steps = [fit["best_step"] for fit in selected_folds]
    final_step = min(base.CHECKPOINTS, key=lambda step: (abs(step - sum(selected_steps) / 2), step))
    final_axis = base.refit(backend, contexts8 + contexts10, limits8 + limits10, pooled, selected_arm, final_step)
    sealed = {
        "selected": base.hard.evaluate_axis(backend, contexts12, limits12, final_axis),
        "pooled": base.hard.evaluate_axis(backend, contexts12, limits12, pooled),
    }
    fold_cosine = float((selected_folds[0]["best_axis"].T @ selected_folds[1]["best_axis"]).abs())
    pooled_folds = [base.hard.evaluate_axis(backend, valid, valid_limits, pooled) for _, _, _, valid, valid_limits in folds]
    pred_b = selected_arm != "no_reg" and all(
        fit["best_step"] > 0
        and base.score(fit["best_report"]) < base.score(by_arm["no_reg"][index]["best_report"])
        and base.score(fit["best_report"]) < base.score(pooled_folds[index])
        for index, fit in enumerate(selected_folds)
    )
    pred_c = fold_cosine >= 0.8
    pred_d = sealed["selected"]["feasible"] and base.score(sealed["selected"]) < base.score(sealed["pooled"])
    pred_e = sealed["selected"]["feasible"] and all(
        sum(sealed["selected"]["environments"][index]["secondary_parts"][key] for key in ("vocab_match", "vocab_inert"))
        < sum(sealed["pooled"]["environments"][index]["secondary_parts"][key] for key in ("vocab_match", "vocab_inert"))
        for index in range(2)
    )
    row_ids = [{row["row_id"] for row in rows} for rows in (rows8, rows10, rows12)]
    updates = 8 * base.STEPS + final_step
    total_forwards = native_count + manual_count
    atomic = {
        "authority_hashes": observed == EXPECTED,
        "predecessor_terminal": authority["terminal"] == "family_memorization",
        "predecessor_source": selection["selected_source"] == "deterministic_pooled",
        "predecessor_step_zero": selection["selected_step"] == 0,
        "capability_manifests": all(cap["terminal"] == "manifest" for cap in caps),
        "families_row_disjoint": not (row_ids[0] & row_ids[1] or row_ids[0] & row_ids[2] or row_ids[1] & row_ids[2]),
        "manual_reader_closure": closure <= 1e-4,
        "all_metrics_finite": finite_numbers([arm_scores, [base.strip(fit) for fit in fits], sealed, fold_cosine, closure]),
        "forward_price": total_forwards <= base.MAX_FORWARDS,
        "update_price": updates <= base.MAX_UPDATES,
    }
    pred_a = all(atomic.values())
    predictions = {
        "pred_a_authority_disjointness_instrument_finiteness_and_price": bool(pred_a),
        "pred_b_regularization_wins_both_family_folds": bool(pred_b),
        "pred_c_regularized_axes_are_cross_family_stable": bool(pred_c),
        "pred_d_regularized_refit_beats_pooled_on_sealed_v12": bool(pred_d),
        "pred_e_improvement_is_not_target_sacrifice": bool(pred_e),
    }
    moved = any(fit["best_step"] > 0 for fit in fits)
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "regularized_cross_family_das_candidate"
    elif not moved:
        terminal = "rank_one_inadequate"
    elif selected_arm != "no_reg" and not pred_b:
        terminal = "regularization_fold_heterogeneity"
    elif selected_arm == "no_reg":
        terminal = "objective_miss"
    else:
        terminal = "residual_family_memorization"
    result = {
        "schema": "temporal_h3_das_family_crossvalidated_regularization_tournament_result_v2",
        "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter() - tic,
        "authority_sha256": EXPECTED, "authority_checks": atomic,
        "row_counts": {"v8": len(rows8), "v10": len(rows10), "v12": len(rows12)},
        "manual_reader_closure_max_abs": closure, "arm_scores": arm_scores,
        "selected_arm": selected_arm, "selected_fold_steps": selected_steps,
        "selected_fold_axis_cosine": fold_cosine, "final_step": final_step,
        "fits": [base.strip(fit) for fit in fits], "sealed": sealed,
        "predictions": predictions, "terminal": terminal,
        "price": {
            "native_forwards_observed": native_count, "manual_reader_forwards_observed": manual_count,
            "model_forwards_observed": total_forwards, "model_forwards_max": base.MAX_FORWARDS,
            "model_updates_observed": updates, "model_updates_max": base.MAX_UPDATES, "fit_parameters": 128,
        },
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("authority_checks", "manual_reader_closure_max_abs", "arm_scores", "selected_arm", "selected_fold_steps", "selected_fold_axis_cosine", "final_step", "sealed", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
