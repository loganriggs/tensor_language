#!/usr/bin/env python3
"""Nested construction-adaptive complement DAS with prospective v15 outcome."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_closure_finiteness_and_price pred_b_nested_regularization_beats_both_outer_controls pred_c_inner_selection_is_direction_stable pred_d_global_refit_beats_pooled_on_v15 pred_e_v15_improvement_is_not_target_sacrifice pred_f_generalization_gap_shrinks
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as v8
import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as v10
import circuit_candidate_temporal_auxiliary_fresh_cues_v15 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_h3_das_hard_feasible_kl_stability_v1 as hard
import run_temporal_h3_das_noisy_worst_environment_multi_reader_v1 as fitlib

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_h3_das_nested_construction_adaptive_regularization_v1.json"
PREVIOUS = ROOT / "circuits/followups/temporal_h3_das_family_crossvalidated_regularization_tournament_v2_result.json"
REDTEAM = ROOT.parent / "polynomial_causal/CONSTRAINED_DAS_REGULARIZATION_REDTEAM_2026-09-07.md"
CAP8 = hard.CAP8
CAP10 = hard.CAP10
CAP15 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v15_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_h3_das_nested_construction_adaptive_regularization_v1_result.json"
EXPECTED = {
    "prior": "25afcd83bec3f53aaad53368f6c6cf45b489aeec1928c939d0cb0cdcb1b4957c",
    "previous": "889b79b787a861b2292cf1620506a35f670e783b7f6ffdfe805aeeb0e27f7729",
    "redteam": "0d947a24eac200057dd977830532ee7a7091c064ac96a276e7b887217214f817",
    "v8_builder": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "v8_capability": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "v10_builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "v10_capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "v15_builder": "e11f490895dcbfaf06039640b6b980c9d0ef7f8e25d823a4b95803fd67617418",
    "v15_capability": "5d60c5733147419ea3a988b842c5bee96652a0f6f939910cd03e1687b5478867",
    "hard_library": "2705c330d1254f544d8ca6477e649aebdb5df5dec8cce965bcbda8c9e81746cc",
    "fit_library": "bf806b077da4e2fb43612f8f3b5ca318e0d45edac6b2c26ecc2f6accd218b413",
}
CONFIGS = {
    "no_reg": (0.0, 0.0), "kl_025": (0.25, 0.0), "kl_1": (1.0, 0.0),
    "kl_4": (4.0, 0.0), "noise_10": (0.0, 0.10),
    "noise_10_kl_025": (0.25, 0.10), "noise_10_kl_1": (1.0, 0.10),
}
STEPS = 10
CHECKPOINTS = (0, 5, 10)
LR = 0.025
TAU = 0.10
BARRIER = 100.0
VARIANCE_WEIGHT = 0.5
MAX_FORWARDS = 5000
MAX_UPDATES = 350
ALL_TERMS = ("margin_match", "l15_match", "margin_inert", "l15_inert", "vocab_match", "vocab_inert")
CORE_TERMS = ("margin_match", "l15_match", "margin_inert", "l15_inert")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def unit(value):
    return value / value.norm().clamp_min(1e-30)


def score(report):
    return report["secondary_worst"], report["secondary_mean"]


def strictly_better(left, right):
    return left["secondary_worst"] < right["secondary_worst"] and left["secondary_mean"] < right["secondary_mean"]


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def noisy_axes(torch, axis, sigma, generator):
    if sigma == 0.0:
        return (axis,)
    noise = torch.randn(axis.shape, generator=generator).to(axis.device)
    tangent = noise - axis * (axis.T @ noise)
    tangent = tangent / tangent.norm().clamp_min(1e-30)
    return tuple(unit(axis + sign * sigma * tangent) for sign in (-1.0, 1.0))


def calibrate(backend, contexts, pooled):
    pieces = hard.pieces_for_axis(backend, contexts, pooled)
    limits = [{term: float(item[term]) + hard.FEASIBILITY_SLACK for term in hard.TARGET_TERMS} for item in pieces]
    normalizers = [{term: max(abs(float(item[term])), 0.02) for term in ALL_TERMS} for item in pieces]
    return limits, normalizers


def objective(backend, contexts, limits, normalizers, raw, config, generator):
    torch = backend.torch
    kl_weight, sigma = CONFIGS[config]
    axes = noisy_axes(torch, unit(raw), sigma, generator)
    environment_values = []
    environment_vectors = []
    violations = []
    for context, bounds, norms in zip(contexts, limits, normalizers):
        values, vectors, failed = [], [], []
        for axis in axes:
            pieces = fitlib.environment_loss(backend, context, axis @ axis.T, grad=True)[1]
            values.append(sum(pieces[term] for term in CORE_TERMS) + kl_weight * (pieces["vocab_match"] + pieces["vocab_inert"]))
            vectors.append(torch.stack(tuple(pieces[term] / norms[term] for term in ALL_TERMS)))
            failed.append(sum(torch.relu(pieces[term] - bounds[term]).square() for term in hard.TARGET_TERMS))
        environment_values.append(sum(values) / len(values))
        environment_vectors.append(sum(vectors) / len(vectors))
        violations.append(sum(failed) / len(failed))
    values = torch.stack(environment_values)
    vectors = torch.stack(environment_vectors)
    robust = TAU * torch.logsumexp(values / TAU, dim=0) - TAU * math.log(len(contexts))
    variance = (vectors - vectors.mean(0, keepdim=True)).square().mean()
    return robust + VARIANCE_WEIGHT * variance + BARRIER * torch.stack(violations).sum()


def fit_config(backend, train, train_limits, train_norms, select, select_limits, initial, config, label, steps=STEPS):
    torch = backend.torch
    raw = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)
    generator = torch.Generator(device="cpu").manual_seed(20260907 + sum(map(ord, config + label)))
    candidates, trace = [], []

    def checkpoint(step):
        axis = unit(raw).detach().clone()
        report = hard.evaluate_axis(backend, select, select_limits, axis)
        candidates.append((step, axis, report))
        trace.append({"step": step, **{key: report[key] for key in ("feasible", "max_violation", "secondary_mean", "secondary_worst")}})

    checkpoint(0)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        objective(backend, train, train_limits, train_norms, raw, config, generator).backward()
        optimizer.step()
        if step in CHECKPOINTS:
            checkpoint(step)
    eligible = [item for item in candidates if item[2]["feasible"]]
    best = min(eligible, key=lambda item: (*score(item[2]), item[0])) if eligible else min(candidates, key=lambda item: (item[2]["max_violation"], *score(item[2]), item[0]))
    return {"config": config, "label": label, "best_step": best[0], "best_axis": best[1], "best_report": best[2], "trace": trace}


def refit(backend, contexts, limits, normalizers, initial, config, steps, label):
    torch = backend.torch
    raw = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)
    generator = torch.Generator(device="cpu").manual_seed(20260907 + sum(map(ord, config + label)))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective(backend, contexts, limits, normalizers, raw, config, generator).backward()
        optimizer.step()
    return unit(raw).detach()


def make_family(backend, builder, capability, pooled):
    rows = hard.rows_for(builder, capability)
    directions = sorted({row["direction_id"] for row in rows})
    contexts = {}
    for direction in directions:
        contexts[direction] = [
            fitlib.attach_targets(backend, [row for row in rows if row["direction_id"] == direction and row["transform_id"] == panel])
            for panel in ("A1", "A2")
        ]
    all_contexts = [context for direction in directions for context in contexts[direction]]
    limits, normalizers = calibrate(backend, all_contexts, pooled)
    by_direction = {}
    for index, direction in enumerate(directions):
        by_direction[direction] = {
            "contexts": contexts[direction],
            "limits": limits[2 * index:2 * index + 2],
            "normalizers": normalizers[2 * index:2 * index + 2],
        }
    return {"rows": rows, "directions": directions, "by_direction": by_direction, "contexts": all_contexts, "limits": limits, "normalizers": normalizers}


def stripped(fit):
    return {key: fit[key] for key in ("config", "label", "best_step", "best_report", "trace")}


def aggregate(fits):
    return {
        "worst": max(fit["best_report"]["secondary_worst"] for fit in fits),
        "mean": sum(fit["best_report"]["secondary_mean"] for fit in fits) / len(fits),
        "updates": sum(fit["best_step"] for fit in fits),
    }


def main():
    candidate_id = "temporal_auxiliary.will_vs_had.h3_das_nested_construction_adaptive_regularization_v1"
    dry = {
        "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "configs": CONFIGS, "outer_folds": ["v8_to_v10", "v10_to_v8"],
        "inner_folds": 2, "steps": STEPS, "checkpoints": CHECKPOINTS,
        "model_reader_evaluations_max": MAX_FORWARDS, "model_updates_max": MAX_UPDATES, "fit_parameters": 128,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {
        "prior": PRIOR, "previous": PREVIOUS, "redteam": REDTEAM,
        "v8_builder": hard.V8, "v8_capability": CAP8, "v10_builder": hard.V10, "v10_capability": CAP10,
        "v15_builder": ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v15.py", "v15_capability": CAP15,
        "hard_library": ROOT / "ops/run_temporal_h3_das_hard_feasible_kl_stability_v1.py", "fit_library": hard.FITLIB,
    }
    observed = {key: sha(path) for key, path in paths.items()}
    started, tic = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    native_original = backend.native
    manual_original = fitlib.manual_readers
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
    fitlib.manual_readers = counted_manual
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    previous = json.loads(PREVIOUS.read_text())
    authority = json.loads(hard.HARD.read_text())
    selection = authority["selection"]
    pooled_axis = unit(torch.tensor(selection["selected_axis"], device=backend.device).float().unsqueeze(1))
    capabilities = {"v8": json.loads(CAP8.read_text()), "v10": json.loads(CAP10.read_text()), "v15": json.loads(CAP15.read_text())}
    families = {
        "v8": make_family(backend, v8, capabilities["v8"], pooled_axis),
        "v10": make_family(backend, v10, capabilities["v10"], pooled_axis),
    }
    inner_fits = {}
    for family_name in ("v8", "v10"):
        family = families[family_name]
        inner_fits[family_name] = []
        for direction in family["directions"]:
            other = next(item for item in family["directions"] if item != direction)
            train = family["by_direction"][direction]
            select = family["by_direction"][other]
            for config in CONFIGS:
                inner_fits[family_name].append(fit_config(
                    backend, train["contexts"], train["limits"], train["normalizers"],
                    select["contexts"], select["limits"], pooled_axis, config, f"{family_name}:{direction}",
                ))

    config_scores = {
        family_name: {config: aggregate([fit for fit in fits if fit["config"] == config]) for config in CONFIGS}
        for family_name, fits in inner_fits.items()
    }
    selected = {
        family_name: min(CONFIGS, key=lambda config: (config_scores[family_name][config]["worst"], config_scores[family_name][config]["mean"], config_scores[family_name][config]["updates"], list(CONFIGS).index(config)))
        for family_name in ("v8", "v10")
    }
    per_direction_winners = {
        family_name: [
            min(CONFIGS, key=lambda config: (*score(next(fit for fit in fits if fit["config"] == config and fit["label"] == label)["best_report"]), next(fit for fit in fits if fit["config"] == config and fit["label"] == label)["best_step"], list(CONFIGS).index(config)))
            for label in [f"{family_name}:{direction}" for direction in families[family_name]["directions"]]
        ]
        for family_name, fits in inner_fits.items()
    }
    outer = {}
    updates = 2 * 2 * len(CONFIGS) * STEPS
    for train_name, test_name in (("v8", "v10"), ("v10", "v8")):
        train, test = families[train_name], families[test_name]
        chosen = selected[train_name]
        chosen_fits = [fit for fit in inner_fits[train_name] if fit["config"] == chosen]
        chosen_step = min(CHECKPOINTS, key=lambda step: (abs(step - sum(fit["best_step"] for fit in chosen_fits) / 2), step))
        no_reg_fits = [fit for fit in inner_fits[train_name] if fit["config"] == "no_reg"]
        no_reg_step = min(CHECKPOINTS, key=lambda step: (abs(step - sum(fit["best_step"] for fit in no_reg_fits) / 2), step))
        chosen_axis = refit(backend, train["contexts"], train["limits"], train["normalizers"], pooled_axis, chosen, chosen_step, f"outer:{train_name}")
        no_reg_axis = refit(backend, train["contexts"], train["limits"], train["normalizers"], pooled_axis, "no_reg", no_reg_step, f"outer-no-reg:{train_name}")
        updates += chosen_step + no_reg_step
        outer[train_name] = {
            "test_family": test_name, "selected_config": chosen, "selected_step": chosen_step, "no_reg_step": no_reg_step,
            "selected": hard.evaluate_axis(backend, test["contexts"], test["limits"], chosen_axis),
            "no_reg": hard.evaluate_axis(backend, test["contexts"], test["limits"], no_reg_axis),
            "pooled": hard.evaluate_axis(backend, test["contexts"], test["limits"], pooled_axis),
            "selected_inner": config_scores[train_name][chosen], "no_reg_inner": config_scores[train_name]["no_reg"],
        }
    global_scores = {config: aggregate([fit for fits in inner_fits.values() for fit in fits if fit["config"] == config]) for config in CONFIGS}
    global_config = min(CONFIGS, key=lambda config: (global_scores[config]["worst"], global_scores[config]["mean"], global_scores[config]["updates"], list(CONFIGS).index(config)))
    global_fits = [fit for fits in inner_fits.values() for fit in fits if fit["config"] == global_config]
    global_step = min(CHECKPOINTS, key=lambda step: (abs(step - sum(fit["best_step"] for fit in global_fits) / len(global_fits)), step))
    combined_contexts = families["v8"]["contexts"] + families["v10"]["contexts"]
    combined_limits = families["v8"]["limits"] + families["v10"]["limits"]
    combined_norms = families["v8"]["normalizers"] + families["v10"]["normalizers"]
    final_axis = refit(backend, combined_contexts, combined_limits, combined_norms, pooled_axis, global_config, global_step, "global")
    updates += global_step
    # The prospective family remains unopened until configuration, step, and final
    # axis are frozen entirely from v8/v10 inner evidence.
    families["v15"] = make_family(backend, v15, capabilities["v15"], pooled_axis)
    sealed = {
        "selected": hard.evaluate_axis(backend, families["v15"]["contexts"], families["v15"]["limits"], final_axis),
        "pooled": hard.evaluate_axis(backend, families["v15"]["contexts"], families["v15"]["limits"], pooled_axis),
    }
    closure = max(context["manual_base_margin_max_abs"] for family in families.values() for context in family["contexts"])
    row_ids = [{row["row_id"] for row in families[name]["rows"]} for name in ("v8", "v10", "v15")]
    total_forwards = native_count + manual_count
    pred_b = all(
        item["selected_config"] != "no_reg" and item["selected_step"] > 0 and item["selected"]["feasible"]
        and strictly_better(item["selected"], item["no_reg"]) and strictly_better(item["selected"], item["pooled"])
        for item in outer.values()
    )
    pred_c = all(len(set(winners)) == 1 for winners in per_direction_winners.values())
    pred_d = sealed["selected"]["feasible"] and strictly_better(sealed["selected"], sealed["pooled"])
    pred_e = sealed["selected"]["feasible"] and all(
        sum(sealed["selected"]["environments"][index]["secondary_parts"][term] for term in ("vocab_match", "vocab_inert"))
        < sum(sealed["pooled"]["environments"][index]["secondary_parts"][term] for term in ("vocab_match", "vocab_inert"))
        for index in range(len(sealed["selected"]["environments"]))
    )
    gaps = {
        name: {
            "selected": item["selected"]["secondary_worst"] - item["selected_inner"]["worst"],
            "no_reg": item["no_reg"]["secondary_worst"] - item["no_reg_inner"]["worst"],
        }
        for name, item in outer.items()
    }
    pred_f = all(value["selected"] < value["no_reg"] for value in gaps.values())
    atomic = {
        "authority_hashes": observed == EXPECTED,
        "previous_terminal": previous["terminal"] == "regularization_fold_heterogeneity",
        "pooled_authority": authority["terminal"] == "family_memorization" and selection["selected_source"] == "deterministic_pooled" and selection["selected_step"] == 0,
        "capability_manifests": all(capability["terminal"] == "manifest" for capability in capabilities.values()),
        "directions_and_panels": all(len(family["directions"]) == 2 and len(family["contexts"]) == 4 for family in families.values()),
        "families_row_disjoint": not (row_ids[0] & row_ids[1] or row_ids[0] & row_ids[2] or row_ids[1] & row_ids[2]),
        "manual_reader_closure": closure <= 1e-4,
        "all_metrics_finite": finite([config_scores, outer, global_scores, sealed, gaps, closure]),
        "forward_price": total_forwards <= MAX_FORWARDS,
        "update_price": updates <= MAX_UPDATES,
    }
    pred_a = all(atomic.values())
    predictions = {
        "pred_a_authority_disjointness_closure_finiteness_and_price": bool(pred_a),
        "pred_b_nested_regularization_beats_both_outer_controls": bool(pred_b),
        "pred_c_inner_selection_is_direction_stable": bool(pred_c),
        "pred_d_global_refit_beats_pooled_on_v15": bool(pred_d),
        "pred_e_v15_improvement_is_not_target_sacrifice": bool(pred_e),
        "pred_f_generalization_gap_shrinks": bool(pred_f),
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "nested_regularized_das_candidate"
    elif pred_b and pred_d and pred_e and not pred_c:
        terminal = "adaptive_but_not_global_objective"
    elif not pred_b:
        terminal = "inner_selection_mispredicts_construction"
    else:
        terminal = "residual_family_memorization"
    result = {
        "schema": "temporal_h3_das_nested_construction_adaptive_regularization_result_v1",
        "candidate_id": candidate_id, "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter() - tic,
        "authority_sha256": EXPECTED, "authority_checks": atomic, "row_counts": {name: len(family["rows"]) for name, family in families.items()},
        "manual_reader_closure_max_abs": closure, "inner_fits": {name: [stripped(fit) for fit in fits] for name, fits in inner_fits.items()},
        "config_scores": config_scores, "selected_config": selected, "per_direction_winners": per_direction_winners,
        "outer": outer, "global_scores": global_scores, "global_config": global_config, "global_step": global_step,
        "global_axis": [float(value) for value in final_axis[:, 0]], "sealed_v15": sealed, "generalization_gaps": gaps,
        "predictions": predictions, "terminal": terminal,
        "price": {"native_forwards_observed": native_count, "manual_reader_forwards_observed": manual_count, "model_reader_evaluations_observed": total_forwards, "model_reader_evaluations_max": MAX_FORWARDS, "model_updates_observed": updates, "model_updates_max": MAX_UPDATES, "fit_parameters": 128},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("authority_checks", "row_counts", "manual_reader_closure_max_abs", "config_scores", "selected_config", "per_direction_winners", "outer", "global_scores", "global_config", "global_step", "sealed_v15", "generalization_gaps", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
