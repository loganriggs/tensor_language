#!/usr/bin/env python3
"""Hard-feasible rank-one H3 DAS with KL/complement secondary selection."""

# BQGATE: EXPERIMENT pred_a_authority_instrument_finiteness_and_price pred_b_nonbaseline_feasible_checkpoint_exists pred_c_feasible_optimization_improves_v8_secondary pred_d_selected_generalizes_to_sealed_families pred_e_regularization_is_stable_not_memorizing
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fit_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as fit_v2
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as select_v8
import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as test_v10
import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as test_v11
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_h3_das_noisy_worst_environment_multi_reader_v1 as fitlib
import run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2 as evaluator

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_h3_das_hard_feasible_kl_stability_v1.json"
REDTEAM = ROOT.parent / "polynomial_causal/CONSTRAINED_DAS_MEMORIZATION_REDTEAM_2026-09-07.md"
PREVIOUS = ROOT / "circuits/followups/temporal_h3_das_complete_family_holdout_v1_result.json"
REGULARIZED = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_regularized_cdas_v1_result.json"
ALIGNED = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1_result.json"
POOLED = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAP8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
CAP10 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
CAP11 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v11_capability_v1_result.json"
V1 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
V2 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v2.py"
V8 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py"
V10 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
V11 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v11.py"
FITLIB = ROOT / "ops/run_temporal_h3_das_noisy_worst_environment_multi_reader_v1.py"
EVALUATOR = ROOT / "ops/run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
OUT = ROOT / "circuits/followups/temporal_h3_das_hard_feasible_kl_stability_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_das_hard_feasible_kl_stability_v1"
EXPECTED = {
    "prior": "660ca7a596b1b52ecf94278f845b5ddf9eaeee28879b3f608938d6869c79f2e3",
    "redteam": "f39cdea34e1006d9e5f9db47cdd6f8d710c8a69178577148f36e53837af69b1b",
    "previous": "0a7e46a037b53fb6c00fd146c816f09df655c2f9beffb9fcf99c144a108a38b4",
    "regularized": "f7d53dd6530dbdbebba7610236adc862b3c595bd83fb6c1b24d8fd4365543163",
    "aligned": "3aea84323bae1c2e46a430ef5f08b838826504693e6b1ba8a05027ca065b379d",
    "pooled": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "cap8": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "cap10": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "cap11": "0330dc5a4f85bc68c4da6f98af2f4208335e65c644ddedd5d8cc487368091026",
    "v1": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "v2": "adbfaf91ed2889cc42da85255edf9f5074f1002e9ad93dc1d4ff706de66d1144",
    "v8": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "v10": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "v11": "f75b17669a5fc5299d21f5b44e91530c03c71d75181683c7b6728cb95c862450",
    "fitlib": "bf806b077da4e2fb43612f8f3b5ca318e0d45edac6b2c26ecc2f6accd218b413",
    "evaluator": "561b40b093e0a46469fa95b01c23e1b0d7d294201aaccb63b918b86900359303",
    "unit_lib": "db187788e35c136bbb081a3aa25232745b8952f6d8df0594e690a5da45def2c8",
}
STEPS = 40
CHECKPOINTS = (0, 10, 20, 30, 40)
LR = 0.025
NOISE_SIGMA = 0.03
FEASIBILITY_SLACK = 0.02
BARRIER = 100.0
TAU = 0.10
TARGET_TERMS = ("margin_match", "l15_match")
SECONDARY_TERMS = ("margin_inert", "l15_inert", "vocab_match", "vocab_inert")
CONDITIONS = ("deterministic", "noise")
STARTS = ("dim", "pooled")
PRICE = {
    "model_forwards_max": 2500,
    "transformer_backward_forwards": 1920,
    "model_updates": 160,
    "fit_updates": 160,
    "transformer_backwards": 160,
    "fit_parameters_per_restart": 128,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def unit(raw):
    return raw / raw.norm().clamp_min(1e-30)


def rows_for(builder, capability):
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    return [row for row in builder.build_rows()
            if row["transform_id"] in ("A1", "A2") and row["row_id"] in allowed]


def pieces_for_axis(backend, contexts, axis, *, grad=False):
    projection = axis @ axis.T
    return [fitlib.environment_loss(backend, context, projection, grad=grad)[1]
            for context in contexts]


def limits_for(backend, contexts, dim):
    with backend.torch.no_grad():
        pieces = pieces_for_axis(backend, contexts, dim)
    return [{term: float(item[term]) + FEASIBILITY_SLACK for term in TARGET_TERMS}
            for item in pieces]


def summarize_pieces(pieces, limits):
    records = []
    for index, (item, bounds) in enumerate(zip(pieces, limits)):
        target = {term: float(item[term]) for term in TARGET_TERMS}
        secondary_parts = {term: float(item[term]) for term in SECONDARY_TERMS}
        violations = {term: max(0.0, target[term] - bounds[term]) for term in TARGET_TERMS}
        records.append({
            "environment": index,
            "target": target,
            "limits": bounds,
            "violations": violations,
            "secondary_parts": secondary_parts,
            "secondary": sum(secondary_parts.values()),
        })
    return {
        "feasible": all(value <= 1e-7 for record in records for value in record["violations"].values()),
        "max_violation": max(value for record in records for value in record["violations"].values()),
        "secondary_mean": sum(record["secondary"] for record in records) / len(records),
        "secondary_worst": max(record["secondary"] for record in records),
        "environments": records,
    }


def evaluate_axis(backend, contexts, limits, axis):
    with backend.torch.no_grad():
        return summarize_pieces(pieces_for_axis(backend, contexts, axis), limits)


def noisy_axes(torch, axis, generator, enabled):
    if not enabled:
        return (axis,)
    noise = torch.randn(axis.shape, generator=generator).to(axis.device)
    tangent = noise - axis * (axis.T @ noise)
    tangent = tangent / tangent.norm().clamp_min(1e-30)
    return tuple(unit(axis + sign * NOISE_SIGMA * tangent) for sign in (-1.0, 1.0))


def training_objective(backend, contexts, limits, raw, *, noise, generator):
    torch = backend.torch
    axes = noisy_axes(torch, unit(raw), generator, noise)
    secondary_values = []
    violation_values = []
    for context, bounds in zip(contexts, limits):
        variant_secondary = []
        variant_violation = []
        for axis in axes:
            pieces = fitlib.environment_loss(backend, context, axis @ axis.T, grad=True)[1]
            variant_secondary.append(sum(pieces[term] for term in SECONDARY_TERMS))
            variant_violation.append(sum(torch.relu(pieces[term] - bounds[term]).square()
                                         for term in TARGET_TERMS))
        secondary_values.append(sum(variant_secondary) / len(variant_secondary))
        violation_values.append(sum(variant_violation) / len(variant_violation))
    secondary_values = torch.stack(secondary_values)
    smooth_worst = TAU * torch.logsumexp(secondary_values / TAU, dim=0) - TAU * math.log(len(contexts))
    return smooth_worst + BARRIER * torch.stack(violation_values).sum()


def fit_one(backend, train, train_limits, selection, selection_limits, initial, condition, start_name):
    torch = backend.torch
    generator = torch.Generator(device="cpu").manual_seed(
        20260907 + sum(map(ord, condition + start_name)))
    raw = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)
    trace = []
    candidates = []

    def checkpoint(step):
        axis = unit(raw).detach().clone()
        report = evaluate_axis(backend, selection, selection_limits, axis)
        summary = {key: report[key] for key in (
            "feasible", "max_violation", "secondary_mean", "secondary_worst")}
        summary["step"] = step
        trace.append(summary)
        candidates.append({"axis": axis, "report": report, "step": step})

    checkpoint(0)
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = training_objective(
            backend, train, train_limits, raw,
            noise=condition == "noise", generator=generator)
        loss.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            checkpoint(step)
    feasible = [candidate for candidate in candidates if candidate["report"]["feasible"]]
    ordering = lambda candidate: (
        candidate["report"]["secondary_worst"],
        candidate["report"]["secondary_mean"], candidate["step"])
    best = min(feasible, key=ordering) if feasible else min(
        candidates, key=lambda candidate: (
            candidate["report"]["max_violation"],) + ordering(candidate))
    return {
        "condition": condition,
        "start": start_name,
        "trace": trace,
        "candidates": candidates,
        "best": best,
    }


def strip_fit(fit):
    return {
        "condition": fit["condition"],
        "start": fit["start"],
        "trace": fit["trace"],
        "best": {
            "step": fit["best"]["step"],
            "axis_sha256": tensor_sha(fit["best"]["axis"]),
            **{key: fit["best"]["report"][key] for key in (
                "feasible", "max_violation", "secondary_mean", "secondary_worst")},
        },
    }


def main():
    paths = {
        "prior": PRIOR, "redteam": REDTEAM, "previous": PREVIOUS,
        "regularized": REGULARIZED, "aligned": ALIGNED, "pooled": POOLED,
        "cap8": CAP8, "cap10": CAP10, "cap11": CAP11,
        "v1": V1, "v2": V2, "v8": V8, "v10": V10, "v11": V11,
        "fitlib": FITLIB, "evaluator": EVALUATOR, "unit_lib": UNIT_LIB,
    }
    if {key: sha(path) for key, path in paths.items()} != EXPECTED:
        raise RuntimeError("hard-feasible DAS authority changed")
    prior, previous, regularized, aligned, pooled, cap8, cap10, cap11 = [
        json.loads(path.read_text()) for path in (
            PRIOR, PREVIOUS, REGULARIZED, ALIGNED, POOLED, CAP8, CAP10, CAP11)]
    train_rows = {
        (bank, panel): [row for row in builder.build_rows()
                        if row["transform_id"] == panel][0::2]
        for bank, builder in (("v1", fit_v1), ("v2", fit_v2))
        for panel in ("A1", "A2")
    }
    rows8 = rows_for(select_v8, cap8)
    rows10 = rows_for(test_v10, cap10)
    rows11 = rows_for(test_v11, cap11)
    if (prior.get("candidate_id") != CANDIDATE_ID
            or previous.get("terminal") != "family_selector_rejects_update"
            or any(cap.get("terminal") != "manifest" for cap in (cap8, cap10, cap11))
            or any(len(rows) != 16 for rows in train_rows.values())
            or [len(rows8), len(rows10), len(rows11)] != [60, 63, 63]):
        raise RuntimeError("hard-feasible DAS population or terminal changed")
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rank": 1,
        "gradient_environments": [f"{bank}_{panel}_even" for bank, panel in train_rows],
        "selection_environments": ["v8_A1_complete", "v8_A2_complete"],
        "sealed_environments": ["v10_A1", "v10_A2", "v11_A1", "v11_A2"],
        "conditions": list(CONDITIONS), "starts": list(STARTS),
        "steps": STEPS, "checkpoints": list(CHECKPOINTS),
        "feasibility_slack": FEASIBILITY_SLACK, "barrier": BARRIER,
        "noise_sigma": NOISE_SIGMA, "target_terms": list(TARGET_TERMS),
        "secondary_terms": list(SECONDARY_TERMS), **PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    pooled_axis = unit(torch.tensor(
        pooled["axis_artifacts"]["pooled_aligned_rank1"], device=backend.device).float().unsqueeze(1))
    prep = g.prepare(backend, train_rows[("v1", "A1")])
    dim = unit(g.diff_in_means_direction(backend, prep, ("attn:11:head:03",)))
    kl = unit(torch.tensor(
        regularized["fits"]["kl"]["coordinates"], device=backend.device).float().unsqueeze(1))
    aligned_fit = next(item for item in aligned["fits"] if float(item["weight"]) == 0.3)
    aligned_axis = unit(torch.tensor(
        aligned_fit["coordinates"], device=backend.device).float().unsqueeze(1))

    train = [fitlib.attach_targets(backend, rows) for rows in train_rows.values()]
    selection = [fitlib.attach_targets(
        backend, [row for row in rows8 if row["transform_id"] == panel])
        for panel in ("A1", "A2")]
    sealed_contexts = {
        f"{bank}_{panel}": fitlib.attach_targets(
            backend, [row for row in rows if row["transform_id"] == panel])
        for bank, rows in (("v10", rows10), ("v11", rows11))
        for panel in ("A1", "A2")
    }
    closure = max(context["manual_base_margin_max_abs"]
                  for context in train + selection + list(sealed_contexts.values()))
    train_limits = limits_for(backend, train, dim)
    selection_limits = limits_for(backend, selection, dim)
    sealed_limits = {
        name: limits_for(backend, [context], dim)[0]
        for name, context in sealed_contexts.items()
    }

    frozen_axes = {
        "dim": dim, "pooled": pooled_axis, "weighted_kl": kl,
        "aligned_0.3": aligned_axis,
    }
    frozen_reports = {
        name: evaluate_axis(backend, selection, selection_limits, axis)
        for name, axis in frozen_axes.items()
    }
    fits = [
        fit_one(backend, train, train_limits, selection, selection_limits,
                frozen_axes[start_name], condition, start_name)
        for condition in CONDITIONS for start_name in STARTS
    ]
    eligible = [
        {"source": name, "kind": "frozen", "step": 0,
         "axis": frozen_axes[name], "report": report}
        for name, report in frozen_reports.items() if report["feasible"]
    ]
    for fit in fits:
        eligible.extend(
            {"source": f"{fit['condition']}_{fit['start']}", "kind": "learned",
             "step": candidate["step"], "axis": candidate["axis"],
             "report": candidate["report"]}
            for candidate in fit["candidates"] if candidate["report"]["feasible"]
        )
    source_order = {name: index for index, name in enumerate(
        ("dim", "pooled", "weighted_kl", "aligned_0.3",
         "deterministic_dim", "deterministic_pooled", "noise_dim", "noise_pooled"))}
    selected = min(eligible, key=lambda item: (
        item["report"]["secondary_worst"], item["report"]["secondary_mean"],
        source_order[item["source"]], item["step"]))

    sealed = {}
    for method, axis in (("selected", selected["axis"]), ("dim", dim)):
        cells = {}
        for name, context in sealed_contexts.items():
            cells[name] = evaluate_axis(backend, [context], [sealed_limits[name]], axis)
        sealed[method] = {
            "feasible": all(cell["feasible"] for cell in cells.values()),
            "secondary_mean": sum(cell["secondary_mean"] for cell in cells.values()) / len(cells),
            "secondary_worst": max(cell["secondary_worst"] for cell in cells.values()),
            "cells": cells,
        }

    rich_evaluations = {}
    max_instrument = 0.0
    for method, axis in (("selected", selected["axis"]), ("dim", dim)):
        rich_evaluations[method] = {}
        for bank, rows in (("v10", rows10), ("v11", rows11)):
            report = evaluator.evaluate_bank(backend, bank, rows, axis, axis @ axis.T)
            max_instrument = max(max_instrument, *report["instrument"].values())
            rich_evaluations[method][bank] = {
                key: value for key, value in report.items()
                if key not in ("records", "forwards", "evaluations")}

    learned_feasible = [item for item in eligible
                        if item["kind"] == "learned" and item["step"] > 0]
    dim_selection = frozen_reports["dim"]
    selected_is_learned = selected["kind"] == "learned" and selected["step"] > 0
    pred_b = bool(learned_feasible)
    pred_c = bool(
        selected_is_learned
        and selected["report"]["secondary_worst"] <= 0.95 * dim_selection["secondary_worst"])
    pred_d = bool(
        sealed["selected"]["feasible"]
        and sealed["selected"]["secondary_mean"] < sealed["dim"]["secondary_mean"]
        and sealed["selected"]["secondary_worst"] < sealed["dim"]["secondary_worst"])
    if selected["kind"] == "frozen":
        stability_cosine = None
        pred_e = True
    else:
        condition = selected["source"].split("_", 1)[0]
        matching = [fit for fit in fits if fit["condition"] == condition]
        stability_cosine = float((matching[0]["best"]["axis"].T
                                  @ matching[1]["best"]["axis"]).abs())
        pred_e = stability_cosine >= 0.80
    numeric = [
        value for fit in fits for point in fit["trace"] for value in point.values()
        if isinstance(value, (int, float))
    ]
    pred_a = bool(
        closure <= 1e-4 and max_instrument <= 1e-4
        and all(math.isfinite(value) for value in numeric)
        and len(fits) == 4 and len(eligible) >= 1)
    predictions = {
        "pred_a_authority_instrument_finiteness_and_price": pred_a,
        "pred_b_nonbaseline_feasible_checkpoint_exists": pred_b,
        "pred_c_feasible_optimization_improves_v8_secondary": pred_c,
        "pred_d_selected_generalizes_to_sealed_families": pred_d,
        "pred_e_regularization_is_stable_not_memorizing": pred_e,
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "hard_feasible_das_candidate"
    elif pred_b and (not pred_c or not pred_d):
        terminal = "family_memorization"
    elif not pred_b and selected["kind"] == "frozen":
        terminal = "dim_frontier"
    else:
        terminal = "partial"
    result = {
        "schema": "temporal_h3_das_hard_feasible_kl_stability_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"manual_margin_max_abs": closure,
                       "sealed_evaluator_max_abs": max_instrument},
        "feasibility": {"slack": FEASIBILITY_SLACK,
                        "train_limits": train_limits,
                        "selection_limits": selection_limits,
                        "sealed_limits": sealed_limits},
        "selection": {
            "selected_source": selected["source"], "selected_kind": selected["kind"],
            "selected_step": selected["step"],
            "selected_axis_sha256": tensor_sha(selected["axis"]),
            "selected_axis": selected["axis"].flatten().cpu().tolist(),
            "selected_report": selected["report"],
            "frozen_reports": frozen_reports,
            "learned_feasible_checkpoint_count": len(learned_feasible),
            "winning_condition_restart_cosine": stability_cosine,
            "fits": [strip_fit(fit) for fit in fits],
        },
        "sealed": sealed, "rich_evaluations": rich_evaluations,
        "predictions": predictions, "terminal": terminal, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "instrument": result["instrument"],
        "selection": {key: result["selection"][key] for key in (
            "selected_source", "selected_kind", "selected_step",
            "learned_feasible_checkpoint_count", "winning_condition_restart_cosine")},
        "sealed": sealed, "predictions": predictions,
        "terminal": terminal, "price": PRICE,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
