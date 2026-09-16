#!/usr/bin/env python3
# BQGATE:48prefixes;5writers;31lambda local MLP9 scans;120seconds;no suffix or readout.
"""Discover a balanced most-additive CrossFirst split at the MLP9 consumer."""
import io
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
HERE = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(HERE), str(P), str(ROOT)]

import torch

import live_crossfirst_prefix_v1 as live
import run_crossfirst_per_layer_causal_hessian_v1 as allocation
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest

STEM = "CROSSFIRST_MOST_ADDITIVE_SPLIT_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
ROWS = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ROWS.json"
PROGRAM = P / "MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt"
LAMBDAS = torch.arange(31, dtype=torch.float64) / 40
RANDOM_SEEDS = (2026091611, 2026091612, 2026091613, 2026091614)


def atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def ratio(numerator, denominator):
    return float(numerator.norm() / denominator.norm().clamp_min(1e-30))


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    rows = json.loads(ROWS.read_text())["rows"]
    validate(rows)
    if len(rows) != 48:
        raise ValueError("expected 48 rows")
    return rows


def plan():
    rows = load_bound()
    return {"schema": "crossfirst_most_additive_split_discovery_v1_plan", "rows": len(rows), "writers": 5, "lambdas": len(LAMBDAS), "local_stage": "mlp9", "suffix_calls": 0, "readout_calls": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def writer_directions(direction):
    unit = direction / direction.norm().clamp_min(1e-30)
    orthonormal = [unit]
    directions = [direction]
    for seed in RANDOM_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        candidate = torch.randn(direction.numel(), generator=generator, dtype=torch.float32).to(direction.device)
        for previous in orthonormal:
            candidate -= (candidate * previous).sum() * previous
        candidate /= candidate.norm().clamp_min(1e-30)
        orthonormal.append(candidate)
        directions.append(candidate * direction.norm())
    return directions


def aggregate(row_norms, rows, families):
    indices = torch.tensor([index for index, row in enumerate(rows) if row["family"] in families], dtype=torch.long)
    squared = row_norms[indices].square().sum(0)
    child, remainder, total, interaction = squared.unbind(-1)
    child, remainder, total, interaction = (value.sqrt() for value in (child, remainder, total, interaction))
    smaller = torch.minimum(child, remainder).clamp_min(1e-30)
    return {
        "child_over_total": child / total.clamp_min(1e-30),
        "remainder_over_total": remainder / total.clamp_min(1e-30),
        "interaction_over_smaller_single": interaction / smaller,
        "interaction_over_total": interaction / total.clamp_min(1e-30),
    }


def select(row_norms, rows, included_families):
    reports = []
    feasible = torch.ones(len(LAMBDAS), dtype=torch.bool)
    minimax = torch.zeros(len(LAMBDAS), dtype=torch.float64)
    for family in included_families:
        metrics = aggregate(row_norms, rows, {family})
        family_feasible = (metrics["child_over_total"] >= .50) & (metrics["remainder_over_total"] >= .25) & (metrics["child_over_total"] <= 1.25) & (metrics["remainder_over_total"] <= 1.25)
        feasible &= family_feasible
        minimax = torch.maximum(minimax, metrics["interaction_over_smaller_single"])
        reports.append({"family": family, **{key: value.tolist() for key, value in metrics.items()}, "feasible": family_feasible.tolist()})
    eligible = torch.where(feasible, minimax, torch.full_like(minimax, float("inf")))
    selected_index = int(eligible.argmin()) if bool(feasible.any()) else None
    return {"included_families": list(included_families), "feasible": feasible.tolist(), "minimax": minimax.tolist(), "selected_index": selected_index, "selected_lambda": float(LAMBDAS[selected_index]) if selected_index is not None else None, "selected_minimax": float(minimax[selected_index]) if selected_index is not None else None, "family_reports": reports}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists() or ARTIFACT.exists():
        raise FileExistsError(OUT if OUT.exists() else ARTIFACT)
    signal.alarm(120)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    rows = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = torch.load(PROGRAM, weights_only=True)["direction"].cuda().float()
    directions = writer_directions(direction)
    row_norms = torch.empty(5, 48, len(LAMBDAS), 4, dtype=torch.float64)
    partition_closure = []
    state_replay = []
    started = time.perf_counter()

    with torch.no_grad():
        for row_index, row in enumerate(rows):
            ids = torch.tensor([row["ids"]], device="cuda")
            native = live.prepare(model, ids, weights)
            _, stages = allocation.make_stages(model, native)
            stage = stages[0]
            baseline = stage(native["z9"])
            state_replay.append(ratio(baseline - native["h9"], native["h9"]))
            child = native["child"].to(native["z9"].dtype)
            remainder = native["remainder"].to(native["z9"].dtype)
            for writer_index, writer in enumerate(directions):
                total_edit = -((child + remainder) * writer)
                child_edits = [-(child + float(value) * remainder) * writer for value in LAMBDAS]
                remainder_edits = [-(1 - float(value)) * remainder * writer for value in LAMBDAS]
                partition_closure.extend(ratio(child_edits[index] + remainder_edits[index] - total_edit, total_edit) for index in range(len(LAMBDAS)))
                values = torch.cat([native["z9"] + total_edit] + [native["z9"] + edit for edit in child_edits] + [native["z9"] + edit for edit in remainder_edits], dim=0)
                outputs = stage(values)
                joint = outputs[0:1]
                children = outputs[1:1 + len(LAMBDAS)]
                remainders = outputs[1 + len(LAMBDAS):]
                base = baseline.expand_as(children)
                total_effect = joint.expand_as(children) - base
                child_effect = children - base
                remainder_effect = remainders - base
                interaction = joint.expand_as(children) - children - remainders + base
                for lambda_index in range(len(LAMBDAS)):
                    row_norms[writer_index, row_index, lambda_index] = torch.stack([child_effect[lambda_index].double().norm(), remainder_effect[lambda_index].double().norm(), total_effect[lambda_index].double().norm(), interaction[lambda_index].double().norm()]).cpu()

    full = select(row_norms[0], rows, (0, 1, 2, 3))
    leave_one_out = [select(row_norms[0], rows, tuple(family for family in range(4) if family != omitted)) for omitted in range(4)]
    control_selections = [select(row_norms[index], rows, (0, 1, 2, 3)) for index in range(1, 5)]
    selected = full["selected_index"]
    original_metrics = []
    selected_metrics = []
    omitted_feasible = []
    if selected is not None:
        for family in range(4):
            metrics = aggregate(row_norms[0], rows, {family})
            original_metrics.append(float(metrics["interaction_over_smaller_single"][0]))
            selected_metrics.append(float(metrics["interaction_over_smaller_single"][selected]))
            omitted_feasible.append(bool(.50 <= metrics["child_over_total"][selected] <= 1.25 and .25 <= metrics["remainder_over_total"][selected] <= 1.25))
    original_minimax = max(original_metrics) if original_metrics else None
    selected_minimax = max(selected_metrics) if selected_metrics else None
    improvement = 1 - selected_minimax / original_minimax if selected is not None else None
    control_minimax = [report["selected_minimax"] for report in control_selections if report["selected_minimax"] is not None]
    control_median = float(torch.tensor(control_minimax).median()) if control_minimax else None

    pred_a = bool(max(state_replay) <= 2e-6 and max(partition_closure) <= 2e-6)
    pred_b = bool(pred_a and selected is not None)
    pred_c = bool(pred_b and improvement >= .30 and selected_minimax <= .10)
    pred_d = bool(pred_b and all(report["selected_lambda"] is not None and abs(report["selected_lambda"] - full["selected_lambda"]) <= .05 for report in leave_one_out) and all(omitted_feasible))
    pred_e = bool(pred_b and control_median is not None and selected_minimax + .02 <= control_median)
    pred_f = bool(row_norms.shape == (5, 48, 31, 4))
    predictions = {"pred_a_instrument": pred_a, "pred_b_nondegenerate_candidate": pred_b, "pred_c_local_additivity": pred_c, "pred_d_leave_family_stability": pred_d, "pred_e_direction_specificity": pred_e, "pred_f_complete_audit": pred_f}
    terminal = "crossfirst_most_additive_split_candidate" if all(predictions.values()) else "valid_crossfirst_most_additive_split_discovery_null" if pred_a else "invalid"

    artifact = {"schema": "crossfirst_most_additive_split_discovery_v1_artifact", "lambdas": LAMBDAS, "writer_labels": ["crossfirst", "random_1", "random_2", "random_3", "random_4"], "row_norm_labels": ["child_effect", "remainder_effect", "total_effect", "finite_interaction"], "row_norms": row_norms, "full_selection": full, "leave_one_family_out": leave_one_out, "control_selections": control_selections}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {"schema": "crossfirst_most_additive_split_discovery_v1_result", "terminal": terminal, "predictions": predictions, "selected_lambda": full["selected_lambda"], "selected_minimax_interaction_over_smaller_single": selected_minimax, "original_minimax_interaction_over_smaller_single": original_minimax, "fractional_improvement": improvement, "selected_family_interaction_over_smaller_single": selected_metrics, "original_family_interaction_over_smaller_single": original_metrics, "leave_one_family_out_selected_lambdas": [report["selected_lambda"] for report in leave_one_out], "candidate_feasible_on_each_family": omitted_feasible, "random_writer_selected_minimax": control_minimax, "random_writer_median_selected_minimax": control_median, "maximum_partition_closure_relative_l2": max(partition_closure), "maximum_state_replay_relative_l2": max(state_replay), "full_selection": full, "artifact_relative": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": digest(ARTIFACT), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": {"checkpoint_loads": 1, "opened_prefixes": 48, "writer_directions": 5, "lambdas": 31, "local_mlp9_stage_batches": 240, "suffix_calls": 0, "readout_calls": 0, "fits": 0, "gradients": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Opened-panel selection of one global child/remainder reallocation scalar using only exact finite interaction at local MLP9; native child and remainder generators remain ports; no suffix outcome, OOD, extraction, or removal claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_lambda": full["selected_lambda"], "original_minimax": original_minimax, "selected_minimax": selected_minimax, "improvement": improvement, "leave_one_out": result["leave_one_family_out_selected_lambdas"], "control_minimax": control_minimax, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
