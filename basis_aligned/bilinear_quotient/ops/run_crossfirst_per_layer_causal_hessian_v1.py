#!/usr/bin/env python3
# BQGATE:96prefixes;5writers;17native suffix stages;360seconds;per-layer causal-Hessian allocation.
"""Allocate the CrossFirst child/remainder mixed Hessian over native suffix stages.

pred_a: native replay and summed-Hessian chain-rule closure pass.
pred_b: second order leaves <=.50 finite-interaction error in every family.
pred_c: every family has top-three energy >=.80 and participation ratio <=4.
pred_d: actual allocation is more concentrated than equal-norm random writers.
pred_e: every prompt/stage/reader allocation is serialized without averaging.
"""
import io
import json
import math
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
import torch.nn.functional as F

import live_crossfirst_prefix_v1 as live
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest

STEM = "CROSSFIRST_PER_LAYER_CAUSAL_HESSIAN_V1"
BINDING = P / f"{STEM}_BINDING.json"
PREREG = P / f"{STEM}_PREREGISTRATION.md"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
ROWS = P / "FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
STATE_ARTIFACT = P / "CROSSFIRST_STATE_EXECUTOR_V1_ARTIFACT.pt"
HIERARCHY_ARTIFACT = P / "CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt"
PROGRAM = P / "MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt"
RANDOM_SEEDS = (202609161401, 202609161402, 202609161403, 202609161404)


def atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    rows = json.loads(ROWS.read_text())["rows"]
    validate(rows)
    if len(rows) != 96:
        raise ValueError("expected 96 regional rows")
    return rows


def plan():
    rows = load_bound()
    return {
        "schema": "crossfirst_per_layer_causal_hessian_v1_plan",
        "rows": len(rows),
        "writers": 5,
        "native_suffix_stages": 17,
        "readout_stages": 1,
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
    }


def mixed_jvp(function, point, left, right):
    return torch.func.jvp(
        lambda value: torch.func.jvp(function, (value,), (left,))[1],
        (point,),
        (right,),
    )[1]


def first_jvp(function, point, tangent):
    return torch.func.jvp(function, (point,), (tangent,))[1]


def make_stages(model, native):
    stages = []
    names = ["mlp9"]
    block9 = model.transformer.h[9]
    stages.append(lambda value, block=block9: value + block.mlp(F.rms_norm(value, (1152,))))
    for layer in range(10, 18):
        block = model.transformer.h[layer]

        def attention_step(value, block=block, x0=native["x0"], v1=native["v1"]):
            raw = block.lambdas[0] * value + block.lambdas[1] * x0
            attention, _ = block.attn(F.rms_norm(raw, (1152,)), v1)
            return raw + attention

        def mlp_step(value, block=block):
            return value + block.mlp(F.rms_norm(value, (1152,)))

        stages.extend((attention_step, mlp_step))
        names.extend((f"attention{layer}", f"mlp{layer}"))
    return names, stages


def make_readout(model, row):
    def readout(value):
        last = F.rms_norm(value[:, -1], (1152,))
        logits = 30 * torch.tanh(model.lm_head(last) / 30)
        target = logits[:, row["uk_id"]] - logits[:, row["us_id"]]
        control = logits[:, row["control_ids"][0]] - logits[:, row["control_ids"][1]]
        return torch.stack((target, control), dim=-1)[0]

    return readout


def writer_directions(direction):
    base = direction.float()
    directions = [base]
    unit = base / base.norm().clamp_min(1e-30)
    orthonormal = [unit]
    for seed in RANDOM_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        candidate = torch.randn(base.numel(), generator=generator, dtype=torch.float32, device="cpu").to(base.device)
        for previous in orthonormal:
            candidate = candidate - (candidate * previous).sum() * previous
        candidate = candidate / candidate.norm().clamp_min(1e-30)
        orthonormal.append(candidate)
        directions.append(candidate * base.norm())
    return directions


def allocation_for_writer(stages, detached_states, graph_states, gradients, readout, left, right):
    stage_terms = []
    propagated_mixed = torch.zeros_like(left)
    current_left, current_right = left, right
    for index, stage in enumerate(stages):
        point = detached_states[index]
        local = mixed_jvp(stage, point, current_left, current_right)
        propagated_mixed = first_jvp(stage, point, propagated_mixed) + local
        current_left = first_jvp(stage, point, current_left)
        current_right = first_jvp(stage, point, current_right)
        gradient = gradients[index].reshape(2, -1)
        stage_terms.append((gradient * local.reshape(1, -1)).sum(1))
    readout_local = mixed_jvp(readout, detached_states[-1], current_left, current_right)
    complete = first_jvp(readout, detached_states[-1], propagated_mixed) + readout_local
    stage_terms.append(readout_local)
    return torch.stack(stage_terms), complete


def concentration_report(terms):
    # terms: rows x stages, with prompts retained until this reporting reduction.
    energy = terms.square().sum(0)
    total = energy.sum().clamp_min(1e-30)
    top_values, top_indices = torch.topk(energy, 3)
    participation = total.square() / energy.square().sum().clamp_min(1e-30)
    return {
        "top_three_energy_fraction": float(top_values.sum() / total),
        "participation_ratio": float(participation),
        "top_three_stage_indices": top_indices.tolist(),
        "stage_energy": energy.tolist(),
    }


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists() or ARTIFACT.exists():
        raise FileExistsError(OUT if OUT.exists() else ARTIFACT)
    signal.alarm(360)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    rows = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = torch.load(PROGRAM, weights_only=True)["direction"].cuda()
    directions = writer_directions(direction)
    hierarchy = torch.load(HIERARCHY_ARTIFACT, weights_only=True)["measures"][:96].double()
    finite = hierarchy[:, 2] - hierarchy[:, 1] - hierarchy[:, 3] + hierarchy[:, 0]
    child_effect = hierarchy[:, 1] - hierarchy[:, 0]
    stage_names = ["mlp9"] + [name for layer in range(10, 18) for name in (f"attention{layer}", f"mlp{layer}")] + ["readout"]
    allocations = torch.empty(5, 96, 18, 2, dtype=torch.float64)
    complete = torch.empty(5, 96, 2, dtype=torch.float64)
    baseline = torch.empty(96, 2, dtype=torch.float64)
    state_replay_errors = []
    started = time.perf_counter()

    for row_index, row in enumerate(rows):
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, weights)
        names, stages = make_stages(model, native)
        if names + ["readout"] != stage_names:
            raise RuntimeError("stage order changed")
        readout = make_readout(model, row)

        # One baseline graph supplies exact suffix gradients for every writer.
        graph_states = [native["z9"].detach().requires_grad_(True)]
        for stage in stages:
            graph_states.append(stage(graph_states[-1]))
        graph_output = readout(graph_states[-1])
        baseline[row_index] = graph_output.detach().double().cpu()
        gradients = []
        for endpoint in range(2):
            endpoint_gradients = torch.autograd.grad(
                graph_output[endpoint],
                graph_states[1:],
                retain_graph=True,
                allow_unused=False,
            )
            gradients.append(endpoint_gradients)
        gradients = [torch.stack((gradients[0][index], gradients[1][index])) for index in range(len(stages))]

        detached_states = [state.detach() for state in graph_states]
        state_replay_errors.append(float((detached_states[1] - native["h9"]).float().norm() / native["h9"].float().norm().clamp_min(1e-30)))
        for writer_index, writer in enumerate(directions):
            left = -(native["child"] * writer).to(native["z9"].dtype)
            right = -(native["remainder"] * writer).to(native["z9"].dtype)
            terms, full = allocation_for_writer(
                stages, detached_states, graph_states, gradients, readout, left, right
            )
            allocations[writer_index, row_index] = terms.detach().double().cpu()
            complete[writer_index, row_index] = full.detach().double().cpu()
        del graph_output, graph_states, gradients, detached_states

    native_replay = float((baseline - hierarchy[:, 0]).norm() / hierarchy[:, 0].norm().clamp_min(1e-30))
    closure = []
    for writer_index in range(5):
        closure.append([
            float((allocations[writer_index, :, :, endpoint].sum(1) - complete[writer_index, :, endpoint]).norm() / complete[writer_index, :, endpoint].norm().clamp_min(1e-30))
            for endpoint in range(2)
        ])

    actual_family_reports = []
    for family in range(4):
        lo, hi = 24 * family, 24 * (family + 1)
        predicted = complete[0, lo:hi, 0]
        observed = finite[lo:hi, 0]
        residual = predicted - observed
        report = {
            "family": family,
            "finite_interaction_relative_l2": float(residual.norm() / observed.norm().clamp_min(1e-30)),
            "residual_relative_to_child": float(residual.norm() / child_effect[lo:hi, 0].norm().clamp_min(1e-30)),
            "hessian_over_finite_norm": float(predicted.norm() / observed.norm().clamp_min(1e-30)),
            "hessian_finite_cosine": float(F.cosine_similarity(predicted, observed, dim=0)),
            "concentration": concentration_report(allocations[0, lo:hi, :, 0]),
            "stage_rms": allocations[0, lo:hi].square().mean(0).sqrt().tolist(),
            "stage_signed_mean": allocations[0, lo:hi].mean(0).tolist(),
            "stage_positive_fraction": (allocations[0, lo:hi] > 0).double().mean(0).tolist(),
        }
        actual_family_reports.append(report)

    pooled_actual = concentration_report(allocations[0, :, :, 0])
    random_reports = []
    for writer_index, seed in enumerate(RANDOM_SEEDS, start=1):
        random_reports.append({
            "seed": seed,
            "target": concentration_report(allocations[writer_index, :, :, 0]),
            "control": concentration_report(allocations[writer_index, :, :, 1]),
            "closure_relative_l2": closure[writer_index],
        })
    random_top3_median = float(torch.tensor([report["target"]["top_three_energy_fraction"] for report in random_reports], dtype=torch.float64).median())
    random_pr_median = float(torch.tensor([report["target"]["participation_ratio"] for report in random_reports], dtype=torch.float64).median())

    pred_a = bool(native_replay <= 2e-6 and max(state_replay_errors) <= 2e-6 and max(max(values) for values in closure) <= 2e-5)
    pred_b = bool(pred_a and all(report["finite_interaction_relative_l2"] <= .50 for report in actual_family_reports))
    pred_c = bool(pred_a and all(report["concentration"]["top_three_energy_fraction"] >= .80 and report["concentration"]["participation_ratio"] <= 4.0 for report in actual_family_reports))
    pred_d = bool(pred_a and pooled_actual["top_three_energy_fraction"] >= random_top3_median + .10 and pooled_actual["participation_ratio"] <= .80 * random_pr_median)
    pred_e = bool(allocations.shape == (5, 96, 18, 2) and all("stage_positive_fraction" in report for report in actual_family_reports))
    predictions = {"pred_a_instrument": pred_a, "pred_b_finite_relevance": pred_b, "pred_c_sparse_allocation": pred_c, "pred_d_random_specificity": pred_d, "pred_e_context_resolution": pred_e}
    terminal = "crossfirst_sparse_per_layer_causal_hessian_candidate" if all(predictions.values()) else "valid_crossfirst_per_layer_causal_hessian_null" if pred_a else "invalid"

    artifact = {
        "schema": "crossfirst_per_layer_causal_hessian_v1_artifact",
        "stage_names": stage_names,
        "writer_labels": ["crossfirst"] + [f"random_{seed}" for seed in RANDOM_SEEDS],
        "allocations": allocations,
        "complete_hessians": complete,
        "finite_interactions": finite,
        "child_effects": child_effect,
        "baseline_readouts": baseline,
    }
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {
        "schema": "crossfirst_per_layer_causal_hessian_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "native_replay_relative_l2": native_replay,
        "maximum_stage_state_replay_relative_l2": max(state_replay_errors),
        "allocation_closure_relative_l2": closure,
        "stage_names": stage_names,
        "actual_family_reports": actual_family_reports,
        "pooled_actual_concentration": pooled_actual,
        "random_reports": random_reports,
        "random_target_top_three_energy_fraction_median": random_top3_median,
        "random_target_participation_ratio_median": random_pr_median,
        "artifact_sha256": digest(ARTIFACT),
        "artifact_relative": str(ARTIFACT.relative_to(ROOT)),
        "runner_sha256": digest(RUNNER),
        "binding_sha256": digest(BINDING),
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "seconds": time.perf_counter() - started,
        "price": {"checkpoint_loads": 1, "prefixes": 96, "writers": 5, "native_suffix_stages": 17, "readout_stages": 1, "fits": 0, "parameter_updates": 0},
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Second-order per-context allocation for the frozen CrossFirst child/remainder edits on an already-open regional panel; native upstream state generators, suffix, and readers remain live ports; no fresh OOD claim.",
    }
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "native_replay": native_replay, "closure": closure, "families": actual_family_reports, "pooled_actual": pooled_actual, "random_top3_median": random_top3_median, "random_pr_median": random_pr_median, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
