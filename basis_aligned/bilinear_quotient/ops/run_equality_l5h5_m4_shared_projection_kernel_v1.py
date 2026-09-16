#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;192 executions;180seconds;M4 shared-projection kernel.
"""Fold the four-score M4 interaction graph through twelve shared Q/K projections."""
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
BQ = ROOT / "basis_aligned/bilinear_quotient"
HERE = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction
import equality_matcher_causal_action_quotient_rung498 as action_parent
import run_equality_l5h5_m4_contracted_mode_graph_v1 as mode_parent
import run_equality_l5h5_m4_explicit_interaction_graph_v1 as graph_parent
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_SHARED_PROJECTION_KERNEL_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_EXPLICIT_INTERACTION_GRAPH_V1_RESULT.json"
DOCUMENTS = 192
CUT = 128
RANK = 256


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    parent = json.loads(PARENT_RESULT.read_text())
    if parent["terminal"] != "equality_l5h5_m4_explicit_interaction_graph_ood" or not all(parent["predictions"].values()):
        raise ValueError("explicit interaction parent changed")
    roles, scales, split_result, contracted_parent, port_parent, manifest = graph_parent.load_bound()
    return roles, parent, manifest


def plan():
    roles, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_shared_projection_kernel_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 192, "authority_qk_projections": 16, "shared_qk_projections": 12, "complete_behavior_forwards": 0, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def mode_residuals(groups, product, mlp, basis, writers, scale, dtype):
    modes = torch.arange(RANK, device=product.device)
    output = {}
    for name, indices in (("baseline", modes[:0]), ("child", modes[:CUT]), ("remainder", modes[CUT:]), ("joint", modes)):
        selected_basis = basis.index_select(0, indices)
        selected_writers = writers.index_select(1, indices)
        write = mode_parent.mode_write(product, mlp, selected_basis, selected_writers, len(indices), scale)
        changed = dict(groups)
        changed["M4"] = write
        output[name] = mode_parent.module_parent.merge_fine(changed, mode_parent.FIVE_MASK, dtype)
    return output


def rotate(value, cos, sin):
    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    return torch.cat((first * cos + second * sin, first * (-sin) + second * cos), dim=-1).type_as(value)


def score_from_raw(raw_ports, cos, sin):
    ports = [rotate(F.rms_norm(raw, (raw.shape[-1],)), cos, sin) for raw in raw_ports]
    q1, k1, q2, k2 = ports
    first = torch.einsum("bqd,bkd->bqk", q1, k1) / q1.shape[-1]
    second = torch.einsum("bqd,bkd->bqk", q2, k2) / q2.shape[-1]
    score = first * second
    causal = torch.ones(score.shape[-2:], dtype=torch.bool, device=score.device).tril()
    return score.masked_fill(~causal, 0).float()


def rms_denominator(residual):
    epsilon = torch.finfo(residual.dtype).eps
    return (residual.float().square().mean(dim=-1, keepdim=True) + epsilon).sqrt()


def shared_scores(residuals, weights, cos, sin):
    baseline = residuals["baseline"]
    child_delta = residuals["child"].float() - baseline.float()
    remainder_delta = residuals["remainder"].float() - baseline.float()
    projected = []
    for weight in weights:
        weight32 = weight.float()
        projected.append((F.linear(baseline.float(), weight32), F.linear(child_delta, weight32), F.linear(remainder_delta, weight32)))
    scores = {}
    combinations = {"baseline": (0, 0), "child": (1, 0), "remainder": (0, 1), "joint": (1, 1)}
    for name, (use_child, use_remainder) in combinations.items():
        denominator = rms_denominator(residuals[name])
        raw_ports = []
        for base_raw, child_raw, remainder_raw in projected:
            numerator = base_raw + use_child * child_raw + use_remainder * remainder_raw
            raw_ports.append((numerator / denominator).to(baseline.dtype))
        scores[name] = score_from_raw(raw_ports, cos, sin)
    scores["interaction"] = scores["joint"] - scores["child"] - scores["remainder"] + scores["baseline"]
    scores["composed"] = ((scores["baseline"] + (scores["child"] - scores["baseline"])) + (scores["remainder"] - scores["baseline"])) + scores["interaction"]
    return scores


def authority_scores(residuals, attention):
    scores = {name: mode_parent.module_parent.score_residual(residual, attention) for name, residual in residuals.items()}
    scores["interaction"] = scores["joint"] - scores["child"] - scores["remainder"] + scores["baseline"]
    scores["composed"] = ((scores["baseline"] + (scores["child"] - scores["baseline"])) + (scores["remainder"] - scores["baseline"])) + scores["interaction"]
    return scores


def empty_stats():
    return {key: torch.zeros(4, dtype=torch.float64) for key in ("baseline", "joint", "interaction", "closure")}


def accumulate(stats, shared, authority, support):
    for key in ("baseline", "joint", "interaction"):
        mode_parent.atom_parent.accumulate_pair(stats[key], shared[key], authority[key], support)
    mode_parent.atom_parent.accumulate_pair(stats["closure"], shared["composed"], shared["joint"], support)


def finish(stats):
    return {key: mode_parent.atom_parent.finish_pair(value) for key, value in stats.items()}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = split_parent.compile_basis(model)
    stats = {"natural": empty_stats(), "code": empty_stats(), "half_0": empty_stats(), "half_1": empty_stats()}
    for role, label in (("final_natural", "natural"), ("ood_code", "code")):
        for start in range(0, DOCUMENTS, action_parent.BATCH):
            tokens = roles[role][start:start + action_parent.BATCH, :-1].cuda()
            residual, groups, attention, product, mlp4, scale5 = mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
            residuals = mode_residuals(groups, product, mlp4, basis, writers, scale5, residual.dtype)
            weights = residual_parent.head_weights(attention)
            cos, sin = residual_parent.rotary_ports(attention, residual)
            shared = shared_scores(residuals, weights, cos, sin)
            authority = authority_scores(residuals, attention)
            support = induction.induction_fetch_mask(tokens)
            accumulate(stats[label], shared, authority, support)
            if label == "code":
                accumulate(stats["half_0" if start < 96 else "half_1"], shared, authority, support)
    reports = {name: finish(value) for name, value in stats.items()}
    pred_a = bool(parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5)
    pred_b = bool(reports["natural"]["baseline"]["relative_l2"] <= .01 and reports["natural"]["baseline"]["cosine"] >= .999 and reports["code"]["baseline"]["relative_l2"] <= .01 and reports["code"]["baseline"]["cosine"] >= .999)
    pred_c = bool(reports["natural"]["joint"]["relative_l2"] <= .01 and reports["natural"]["joint"]["cosine"] >= .999 and reports["code"]["joint"]["relative_l2"] <= .01 and reports["code"]["joint"]["cosine"] >= .999 and reports["half_0"]["joint"]["relative_l2"] <= .015 and reports["half_1"]["joint"]["relative_l2"] <= .015)
    pred_d = bool(reports["natural"]["interaction"]["relative_l2"] <= .10 and reports["natural"]["interaction"]["cosine"] >= .95 and reports["code"]["interaction"]["relative_l2"] <= .10 and reports["code"]["interaction"]["cosine"] >= .95 and reports["half_0"]["interaction"]["relative_l2"] <= .15 and reports["half_1"]["interaction"]["relative_l2"] <= .15)
    pred_e = bool(all(reports[name]["closure"]["relative_l2"] <= 2e-6 for name in reports) and planned["shared_qk_projections"] == 12 and planned["authority_qk_projections"] == 16 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_parent_and_bridge": pred_a, "pred_b_valid_shared_baseline": pred_b, "pred_c_shared_joint_fidelity": pred_c, "pred_d_shared_interaction_fidelity": pred_d, "pred_e_shared_kernel_composition_and_price": pred_e}
    terminal = "equality_l5h5_m4_shared_projection_kernel" if all(predictions.values()) else "valid_equality_l5h5_m4_shared_projection_kernel_null" if pred_a and pred_b else "invalid"
    result = {"schema": "equality_l5h5_m4_shared_projection_kernel_v1_result", "terminal": terminal, "predictions": predictions, "reports": reports, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "shared_projection_components": ["baseline", "child_delta", "remainder_delta"], "omitted_joint_rounding_component": True}, "program_price": {"authority_qk_projections": 16, "shared_qk_projections": 12, "reduction_fraction": .25, "score_cells": 4, "learned_parameters": 0}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Twelve-projection shared residual-to-Q/K kernel for the explicit M4 child/remainder interaction graph. Exact residual RMS denominators remain open context scalars."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "reports": reports, "bridge_error": bridge_error, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
