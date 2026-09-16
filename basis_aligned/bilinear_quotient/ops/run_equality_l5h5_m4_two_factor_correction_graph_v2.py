#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;312 forwards;180seconds;M4 two-QK graph plus native arithmetic residual.
"""Precision-correct the two-factor graph with an explicit arithmetic node."""
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
import run_equality_l5h5_m4_two_factor_correction_graph_v1 as v1
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V2"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
V1_RESULT = P / "EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V1_RESULT.json"
DOCUMENTS = v1.DOCUMENTS
CELLS = v1.CELLS
NODES = ("first", "second", "cross", "arithmetic")
VARIANTS = ("full", "base", "drop_first", "drop_second", "drop_cross", "drop_arithmetic", "rolled_arithmetic")
v1.NODES = NODES
v1.VARIANTS = VARIANTS


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
    invalid = json.loads(V1_RESULT.read_text())
    if invalid["terminal"] != "invalid" or invalid["predictions"]["pred_a_exact_two_factor_graph"] or invalid["score_census"]["code"]["closure_relative_l2"] <= .1:
        raise ValueError("V1 precision failure changed")
    roles, scales, parent, explicit_parent, manifest = v1.load_bound()
    return roles, scales, invalid, parent, explicit_parent, manifest


def plan():
    roles, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_two_factor_correction_graph_v2_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "macro_nodes": list(NODES), "variants": list(VARIANTS), "execution_count": 312, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def factor_scores(residuals, attention):
    ports = {name: v1.selection_parent.native_ports(attention, residuals[name]) for name in ("baseline", "remainder", "joint", "child")}
    denominators = {name: v1.kernel_parent.rms_denominator(residuals[name]) for name in residuals}
    derived = []
    for index in range(4):
        value = ports["joint"][index].float() * (denominators["joint"] / denominators["child"])
        value = value - ports["remainder"][index].float() * (denominators["remainder"] / denominators["child"])
        value = value + ports["baseline"][index].float() * (denominators["baseline"] / denominators["child"])
        derived.append(value.to(residuals["child"].dtype))
    cos, sin = v1.mode_parent.module_parent.parent.rotary_ports(attention, residuals["joint"])
    transform = lambda raw: v1.kernel_parent.rotate(F.rms_norm(raw, (raw.shape[-1],)), cos, sin).float()
    derived = [transform(value) for value in derived]
    native = [transform(value) for value in ports["child"]]
    a0, a1 = v1.dot_score(derived[0], derived[1]), v1.dot_score(native[0], native[1])
    b0, b1 = v1.dot_score(derived[2], derived[3]), v1.dot_score(native[2], native[3])
    nodes = {"base": a0 * b0, "first": (a1 - a0) * b0, "second": a0 * (b1 - b0), "cross": (a1 - a0) * (b1 - b0)}
    causal = torch.ones(nodes["base"].shape[-2:], dtype=torch.bool, device=nodes["base"].device).tril()
    nodes = {name: value.masked_fill(~causal, 0).float() for name, value in nodes.items()}
    native_child = v1.kernel_parent.score_from_raw(ports["child"], cos, sin)
    algebraic = ((nodes["base"] + nodes["first"]) + nodes["second"]) + nodes["cross"]
    nodes["arithmetic"] = native_child - algebraic
    child = {
        "full": algebraic + nodes["arithmetic"],
        "base": nodes["base"],
        "drop_first": native_child - nodes["first"],
        "drop_second": native_child - nodes["second"],
        "drop_cross": native_child - nodes["cross"],
        "drop_arithmetic": algebraic,
        "rolled_arithmetic": algebraic + torch.roll(nodes["arithmetic"], shifts=1, dims=1),
    }
    fixed = {name: v1.kernel_parent.score_from_raw(ports[name], cos, sin) for name in ("baseline", "remainder", "joint")}
    additive = {name: (fixed["baseline"] + (value - fixed["baseline"])) + (fixed["remainder"] - fixed["baseline"]) for name, value in child.items()}
    return additive, nodes, child["full"] - native_child, fixed


v1.factor_scores = factor_scores


@torch.no_grad()
def score_census(model, rows, basis, writers):
    node_sq = {name: 0.0 for name in NODES}
    total_sq = 0.0
    closure_sq = 0.0
    rolled_sq = 0.0
    arithmetic_sq = 0.0
    values = 0
    for start in range(0, DOCUMENTS, v1.action_parent.BATCH):
        tokens = rows[start:start + v1.action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = v1.mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        residuals = v1.kernel_parent.mode_residuals(groups, product, mlp4, basis, writers, scale5, residual.dtype)
        _, nodes, closure, _ = factor_scores(residuals, attention)
        support = induction.induction_fetch_mask(tokens)
        total = sum((nodes[name] for name in NODES), torch.zeros_like(nodes["base"]))[support].double()
        total_sq += float(total.square().sum())
        closure_sq += float(closure[support].double().square().sum())
        for name in NODES:
            node_sq[name] += float(nodes[name][support].double().square().sum())
        arithmetic_sq += float(nodes["arithmetic"].double().square().sum())
        rolled_sq += float(torch.roll(nodes["arithmetic"], shifts=1, dims=1).double().square().sum())
        values += total.numel()
    denominator = max(total_sq, 1e-30)
    return {"node_relative_norm": {name: math.sqrt(value / denominator) for name, value in node_sq.items()}, "closure_relative_l2": math.sqrt(closure_sq / denominator), "rolled_arithmetic_to_arithmetic_full_norm_ratio": math.sqrt(rolled_sq / max(arithmetic_sq, 1e-30)), "values": values}


def reports(nll, masks):
    output = {name: {} for name in ("control",) + NODES + ("rolled_arithmetic",)}
    node_variant = {name: f"drop_{name}" for name in NODES}
    for cell in CELLS:
        selected = masks[cell]
        denominator = (nll["authority_additive"] - nll["authority_direct"])[selected].double()
        control = (nll["full"] - nll["authority_direct"])[selected].double()
        output["control"][cell] = {"relative_l2": float((control - denominator).norm() / denominator.norm().clamp_min(1e-30)), "cosine": float((control * denominator).sum() / (control.norm() * denominator.norm()).clamp_min(1e-30)), "tokens": int(selected.sum())}
        effects = {}
        for name, variant in node_variant.items():
            effect = (nll[variant] - nll["full"])[selected].double()
            effects[name] = effect
            output[name][cell] = {"relative_to_interaction_removal": float(effect.norm() / denominator.norm().clamp_min(1e-30)), "tokens": int(selected.sum())}
        rolled = (nll["rolled_arithmetic"] - nll["full"])[selected].double()
        true = effects["arithmetic"]
        output["rolled_arithmetic"][cell] = {"relative_to_interaction_removal": float(rolled.norm() / denominator.norm().clamp_min(1e-30)), "cosine_to_true_arithmetic_removal": float((rolled * true).sum() / (rolled.norm() * true.norm()).clamp_min(1e-30)), "tokens": int(selected.sum())}
    return output


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, scales, invalid, parent, explicit_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = v1.split_parent.compile_basis(model)
    census = {label: score_census(model, roles[role], basis, writers) for label, role in (("natural", "final_natural"), ("code", "ood_code"))}
    behavior = {}
    noncopy = {}
    minimum_rms = math.inf
    for label, role in (("natural", "final_natural"), ("code", "ood_code")):
        masks = v1.selection_parent.masks_for(roles[role])
        nll, rms = v1.collect(model, roles[role], scales["L5H5"], basis, writers)
        minimum_rms = min(minimum_rms, rms)
        behavior[label] = reports(nll, masks)
        noncopy[label] = {name: float((nll[f"drop_{name}"] - nll["full"])[masks["all_noncopy"]].mean()) for name in NODES}
    pred_a = bool(explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5 and all(census[label]["closure_relative_l2"] <= 2e-6 for label in census) and all(behavior[label]["control"][cell]["relative_l2"] <= 2e-5 and behavior[label]["control"][cell]["cosine"] >= .99999 for label in behavior for cell in CELLS))
    pred_b = bool(all(behavior[label][name]["copy_positive"]["relative_to_interaction_removal"] >= .02 and behavior[label][name]["half_0"]["relative_to_interaction_removal"] > 0 and behavior[label][name]["half_1"]["relative_to_interaction_removal"] > 0 for label in behavior for name in NODES))
    pred_c = bool(all(abs(noncopy[label][name]) <= .01 for label in noncopy for name in NODES))
    pred_d = bool(.99 <= census["code"]["rolled_arithmetic_to_arithmetic_full_norm_ratio"] <= 1.01 and behavior["code"]["rolled_arithmetic"]["copy_positive"]["cosine_to_true_arithmetic_removal"] <= .80)
    pred_e = bool(len(NODES) == 4 and manifest["learned_parameters"] == 0 and minimum_rms > 0)
    predictions = {"pred_a_exact_precision_corrected_factor_graph": pred_a, "pred_b_all_four_nodes_behaviorally_active": pred_b, "pred_c_selective_node_removals": pred_c, "pred_d_arithmetic_node_direction_specificity": pred_d, "pred_e_sparse_graph_price": pred_e}
    terminal = "equality_l5h5_m4_precision_corrected_factor_graph_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_precision_corrected_factor_node_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_two_factor_correction_graph_v2_result", "terminal": terminal, "predictions": predictions, "graph": {"baseline": "derived_child_score", "nodes": ["first_qk_correction_times_second_baseline", "first_baseline_times_second_qk_correction", "algebraic_qk_correction_cross_product", "native_arithmetic_residual"], "native_qk_projections": 16, "learned_parameters": 0}, "score_census": census, "behavior": behavior, "noncopy_incremental_mean_nat": noncopy, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "minimum_donor_term_rms": minimum_rms, "invalid_v1_code_closure_relative_l2": invalid["score_census"]["code"]["closure_relative_l2"]}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Precision-corrected four-node graph of the two multiplicative QK-factor corrections plus the native BF16 arithmetic residual, with OOD node removal and direction-specificity."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "score_census": census, "code_behavior": behavior["code"], "noncopy": noncopy, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
