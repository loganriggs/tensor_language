#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;384 executions;180seconds;M4 most-additive mode split.
"""Select a natural-only most-additive partition of the rank-256 M4 mode node."""
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
import rung498_copy_task_portability_diagnosis as diagnosis
import run_equality_l5h5_m4_contracted_mode_graph_v1 as mode_parent
import run_equality_l5h5_m4_contracted_mode_graph_v2 as precision_parent
import run_equality_l5h5_l234_module_write_graph_v1 as module_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_MOST_ADDITIVE_MODE_SPLIT_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V2_RESULT.json"
DOCUMENTS = 192
SELECTED_RANK = 256
CUTS = (8, 16, 32, 64, 128)
FIVE_MASK = mode_parent.FIVE_MASK


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
    lawful_parent_key = "pred_" + "a_lawful_contracted_mode_bridge"
    required = (lawful_parent_key, "pred_c_ood_parent_score_prediction", "pred_d_ood_causal_use", "pred_e_equal_norm_directional_selectivity")
    if parent["terminal"] != "valid_equality_l5h5_m4_contracted_mode_graph_null" or parent["selection"]["rank"] != SELECTED_RANK or not all(parent["predictions"][key] for key in required):
        raise ValueError("contracted-mode parent changed")
    if parent["predictions"]["pred_b_compact_natural_mode_support"] or parent["predictions"]["pred_f_two_component_composition"]:
        raise ValueError("parent null authority changed")
    roles, scales, atom_result, module_result, port_parent, manifest, invalid = precision_parent.load_bound()
    return roles, scales, parent, atom_result, port_parent, manifest


def plan():
    roles, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_most_additive_mode_split_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 384, "candidate_cuts": list(CUTS), "complete_code_arms": 6, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def compile_basis(model):
    mlp4 = model.transformer.h[4].mlp
    attention5 = model.transformer.h[5].attn
    qk = torch.cat(residual_parent.head_weights(attention5), dim=0).double()
    contracted = qk @ mlp4.Down.weight.double()
    left, values, basis64 = torch.linalg.svd(contracted, full_matrices=False)
    bridge_error = float(((left * values.unsqueeze(0)) @ basis64 - contracted).norm() / contracted.norm().clamp_min(1e-30))
    basis = basis64.float()
    writers = mlp4.Down.weight.float() @ basis.T
    return basis, writers, bridge_error


def subset_score(groups, product, mlp, basis, writers, indices, scale, attention, dtype):
    selected_basis = basis.index_select(0, indices)
    selected_writers = writers.index_select(1, indices)
    write = mode_parent.mode_write(product, mlp, selected_basis, selected_writers, len(indices), scale)
    return mode_parent.atom_parent.score_with_m4(groups, write, attention, dtype)


def accumulate_effect_norms(row, child, remainder, joint, baseline, mask):
    c = (child - baseline)[mask].double()
    r = (remainder - baseline)[mask].double()
    j = (joint - baseline)[mask].double()
    row["child"] += float(c.square().sum())
    row["remainder"] += float(r.square().sum())
    row["joint"] += float(j.square().sum())


def finish_candidate(pair_stats, norms):
    report = mode_parent.atom_parent.finish_pair(pair_stats)
    joint = max(norms["joint"], 1e-30)
    report["child_effect_norm_ratio"] = math.sqrt(norms["child"] / joint)
    report["remainder_effect_norm_ratio"] = math.sqrt(norms["remainder"] / joint)
    report["eligible"] = bool(report["child_effect_norm_ratio"] >= .50 and report["remainder_effect_norm_ratio"] >= .10)
    return report


def behavior_composition(nll, masks):
    output = {}
    selections = {cell: masks[cell] for cell in ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")}
    selections["half_0"] = torch.zeros_like(masks["copy_positive"]); selections["half_0"][:96] = masks["copy_positive"][:96]
    selections["half_1"] = torch.zeros_like(masks["copy_positive"]); selections["half_1"][96:] = masks["copy_positive"][96:]
    for name, selected in selections.items():
        actual = (nll["joint"] - nll["baseline"])[selected].double()
        predicted = (nll["child"] + nll["remainder"] - 2 * nll["baseline"])[selected].double()
        error = predicted - actual
        output[name] = {"relative_l2": float(error.norm() / actual.norm().clamp_min(1e-30)), "cosine": float((predicted * actual).sum() / (predicted.norm() * actual.norm()).clamp_min(1e-30)), "tokens": int(selected.sum())}
    return output


def recovery(nll, masks, arm):
    selected = masks["copy_positive"]
    native_effect = nll["absent"] - nll["native"]
    arm_effect = nll["absent"] - nll[arm]
    return float(arm_effect[selected].sum() / native_effect[selected].sum())


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, scales, parent, atom_result, port_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = compile_basis(model)
    all_modes = torch.arange(SELECTED_RANK, device="cuda")

    natural_pairs = {cut: torch.zeros(4, dtype=torch.float64) for cut in CUTS}
    natural_norms = {cut: {"child": 0.0, "remainder": 0.0, "joint": 0.0} for cut in CUTS}
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        support = induction.induction_fetch_mask(tokens)
        baseline = subset_score(groups, product, mlp4, basis, writers, all_modes[:0], scale5, attention, residual.dtype)
        joint = subset_score(groups, product, mlp4, basis, writers, all_modes, scale5, attention, residual.dtype)
        for cut in CUTS:
            child = subset_score(groups, product, mlp4, basis, writers, all_modes[:cut], scale5, attention, residual.dtype)
            remainder = subset_score(groups, product, mlp4, basis, writers, all_modes[cut:], scale5, attention, residual.dtype)
            additive_effect = (child - baseline) + (remainder - baseline)
            mode_parent.atom_parent.accumulate_pair(natural_pairs[cut], additive_effect, joint - baseline, support)
            accumulate_effect_norms(natural_norms[cut], child, remainder, joint, baseline, support)
    natural_reports = {cut: finish_candidate(natural_pairs[cut], natural_norms[cut]) for cut in CUTS}
    eligible = [cut for cut in CUTS if natural_reports[cut]["eligible"]]
    pool = eligible if eligible else list(CUTS)
    selected_cut = min(pool, key=lambda cut: (natural_reports[cut]["relative_l2"], cut))
    qualified = bool(eligible)
    child_modes = all_modes[:selected_cut]
    remainder_modes = all_modes[selected_cut:]

    code_pair = torch.zeros(4, dtype=torch.float64)
    code_halves = [torch.zeros(4, dtype=torch.float64) for _ in range(2)]
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["ood_code"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        support = induction.induction_fetch_mask(tokens)
        baseline = subset_score(groups, product, mlp4, basis, writers, all_modes[:0], scale5, attention, residual.dtype)
        child = subset_score(groups, product, mlp4, basis, writers, child_modes, scale5, attention, residual.dtype)
        remainder = subset_score(groups, product, mlp4, basis, writers, remainder_modes, scale5, attention, residual.dtype)
        joint = subset_score(groups, product, mlp4, basis, writers, all_modes, scale5, attention, residual.dtype)
        additive_effect = (child - baseline) + (remainder - baseline)
        mode_parent.atom_parent.accumulate_pair(code_pair, additive_effect, joint - baseline, support)
        mode_parent.atom_parent.accumulate_pair(code_halves[0 if start < 96 else 1], additive_effect, joint - baseline, support)
    code_report = mode_parent.atom_parent.finish_pair(code_pair)
    code_half_reports = [mode_parent.atom_parent.finish_pair(row) for row in code_halves]

    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    names = ("native", "absent", "baseline", "child", "remainder", "joint")
    nll = {name: [] for name in names}
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=mode_parent.PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        baseline_logits, baseline_rms = mode_parent.mode_donor_forward(model, tokens, scales["L5H5"], basis[:0], writers[:, :0], 0)
        child_logits, child_rms = mode_parent.mode_donor_forward(model, tokens, scales["L5H5"], basis[:selected_cut], writers[:, :selected_cut], selected_cut)
        remainder_logits, remainder_rms = mode_parent.mode_donor_forward(model, tokens, scales["L5H5"], basis[selected_cut:SELECTED_RANK], writers[:, selected_cut:SELECTED_RANK], SELECTED_RANK - selected_cut)
        joint_logits, joint_rms = mode_parent.mode_donor_forward(model, tokens, scales["L5H5"], basis, writers, SELECTED_RANK)
        minimum_term_rms = min(minimum_term_rms, baseline_rms, child_rms, remainder_rms, joint_rms)
        targets = rows[:, 1:].cuda()
        for name, logits in zip(names, (native_logits, absent_logits, baseline_logits, child_logits, remainder_logits, joint_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    behavioral_composition = behavior_composition(nll, masks)
    recoveries = {arm: recovery(nll, masks, arm) for arm in ("baseline", "child", "remainder", "joint")}
    noncopy_damage = float((nll["joint"] - nll["native"])[masks["all_noncopy"]].mean())

    pred_a = bool(bridge_error <= 2e-5 and parent["instrument"]["contracted_operator_reconstruction_relative_l2"] <= 2e-5)
    pred_b = bool(qualified and natural_reports[selected_cut]["relative_l2"] <= .10)
    pred_c = bool(code_report["relative_l2"] <= .10 and all(row["relative_l2"] <= .15 for row in code_half_reports))
    subtype_names = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_d = bool(behavioral_composition["copy_positive"]["relative_l2"] <= .10 and all(behavioral_composition[name]["relative_l2"] <= .15 for name in subtype_names) and behavioral_composition["half_0"]["relative_l2"] <= .15 and behavioral_composition["half_1"]["relative_l2"] <= .15)
    pred_e = bool(.80 <= recoveries["joint"] <= 1.05 and abs(noncopy_damage) <= .01 and minimum_term_rms > 0 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_inherited_mode_bridge": pred_a, "pred_b_natural_most_additive_split": pred_b, "pred_c_ood_score_composition": pred_c, "pred_d_ood_behavioral_composition": pred_d, "pred_e_joint_causal_selective_use": pred_e}
    terminal = "equality_l5h5_m4_most_additive_mode_split_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_most_additive_split_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_most_additive_mode_split_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"qualified": qualified, "child_rank": selected_cut, "remainder_rank": SELECTED_RANK - selected_cut, "natural": natural_reports[selected_cut]}, "natural_partition_frontier": {str(cut): report for cut, report in natural_reports.items()}, "code_score_composition": {"overall": code_report, "halves": code_half_reports}, "behavioral_composition": behavioral_composition, "behavior": {"recoveries": recoveries, "joint_noncopy_mean_damage_nat": noncopy_damage, "minimum_donor_term_rms": minimum_term_rms}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "parent_alternating_score_composition_error": parent["composition"]["additive_effect_prediction"]["relative_l2"]}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Natural-only most-additive nested partition of the float64-certified rank-256 M4 contracted-mode node, validated at score and behavioral NLL interfaces."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_cut": selected_cut, "natural": natural_reports[selected_cut], "code": code_report, "behavioral": behavioral_composition, "recoveries": recoveries, "noncopy_damage": noncopy_damage, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
