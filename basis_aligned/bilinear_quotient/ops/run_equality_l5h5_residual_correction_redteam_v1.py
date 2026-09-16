#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;336 executions;180seconds;correction-port red-team.
"""Test whether the L5H5 residual roundoff port is semantically necessary."""
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
import run_equality_l5h5_residual_source_graph_v1 as parent
import run_equality_reusable_score_port_code_ood_v1 as score_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_RESIDUAL_CORRECTION_REDTEAM_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_RESIDUAL_SOURCE_GRAPH_V1_RESULT.json"
DOCUMENTS = 192
PAIR = action_parent.PAIRS[0]


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
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt["terminal"] != "equality_l5h5_sparse_residual_source_graph_ood" or not all(receipt["predictions"].values()) or receipt["selection"]["sources"] != ["L2", "L3", "L4"]:
        raise ValueError("residual-source parent changed")
    roles, scales, _, port_parent, manifest = parent.load_bound()
    return roles, scales, receipt, port_parent, manifest


def plan():
    roles, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_residual_correction_redteam_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 336, "complete_code_forwards": 240, "prefix_executions": 96, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def transformed_forward(model, tokens, scale, selected_mask, transform):
    original = parent.residual_node.merge_sources
    parent.residual_node.merge_sources = lambda sources, correction: original(sources, transform(correction))
    try:
        return parent.residual_donor_forward(model, tokens, scale, selected_mask)
    finally:
        parent.residual_node.merge_sources = original


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, scales, parent_result, port_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()

    natural_stats = parent.empty_geometry()
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, correction, attention = parent.residual_parts(model, tokens)
        zero = torch.zeros_like(correction)
        scores = parent.subset_scores(groups, zero, attention, range(1, parent.FULL_MASK + 1), residual.dtype)
        target = parent.subset_scores(groups, correction, attention, (parent.FULL_MASK,), residual.dtype)[parent.FULL_MASK]
        parent.accumulate_geometry(natural_stats, scores, target, induction.induction_fetch_mask(tokens))
    natural_reports = {mask: parent.finish_geometry(row) for mask, row in natural_stats.items()}
    selected_mask, qualified = parent.select_support(natural_reports)
    selected_names = parent.source_names(selected_mask)

    code_stats = {selected_mask: torch.zeros(4, dtype=torch.float64)}
    rolled_stats = {selected_mask: torch.zeros(4, dtype=torch.float64)}
    code_half_stats = [{selected_mask: torch.zeros(4, dtype=torch.float64)} for _ in range(2)]
    composition_max = 0.0
    selected_indices = [index for index in range(len(parent.GROUPS)) if selected_mask & (1 << index)]
    local_masks = [sum(1 << selected_indices[index] for index in range(len(selected_indices)) if local & (1 << index)) for local in range(1 << len(selected_indices))]
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["ood_code"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, correction, attention = parent.residual_parts(model, tokens)
        required = sorted(set(local_masks + [parent.FULL_MASK, selected_mask]))
        target = parent.subset_scores(groups, correction, attention, (parent.FULL_MASK,), residual.dtype)[parent.FULL_MASK]
        zero_scores = parent.subset_scores(groups, torch.zeros_like(correction), attention, required, residual.dtype)
        rolled_score = parent.subset_scores(groups, correction.roll(1, dims=1), attention, (selected_mask,), residual.dtype)[selected_mask]
        support = induction.induction_fetch_mask(tokens)
        parent.accumulate_geometry(code_stats, {selected_mask: zero_scores[selected_mask]}, target, support)
        parent.accumulate_geometry(rolled_stats, {selected_mask: rolled_score}, target, support)
        half = 0 if start < 96 else 1
        parent.accumulate_geometry(code_half_stats[half], {selected_mask: zero_scores[selected_mask]}, target, support)
        closure, _ = parent.mobius_closure(zero_scores, selected_mask)
        composition_max = max(composition_max, closure)
    code_report = parent.finish_geometry(code_stats[selected_mask])
    rolled_report = parent.finish_geometry(rolled_stats[selected_mask])
    code_halves = [parent.finish_geometry(stats[selected_mask]) for stats in code_half_stats]

    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    names = ("native", "absent", "factor_donor", "zero_correction", "rolled_correction")
    nll = {name: [] for name in names}
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        factor_logits, _ = score_parent.exact_score_donor_forward(model, tokens, PAIR, scales["L5H5"])
        zero_logits, zero_diag = transformed_forward(model, tokens, scales["L5H5"], selected_mask, torch.zeros_like)
        rolled_logits, rolled_diag = transformed_forward(model, tokens, scales["L5H5"], selected_mask, lambda correction: correction.roll(1, dims=1))
        minimum_term_rms = min(minimum_term_rms, zero_diag["term_rms"], rolled_diag["term_rms"])
        targets = batch_rows[:, 1:].cuda()
        for name, logits in zip(names, (native_logits, absent_logits, factor_logits, zero_logits, rolled_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    native_effect = nll["absent"] - nll["native"]
    behavior = {}
    for arm in ("factor_donor", "zero_correction", "rolled_correction"):
        effect = nll["absent"] - nll[arm]
        behavior[arm] = {}
        for cell in parent.CELLS:
            selected = masks[cell]
            native_sum = float(native_effect[selected].sum()); arm_sum = float(effect[selected].sum())
            behavior[arm][cell] = {"tokens": int(selected.sum()), "native_effect_sum_nat": native_sum, "recovery": arm_sum / native_sum if abs(native_sum) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["native"])[selected].mean())}
        behavior[arm]["halves"] = []
        for lo, hi in ((0, 96), (96, 192)):
            selected = masks["copy_positive"][lo:hi]
            native_sum = float(native_effect[lo:hi][selected].sum()); arm_sum = float(effect[lo:hi][selected].sum())
            behavior[arm]["halves"].append({"native_effect_sum_nat": native_sum, "recovery": arm_sum / native_sum if abs(native_sum) > 1e-30 else None})

    zero_recovery = behavior["zero_correction"]["copy_positive"]["recovery"]
    rolled_recovery = behavior["rolled_correction"]["copy_positive"]["recovery"]
    parent_recovery = parent_result["behavior"]["selected_donor"]["copy_positive"]["recovery"]
    parent_code_error = parent_result["code_score"]["pooled"]["relative_l2"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_a = bool(all(parent_result["predictions"].values()) and minimum_term_rms > 0 and behavior["factor_donor"]["copy_positive"]["recovery"] > .85)
    selected_natural = natural_reports[selected_mask]
    pred_b = bool(qualified and selected_names == ["L2", "L3", "L4"] and selected_natural["relative_l2"] <= .15 and selected_natural["cosine"] >= .98)
    pred_c = bool(code_report["relative_l2"] <= .20 and code_report["cosine"] >= .95 and all(report["relative_l2"] <= .25 for report in code_halves))
    pred_d = bool(.85 <= zero_recovery <= 1.05 and all(behavior["zero_correction"][cell]["recovery"] > .65 for cell in stable_cells) and all(row["recovery"] > .65 and row["native_effect_sum_nat"] > 0 for row in behavior["zero_correction"]["halves"]) and abs(behavior["zero_correction"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01)
    pred_e = bool(abs(zero_recovery - parent_recovery) <= .03 and code_report["relative_l2"] - parent_code_error <= .03 and abs(rolled_recovery - zero_recovery) <= .05)
    pred_f = bool(composition_max <= 2e-6)
    predictions = {"pred_a_live_bound_instrument": pred_a, "pred_b_same_correction_free_support": pred_b, "pred_c_correction_free_ood_score": pred_c, "pred_d_correction_free_behavior": pred_d, "pred_e_correction_dispensable": pred_e, "pred_f_correction_free_composition": pred_f}
    terminal = "equality_l5h5_correction_free_sparse_source_graph" if all(predictions.values()) else "valid_equality_l5h5_residual_correction_redteam_null" if pred_a and pred_f else "invalid"
    result = {"schema": "equality_l5h5_residual_correction_redteam_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"mask": selected_mask, "sources": selected_names, "qualified": qualified, "natural": selected_natural}, "code_score": {"zero_correction": code_report, "rolled_correction": rolled_report, "halves_zero": code_halves, "parent_error": parent_code_error}, "behavior": behavior, "comparisons": {"parent_selected_recovery": parent_recovery, "zero_recovery": zero_recovery, "rolled_recovery": rolled_recovery, "zero_minus_parent": zero_recovery - parent_recovery, "rolled_minus_zero": rolled_recovery - zero_recovery, "zero_score_error_minus_parent": code_report["relative_l2"] - parent_code_error}, "maximum_mobius_closure_relative_l2": composition_max, "minimum_term_rms": minimum_term_rms, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Red-team of the L5H5 residual provenance correction port. Zero and equal-norm position-rolled controls test whether three-source sparsity depends on roundoff direction after Q/K normalization."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_sources": selected_names, "natural": selected_natural, "code": code_report, "recoveries": result["comparisons"], "composition_error": composition_max, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
