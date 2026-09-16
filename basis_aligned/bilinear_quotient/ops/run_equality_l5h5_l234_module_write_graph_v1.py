#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;288 executions;180seconds;L2-L4 module-write graph.
"""Refine the L2+L3+L4 L5H5 source graph into attention/MLP write ports."""
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
import run_equality_l8h4_exact_order_node_code_ood_v2 as edge_v2
import run_equality_reusable_score_port_code_ood_v1 as score_parent
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_L234_MODULE_WRITE_GRAPH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_RESIDUAL_CORRECTION_REDTEAM_V1_RESULT.json"
SOURCE_PARENT = P / "EQUALITY_L5H5_RESIDUAL_SOURCE_GRAPH_V1_RESULT.json"
DOCUMENTS = 192
PAIR = action_parent.PAIRS[0]
GROUPS = ("A2", "M2", "A3", "M3", "A4", "M4")
FULL_MASK = (1 << len(GROUPS)) - 1


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
    correction_parent = json.loads(PARENT_RESULT.read_text())
    source_parent = json.loads(SOURCE_PARENT.read_text())
    if correction_parent["terminal"] != "equality_l5h5_correction_free_sparse_source_graph" or not all(correction_parent["predictions"].values()) or correction_parent["selection"]["sources"] != ["L2", "L3", "L4"]:
        raise ValueError("correction-free parent changed")
    if source_parent["terminal"] != "equality_l5h5_sparse_residual_source_graph_ood" or not all(source_parent["predictions"].values()):
        raise ValueError("source-graph authority changed")
    roles, scales, _, port_parent, manifest = parent.load_bound()
    return roles, scales, correction_parent, source_parent, port_parent, manifest


def plan():
    roles, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_l234_module_write_graph_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 288, "complete_code_forwards": 192, "prefix_executions": 96, "subset_node_calls_per_natural_batch": 63, "maximum_mobius_node_calls_per_code_batch": 64, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def fine_parts(model, tokens):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    parts = {"E": x.float()}
    for site in range(5):
        block = model.transformer.h[site]
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        scale = block.lambdas[0].float()
        parts = {name: value * scale for name, value in parts.items()}
        parts["E"] = parts["E"] + block.lambdas[1].float() * x0.float()
        attention_write, v1 = block.attn(F.rms_norm(x, (1152,)), v1)
        parts[f"A{site}"] = attention_write.float()
        pre = x + attention_write
        mlp_write = block.mlp(F.rms_norm(pre, (1152,)))
        parts[f"M{site}"] = mlp_write.float()
        x = pre + mlp_write
    block = model.transformer.h[5]
    x = block.lambdas[0] * x + block.lambdas[1] * x0
    scale = block.lambdas[0].float()
    parts = {name: value * scale for name, value in parts.items()}
    fine = {name: parts[name] for name in GROUPS}
    return x, fine, block.attn


def merge_fine(groups, mask, dtype):
    chosen = [groups[name] for index, name in enumerate(GROUPS) if mask & (1 << index)]
    if not chosen:
        return torch.zeros_like(next(iter(groups.values())), dtype=dtype)
    return sum((value.float() for value in chosen), start=torch.zeros_like(chosen[0].float())).to(dtype)


def parent_residual(groups, dtype):
    layer2 = groups["A2"] + groups["M2"]
    layer3 = groups["A3"] + groups["M3"]
    layer4 = groups["A4"] + groups["M4"]
    return (torch.zeros_like(layer2) + layer2 + layer3 + layer4).to(dtype)


def score_residual(residual, attention):
    weights = parent.head_weights(attention)
    cos, sin = parent.rotary_ports(attention, residual)
    return parent.residual_node.execute(residual, weights, cos, sin).float()


def subset_scores(groups, attention, masks, dtype):
    return {mask: score_residual(merge_fine(groups, mask, dtype), attention) for mask in masks}


def empty_geometry():
    return {mask: torch.zeros(4, dtype=torch.float64) for mask in range(1, FULL_MASK + 1)}


def names(mask):
    return [name for index, name in enumerate(GROUPS) if mask & (1 << index)]


def select_support(reports):
    qualifying = [mask for mask, report in reports.items() if report["relative_l2"] <= .10 and report["cosine"] >= .995]
    if qualifying:
        return min(qualifying, key=lambda mask: (mask.bit_count(), reports[mask]["relative_l2"], mask)), True
    return min(reports, key=lambda mask: (reports[mask]["relative_l2"], mask.bit_count(), mask)), False


@torch.no_grad()
def fine_donor_forward(model, tokens, scale, selected_mask):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    parts = {"E": x.float()}
    selected_score = None
    diagnostics = {"parent_score_error": 0.0, "term_rms": 0.0}
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        if site <= 5:
            scale0 = block.lambdas[0].float()
            parts = {name: value * scale0 for name, value in parts.items()}
            parts["E"] = parts["E"] + block.lambdas[1].float() * x0.float()
        attention_state = F.rms_norm(x, (1152,))
        if site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, v1 = block.attn(attention_state, v1)
        else:
            attention_write, factors, support, _ = action_parent.factor_parent._factor_site(attention_state, v1, block.attn, site, tokens)
            if site == action_parent.factor_parent.TERMS[PAIR[0]][1]:
                groups = {name: parts[name] for name in GROUPS}
                selected_residual = merge_fine(groups, selected_mask, x.dtype)
                selected_score = score_residual(selected_residual, block.attn)
                full_fine = score_residual(merge_fine(groups, FULL_MASK, x.dtype), block.attn)
                target_parent = score_residual(parent_residual(groups, x.dtype), block.attn)
                diagnostics["parent_score_error"] = float((full_fine - target_parent).norm() / target_parent.norm().clamp_min(1e-30))
            if site == action_parent.factor_parent.TERMS[PAIR[1]][1]:
                if selected_score is None:
                    raise RuntimeError("selected fine score unavailable")
                late = factors[PAIR[1]]
                late_head = action_parent.factor_parent.TERMS[PAIR[1]][2]
                raw_payload, output_weight = edge_v2.raw_head_payload(attention_state, v1, block.attn, late_head)
                donor_term = edge_node.execute(selected_score.float() * scale["score_ratio"], raw_payload, support, output_weight)
                diagnostics["term_rms"] = float(donor_term.float().square().mean().sqrt())
                attention_write = attention_write - late["native_term"] + donor_term.to(attention_write.dtype)
        pre = x + attention_write
        mlp_write = block.mlp(F.rms_norm(pre, (1152,)))
        if site < 5:
            parts[f"A{site}"] = attention_write.float()
            parts[f"M{site}"] = mlp_write.float()
        x = pre + mlp_write
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, diagnostics


def summarize(nll, masks):
    native_effect = nll["absent"] - nll["native"]
    output = {}
    for arm in ("factor_donor", "fine_donor"):
        effect = nll["absent"] - nll[arm]
        output[arm] = {}
        for cell in parent.CELLS:
            selected = masks[cell]
            native_sum = float(native_effect[selected].sum()); arm_sum = float(effect[selected].sum())
            output[arm][cell] = {"tokens": int(selected.sum()), "native_effect_sum_nat": native_sum, "recovery": arm_sum / native_sum if abs(native_sum) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["native"])[selected].mean())}
        output[arm]["halves"] = []
        for lo, hi in ((0, 96), (96, 192)):
            selected = masks["copy_positive"][lo:hi]
            native_sum = float(native_effect[lo:hi][selected].sum()); arm_sum = float(effect[lo:hi][selected].sum())
            output[arm]["halves"].append({"native_effect_sum_nat": native_sum, "recovery": arm_sum / native_sum if abs(native_sum) > 1e-30 else None})
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
    roles, scales, correction_parent, source_parent, port_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()

    natural_stats = empty_geometry()
    full_parent_error_max = 0.0
    source_energy = {name: 0.0 for name in GROUPS}
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention = fine_parts(model, tokens)
        scores = subset_scores(groups, attention, range(1, FULL_MASK + 1), residual.dtype)
        target = score_residual(parent_residual(groups, residual.dtype), attention)
        full_parent_error_max = max(full_parent_error_max, float((scores[FULL_MASK] - target).norm() / target.norm().clamp_min(1e-30)))
        parent.accumulate_geometry(natural_stats, scores, target, induction.induction_fetch_mask(tokens))
        for name in GROUPS:
            source_energy[name] += float(groups[name].double().square().sum())
    natural_reports = {mask: parent.finish_geometry(row) for mask, row in natural_stats.items()}
    selected_mask, qualified = select_support(natural_reports)
    selected_names = names(selected_mask)

    code_parent_stats = {selected_mask: torch.zeros(4, dtype=torch.float64)}
    code_native_stats = {selected_mask: torch.zeros(4, dtype=torch.float64)}
    code_half_stats = [{selected_mask: torch.zeros(4, dtype=torch.float64)} for _ in range(2)]
    composition_max = 0.0
    selected_indices = [index for index in range(len(GROUPS)) if selected_mask & (1 << index)]
    local_masks = [sum(1 << selected_indices[index] for index in range(len(selected_indices)) if local & (1 << index)) for local in range(1 << len(selected_indices))]
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["ood_code"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention = fine_parts(model, tokens)
        required = sorted(set(local_masks + [FULL_MASK, selected_mask]))
        scores = subset_scores(groups, attention, required, residual.dtype)
        target_parent = score_residual(parent_residual(groups, residual.dtype), attention)
        target_native = score_residual(residual, attention)
        full_parent_error_max = max(full_parent_error_max, float((scores[FULL_MASK] - target_parent).norm() / target_parent.norm().clamp_min(1e-30)))
        support = induction.induction_fetch_mask(tokens)
        parent.accumulate_geometry(code_parent_stats, {selected_mask: scores[selected_mask]}, target_parent, support)
        parent.accumulate_geometry(code_native_stats, {selected_mask: scores[selected_mask]}, target_native, support)
        half = 0 if start < 96 else 1
        parent.accumulate_geometry(code_half_stats[half], {selected_mask: scores[selected_mask]}, target_parent, support)
        closure, _ = parent.mobius_closure(scores, selected_mask)
        composition_max = max(composition_max, closure)
    code_parent = parent.finish_geometry(code_parent_stats[selected_mask])
    code_native = parent.finish_geometry(code_native_stats[selected_mask])
    code_halves = [parent.finish_geometry(stats[selected_mask]) for stats in code_half_stats]

    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    names_forward = ("native", "absent", "factor_donor", "fine_donor")
    nll = {name: [] for name in names_forward}
    minimum_term_rms = math.inf
    forward_parent_error_max = 0.0
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        factor_logits, _ = score_parent.exact_score_donor_forward(model, tokens, PAIR, scales["L5H5"])
        fine_logits, diag = fine_donor_forward(model, tokens, scales["L5H5"], selected_mask)
        minimum_term_rms = min(minimum_term_rms, diag["term_rms"])
        forward_parent_error_max = max(forward_parent_error_max, diag["parent_score_error"])
        targets = batch_rows[:, 1:].cuda()
        for name, logits in zip(names_forward, (native_logits, absent_logits, factor_logits, fine_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    behavior = summarize(nll, masks)
    recovery = behavior["fine_donor"]["copy_positive"]["recovery"]
    parent_recovery = correction_parent["behavior"]["zero_correction"]["copy_positive"]["recovery"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")

    pred_a = bool(max(full_parent_error_max, forward_parent_error_max) <= 2e-6 and minimum_term_rms > 0 and all(correction_parent["predictions"].values()))
    selected_natural = natural_reports[selected_mask]
    pred_b = bool(qualified and len(selected_names) <= 4 and selected_natural["relative_l2"] <= .10 and selected_natural["cosine"] >= .995)
    pred_c = bool(code_parent["relative_l2"] <= .15 and code_parent["cosine"] >= .98 and all(report["relative_l2"] <= .20 for report in code_halves))
    pred_d = bool(.80 <= recovery <= 1.05 and abs(recovery - parent_recovery) <= .08 and all(behavior["fine_donor"][cell]["recovery"] > .55 for cell in stable_cells) and all(row["recovery"] > .55 and row["native_effect_sum_nat"] > 0 for row in behavior["fine_donor"]["halves"]))
    pred_e = bool(abs(behavior["fine_donor"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01 and port_parent["reports"]["exact_control"]["copy_positive"]["recovery"] < 0)
    pred_f = bool(composition_max <= 2e-6 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_module_refinement": pred_a, "pred_b_sparse_natural_module_support": pred_b, "pred_c_ood_parent_score_prediction": pred_c, "pred_d_ood_causal_use": pred_d, "pred_e_selectivity": pred_e, "pred_f_composition_reuse": pred_f}
    terminal = "equality_l5h5_sparse_module_write_graph_ood" if all(predictions.values()) else "valid_equality_l5h5_module_write_graph_null" if pred_a and pred_f else "invalid"
    result = {"schema": "equality_l5h5_l234_module_write_graph_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"qualified": qualified, "mask": selected_mask, "sources": selected_names, "source_count": len(selected_names), "natural": selected_natural}, "natural_subset_frontier": {str(mask): {"sources": names(mask), **report} for mask, report in sorted(natural_reports.items(), key=lambda item: (item[0].bit_count(), item[1]["relative_l2"], item[0]))}, "code_score": {"versus_parent_l234": code_parent, "versus_native": code_native, "parent_halves": code_halves}, "behavior": behavior, "comparisons": {"parent_recovery": parent_recovery, "fine_recovery": recovery, "fine_minus_parent": recovery - parent_recovery}, "maximum_all_six_to_parent_score_error": max(full_parent_error_max, forward_parent_error_max), "maximum_mobius_closure_relative_l2": composition_max, "minimum_term_rms": minimum_term_rms, "source_squared_norm": source_energy, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Frozen-write refinement of the L2+L3+L4 parent score graph into A2/M2/A3/M3/A4/M4 boundary ports. Selected module writes remain native external producers."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_sources": selected_names, "natural": selected_natural, "code_parent": code_parent, "code_native": code_native, "recovery": recovery, "parent_recovery": parent_recovery, "all_six_error": result["maximum_all_six_to_parent_score_error"], "composition_error": composition_max, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
