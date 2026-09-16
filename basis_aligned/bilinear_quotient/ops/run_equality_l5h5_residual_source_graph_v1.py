#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;336 executions;180seconds;L5H5 residual-source graph.
"""Select and validate a sparse residual-source graph for the L5H5 score."""
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
PACKAGE = P / "extracted_circuits/equality_l5h5_residual_score_node_v1"
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction
import equality_matcher_causal_action_quotient_rung498 as action_parent
import rung498_copy_task_portability_diagnosis as diagnosis
import run_equality_l8h4_exact_order_node_code_ood_v2 as edge_v2
import run_equality_reusable_score_port_code_ood_v1 as score_parent
from extracted_circuits.equality_l5h5_residual_score_node_v1 import node as residual_node
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_RESIDUAL_SOURCE_GRAPH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
NATURAL = BQ / ".rowcache_induction_equality_tensor_final_ood_v2/final_natural.pt"
CODE = BQ / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
PARENT = P / "EQUALITY_L5H5_BILINEAR_SCORE_NODE_CODE_OOD_V2_RESULT.json"
PORT_PARENT = P / "EQUALITY_REUSABLE_SCORE_PORT_CODE_OOD_V1_RESULT.json"
MANIFEST = PACKAGE / "manifest.json"
DOCUMENTS = 192
PAIR = action_parent.PAIRS[0]
HEAD = action_parent.factor_parent.TERMS[PAIR[0]][2]
GROUPS = ("E", "L0", "L1", "L2", "L3", "L4")
FULL_MASK = (1 << len(GROUPS)) - 1
CELLS = score_parent.CELLS


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
    parent = json.loads(PARENT.read_text())
    port_parent = json.loads(PORT_PARENT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if parent["terminal"] != "equality_l5h5_bilinear_score_node_extracted_ood" or not all(parent["predictions"].values()):
        raise ValueError("bilinear score-node authority changed")
    if port_parent["terminal"] != "equality_l5h5_score_reusable_ood_port" or not all(port_parent["predictions"].values()):
        raise ValueError("score-port authority changed")
    if manifest.get("learned_parameters") != 0 or manifest.get("native_parameters_reused") != 589824:
        raise ValueError("residual-node manifest changed")
    roles = {}
    for name, path in (("final_natural", NATURAL), ("ood_code", CODE)):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("role") != name or list(payload["rows"].shape) != [DOCUMENTS, 257]:
            raise ValueError(f"{name} row authority changed")
        roles[name] = payload["rows"]
    _, _, _, _, scales, _ = action_parent.validate_inputs()
    return roles, scales, parent, port_parent, manifest


def plan():
    roles, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_residual_source_graph_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "tokens_per_document": 256, "execution_count": 336, "complete_code_forwards": 240, "prefix_executions": 96, "subset_node_calls_per_natural_batch": 63, "maximum_mobius_node_calls_per_code_batch": 64, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def head_weights(attention):
    width = action_parent.factor_parent.stage1.HEAD_DIM
    lo, hi = HEAD * width, (HEAD + 1) * width
    return tuple(layer.weight[lo:hi] for layer in (attention.c_q, attention.c_k, attention.c_q2, attention.c_k2))


def rotary_ports(attention, residual):
    width = action_parent.factor_parent.stage1.HEAD_DIM
    heads = action_parent.factor_parent.stage1.HEADS
    dummy = torch.empty(residual.shape[0], residual.shape[1], heads, width,
                        dtype=residual.dtype, device=residual.device)
    cos, sin = attention.rotary(dummy)
    return cos[:, :, 0], sin[:, :, 0]


@torch.no_grad()
def residual_parts(model, tokens):
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
    parts["E"] = parts["E"] + block.lambdas[1].float() * x0.float()
    groups = {"E": parts["E"]}
    for site in range(5):
        groups[f"L{site}"] = parts[f"A{site}"] + parts[f"M{site}"]
    semantic_sum = sum(groups.values(), start=torch.zeros_like(x.float()))
    correction = x.float() - semantic_sum
    return x, groups, correction, block.attn


def subset_residual(groups, correction, mask, dtype):
    chosen = [groups[name] for index, name in enumerate(GROUPS) if mask & (1 << index)]
    return residual_node.merge_sources(chosen, correction).to(dtype)


def subset_scores(groups, correction, attention, masks, dtype):
    reference = next(iter(groups.values()))
    weights = head_weights(attention)
    cos, sin = rotary_ports(attention, reference.to(dtype))
    return {mask: residual_node.execute(subset_residual(groups, correction, mask, dtype), weights, cos, sin).float() for mask in masks}


def empty_geometry():
    return {mask: torch.zeros(4, dtype=torch.float64) for mask in range(1, FULL_MASK + 1)}


def accumulate_geometry(stats, scores, target, support):
    target_selected = target[support].double()
    target2 = float(target_selected.square().sum())
    for mask, score in scores.items():
        value = score[support].double()
        stats[mask] += torch.tensor([float((value * target_selected).sum()), float(value.square().sum()), target2, int(support.sum())], dtype=torch.float64)


def finish_geometry(row):
    cross, predicted2, target2, edges = [float(value) for value in row]
    error = math.sqrt(max(predicted2 + target2 - 2 * cross, 0.0) / max(target2, 1e-30))
    cosine = cross / math.sqrt(max(predicted2 * target2, 1e-30))
    return {"relative_l2": error, "cosine": cosine, "edges": int(edges)}


def select_support(reports):
    qualifying = [mask for mask, report in reports.items() if report["relative_l2"] <= .15 and report["cosine"] >= .98]
    if qualifying:
        return min(qualifying, key=lambda mask: (mask.bit_count(), reports[mask]["relative_l2"], mask)), True
    return min(reports, key=lambda mask: (reports[mask]["relative_l2"], mask.bit_count(), mask)), False


def source_names(mask):
    return [name for index, name in enumerate(GROUPS) if mask & (1 << index)]


def mobius_closure(scores, selected_global_mask):
    selected_indices = [index for index in range(len(GROUPS)) if selected_global_mask & (1 << index)]
    local_full = (1 << len(selected_indices)) - 1
    local_scores = {}
    for local_mask in range(local_full + 1):
        global_mask = sum(1 << selected_indices[index] for index in range(len(selected_indices)) if local_mask & (1 << index))
        local_scores[local_mask] = scores[global_mask]
    terms = {}
    for mask in range(local_full + 1):
        if mask == 0:
            terms[mask] = local_scores[mask]
            continue
        term = local_scores[mask].clone()
        submask = (mask - 1) & mask
        while True:
            term = term - terms[submask]
            if submask == 0:
                break
            submask = (submask - 1) & mask
        terms[mask] = term
    reconstructed = sum(terms.values(), start=torch.zeros_like(local_scores[local_full]))
    error = float((reconstructed - local_scores[local_full]).norm() / local_scores[local_full].norm().clamp_min(1e-30))
    energies = {}
    for mask, term in terms.items():
        order = str(mask.bit_count())
        energies[order] = energies.get(order, 0.0) + float(term.double().square().sum())
    return error, energies


@torch.no_grad()
def residual_donor_forward(model, tokens, scale, selected_mask):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    parts = {"E": x.float()}
    selected_score = None
    diagnostics = {"residual_error": 0.0, "correction_ratio": 0.0, "full_score_error": 0.0, "term_rms": 0.0}
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
                groups = {"E": parts["E"]}
                for prior in range(5):
                    groups[f"L{prior}"] = parts[f"A{prior}"] + parts[f"M{prior}"]
                semantic_sum = sum(groups.values(), start=torch.zeros_like(x.float()))
                correction = x.float() - semantic_sum
                merged = residual_node.merge_sources(tuple(groups.values()), correction)
                diagnostics["residual_error"] = float((merged - x.float()).norm() / x.float().norm().clamp_min(1e-30))
                diagnostics["correction_ratio"] = float(correction.norm() / x.float().norm().clamp_min(1e-30))
                weights = head_weights(block.attn)
                cos, sin = rotary_ports(block.attn, x)
                full_score = residual_node.execute(merged.to(x.dtype), weights, cos, sin)
                selected_residual = subset_residual(groups, correction, selected_mask, x.dtype)
                selected_score = residual_node.execute(selected_residual, weights, cos, sin)
                diagnostics["full_score_error"] = float((full_score.float() - factors[PAIR[0]]["p"]).norm() / factors[PAIR[0]]["p"].norm().clamp_min(1e-30))
            if site == action_parent.factor_parent.TERMS[PAIR[1]][1]:
                if selected_score is None:
                    raise RuntimeError("selected L5H5 score unavailable")
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
    for arm in ("factor_donor", "full_residual_donor", "selected_donor"):
        effect = nll["absent"] - nll[arm]
        output[arm] = {}
        for cell in CELLS:
            selected = masks[cell]
            native_sum = float(native_effect[selected].sum())
            arm_sum = float(effect[selected].sum())
            output[arm][cell] = {"tokens": int(selected.sum()), "native_effect_sum_nat": native_sum, "recovery": arm_sum / native_sum if abs(native_sum) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["native"])[selected].mean())}
        output[arm]["halves"] = []
        for lo, hi in ((0, 96), (96, 192)):
            selected = masks["copy_positive"][lo:hi]
            native_sum = float(native_effect[lo:hi][selected].sum())
            arm_sum = float(effect[lo:hi][selected].sum())
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
    roles, scales, parent, port_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()

    natural_stats = empty_geometry()
    source_energy = {name: 0.0 for name in GROUPS}
    residual_energy = 0.0
    natural_correction_max = 0.0
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, correction, attention = residual_parts(model, tokens)
        semantic = sum(groups.values(), start=torch.zeros_like(residual.float()))
        natural_correction_max = max(natural_correction_max, float(correction.norm() / residual.float().norm().clamp_min(1e-30)))
        residual_energy += float(residual.double().square().sum())
        for name in GROUPS:
            source_energy[name] += float(groups[name].double().square().sum())
        scores = subset_scores(groups, correction, attention, range(1, FULL_MASK + 1), residual.dtype)
        target = scores[FULL_MASK]
        accumulate_geometry(natural_stats, scores, target, induction.induction_fetch_mask(tokens))
    natural_reports = {mask: finish_geometry(row) for mask, row in natural_stats.items()}
    selected_mask, qualified = select_support(natural_reports)
    selected_names = source_names(selected_mask)

    code_stats = {selected_mask: torch.zeros(4, dtype=torch.float64)}
    code_half_stats = [{selected_mask: torch.zeros(4, dtype=torch.float64)} for _ in range(2)]
    composition_max = 0.0
    mobius_energy = {}
    code_correction_max = 0.0
    selected_indices = [index for index in range(len(GROUPS)) if selected_mask & (1 << index)]
    local_masks = []
    for local_mask in range(1 << len(selected_indices)):
        local_masks.append(sum(1 << selected_indices[index] for index in range(len(selected_indices)) if local_mask & (1 << index)))
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["ood_code"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, correction, attention = residual_parts(model, tokens)
        code_correction_max = max(code_correction_max, float(correction.norm() / residual.float().norm().clamp_min(1e-30)))
        required = sorted(set(local_masks + [FULL_MASK, selected_mask]))
        scores = subset_scores(groups, correction, attention, required, residual.dtype)
        support = induction.induction_fetch_mask(tokens)
        accumulate_geometry(code_stats, {selected_mask: scores[selected_mask]}, scores[FULL_MASK], support)
        half = 0 if start < 96 else 1
        accumulate_geometry(code_half_stats[half], {selected_mask: scores[selected_mask]}, scores[FULL_MASK], support)
        closure, energies = mobius_closure(scores, selected_mask)
        composition_max = max(composition_max, closure)
        for order, energy in energies.items():
            mobius_energy[order] = mobius_energy.get(order, 0.0) + energy
    code_report = finish_geometry(code_stats[selected_mask])
    code_halves = [finish_geometry(stats[selected_mask]) for stats in code_half_stats]

    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    names = ("native", "absent", "factor_donor", "full_residual_donor", "selected_donor")
    nll = {name: [] for name in names}
    maxima = {"residual": 0.0, "correction": 0.0, "full_score": 0.0, "full_logits": 0.0}
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        factor_logits, _ = score_parent.exact_score_donor_forward(model, tokens, PAIR, scales["L5H5"])
        full_logits, full_diag = residual_donor_forward(model, tokens, scales["L5H5"], FULL_MASK)
        selected_logits, selected_diag = residual_donor_forward(model, tokens, scales["L5H5"], selected_mask)
        maxima["residual"] = max(maxima["residual"], full_diag["residual_error"], selected_diag["residual_error"])
        maxima["correction"] = max(maxima["correction"], full_diag["correction_ratio"], selected_diag["correction_ratio"], natural_correction_max, code_correction_max)
        maxima["full_score"] = max(maxima["full_score"], full_diag["full_score_error"], selected_diag["full_score_error"])
        maxima["full_logits"] = max(maxima["full_logits"], float((full_logits - factor_logits).norm() / factor_logits.norm().clamp_min(1e-30)))
        minimum_term_rms = min(minimum_term_rms, selected_diag["term_rms"])
        targets = batch_rows[:, 1:].cuda()
        for name, logits in zip(names, (native_logits, absent_logits, factor_logits, full_logits, selected_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    behavior = summarize(nll, masks)
    selected_recovery = behavior["selected_donor"]["copy_positive"]["recovery"]
    factor_recovery = behavior["factor_donor"]["copy_positive"]["recovery"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")

    pred_a = bool(max(maxima["residual"], maxima["full_score"], maxima["full_logits"]) <= 2e-6 and maxima["correction"] <= .01 and minimum_term_rms > 0)
    selected_natural = natural_reports[selected_mask]
    pred_b = bool(len(selected_names) <= 3 and qualified and selected_natural["relative_l2"] <= .15 and selected_natural["cosine"] >= .98)
    pred_c = bool(code_report["relative_l2"] <= .20 and code_report["cosine"] >= .95 and all(report["relative_l2"] <= .25 for report in code_halves))
    pred_d = bool(.85 <= selected_recovery <= 1.05 and abs(selected_recovery - factor_recovery) <= .10 and all(behavior["selected_donor"][cell]["recovery"] > .65 for cell in stable_cells) and all(row["recovery"] > .65 and row["native_effect_sum_nat"] > 0 for row in behavior["selected_donor"]["halves"]))
    pred_e = bool(abs(behavior["selected_donor"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01 and port_parent["reports"]["exact_control"]["copy_positive"]["recovery"] < 0)
    pred_f = bool(composition_max <= 2e-6 and manifest["learned_parameters"] == 0 and len(selected_names) == selected_mask.bit_count())
    predictions = {"pred_a_exact_residual_boundary": pred_a, "pred_b_sparse_natural_support": pred_b, "pred_c_ood_score_prediction": pred_c, "pred_d_ood_installation_removal": pred_d, "pred_e_selectivity": pred_e, "pred_f_composition_reuse": pred_f}
    terminal = "equality_l5h5_sparse_residual_source_graph_ood" if all(predictions.values()) else "valid_equality_l5h5_residual_source_graph_null" if pred_a and pred_f else "invalid"
    result = {"schema": "equality_l5h5_residual_source_graph_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"rule_qualified": qualified, "mask": selected_mask, "sources": selected_names, "source_count": len(selected_names), "natural": selected_natural}, "natural_subset_frontier": {str(mask): {"sources": source_names(mask), **report} for mask, report in sorted(natural_reports.items(), key=lambda item: (item[0].bit_count(), item[1]["relative_l2"], item[0]))}, "code_score": {"pooled": code_report, "halves": code_halves}, "behavior": behavior, "maximum_exactness_errors": maxima, "minimum_selected_term_rms": minimum_term_rms, "composition": {"maximum_mobius_closure_relative_l2": composition_max, "term_squared_norm_by_order": mobius_energy}, "provenance": {"groups": list(GROUPS), "source_squared_norm_over_residual_squared": {name: source_energy[name] / max(residual_energy, 1e-30) for name in GROUPS}, "correction_is_typed_implementation_port": True}, "package": {"manifest_sha256": digest(MANIFEST), "source_sha256": digest(PACKAGE / "node.py"), "learned_parameters": manifest["learned_parameters"], "native_parameters_reused": manifest["native_parameters_reused"]}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Exact one-residual-port L5H5 score executor plus frozen natural-to-code semantic-source support test. Source ablations are frozen-write boundary interventions with RMS recomputed; they are not recursive removals of upstream modules."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_sources": selected_names, "natural_score": selected_natural, "code_score": code_report, "selected_recovery": selected_recovery, "factor_recovery": factor_recovery, "maximum_errors": maxima, "composition_error": composition_max, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
