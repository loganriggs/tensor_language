#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;480 executions;180seconds;M4 native-product graph.
"""Refine the equality five-write boundary into selected native M4 product atoms."""
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
import run_equality_l5h5_l234_module_write_graph_v1 as module_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
import run_equality_l8h4_exact_order_node_code_ood_v2 as edge_v2
import run_equality_reusable_score_port_code_ood_v1 as score_parent
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_PRODUCT_CHANNEL_GRAPH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_L234_MODULE_WRITE_GRAPH_V2_RESULT.json"
DOCUMENTS = 192
PAIR = action_parent.PAIRS[0]
FIVE_MASK = sum(1 << module_parent.GROUPS.index(name) for name in ("M2", "A3", "M3", "A4", "M4"))
WIDTHS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4608)


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
    parent_result = json.loads(PARENT_RESULT.read_text())
    if parent_result["terminal"] != "valid_equality_l5h5_module_write_graph_null":
        raise ValueError("module-write parent changed")
    if not parent_result["predictions"]["pred_a_lawful_module_refinement"] or parent_result["selection"]["sources"] != ["M2", "A3", "M3", "A4", "M4"]:
        raise ValueError("five-write boundary changed")
    roles, scales, correction_parent, source_parent, port_parent, manifest, v1 = module_parent.load_bound()
    return roles, scales, parent_result, correction_parent, port_parent, manifest


def plan():
    roles, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_product_channel_graph_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 480, "natural_prefix_passes": 2, "code_geometry_passes": 1, "complete_code_arms": 6, "candidate_widths": list(WIDTHS), "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def fine_parts_with_m4(model, tokens):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    parts = {"E": x.float()}
    product = None
    mlp4 = None
    for site in range(5):
        block = model.transformer.h[site]
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        scale = block.lambdas[0].float()
        parts = {name: value * scale for name, value in parts.items()}
        parts["E"] = parts["E"] + block.lambdas[1].float() * x0.float()
        attention_write, v1 = block.attn(F.rms_norm(x, (1152,)), v1)
        parts[f"A{site}"] = attention_write.float()
        pre = x + attention_write
        mlp_state = F.rms_norm(pre, (1152,))
        if site == 4:
            mlp4 = block.mlp
            product = mlp4.Left(mlp_state) * mlp4.Right(mlp_state)
        mlp_write = block.mlp(mlp_state)
        parts[f"M{site}"] = mlp_write.float()
        x = pre + mlp_write
    block5 = model.transformer.h[5]
    x = block5.lambdas[0] * x + block5.lambdas[1] * x0
    scale5 = block5.lambdas[0].float()
    parts = {name: value * scale5 for name, value in parts.items()}
    parts["E"] = parts["E"] + block5.lambdas[1].float() * x0.float()
    groups = {name: parts[name] for name in module_parent.GROUPS}
    return x, groups, block5.attn, product, mlp4, scale5


def product_write(product, mlp, indices, scale, roll=False):
    if len(indices) == product.shape[-1]:
        write = mlp.Down(product)
    elif len(indices) == 0:
        write = torch.zeros((*product.shape[:-1], mlp.Down.weight.shape[0]), device=product.device, dtype=product.dtype)
    else:
        ordered = torch.sort(indices).values
        write = F.linear(product.index_select(-1, ordered), mlp.Down.weight.index_select(1, ordered).to(product.dtype))
    if roll:
        write = torch.roll(write, shifts=1, dims=1)
    return (write + mlp.Down_bias.to(write.dtype)).float() * scale


def score_with_m4(groups, m4_write, attention, dtype):
    changed = dict(groups)
    changed["M4"] = m4_write
    residual = module_parent.merge_fine(changed, FIVE_MASK, dtype)
    return module_parent.score_residual(residual, attention)


def edge_position_mask(tokens):
    support = induction.induction_fetch_mask(tokens)
    return support.any(dim=-1) | support.any(dim=-2)


def finish_pair(stats):
    dot, left, right, count = (float(value) for value in stats)
    denominator = max(right, 1e-30)
    relative_l2 = math.sqrt(max(left + right - 2 * dot, 0.0) / denominator)
    cosine = dot / math.sqrt(max(left * right, 1e-30))
    return {"relative_l2": relative_l2, "cosine": cosine, "values": int(count)}


def accumulate_pair(stats, predicted, target, mask):
    selected_predicted = predicted[mask].double()
    selected_target = target[mask].double()
    stats += torch.tensor([float((selected_predicted * selected_target).sum()), float(selected_predicted.square().sum()), float(selected_target.square().sum()), selected_target.numel()], dtype=torch.float64)


def select_width(reports):
    qualifying = [width for width, report in reports.items() if report["relative_l2"] <= .10 and report["cosine"] >= .995]
    if qualifying:
        return min(qualifying), True
    return min(reports, key=lambda width: (reports[width]["relative_l2"], width)), False


@torch.no_grad()
def product_donor_forward(model, tokens, scale, indices, roll=False):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    parts = {"E": x.float()}
    selected_score = None
    product4 = None
    mlp4 = None
    scale5 = None
    term_rms = math.inf
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        if site <= 5:
            scale0 = block.lambdas[0].float()
            parts = {name: value * scale0 for name, value in parts.items()}
            parts["E"] = parts["E"] + block.lambdas[1].float() * x0.float()
            if site == 5:
                scale5 = scale0
        attention_state = F.rms_norm(x, (1152,))
        if site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, v1 = block.attn(attention_state, v1)
        else:
            attention_write, factors, support, _ = action_parent.factor_parent._factor_site(attention_state, v1, block.attn, site, tokens)
            if site == action_parent.factor_parent.TERMS[PAIR[0]][1]:
                groups = {name: parts[name] for name in module_parent.GROUPS}
                groups["M4"] = product_write(product4, mlp4, indices, scale5, roll=roll)
                selected_score = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, x.dtype), block.attn)
            if site == action_parent.factor_parent.TERMS[PAIR[1]][1]:
                late = factors[PAIR[1]]
                late_head = action_parent.factor_parent.TERMS[PAIR[1]][2]
                raw_payload, output_weight = edge_v2.raw_head_payload(attention_state, v1, block.attn, late_head)
                donor_term = edge_node.execute(selected_score.float() * scale["score_ratio"], raw_payload, support, output_weight)
                term_rms = float(donor_term.float().square().mean().sqrt())
                attention_write = attention_write - late["native_term"] + donor_term.to(attention_write.dtype)
        pre = x + attention_write
        mlp_state = F.rms_norm(pre, (1152,))
        if site == 4:
            mlp4 = block.mlp
            product4 = mlp4.Left(mlp_state) * mlp4.Right(mlp_state)
        mlp_write = block.mlp(mlp_state)
        if site < 5:
            parts[f"A{site}"] = attention_write.float()
            parts[f"M{site}"] = mlp_write.float()
        x = pre + mlp_write
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, term_rms


def summarize(nll, masks):
    native_effect = nll["absent"] - nll["native"]
    output = {}
    for arm in ("factor_donor", "five_donor", "product_donor", "rolled_donor"):
        effect = nll["absent"] - nll[arm]
        output[arm] = {}
        for cell in residual_parent.CELLS:
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
    roles, scales, parent_result, correction_parent, port_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()

    activation_energy = torch.zeros(4608, device="cuda", dtype=torch.float64)
    activation_count = 0
    exact_write_error = 0.0
    exact_score_error = 0.0
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = fine_parts_with_m4(model, tokens)
        position_mask = edge_position_mask(tokens)
        selected_product = product[position_mask].double()
        activation_energy += selected_product.square().sum(dim=0)
        activation_count += selected_product.shape[0]
        reconstructed = product_write(product, mlp4, torch.arange(4608, device="cuda"), scale5)
        exact_write_error = max(exact_write_error, float((reconstructed - groups["M4"]).norm() / groups["M4"].norm().clamp_min(1e-30)))
        exact_score = score_with_m4(groups, reconstructed, attention, residual.dtype)
        target_score = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
        exact_score_error = max(exact_score_error, float((exact_score - target_score).norm() / target_score.norm().clamp_min(1e-30)))

    mlp4 = model.transformer.h[4].mlp
    attention5 = model.transformer.h[5].attn
    qk_weights = torch.cat(residual_parent.head_weights(attention5), dim=0).float()
    contracted = qk_weights @ mlp4.Down.weight.float()
    coupling = contracted.double().square().sum(dim=0).sqrt()
    activation_rms = (activation_energy / max(activation_count, 1)).sqrt()
    salience = activation_rms * coupling
    ranking = torch.argsort(salience, descending=True, stable=True)

    natural_stats = {width: torch.zeros(4, dtype=torch.float64) for width in WIDTHS}
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = fine_parts_with_m4(model, tokens)
        target = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
        support = induction.induction_fetch_mask(tokens)
        for width in WIDTHS:
            write = product_write(product, mlp4, ranking[:width], scale5)
            predicted = score_with_m4(groups, write, attention, residual.dtype)
            accumulate_pair(natural_stats[width], predicted, target, support)
    natural_reports = {width: finish_pair(stats) for width, stats in natural_stats.items()}
    selected_width, qualified = select_width(natural_reports)
    selected_indices = ranking[:selected_width]
    first_indices = selected_indices[0::2]
    second_indices = selected_indices[1::2]

    code_stats = torch.zeros(4, dtype=torch.float64)
    code_native_stats = torch.zeros(4, dtype=torch.float64)
    rolled_stats = torch.zeros(4, dtype=torch.float64)
    half_stats = [torch.zeros(4, dtype=torch.float64) for _ in range(2)]
    composition_stats = torch.zeros(4, dtype=torch.float64)
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["ood_code"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = fine_parts_with_m4(model, tokens)
        parent_score = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
        native_score = module_parent.score_residual(residual, attention)
        selected_write = product_write(product, mlp4, selected_indices, scale5)
        selected_score = score_with_m4(groups, selected_write, attention, residual.dtype)
        rolled_score = score_with_m4(groups, product_write(product, mlp4, selected_indices, scale5, roll=True), attention, residual.dtype)
        support = induction.induction_fetch_mask(tokens)
        accumulate_pair(code_stats, selected_score, parent_score, support)
        accumulate_pair(code_native_stats, selected_score, native_score, support)
        accumulate_pair(rolled_stats, rolled_score, parent_score, support)
        accumulate_pair(half_stats[0 if start < 96 else 1], selected_score, parent_score, support)
        score0 = score_with_m4(groups, product_write(product, mlp4, selected_indices[:0], scale5), attention, residual.dtype)
        score1 = score_with_m4(groups, product_write(product, mlp4, first_indices, scale5), attention, residual.dtype)
        score2 = score_with_m4(groups, product_write(product, mlp4, second_indices, scale5), attention, residual.dtype)
        additive = score1 + score2 - score0
        accumulate_pair(composition_stats, additive - score0, selected_score - score0, support)
    code_report = finish_pair(code_stats)
    code_native_report = finish_pair(code_native_stats)
    rolled_report = finish_pair(rolled_stats)
    code_halves = [finish_pair(stats) for stats in half_stats]
    composition = finish_pair(composition_stats)

    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    names_forward = ("native", "absent", "factor_donor", "five_donor", "product_donor", "rolled_donor")
    nll = {name: [] for name in names_forward}
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        factor_logits, _ = score_parent.exact_score_donor_forward(model, tokens, PAIR, scales["L5H5"])
        five_logits, _ = module_parent.fine_donor_forward(model, tokens, scales["L5H5"], FIVE_MASK)
        product_logits, term_rms = product_donor_forward(model, tokens, scales["L5H5"], selected_indices)
        rolled_logits, rolled_term_rms = product_donor_forward(model, tokens, scales["L5H5"], selected_indices, roll=True)
        minimum_term_rms = min(minimum_term_rms, term_rms, rolled_term_rms)
        targets = batch_rows[:, 1:].cuda()
        for name, logits in zip(names_forward, (native_logits, absent_logits, factor_logits, five_logits, product_logits, rolled_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    behavior = summarize(nll, masks)
    recovery = behavior["product_donor"]["copy_positive"]["recovery"]
    rolled_recovery = behavior["rolled_donor"]["copy_positive"]["recovery"]
    parent_recovery = behavior["five_donor"]["copy_positive"]["recovery"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")

    pred_a = bool(max(exact_write_error, exact_score_error) <= 2e-6 and minimum_term_rms > 0)
    pred_b = bool(qualified and selected_width <= 512)
    pred_c = bool(code_report["relative_l2"] <= .15 and code_report["cosine"] >= .98 and all(row["relative_l2"] <= .20 for row in code_halves))
    pred_d = bool(.80 <= recovery <= 1.05 and abs(recovery - parent_recovery) <= .08 and all(behavior["product_donor"][cell]["recovery"] > .55 for cell in stable_cells) and all(row["recovery"] > .55 and row["native_effect_sum_nat"] > 0 for row in behavior["product_donor"]["halves"]))
    pred_e = bool(abs(behavior["product_donor"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01 and recovery - rolled_recovery >= .05 and port_parent["reports"]["exact_control"]["copy_positive"]["recovery"] < 0)
    pred_f = bool(composition["relative_l2"] <= .10 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_native_product_instrument": pred_a, "pred_b_compact_natural_product_support": pred_b, "pred_c_ood_parent_score_prediction": pred_c, "pred_d_ood_causal_use": pred_d, "pred_e_equal_norm_directional_selectivity": pred_e, "pred_f_two_component_composition": pred_f}
    terminal = "equality_l5h5_compact_m4_product_graph_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_product_graph_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_product_channel_graph_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"qualified": qualified, "width": selected_width, "fraction_of_m4_products": selected_width / 4608, "indices": selected_indices.cpu().tolist(), "natural": natural_reports[selected_width]}, "natural_width_curve": {str(width): report for width, report in natural_reports.items()}, "code_score": {"versus_five_write_parent": code_report, "versus_native_residual": code_native_report, "rolled_equal_norm_control": rolled_report, "halves": code_halves}, "composition": {"split": "alternating frozen salience rank", "first_width": len(first_indices), "second_width": len(second_indices), "additive_effect_prediction": composition}, "behavior": behavior, "comparisons": {"five_write_recovery": parent_recovery, "product_recovery": recovery, "product_minus_five": recovery - parent_recovery, "rolled_recovery": rolled_recovery, "product_minus_rolled": recovery - rolled_recovery}, "instrument": {"maximum_native_m4_write_replay_error": exact_write_error, "maximum_native_five_score_replay_error": exact_score_error, "salience": "natural product RMS on equality-edge incident positions times norm of Q/K-contracted Down column", "activation_positions": activation_count, "minimum_donor_term_rms": minimum_term_rms}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "learned_parameters": 0, "native_product_width": 4608, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "One-layer DCT-moment ranking and causal validation of exact native M4 bilinear product atoms inside the five-write equality-score boundary. M4 input state remains an external context port."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_width": selected_width, "natural": natural_reports[selected_width], "code": code_report, "recovery": recovery, "rolled_recovery": rolled_recovery, "composition": composition, "exact_write_error": exact_write_error, "exact_score_error": exact_score_error, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
