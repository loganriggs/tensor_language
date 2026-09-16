#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;384 executions;180seconds;M4 contracted-mode graph.
"""Positive red-team of M4 atom sparsity with weight-only Q/K-contracted modes."""
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
import run_equality_l5h5_m4_product_channel_graph_v1 as atom_parent
import run_equality_l5h5_l234_module_write_graph_v1 as module_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
import run_equality_l8h4_exact_order_node_code_ood_v2 as edge_v2
import run_equality_reusable_score_port_code_ood_v1 as score_parent
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ATOM_RESULT = P / "EQUALITY_L5H5_M4_PRODUCT_CHANNEL_GRAPH_V1_RESULT.json"
DOCUMENTS = 192
PAIR = action_parent.PAIRS[0]
FIVE_MASK = atom_parent.FIVE_MASK
RANKS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


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
    atom_result = json.loads(ATOM_RESULT.read_text())
    if atom_result["terminal"] != "valid_equality_l5h5_m4_product_graph_null" or not atom_result["predictions"]["pred_a_lawful_native_product_instrument"]:
        raise ValueError("native-product null changed")
    if atom_result["selection"]["width"] != 4096 or atom_result["predictions"]["pred_b_compact_natural_product_support"]:
        raise ValueError("native-product sparsity authority changed")
    roles, scales, parent_result, correction_parent, port_parent, manifest = atom_parent.load_bound()
    return roles, scales, atom_result, parent_result, port_parent, manifest


def plan():
    roles, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_contracted_mode_graph_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 384, "natural_geometry_passes": 1, "code_geometry_passes": 1, "complete_code_arms": 6, "candidate_ranks": list(RANKS), "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def mode_write(product, mlp, basis, writers, rank, scale, roll=False):
    if rank == 0:
        write = torch.zeros((*product.shape[:-1], mlp.Down.weight.shape[0]), device=product.device, dtype=torch.float32)
    else:
        coefficients = F.linear(product.float(), basis[:rank])
        write = F.linear(coefficients, writers[:, :rank])
    if roll:
        write = torch.roll(write, shifts=1, dims=1)
    quantized = (write + mlp.Down_bias.float()).to(product.dtype).float()
    return quantized * scale


def score_with_rank(groups, product, mlp, basis, writers, rank, scale, attention, dtype, roll=False):
    return atom_parent.score_with_m4(groups, mode_write(product, mlp, basis, writers, rank, scale, roll=roll), attention, dtype)


def select_rank(reports):
    qualifying = [rank for rank, report in reports.items() if report["relative_l2"] <= .10 and report["cosine"] >= .995]
    if qualifying:
        return min(qualifying), True
    return min(reports, key=lambda rank: (reports[rank]["relative_l2"], rank)), False


@torch.no_grad()
def mode_donor_forward(model, tokens, scale, basis, writers, rank, roll=False):
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
                groups["M4"] = mode_write(product4, mlp4, basis, writers, rank, scale5, roll=roll)
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


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, scales, atom_result, parent_result, port_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()

    mlp4 = model.transformer.h[4].mlp
    attention5 = model.transformer.h[5].attn
    qk_weights = torch.cat(residual_parent.head_weights(attention5), dim=0).float()
    contracted = qk_weights @ mlp4.Down.weight.float()
    left_vectors, singular_values, basis = torch.linalg.svd(contracted, full_matrices=False)
    reconstructed = (left_vectors * singular_values.unsqueeze(0)) @ basis
    bridge_error = float((reconstructed - contracted).norm() / contracted.norm().clamp_min(1e-30))
    writers = mlp4.Down.weight.float() @ basis.T

    natural_stats = {rank: torch.zeros(4, dtype=torch.float64) for rank in RANKS}
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = atom_parent.fine_parts_with_m4(model, tokens)
        target = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
        support = induction.induction_fetch_mask(tokens)
        for rank in RANKS:
            predicted = score_with_rank(groups, product, mlp4, basis, writers, rank, scale5, attention, residual.dtype)
            atom_parent.accumulate_pair(natural_stats[rank], predicted, target, support)
    natural_reports = {rank: atom_parent.finish_pair(stats) for rank, stats in natural_stats.items()}
    selected_rank, qualified = select_rank(natural_reports)

    code_stats = torch.zeros(4, dtype=torch.float64)
    code_native_stats = torch.zeros(4, dtype=torch.float64)
    rolled_stats = torch.zeros(4, dtype=torch.float64)
    half_stats = [torch.zeros(4, dtype=torch.float64) for _ in range(2)]
    composition_stats = torch.zeros(4, dtype=torch.float64)
    first_modes = torch.arange(0, selected_rank, 2, device="cuda")
    second_modes = torch.arange(1, selected_rank, 2, device="cuda")
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["ood_code"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = atom_parent.fine_parts_with_m4(model, tokens)
        parent_score = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
        native_score = module_parent.score_residual(residual, attention)
        selected_score = score_with_rank(groups, product, mlp4, basis, writers, selected_rank, scale5, attention, residual.dtype)
        rolled_score = score_with_rank(groups, product, mlp4, basis, writers, selected_rank, scale5, attention, residual.dtype, roll=True)
        support = induction.induction_fetch_mask(tokens)
        atom_parent.accumulate_pair(code_stats, selected_score, parent_score, support)
        atom_parent.accumulate_pair(code_native_stats, selected_score, native_score, support)
        atom_parent.accumulate_pair(rolled_stats, rolled_score, parent_score, support)
        atom_parent.accumulate_pair(half_stats[0 if start < 96 else 1], selected_score, parent_score, support)
        score0 = score_with_rank(groups, product, mlp4, basis, writers, 0, scale5, attention, residual.dtype)
        score1 = atom_parent.score_with_m4(groups, mode_write(product, mlp4, basis.index_select(0, first_modes), writers.index_select(1, first_modes), len(first_modes), scale5), attention, residual.dtype)
        score2 = atom_parent.score_with_m4(groups, mode_write(product, mlp4, basis.index_select(0, second_modes), writers.index_select(1, second_modes), len(second_modes), scale5), attention, residual.dtype)
        additive = score1 + score2 - score0
        atom_parent.accumulate_pair(composition_stats, additive - score0, selected_score - score0, support)
    code_report = atom_parent.finish_pair(code_stats)
    code_native_report = atom_parent.finish_pair(code_native_stats)
    rolled_report = atom_parent.finish_pair(rolled_stats)
    code_halves = [atom_parent.finish_pair(stats) for stats in half_stats]
    composition = atom_parent.finish_pair(composition_stats)

    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    names_forward = ("native", "absent", "factor_donor", "five_donor", "mode_donor", "rolled_donor")
    nll = {name: [] for name in names_forward}
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        factor_logits, _ = score_parent.exact_score_donor_forward(model, tokens, PAIR, scales["L5H5"])
        five_logits, _ = module_parent.fine_donor_forward(model, tokens, scales["L5H5"], FIVE_MASK)
        mode_logits, term_rms = mode_donor_forward(model, tokens, scales["L5H5"], basis, writers, selected_rank)
        rolled_logits, rolled_term_rms = mode_donor_forward(model, tokens, scales["L5H5"], basis, writers, selected_rank, roll=True)
        minimum_term_rms = min(minimum_term_rms, term_rms, rolled_term_rms)
        targets = batch_rows[:, 1:].cuda()
        for name, logits in zip(names_forward, (native_logits, absent_logits, factor_logits, five_logits, mode_logits, rolled_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    behavior = atom_parent.summarize({"native": nll["native"], "absent": nll["absent"], "factor_donor": nll["factor_donor"], "five_donor": nll["five_donor"], "product_donor": nll["mode_donor"], "rolled_donor": nll["rolled_donor"]}, masks)
    behavior["mode_donor"] = behavior.pop("product_donor")
    recovery = behavior["mode_donor"]["copy_positive"]["recovery"]
    rolled_recovery = behavior["rolled_donor"]["copy_positive"]["recovery"]
    parent_recovery = behavior["five_donor"]["copy_positive"]["recovery"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")

    inherited_exact = atom_result["instrument"]
    pred_a = bool(inherited_exact["maximum_native_m4_write_replay_error"] <= 2e-6 and inherited_exact["maximum_native_five_score_replay_error"] <= 2e-6 and bridge_error <= 2e-5 and minimum_term_rms > 0)
    pred_b = bool(qualified and selected_rank <= 64)
    pred_c = bool(code_report["relative_l2"] <= .15 and code_report["cosine"] >= .98 and all(row["relative_l2"] <= .20 for row in code_halves))
    pred_d = bool(.80 <= recovery <= 1.05 and abs(recovery - parent_recovery) <= .08 and all(behavior["mode_donor"][cell]["recovery"] > .55 for cell in stable_cells) and all(row["recovery"] > .55 and row["native_effect_sum_nat"] > 0 for row in behavior["mode_donor"]["halves"]))
    pred_e = bool(abs(behavior["mode_donor"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01 and recovery - rolled_recovery >= .05 and port_parent["reports"]["exact_control"]["copy_positive"]["recovery"] < 0)
    pred_f = bool(composition["relative_l2"] <= .10 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_contracted_mode_bridge": pred_a, "pred_b_compact_natural_mode_support": pred_b, "pred_c_ood_parent_score_prediction": pred_c, "pred_d_ood_causal_use": pred_d, "pred_e_equal_norm_directional_selectivity": pred_e, "pred_f_two_component_composition": pred_f}
    terminal = "equality_l5h5_compact_m4_contracted_mode_graph_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_contracted_mode_graph_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_contracted_mode_graph_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"qualified": qualified, "rank": selected_rank, "fraction_of_contracted_rank": selected_rank / 512, "natural": natural_reports[selected_rank]}, "natural_rank_curve": {str(rank): report for rank, report in natural_reports.items()}, "code_score": {"versus_five_write_parent": code_report, "versus_native_residual": code_native_report, "rolled_equal_norm_control": rolled_report, "halves": code_halves}, "composition": {"split": "alternating frozen singular rank", "first_rank": len(first_modes), "second_rank": len(second_modes), "additive_effect_prediction": composition}, "behavior": behavior, "comparisons": {"native_atom_selected_width": atom_result["selection"]["width"], "five_write_recovery": parent_recovery, "mode_recovery": recovery, "mode_minus_five": recovery - parent_recovery, "rolled_recovery": rolled_recovery, "mode_minus_rolled": recovery - rolled_recovery}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "inherited_native_m4_write_replay_error": inherited_exact["maximum_native_m4_write_replay_error"], "inherited_native_five_score_replay_error": inherited_exact["maximum_native_five_score_replay_error"], "minimum_donor_term_rms": minimum_term_rms}, "program_price": {"selected_scalar_bilinear_modes": selected_rank, "derived_basis_values": selected_rank * 4608, "derived_writer_values": selected_rank * 1152, "native_product_evaluations_retained": 4608, "m4_input_state_port": True, "learned_parameters": 0}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Weight-only Q/K-contracted SVD compression of M4's product-to-writer map inside the five-write equality score graph. Dense modes retain native M4 input and all native product evaluations."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_rank": selected_rank, "natural": natural_reports[selected_rank], "code": code_report, "recovery": recovery, "rolled_recovery": rolled_recovery, "composition": composition, "bridge_error": bridge_error, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
