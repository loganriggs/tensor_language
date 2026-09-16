#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;432 executions;180seconds;M4 explicit interaction graph.
"""Promote the rank-128/rank-128 M4 cross-difference to a graph node."""
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
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
import run_equality_l5h5_l234_module_write_graph_v1 as module_parent
import run_equality_l8h4_exact_order_node_code_ood_v2 as edge_v2
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_EXPLICIT_INTERACTION_GRAPH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_MOST_ADDITIVE_MODE_SPLIT_V1_RESULT.json"
DOCUMENTS = 192
CUT = 128
RANK = 256
PAIR = mode_parent.PAIR
FIVE_MASK = mode_parent.FIVE_MASK
VARIANTS = ("baseline", "additive", "composed", "direct", "rolled")


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
    if parent["terminal"] != "valid_equality_l5h5_m4_most_additive_split_null" or parent["selection"]["child_rank"] != CUT or parent["selection"]["remainder_rank"] != CUT:
        raise ValueError("most-additive parent changed")
    if not parent["predictions"]["pred_a_lawful_inherited_mode_bridge"] or not parent["predictions"]["pred_e_joint_causal_selective_use"]:
        raise ValueError("lawful causal parent changed")
    roles, scales, contracted_parent, atom_result, port_parent, manifest = split_parent.load_bound()
    return roles, scales, parent, contracted_parent, port_parent, manifest


def plan():
    roles, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_explicit_interaction_graph_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 432, "score_variants": list(VARIANTS), "complete_code_arms": 7, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def score_nodes(groups, product, mlp, basis, writers, scale, attention, dtype):
    modes = torch.arange(RANK, device=product.device)
    baseline = split_parent.subset_score(groups, product, mlp, basis, writers, modes[:0], scale, attention, dtype)
    child = split_parent.subset_score(groups, product, mlp, basis, writers, modes[:CUT], scale, attention, dtype)
    remainder = split_parent.subset_score(groups, product, mlp, basis, writers, modes[CUT:], scale, attention, dtype)
    direct = split_parent.subset_score(groups, product, mlp, basis, writers, modes, scale, attention, dtype)
    child_effect = child - baseline
    remainder_effect = remainder - baseline
    interaction = direct - child - remainder + baseline
    additive = (baseline + child_effect) + remainder_effect
    composed = additive + interaction
    rolled = additive + torch.roll(interaction, shifts=1, dims=1)
    return {"baseline": baseline, "child": child, "remainder": remainder, "direct": direct, "child_effect": child_effect, "remainder_effect": remainder_effect, "interaction": interaction, "additive": additive, "composed": composed, "rolled": rolled}


def empty_graph_stats():
    return {"closure": torch.zeros(4, dtype=torch.float64), "direct_parent": torch.zeros(4, dtype=torch.float64), "interaction_sq": 0.0, "joint_effect_sq": 0.0, "rolled_interaction_sq": 0.0, "values": 0}


def accumulate_graph(stats, nodes, parent_score, support):
    mode_parent.atom_parent.accumulate_pair(stats["closure"], nodes["composed"], nodes["direct"], support)
    mode_parent.atom_parent.accumulate_pair(stats["direct_parent"], nodes["direct"], parent_score, support)
    interaction = nodes["interaction"][support].double()
    joint_effect = (nodes["direct"] - nodes["baseline"])[support].double()
    rolled = torch.roll(nodes["interaction"], shifts=1, dims=1)[support].double()
    stats["interaction_sq"] += float(interaction.square().sum())
    stats["joint_effect_sq"] += float(joint_effect.square().sum())
    stats["rolled_interaction_sq"] += float(rolled.square().sum())
    stats["values"] += interaction.numel()


def finish_graph(stats):
    return {"closure": mode_parent.atom_parent.finish_pair(stats["closure"]), "direct_versus_five_write_parent": mode_parent.atom_parent.finish_pair(stats["direct_parent"]), "interaction_to_joint_effect_norm_ratio": math.sqrt(stats["interaction_sq"] / max(stats["joint_effect_sq"], 1e-30)), "rolled_to_interaction_norm_ratio": math.sqrt(stats["rolled_interaction_sq"] / max(stats["interaction_sq"], 1e-30)), "values": stats["values"]}


@torch.no_grad()
def graph_donor_forward(model, tokens, scale, basis, writers, variant):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    parts = {"E": x.float()}
    product4 = None
    mlp4 = None
    scale5 = None
    selected_score = None
    diagnostics = {}
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
                nodes = score_nodes(groups, product4, mlp4, basis, writers, scale5, block.attn, x.dtype)
                selected_score = nodes[variant]
                diagnostics = {"graph_score_relative_l2": float((nodes["composed"] - nodes["direct"]).norm() / nodes["direct"].norm().clamp_min(1e-30)), "interaction_norm": float(nodes["interaction"].norm()), "rolled_interaction_norm": float(torch.roll(nodes["interaction"], shifts=1, dims=1).norm())}
            if site == action_parent.factor_parent.TERMS[PAIR[1]][1]:
                late = factors[PAIR[1]]
                late_head = action_parent.factor_parent.TERMS[PAIR[1]][2]
                raw_payload, output_weight = edge_v2.raw_head_payload(attention_state, v1, block.attn, late_head)
                donor_term = edge_node.execute(selected_score.float() * scale["score_ratio"], raw_payload, support, output_weight)
                diagnostics["term_rms"] = float(donor_term.float().square().mean().sqrt())
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
    return logits, diagnostics


def effect_report(nll, masks):
    selections = {cell: masks[cell] for cell in ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")}
    selections["half_0"] = torch.zeros_like(masks["copy_positive"]); selections["half_0"][:96] = masks["copy_positive"][:96]
    selections["half_1"] = torch.zeros_like(masks["copy_positive"]); selections["half_1"][96:] = masks["copy_positive"][96:]
    output = {}
    for name, selected in selections.items():
        denominator = (nll["direct"] - nll["baseline"])[selected].double()
        removal = (nll["additive"] - nll["direct"])[selected].double()
        rolled = (nll["rolled"] - nll["direct"])[selected].double()
        replay = (nll["composed"] - nll["direct"])[selected].double()
        output[name] = {"interaction_removal_relative_l2": float(removal.norm() / denominator.norm().clamp_min(1e-30)), "interaction_removal_cosine_to_joint_effect": float((removal * denominator).sum() / (removal.norm() * denominator.norm()).clamp_min(1e-30)), "rolled_effect_cosine_to_removal": float((rolled * removal).sum() / (rolled.norm() * removal.norm()).clamp_min(1e-30)), "composed_replay_relative_l2": float(replay.norm() / denominator.norm().clamp_min(1e-30)), "tokens": int(selected.sum())}
    return output


def arm_recovery(nll, masks, arm):
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
    roles, scales, parent, contracted_parent, port_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = split_parent.compile_basis(model)

    natural_stats = empty_graph_stats()
    code_stats = empty_graph_stats()
    code_halves = [empty_graph_stats(), empty_graph_stats()]
    for role, stats in (("final_natural", natural_stats), ("ood_code", code_stats)):
        for start in range(0, DOCUMENTS, action_parent.BATCH):
            tokens = roles[role][start:start + action_parent.BATCH, :-1].cuda()
            residual, groups, attention, product, mlp4, scale5 = mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
            nodes = score_nodes(groups, product, mlp4, basis, writers, scale5, attention, residual.dtype)
            parent_score = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
            support = induction.induction_fetch_mask(tokens)
            accumulate_graph(stats, nodes, parent_score, support)
            if role == "ood_code":
                accumulate_graph(code_halves[0 if start < 96 else 1], nodes, parent_score, support)
    natural_report = finish_graph(natural_stats)
    code_report = finish_graph(code_stats)
    half_reports = [finish_graph(row) for row in code_halves]

    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    names = ("native", "absent", "baseline", "additive", "composed", "direct", "rolled")
    nll = {name: [] for name in names}
    max_forward_graph_error = 0.0
    max_equal_norm_error = 0.0
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        donor_logits = {}
        for variant in VARIANTS:
            donor_logits[variant], diagnostics = graph_donor_forward(model, tokens, scales["L5H5"], basis, writers, variant)
            max_forward_graph_error = max(max_forward_graph_error, diagnostics["graph_score_relative_l2"])
            max_equal_norm_error = max(max_equal_norm_error, abs(diagnostics["rolled_interaction_norm"] / max(diagnostics["interaction_norm"], 1e-30) - 1))
            minimum_term_rms = min(minimum_term_rms, diagnostics["term_rms"])
        targets = rows[:, 1:].cuda()
        logits_by_name = (native_logits, absent_logits, donor_logits["baseline"], donor_logits["additive"], donor_logits["composed"], donor_logits["direct"], donor_logits["rolled"])
        for name, logits in zip(names, logits_by_name):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    effects = effect_report(nll, masks)
    recoveries = {arm: arm_recovery(nll, masks, arm) for arm in ("baseline", "additive", "composed", "direct", "rolled")}
    noncopy_incremental = float((nll["additive"] - nll["direct"])[masks["all_noncopy"]].mean())

    pred_a = bool(bridge_error <= 2e-5 and natural_report["closure"]["relative_l2"] <= 2e-6 and code_report["closure"]["relative_l2"] <= 2e-6 and all(row["closure"]["relative_l2"] <= 2e-6 for row in half_reports) and max_forward_graph_error <= 2e-6)
    pred_b = bool(code_report["direct_versus_five_write_parent"]["relative_l2"] <= .15 and code_report["direct_versus_five_write_parent"]["cosine"] >= .98 and effects["copy_positive"]["composed_replay_relative_l2"] <= 2e-6)
    target_cells = ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors", "half_0", "half_1")
    pred_c = bool(all(effects[cell]["interaction_removal_relative_l2"] >= .20 for cell in target_cells) and abs(noncopy_incremental) <= .01)
    pred_d = bool(max_equal_norm_error <= 2e-6 and effects["copy_positive"]["rolled_effect_cosine_to_removal"] <= .90)
    pred_e = bool(.80 <= recoveries["direct"] <= 1.05 and .80 <= recoveries["composed"] <= 1.05 and abs(recoveries["direct"] - recoveries["composed"]) <= 2e-4 and minimum_term_rms > 0 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_explicit_interaction_graph": pred_a, "pred_b_ood_joint_prediction_and_replay": pred_b, "pred_c_selective_interaction_removal": pred_c, "pred_d_equal_norm_directional_null": pred_d, "pred_e_causal_composition_reuse": pred_e}
    terminal = "equality_l5h5_m4_explicit_interaction_graph_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_explicit_interaction_graph_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_explicit_interaction_graph_v1_result", "terminal": terminal, "predictions": predictions, "graph": {"nodes": ["bias_only_baseline", "rank128_child_effect", "rank128_remainder_effect", "child_remainder_interaction"], "child_rank": CUT, "remainder_rank": CUT, "learned_parameters": 0, "score_evaluations": 4}, "natural_score": natural_report, "code_score": {"overall": code_report, "halves": half_reports}, "behavioral_effects": effects, "behavior": {"recoveries": recoveries, "interaction_removal_noncopy_mean_nat": noncopy_incremental, "minimum_donor_term_rms": minimum_term_rms}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "maximum_forward_graph_score_relative_l2": max_forward_graph_error, "maximum_equal_norm_relative_error": max_equal_norm_error}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Explicit four-node Möbius graph for the bias baseline, rank-128 child, rank-128 remainder, and their contextual interaction inside the rank-256 M4 equality-score boundary."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "natural": natural_report, "code": code_report, "effects": effects, "recoveries": recoveries, "noncopy_incremental": noncopy_incremental, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
