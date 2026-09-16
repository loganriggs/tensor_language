#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;288 forwards;180seconds;M4 exact two-QK-factor correction graph.
"""Factor the four-port child correction into two QK effects and their cross."""
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
import run_equality_l5h5_m4_behavioral_projection_selection_v1 as selection_parent
import run_equality_l5h5_m4_contracted_mode_graph_v1 as mode_parent
import run_equality_l5h5_m4_explicit_interaction_graph_v1 as graph_parent
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
import run_equality_l5h5_m4_port_mobius_order_v1 as mobius_parent
import run_equality_l5h5_m4_shared_projection_kernel_v1 as kernel_parent
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_PORT_MOBIUS_ORDER_V1_RESULT.json"
DOCUMENTS = 192
PAIR = mode_parent.PAIR
CELLS = selection_parent.CELLS
NODES = ("first", "second", "cross")
VARIANTS = ("full", "base", "drop_first", "drop_second", "drop_cross", "rolled_cross")


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
    if parent["terminal"] != "valid_equality_l5h5_m4_port_mobius_low_order_null" or not parent["predictions"]["pred_a_exact_mobius_and_behavior_control"]:
        raise ValueError("port Möbius authority changed")
    roles, scales, _, explicit_parent, manifest = mobius_parent.load_bound()
    return roles, scales, parent, explicit_parent, manifest


def plan():
    roles, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_two_factor_correction_graph_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "macro_nodes": list(NODES), "variants": list(VARIANTS), "execution_count": 288, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def dot_score(query, key):
    return torch.einsum("bqd,bkd->bqk", query, key) / query.shape[-1]


def factor_scores(residuals, attention):
    ports = {name: selection_parent.native_ports(attention, residuals[name]) for name in ("baseline", "remainder", "joint", "child")}
    denominators = {name: kernel_parent.rms_denominator(residuals[name]) for name in residuals}
    derived = []
    for index in range(4):
        value = ports["joint"][index].float() * (denominators["joint"] / denominators["child"])
        value = value - ports["remainder"][index].float() * (denominators["remainder"] / denominators["child"])
        value = value + ports["baseline"][index].float() * (denominators["baseline"] / denominators["child"])
        derived.append(value.to(residuals["child"].dtype))
    cos, sin = mode_parent.module_parent.parent.rotary_ports(attention, residuals["joint"])
    transform = lambda raw: kernel_parent.rotate(F.rms_norm(raw, (raw.shape[-1],)), cos, sin)
    derived = [transform(value) for value in derived]
    native = [transform(value) for value in ports["child"]]
    a0, a1 = dot_score(derived[0], derived[1]), dot_score(native[0], native[1])
    b0, b1 = dot_score(derived[2], derived[3]), dot_score(native[2], native[3])
    nodes = {"base": a0 * b0, "first": (a1 - a0) * b0, "second": a0 * (b1 - b0), "cross": (a1 - a0) * (b1 - b0)}
    causal = torch.ones(nodes["base"].shape[-2:], dtype=torch.bool, device=nodes["base"].device).tril()
    nodes = {name: value.masked_fill(~causal, 0).float() for name, value in nodes.items()}
    child = {
        "full": ((nodes["base"] + nodes["first"]) + nodes["second"]) + nodes["cross"],
        "base": nodes["base"],
        "drop_first": (nodes["base"] + nodes["second"]) + nodes["cross"],
        "drop_second": (nodes["base"] + nodes["first"]) + nodes["cross"],
        "drop_cross": (nodes["base"] + nodes["first"]) + nodes["second"],
        "rolled_cross": ((nodes["base"] + nodes["first"]) + nodes["second"]) + torch.roll(nodes["cross"], shifts=1, dims=1),
    }
    fixed = {name: kernel_parent.score_from_raw(ports[name], cos, sin) for name in ("baseline", "remainder", "joint")}
    additive = {name: (fixed["baseline"] + (value - fixed["baseline"])) + (fixed["remainder"] - fixed["baseline"]) for name, value in child.items()}
    direct_child = kernel_parent.score_from_raw(ports["child"], cos, sin)
    return additive, nodes, child["full"] - direct_child, fixed


@torch.no_grad()
def donor_forward(model, tokens, scale, basis, writers, variant):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    parts = {"E": x.float()}
    product4 = None
    mlp4 = None
    scale5 = None
    selected_score = None
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
                groups = {name: parts[name] for name in mode_parent.module_parent.GROUPS}
                residuals = kernel_parent.mode_residuals(groups, product4, mlp4, basis, writers, scale5, x.dtype)
                additive, _, _, fixed = factor_scores(residuals, block.attn)
                selected_score = fixed["baseline"] if variant == "baseline" else additive[variant]
            if site == action_parent.factor_parent.TERMS[PAIR[1]][1]:
                late = factors[PAIR[1]]
                late_head = action_parent.factor_parent.TERMS[PAIR[1]][2]
                raw_payload, output_weight = selection_parent.behavior_parent.edge_v2.raw_head_payload(attention_state, v1, block.attn, late_head)
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


@torch.no_grad()
def score_census(model, rows, basis, writers):
    node_sq = {name: 0.0 for name in NODES}
    total_sq = 0.0
    closure_sq = 0.0
    cross_roll_sq = 0.0
    values = 0
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = rows[start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        residuals = kernel_parent.mode_residuals(groups, product, mlp4, basis, writers, scale5, residual.dtype)
        _, nodes, closure, _ = factor_scores(residuals, attention)
        support = induction.induction_fetch_mask(tokens)
        total = sum((nodes[name] for name in NODES), torch.zeros_like(nodes["base"]))[support].double()
        total_sq += float(total.square().sum())
        closure_sq += float(closure[support].double().square().sum())
        for name in NODES:
            node_sq[name] += float(nodes[name][support].double().square().sum())
        cross_roll_sq += float(torch.roll(nodes["cross"], shifts=1, dims=1)[support].double().square().sum())
        values += total.numel()
    denominator = max(total_sq, 1e-30)
    return {"node_relative_norm": {name: math.sqrt(value / denominator) for name, value in node_sq.items()}, "closure_relative_l2": math.sqrt(closure_sq / denominator), "rolled_cross_to_cross_norm_ratio": math.sqrt(cross_roll_sq / max(node_sq["cross"], 1e-30)), "values": values}


def collect(model, rows, scale, basis, writers):
    arms = ["authority_additive", "authority_direct", "baseline"] + list(VARIANTS)
    nll = {arm: [] for arm in arms}
    minimum_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch = rows[start:start + action_parent.BATCH]
        tokens = batch[:, :-1].cuda()
        values = {}
        values["authority_additive"], diag_a = graph_parent.graph_donor_forward(model, tokens, scale, basis, writers, "additive")
        values["authority_direct"], diag_d = graph_parent.graph_donor_forward(model, tokens, scale, basis, writers, "direct")
        values["baseline"], rms_b = donor_forward(model, tokens, scale, basis, writers, "baseline")
        minimum_rms = min(minimum_rms, diag_a["term_rms"], diag_d["term_rms"], rms_b)
        for variant in VARIANTS:
            values[variant], rms = donor_forward(model, tokens, scale, basis, writers, variant)
            minimum_rms = min(minimum_rms, rms)
        targets = batch[:, 1:].cuda()
        for arm in arms:
            logits = values[arm]
            nll[arm].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch), -1).cpu())
    return {arm: torch.cat(parts) for arm, parts in nll.items()}, minimum_rms


def reports(nll, masks):
    output = {name: {} for name in ("control",) + NODES + ("rolled_cross",)}
    node_variant = {"first": "drop_first", "second": "drop_second", "cross": "drop_cross"}
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
        rolled = (nll["rolled_cross"] - nll["full"])[selected].double()
        true = effects["cross"]
        output["rolled_cross"][cell] = {"relative_to_interaction_removal": float(rolled.norm() / denominator.norm().clamp_min(1e-30)), "cosine_to_true_cross_removal": float((rolled * true).sum() / (rolled.norm() * true.norm()).clamp_min(1e-30)), "tokens": int(selected.sum())}
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
    roles, scales, parent, explicit_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = split_parent.compile_basis(model)
    census = {label: score_census(model, roles[role], basis, writers) for label, role in (("natural", "final_natural"), ("code", "ood_code"))}
    behavior = {}
    noncopy = {}
    minimum_rms = math.inf
    for label, role in (("natural", "final_natural"), ("code", "ood_code")):
        masks = selection_parent.masks_for(roles[role])
        nll, rms = collect(model, roles[role], scales["L5H5"], basis, writers)
        minimum_rms = min(minimum_rms, rms)
        behavior[label] = reports(nll, masks)
        noncopy[label] = {name: float((nll[f"drop_{name}"] - nll["full"])[masks["all_noncopy"]].mean()) for name in NODES}
    pred_a = bool(explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5 and all(census[label]["closure_relative_l2"] <= 2e-6 for label in census) and all(behavior[label]["control"][cell]["relative_l2"] <= 2e-5 and behavior[label]["control"][cell]["cosine"] >= .99999 for label in behavior for cell in CELLS))
    pred_b = bool(all(behavior[label][name]["copy_positive"]["relative_to_interaction_removal"] >= .02 and behavior[label][name]["half_0"]["relative_to_interaction_removal"] > 0 and behavior[label][name]["half_1"]["relative_to_interaction_removal"] > 0 for label in behavior for name in NODES))
    pred_c = bool(all(abs(noncopy[label][name]) <= .01 for label in noncopy for name in NODES))
    pred_d = bool(census["code"]["rolled_cross_to_cross_norm_ratio"] >= .99 and census["code"]["rolled_cross_to_cross_norm_ratio"] <= 1.01 and behavior["code"]["rolled_cross"]["copy_positive"]["cosine_to_true_cross_removal"] <= .80)
    pred_e = bool(len(NODES) == 3 and manifest["learned_parameters"] == 0 and minimum_rms > 0)
    predictions = {"pred_a_exact_two_factor_graph": pred_a, "pred_b_all_macro_nodes_behaviorally_active": pred_b, "pred_c_selective_node_removals": pred_c, "pred_d_cross_node_direction_specificity": pred_d, "pred_e_sparse_graph_price": pred_e}
    terminal = "equality_l5h5_m4_two_factor_correction_graph_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_two_factor_graph_node_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_two_factor_correction_graph_v1_result", "terminal": terminal, "predictions": predictions, "graph": {"baseline": "derived_child_score", "nodes": ["first_qk_correction_times_second_baseline", "first_baseline_times_second_qk_correction", "qk_correction_cross_product"], "native_qk_projections": 16, "learned_parameters": 0}, "score_census": census, "behavior": behavior, "noncopy_incremental_mean_nat": noncopy, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "minimum_donor_term_rms": minimum_rms, "parent_four_way_natural_relative_norm": parent["score_census"]["natural"]["terms"]["Q1*K1*Q2*K2"]["relative_norm"]}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Exact three-node factor graph of native-versus-derived child score corrections across the two multiplicative L5H5 QK factors, with OOD node removal and rolled-cross specificity."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "score_census": census, "code_behavior": behavior["code"], "noncopy": noncopy, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
