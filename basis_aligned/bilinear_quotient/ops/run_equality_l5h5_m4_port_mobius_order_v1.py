#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;420 forwards;180seconds;M4 exact four-port Mobius order graph.
"""Test low-order Möbius graphs of the four native child Q/K corrections."""
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
import run_equality_l5h5_m4_shared_projection_kernel_v1 as kernel_parent
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_PORT_MOBIUS_ORDER_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_BEHAVIORAL_PROJECTION_SELECTION_V1_RESULT.json"
DOCUMENTS = 192
MAPS = selection_parent.MAPS
PAIR = mode_parent.PAIR
ORDERS = (0, 1, 2, 3)
CELLS = selection_parent.CELLS


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
    if parent["terminal"] != "valid_equality_l5h5_m4_natural_behavioral_projection_null" or not parent["predictions"]["pred_a_full_native_projection_positive_control"]:
        raise ValueError("behavior-selected projection null changed")
    roles, scales, _, explicit_parent, manifest = selection_parent.load_bound()
    return roles, scales, parent, explicit_parent, manifest


def plan():
    roles, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_port_mobius_order_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "proper_orders": list(ORDERS), "mobius_terms": 15, "positive_controls": 1, "execution_count": 420, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def submasks(mask):
    value = mask
    while True:
        yield value
        if value == 0:
            return
        value = (value - 1) & mask


def port_mobius_scores(residuals, attention):
    ports = {name: selection_parent.native_ports(attention, residuals[name]) for name in ("baseline", "remainder", "joint", "child")}
    denominators = {name: kernel_parent.rms_denominator(residuals[name]) for name in residuals}
    derived = []
    for index in range(4):
        value = ports["joint"][index].float() * (denominators["joint"] / denominators["child"])
        value = value - ports["remainder"][index].float() * (denominators["remainder"] / denominators["child"])
        value = value + ports["baseline"][index].float() * (denominators["baseline"] / denominators["child"])
        derived.append(value.to(residuals["child"].dtype))
    cos, sin = mode_parent.module_parent.parent.rotary_ports(attention, residuals["joint"])
    corner = {}
    for mask in range(16):
        raw = [ports["child"][index] if mask & (1 << index) else derived[index] for index in range(4)]
        corner[mask] = kernel_parent.score_from_raw(raw, cos, sin)
    terms = {}
    for mask in range(1, 16):
        value = torch.zeros_like(corner[0])
        for subset in submasks(mask):
            sign = -1 if (mask.bit_count() - subset.bit_count()) % 2 else 1
            value = value + sign * corner[subset]
        terms[mask] = value
    child_by_order = {0: corner[0]}
    running = corner[0]
    for order in range(1, 5):
        for mask, value in terms.items():
            if mask.bit_count() == order:
                running = running + value
        child_by_order[order] = running
    fixed = {name: kernel_parent.score_from_raw(ports[name], cos, sin) for name in ("baseline", "remainder", "joint")}
    outputs = {}
    for order, child in child_by_order.items():
        child_effect = child - fixed["baseline"]
        remainder_effect = fixed["remainder"] - fixed["baseline"]
        outputs[order] = {"additive": (fixed["baseline"] + child_effect) + remainder_effect, "child": child}
    reconstruction = child_by_order[4] - corner[15]
    return outputs, terms, corner[15] - corner[0], reconstruction, fixed


@torch.no_grad()
def donor_forward(model, tokens, scale, basis, writers, order, variant):
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
                scores, _, _, _, fixed = port_mobius_scores(residuals, block.attn)
                if variant == "baseline":
                    selected_score = fixed["baseline"]
                elif variant == "additive":
                    selected_score = scores[order]["additive"]
                else:
                    raise ValueError(variant)
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


def empty_census():
    return {"term_sq": {str(mask): 0.0 for mask in range(1, 16)}, "term_dot": {str(mask): 0.0 for mask in range(1, 16)}, "total_sq": 0.0, "cumulative_error_sq": {str(order): 0.0 for order in range(5)}, "reconstruction_sq": 0.0}


@torch.no_grad()
def score_census(model, rows, basis, writers):
    stats = empty_census()
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = rows[start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        residuals = kernel_parent.mode_residuals(groups, product, mlp4, basis, writers, scale5, residual.dtype)
        outputs, terms, total, reconstruction, _ = port_mobius_scores(residuals, attention)
        support = induction.induction_fetch_mask(tokens)
        total_values = total[support].double()
        stats["total_sq"] += float(total_values.square().sum())
        stats["reconstruction_sq"] += float(reconstruction[support].double().square().sum())
        for mask, value in terms.items():
            values = value[support].double()
            stats["term_sq"][str(mask)] += float(values.square().sum())
            stats["term_dot"][str(mask)] += float((values * total_values).sum())
        base = outputs[0]["child"]
        full = outputs[4]["child"]
        for order in range(5):
            error = (outputs[order]["child"] - full)[support].double()
            stats["cumulative_error_sq"][str(order)] += float(error.square().sum())
    denominator = max(stats["total_sq"], 1e-30)
    terms_report = {}
    for mask in range(1, 16):
        square = stats["term_sq"][str(mask)]
        terms_report["*".join(MAPS[index] for index in range(4) if mask & (1 << index))] = {"order": mask.bit_count(), "relative_norm": math.sqrt(square / denominator), "cosine_to_total": stats["term_dot"][str(mask)] / max(math.sqrt(square * denominator), 1e-30)}
    return {"terms": terms_report, "cumulative_relative_l2": {str(order): math.sqrt(value / denominator) for order, value in ((int(key), val) for key, val in stats["cumulative_error_sq"].items())}, "reconstruction_relative_l2": math.sqrt(stats["reconstruction_sq"] / denominator)}


def collect_behavior(model, rows, scale, basis, writers, orders):
    arms = ["authority_additive", "authority_direct", "baseline", "control_additive"] + [f"order_{order}_additive" for order in orders]
    nll = {arm: [] for arm in arms}
    minimum_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch = rows[start:start + action_parent.BATCH]
        tokens = batch[:, :-1].cuda()
        values = {}
        values["authority_additive"], diag_add = graph_parent.graph_donor_forward(model, tokens, scale, basis, writers, "additive")
        values["authority_direct"], diag_direct = graph_parent.graph_donor_forward(model, tokens, scale, basis, writers, "direct")
        values["baseline"], rms_b = donor_forward(model, tokens, scale, basis, writers, 0, "baseline")
        values["control_additive"], rms_f = donor_forward(model, tokens, scale, basis, writers, 4, "additive")
        minimum_rms = min(minimum_rms, diag_add["term_rms"], diag_direct["term_rms"], rms_b, rms_f)
        for order in orders:
            values[f"order_{order}_additive"], rms = donor_forward(model, tokens, scale, basis, writers, order, "additive")
            minimum_rms = min(minimum_rms, rms)
        targets = batch[:, 1:].cuda()
        for arm in arms:
            logits = values[arm]
            nll[arm].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch), -1).cpu())
    return {arm: torch.cat(parts) for arm, parts in nll.items()}, minimum_rms


def removal_report(nll, masks, prefix):
    output = {}
    for cell in CELLS:
        selected = masks[cell]
        denominator = (nll["authority_direct"] - nll["baseline"])[selected].double()
        authority = (nll["authority_additive"] - nll["authority_direct"])[selected].double()
        candidate = (nll[f"{prefix}_additive"] - nll["authority_direct"])[selected].double()
        output[cell] = {"removal_relative_to_joint": float(candidate.norm() / denominator.norm().clamp_min(1e-30)), "removal_relative_l2": float((candidate - authority).norm() / authority.norm().clamp_min(1e-30)), "removal_cosine": float((candidate * authority).sum() / (candidate.norm() * authority.norm()).clamp_min(1e-30)), "tokens": int(selected.sum())}
    return output


def passes(report, error=.20, cosine=.90):
    return all(report[cell]["removal_relative_l2"] <= error and report[cell]["removal_cosine"] >= cosine for cell in CELLS)


def rank_key(order, report):
    return (order, max(report[cell]["removal_relative_l2"] for cell in CELLS), -min(report[cell]["removal_cosine"] for cell in CELLS))


def diagnostic_key(order, report):
    return (max(report[cell]["removal_relative_l2"] for cell in CELLS), -min(report[cell]["removal_cosine"] for cell in CELLS), order)


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
    census = {role: score_census(model, roles[source], basis, writers) for role, source in (("natural", "final_natural"), ("code", "ood_code"))}

    natural_masks = selection_parent.masks_for(roles["final_natural"])
    natural_nll, natural_rms = collect_behavior(model, roles["final_natural"], scales["L5H5"], basis, writers, ORDERS)
    natural_reports = {str(order): removal_report(natural_nll, natural_masks, f"order_{order}") for order in ORDERS}
    natural_control = removal_report(natural_nll, natural_masks, "control")
    passing = [order for order in ORDERS if passes(natural_reports[str(order)])]
    selected = min(passing, key=lambda order: rank_key(order, natural_reports[str(order)])) if passing else min(ORDERS, key=lambda order: diagnostic_key(order, natural_reports[str(order)]))

    code_masks = selection_parent.masks_for(roles["ood_code"])
    code_nll, code_rms = collect_behavior(model, roles["ood_code"], scales["L5H5"], basis, writers, (selected,))
    code_report = removal_report(code_nll, code_masks, f"order_{selected}")
    code_control = removal_report(code_nll, code_masks, "control")
    noncopy = float((code_nll[f"order_{selected}_additive"] - code_nll["authority_direct"])[code_masks["all_noncopy"]].mean())
    term_count = sum(math.comb(4, order) for order in range(1, selected + 1))
    control_ok = all(passes(report, error=2e-5, cosine=.99999) for report in (natural_control, code_control))
    pred_a = bool(explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5 and control_ok and all(census[role]["reconstruction_relative_l2"] <= 2e-6 for role in census))
    pred_b = bool(passing and passes(natural_reports[str(selected)]))
    pred_c = bool(passes(code_report))
    pred_d = bool(all(code_report[cell]["removal_relative_to_joint"] >= .20 for cell in CELLS) and abs(noncopy) <= .01 and min(natural_rms, code_rms) > 0)
    pred_e = bool(term_count < 15 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_exact_mobius_and_behavior_control": pred_a, "pred_b_natural_low_order_graph": pred_b, "pred_c_ood_removal_fidelity": pred_c, "pred_d_selective_removal": pred_d, "pred_e_sparse_term_price": pred_e}
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "equality_l5h5_m4_low_order_port_mobius_graph_ood"
    elif not pred_b:
        terminal = "valid_equality_l5h5_m4_port_mobius_low_order_null"
    else:
        terminal = "valid_equality_l5h5_m4_port_mobius_order_ood_null"
    result = {"schema": "equality_l5h5_m4_port_mobius_order_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"maximum_order": selected, "nonconstant_terms": term_count, "natural_pass_count": len(passing), "natural_report": natural_reports[str(selected)]}, "natural_frontier": natural_reports, "score_census": census, "positive_control": {"natural": natural_control, "code": code_control}, "code": {"report": code_report, "interaction_removal_noncopy_mean_nat": noncopy}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "minimum_donor_term_rms": min(natural_rms, code_rms), "parent_best_proper_subset": parent["selection"]["maps"]}, "program_price": {"native_qk_projections": 16, "selected_nonconstant_mobius_terms": term_count, "full_nonconstant_mobius_terms": 15, "learned_parameters": 0}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Exact Boolean Möbius decomposition of four native child Q/K port corrections, with natural-selected cumulative interaction order frozen for code OOD behavioral removal."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selection": result["selection"], "score_census": census, "code": result["code"], "positive_control": result["positive_control"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
