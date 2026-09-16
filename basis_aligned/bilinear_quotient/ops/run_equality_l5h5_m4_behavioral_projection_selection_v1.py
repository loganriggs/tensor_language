#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;900 forwards;180seconds;M4 behavior-selected native projection corrections.
"""Select shared-kernel projection corrections by causal removal fidelity."""
import itertools
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
import equality_matcher_causal_action_quotient_rung498 as action_parent
import rung498_copy_task_portability_diagnosis as diagnosis
import run_equality_l5h5_m4_contracted_mode_graph_v1 as mode_parent
import run_equality_l5h5_m4_explicit_interaction_graph_v1 as graph_parent
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
import run_equality_l5h5_m4_shared_kernel_behavior_v1 as behavior_parent
import run_equality_l5h5_m4_shared_projection_kernel_v1 as kernel_parent
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_BEHAVIORAL_PROJECTION_SELECTION_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_SHARED_KERNEL_BEHAVIOR_V1_RESULT.json"
DOCUMENTS = 192
MAPS = ("Q1", "K1", "Q2", "K2")
PAIR = mode_parent.PAIR
PROPER_SUBSETS = tuple(subset for size in range(4) for subset in itertools.combinations(range(4), size))
FULL = tuple(range(4))
CELLS = ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors", "half_0", "half_1")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def subset_name(subset):
    return "none" if not subset else "+".join(MAPS[index] for index in subset)


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    parent = json.loads(PARENT_RESULT.read_text())
    if parent["terminal"] != "valid_equality_l5h5_m4_shared_kernel_behavior_null":
        raise ValueError("score-selected behavioral null changed")
    if parent["predictions"]["pred_b_ood_behavioral_replay"] or parent["predictions"]["pred_c_interaction_removal_fidelity"]:
        raise ValueError("behavioral-null predicates changed")
    roles, scales, _, explicit_parent, manifest = behavior_parent.load_bound()
    return roles, scales, parent, explicit_parent, manifest


def plan():
    roles, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_behavioral_projection_selection_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "proper_subsets": len(PROPER_SUBSETS), "positive_controls": 1, "maximum_qk_projections": 15, "execution_count": 900, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def native_ports(attention, residual):
    width = action_parent.factor_parent.stage1.HEAD_DIM
    head = mode_parent.module_parent.parent.HEAD
    lo, hi = head * width, (head + 1) * width
    state = F.rms_norm(residual, (residual.shape[-1],))
    return [layer(state)[..., lo:hi] for layer in (attention.c_q, attention.c_k, attention.c_q2, attention.c_k2)]


def native_corner_scores(residuals, attention, subset):
    ports = {name: native_ports(attention, residuals[name]) for name in ("baseline", "remainder", "joint")}
    direct_child = native_ports(attention, residuals["child"]) if subset else None
    denominators = {name: kernel_parent.rms_denominator(residuals[name]) for name in residuals}
    child_ports = []
    for index in range(4):
        if index in subset:
            child_ports.append(direct_child[index])
        else:
            scale_joint = denominators["joint"] / denominators["child"]
            scale_remainder = denominators["remainder"] / denominators["child"]
            scale_baseline = denominators["baseline"] / denominators["child"]
            derived = ports["joint"][index].float() * scale_joint
            derived = derived - ports["remainder"][index].float() * scale_remainder
            derived = derived + ports["baseline"][index].float() * scale_baseline
            child_ports.append(derived.to(residuals["child"].dtype))
    ports["child"] = child_ports
    cos, sin = mode_parent.module_parent.parent.rotary_ports(attention, residuals["joint"])
    scores = {name: kernel_parent.score_from_raw(raw, cos, sin) for name, raw in ports.items()}
    scores["interaction"] = scores["joint"] - scores["child"] - scores["remainder"] + scores["baseline"]
    child_effect = scores["child"] - scores["baseline"]
    remainder_effect = scores["remainder"] - scores["baseline"]
    scores["additive"] = (scores["baseline"] + child_effect) + remainder_effect
    scores["composed"] = scores["additive"] + scores["interaction"]
    return scores


@torch.no_grad()
def donor_forward(model, tokens, scale, basis, writers, subset, variant):
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
                scores = native_corner_scores(residuals, block.attn, subset)
                selected_score = scores[variant]
            if site == action_parent.factor_parent.TERMS[PAIR[1]][1]:
                late = factors[PAIR[1]]
                late_head = action_parent.factor_parent.TERMS[PAIR[1]][2]
                raw_payload, output_weight = behavior_parent.edge_v2.raw_head_payload(attention_state, v1, block.attn, late_head)
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


def masks_for(rows):
    masks = diagnosis.build_masks(rows)
    masks["half_0"] = torch.zeros_like(masks["copy_positive"])
    masks["half_0"][:DOCUMENTS // 2] = masks["copy_positive"][:DOCUMENTS // 2]
    masks["half_1"] = torch.zeros_like(masks["copy_positive"])
    masks["half_1"][DOCUMENTS // 2:] = masks["copy_positive"][DOCUMENTS // 2:]
    valid = torch.zeros_like(masks["copy_positive"])
    valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    return masks


def effect_report(nll, masks, prefix):
    output = {}
    for cell in CELLS:
        selected = masks[cell]
        denominator = (nll["authority_direct"] - nll["baseline"])[selected].double()
        authority_removal = (nll["authority_additive"] - nll["authority_direct"])[selected].double()
        candidate_removal = (nll[f"{prefix}_additive"] - nll[f"{prefix}_composed"])[selected].double()
        replay = (nll[f"{prefix}_composed"] - nll["authority_direct"])[selected].double()
        output[cell] = {
            "replay_relative_l2": float(replay.norm() / denominator.norm().clamp_min(1e-30)),
            "removal_relative_to_joint": float(candidate_removal.norm() / denominator.norm().clamp_min(1e-30)),
            "removal_relative_l2": float((candidate_removal - authority_removal).norm() / authority_removal.norm().clamp_min(1e-30)),
            "removal_cosine": float((candidate_removal * authority_removal).sum() / (candidate_removal.norm() * authority_removal.norm()).clamp_min(1e-30)),
            "tokens": int(selected.sum()),
        }
    return output


def passes(report, replay=.05, removal=.20, cosine=.90):
    return all(report[cell]["replay_relative_l2"] <= replay and report[cell]["removal_relative_l2"] <= removal and report[cell]["removal_cosine"] >= cosine for cell in CELLS)


def control_passes(report):
    return passes(report, replay=2e-5, removal=2e-5, cosine=.99999)


def rank_key(subset, report):
    return (len(subset), max(report[cell]["removal_relative_l2"] for cell in CELLS), max(report[cell]["replay_relative_l2"] for cell in CELLS), subset_name(subset))


def diagnostic_key(subset, report):
    return (max(report[cell]["removal_relative_l2"] for cell in CELLS), max(report[cell]["replay_relative_l2"] for cell in CELLS), -min(report[cell]["removal_cosine"] for cell in CELLS), len(subset), subset_name(subset))


def collect_role(model, rows, scale, basis, writers, subsets, include_recovery):
    arms = ["authority_additive", "authority_direct", "baseline", "control_additive", "control_composed"]
    for subset in subsets:
        name = subset_name(subset)
        arms.extend((f"{name}_additive", f"{name}_composed"))
    if include_recovery:
        arms.extend(("native", "absent"))
    nll = {arm: [] for arm in arms}
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch = rows[start:start + action_parent.BATCH]
        tokens = batch[:, :-1].cuda()
        values = {}
        values["authority_additive"], diag_add = graph_parent.graph_donor_forward(model, tokens, scale, basis, writers, "additive")
        values["authority_direct"], diag_direct = graph_parent.graph_donor_forward(model, tokens, scale, basis, writers, "direct")
        values["baseline"], rms = donor_forward(model, tokens, scale, basis, writers, (), "baseline")
        values["control_additive"], rms_ca = donor_forward(model, tokens, scale, basis, writers, FULL, "additive")
        values["control_composed"], rms_cc = donor_forward(model, tokens, scale, basis, writers, FULL, "composed")
        minimum_term_rms = min(minimum_term_rms, diag_add["term_rms"], diag_direct["term_rms"], rms, rms_ca, rms_cc)
        for subset in subsets:
            name = subset_name(subset)
            values[f"{name}_additive"], rms_a = donor_forward(model, tokens, scale, basis, writers, subset, "additive")
            values[f"{name}_composed"], rms_c = donor_forward(model, tokens, scale, basis, writers, subset, "composed")
            minimum_term_rms = min(minimum_term_rms, rms_a, rms_c)
        if include_recovery:
            values["native"], _, _ = action_parent.run_forward(model, tokens, direct=True)
            values["absent"], _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scale)
        targets = batch[:, 1:].cuda()
        for arm in arms:
            logits = values[arm]
            nll[arm].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch), -1).cpu())
    return {arm: torch.cat(parts) for arm, parts in nll.items()}, minimum_term_rms


def recovery(nll, masks, arm):
    selected = masks["copy_positive"]
    return float(((nll["absent"] - nll[arm])[selected].sum()) / ((nll["absent"] - nll["native"])[selected].sum()))


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

    natural_masks = masks_for(roles["final_natural"])
    natural_nll, natural_minimum_rms = collect_role(model, roles["final_natural"], scales["L5H5"], basis, writers, PROPER_SUBSETS, False)
    natural_reports = {subset_name(subset): effect_report(natural_nll, natural_masks, subset_name(subset)) for subset in PROPER_SUBSETS}
    natural_control = effect_report(natural_nll, natural_masks, "control")
    passing = [subset for subset in PROPER_SUBSETS if passes(natural_reports[subset_name(subset)])]
    selected = min(passing, key=lambda subset: rank_key(subset, natural_reports[subset_name(subset)])) if passing else min(PROPER_SUBSETS, key=lambda subset: diagnostic_key(subset, natural_reports[subset_name(subset)]))
    selected_name = subset_name(selected)

    code_masks = masks_for(roles["ood_code"])
    code_nll, code_minimum_rms = collect_role(model, roles["ood_code"], scales["L5H5"], basis, writers, (selected,), True)
    code_report = effect_report(code_nll, code_masks, selected_name)
    code_control = effect_report(code_nll, code_masks, "control")
    authority_recovery = recovery(code_nll, code_masks, "authority_direct")
    selected_recovery = recovery(code_nll, code_masks, f"{selected_name}_composed")
    selected_noncopy = float((code_nll[f"{selected_name}_additive"] - code_nll[f"{selected_name}_composed"])[code_masks["all_noncopy"]].mean())
    minimum_term_rms = min(natural_minimum_rms, code_minimum_rms)

    pred_a = bool(explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5 and control_passes(natural_control) and control_passes(code_control))
    pred_b = bool(passing and passes(natural_reports[selected_name]))
    pred_c = bool(passes(code_report))
    pred_d = bool(all(code_report[cell]["removal_relative_to_joint"] >= .20 for cell in CELLS) and abs(selected_noncopy) <= .01 and .80 <= selected_recovery <= 1.05 and abs(selected_recovery - authority_recovery) <= .02 and minimum_term_rms > 0)
    pred_e = bool(12 + len(selected) <= 15 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_full_native_projection_positive_control": pred_a, "pred_b_natural_behavioral_subset": pred_b, "pred_c_ood_behavioral_replay_and_removal": pred_c, "pred_d_selective_causal_use": pred_d, "pred_e_projection_price": pred_e}
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "equality_l5h5_m4_behavior_selected_projection_kernel_ood"
    elif not pred_b:
        terminal = "valid_equality_l5h5_m4_natural_behavioral_projection_null"
    else:
        terminal = "valid_equality_l5h5_m4_ood_behavioral_projection_null"
    result = {
        "schema": "equality_l5h5_m4_behavioral_projection_selection_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "selection": {"maps": [MAPS[index] for index in selected], "proper_subset": True, "natural_pass_count": len(passing), "natural_report": natural_reports[selected_name]},
        "natural_frontier": natural_reports,
        "positive_control": {"natural": natural_control, "code": code_control},
        "code": {"report": code_report, "selected_recovery": selected_recovery, "authority_recovery": authority_recovery, "interaction_removal_noncopy_mean_nat": selected_noncopy},
        "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "minimum_donor_term_rms": minimum_term_rms, "parent_score_selected_behavioral_replay_error": parent["effects"]["copy_positive"]["shared_replay_relative_l2"]},
        "program_price": {"authority_qk_projections": 16, "selected_qk_projections": 12 + len(selected), "reduction_fraction": (4 - len(selected)) / 16, "learned_parameters": 0},
        "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True},
        "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Natural behavior-selected proper subset of native child Q/K projections for the M4 interaction kernel, frozen for code OOD with a full native-projection implementation control.",
    }
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selection": result["selection"], "code": result["code"], "positive_control": result["positive_control"], "program_price": result["program_price"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
