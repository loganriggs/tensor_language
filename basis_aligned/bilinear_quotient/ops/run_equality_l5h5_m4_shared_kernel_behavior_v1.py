#!/usr/bin/env python3
# BQGATE:192 frozen code documents;336 forwards;180seconds;M4 corrected shared-kernel behavior.
"""Behavioral validation of the thirteen-projection explicit interaction kernel."""
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
import run_equality_l5h5_m4_shared_projection_correction_v1 as correction_parent
import run_equality_l5h5_m4_shared_projection_kernel_v1 as kernel_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
import run_equality_l8h4_exact_order_node_code_ood_v2 as edge_v2
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_SHARED_KERNEL_BEHAVIOR_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_SHARED_PROJECTION_CORRECTION_V1_RESULT.json"
DOCUMENTS = 192
PAIR = mode_parent.PAIR
CORRECTION_INDEX = 3
ARMS = ("native", "absent", "authority_additive", "authority_direct", "shared_baseline", "shared_additive", "shared_composed")


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
    if parent["terminal"] != "equality_l5h5_m4_shared_projection_one_map_correction" or not all(parent["predictions"].values()) or parent["selection"]["map"] != "K2":
        raise ValueError("corrected shared-kernel parent changed")
    roles, gauge_result, explicit_parent, manifest = correction_parent.load_bound()
    _, scales, _, _, _, _ = graph_parent.load_bound()
    return roles, scales, parent, explicit_parent, manifest


def plan():
    roles, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_shared_kernel_behavior_v1_plan", "ood_documents": len(roles["ood_code"]), "execution_count": 336, "complete_code_arms": len(ARMS), "selected_qk_projections": 13, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def shared_donor_forward(model, tokens, scale, basis, writers, variant):
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
                weights = residual_parent.head_weights(block.attn)
                cos, sin = residual_parent.rotary_ports(block.attn, x)
                scores, _ = correction_parent.corrected_scores(residuals, weights, cos, sin, CORRECTION_INDEX)
                child_effect = scores["child"] - scores["baseline"]
                remainder_effect = scores["remainder"] - scores["baseline"]
                additive = (scores["baseline"] + child_effect) + remainder_effect
                interaction = scores["joint"] - scores["child"] - scores["remainder"] + scores["baseline"]
                choices = {"baseline": scores["baseline"], "additive": additive, "composed": additive + interaction}
                selected_score = choices[variant]
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


def reports(nll, masks):
    selections = {cell: masks[cell] for cell in ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")}
    selections["half_0"] = torch.zeros_like(masks["copy_positive"]); selections["half_0"][:96] = masks["copy_positive"][:96]
    selections["half_1"] = torch.zeros_like(masks["copy_positive"]); selections["half_1"][96:] = masks["copy_positive"][96:]
    output = {}
    for name, selected in selections.items():
        denominator = (nll["authority_direct"] - nll["shared_baseline"])[selected].double()
        authority_removal = (nll["authority_additive"] - nll["authority_direct"])[selected].double()
        shared_removal = (nll["shared_additive"] - nll["shared_composed"])[selected].double()
        replay = (nll["shared_composed"] - nll["authority_direct"])[selected].double()
        output[name] = {"shared_replay_relative_l2": float(replay.norm() / denominator.norm().clamp_min(1e-30)), "shared_removal_relative_to_joint": float(shared_removal.norm() / denominator.norm().clamp_min(1e-30)), "shared_vs_authority_removal_relative_l2": float((shared_removal - authority_removal).norm() / authority_removal.norm().clamp_min(1e-30)), "shared_vs_authority_removal_cosine": float((shared_removal * authority_removal).sum() / (shared_removal.norm() * authority_removal.norm()).clamp_min(1e-30)), "tokens": int(selected.sum())}
    return output


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
    masks = diagnosis.build_masks(roles["ood_code"])
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    nll = {arm: [] for arm in ARMS}
    minimum_term_rms = math.inf
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        rows = roles["ood_code"][start:start + action_parent.BATCH]
        tokens = rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        authority_additive, diag_add = graph_parent.graph_donor_forward(model, tokens, scales["L5H5"], basis, writers, "additive")
        authority_direct, diag_direct = graph_parent.graph_donor_forward(model, tokens, scales["L5H5"], basis, writers, "direct")
        shared_baseline, rms_baseline = shared_donor_forward(model, tokens, scales["L5H5"], basis, writers, "baseline")
        shared_additive, rms_additive = shared_donor_forward(model, tokens, scales["L5H5"], basis, writers, "additive")
        shared_composed, rms_composed = shared_donor_forward(model, tokens, scales["L5H5"], basis, writers, "composed")
        minimum_term_rms = min(minimum_term_rms, diag_add["term_rms"], diag_direct["term_rms"], rms_baseline, rms_additive, rms_composed)
        targets = rows[:, 1:].cuda()
        logits = (native_logits, absent_logits, authority_additive, authority_direct, shared_baseline, shared_additive, shared_composed)
        for arm, values in zip(ARMS, logits):
            nll[arm].append(F.cross_entropy(values.reshape(-1, values.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(rows), -1).cpu())
    nll = {arm: torch.cat(parts) for arm, parts in nll.items()}
    effect_reports = reports(nll, masks)
    recoveries = {arm: recovery(nll, masks, arm) for arm in ("authority_direct", "shared_additive", "shared_composed")}
    noncopy_incremental = float((nll["shared_additive"] - nll["shared_composed"])[masks["all_noncopy"]].mean())
    cells = tuple(effect_reports)
    pred_a = bool(all(parent["predictions"].values()) and explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5)
    pred_b = bool(all(effect_reports[cell]["shared_replay_relative_l2"] <= .05 for cell in cells))
    pred_c = bool(all(effect_reports[cell]["shared_vs_authority_removal_relative_l2"] <= .20 and effect_reports[cell]["shared_vs_authority_removal_cosine"] >= .90 for cell in cells))
    pred_d = bool(all(effect_reports[cell]["shared_removal_relative_to_joint"] >= .20 for cell in cells) and abs(noncopy_incremental) <= .01)
    pred_e = bool(.80 <= recoveries["shared_composed"] <= 1.05 and abs(recoveries["shared_composed"] - recoveries["authority_direct"]) <= .02 and minimum_term_rms > 0 and planned["selected_qk_projections"] == 13 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_corrected_shared_kernel": pred_a, "pred_b_ood_behavioral_replay": pred_b, "pred_c_interaction_removal_fidelity": pred_c, "pred_d_selective_interaction_removal": pred_d, "pred_e_causal_use_and_price": pred_e}
    terminal = "equality_l5h5_m4_corrected_shared_kernel_behavior_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_shared_kernel_behavior_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_shared_kernel_behavior_v1_result", "terminal": terminal, "predictions": predictions, "effects": effect_reports, "behavior": {"recoveries": recoveries, "interaction_removal_noncopy_mean_nat": noncopy_incremental, "minimum_donor_term_rms": minimum_term_rms}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "selected_correction_map": parent["selection"]["map"], "code_interaction_score_relative_l2": parent["code"]["overall"]["interaction"]["relative_l2"]}, "program_price": {"authority_qk_projections": 16, "selected_qk_projections": 13, "reduction_fraction": .1875, "learned_parameters": 0}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Behavioral replay and interaction-removal validation for the natural-selected K2-corrected thirteen-projection M4 interaction kernel."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "effects": effect_reports, "recoveries": recoveries, "noncopy_incremental": noncopy_incremental, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
