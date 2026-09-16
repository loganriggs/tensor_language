#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;384 executions;180seconds;M4 contracted-mode graph precision correction.
"""Float64 bridge correction for the Q/K-contracted M4 mode graph."""
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
import run_equality_l5h5_m4_contracted_mode_graph_v1 as v1
import run_equality_l5h5_l234_module_write_graph_v1 as module_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
import run_equality_reusable_score_port_code_ood_v1 as score_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V2"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
V1_RESULT = P / "EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V1_RESULT.json"
DOCUMENTS = v1.DOCUMENTS
PAIR = v1.PAIR
FIVE_MASK = v1.FIVE_MASK
RANKS = v1.RANKS


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
    invalid = json.loads(V1_RESULT.read_text())
    if invalid["terminal"] != "invalid" or invalid["predictions"]["pred_a_lawful_contracted_mode_bridge"] or invalid["instrument"]["contracted_operator_reconstruction_relative_l2"] <= 2e-5:
        raise ValueError("V1 precision-failure authority changed")
    roles, scales, atom_result, parent_result, port_parent, manifest = v1.load_bound()
    return roles, scales, atom_result, parent_result, port_parent, manifest, invalid


def plan():
    roles, _, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_contracted_mode_graph_v2_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 384, "natural_geometry_passes": 1, "code_geometry_passes": 1, "complete_code_arms": 6, "candidate_ranks": list(RANKS), "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, scales, atom_result, parent_result, port_parent, manifest, invalid = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()

    mlp4 = model.transformer.h[4].mlp
    attention5 = model.transformer.h[5].attn
    qk_weights = torch.cat(residual_parent.head_weights(attention5), dim=0).double()
    contracted = qk_weights @ mlp4.Down.weight.double()
    left_vectors, singular_values, basis64 = torch.linalg.svd(contracted, full_matrices=False)
    reconstructed = (left_vectors * singular_values.unsqueeze(0)) @ basis64
    bridge_error = float((reconstructed - contracted).norm() / contracted.norm().clamp_min(1e-30))
    basis = basis64.float()
    writers = mlp4.Down.weight.float() @ basis.T
    del left_vectors, singular_values, basis64, reconstructed, contracted, qk_weights

    natural_stats = {rank: torch.zeros(4, dtype=torch.float64) for rank in RANKS}
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["final_natural"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = v1.atom_parent.fine_parts_with_m4(model, tokens)
        target = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
        support = induction.induction_fetch_mask(tokens)
        for rank in RANKS:
            predicted = v1.score_with_rank(groups, product, mlp4, basis, writers, rank, scale5, attention, residual.dtype)
            v1.atom_parent.accumulate_pair(natural_stats[rank], predicted, target, support)
    natural_reports = {rank: v1.atom_parent.finish_pair(stats) for rank, stats in natural_stats.items()}
    selected_rank, qualified = v1.select_rank(natural_reports)

    code_stats = torch.zeros(4, dtype=torch.float64)
    code_native_stats = torch.zeros(4, dtype=torch.float64)
    rolled_stats = torch.zeros(4, dtype=torch.float64)
    half_stats = [torch.zeros(4, dtype=torch.float64) for _ in range(2)]
    composition_stats = torch.zeros(4, dtype=torch.float64)
    first_modes = torch.arange(0, selected_rank, 2, device="cuda")
    second_modes = torch.arange(1, selected_rank, 2, device="cuda")
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        tokens = roles["ood_code"][start:start + action_parent.BATCH, :-1].cuda()
        residual, groups, attention, product, mlp4, scale5 = v1.atom_parent.fine_parts_with_m4(model, tokens)
        parent_score = module_parent.score_residual(module_parent.merge_fine(groups, FIVE_MASK, residual.dtype), attention)
        native_score = module_parent.score_residual(residual, attention)
        selected_score = v1.score_with_rank(groups, product, mlp4, basis, writers, selected_rank, scale5, attention, residual.dtype)
        rolled_score = v1.score_with_rank(groups, product, mlp4, basis, writers, selected_rank, scale5, attention, residual.dtype, roll=True)
        support = induction.induction_fetch_mask(tokens)
        v1.atom_parent.accumulate_pair(code_stats, selected_score, parent_score, support)
        v1.atom_parent.accumulate_pair(code_native_stats, selected_score, native_score, support)
        v1.atom_parent.accumulate_pair(rolled_stats, rolled_score, parent_score, support)
        v1.atom_parent.accumulate_pair(half_stats[0 if start < 96 else 1], selected_score, parent_score, support)
        score0 = v1.score_with_rank(groups, product, mlp4, basis, writers, 0, scale5, attention, residual.dtype)
        score1 = v1.atom_parent.score_with_m4(groups, v1.mode_write(product, mlp4, basis.index_select(0, first_modes), writers.index_select(1, first_modes), len(first_modes), scale5), attention, residual.dtype)
        score2 = v1.atom_parent.score_with_m4(groups, v1.mode_write(product, mlp4, basis.index_select(0, second_modes), writers.index_select(1, second_modes), len(second_modes), scale5), attention, residual.dtype)
        additive = score1 + score2 - score0
        v1.atom_parent.accumulate_pair(composition_stats, additive - score0, selected_score - score0, support)
    code_report = v1.atom_parent.finish_pair(code_stats)
    code_native_report = v1.atom_parent.finish_pair(code_native_stats)
    rolled_report = v1.atom_parent.finish_pair(rolled_stats)
    code_halves = [v1.atom_parent.finish_pair(stats) for stats in half_stats]
    composition = v1.atom_parent.finish_pair(composition_stats)

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
        mode_logits, term_rms = v1.mode_donor_forward(model, tokens, scales["L5H5"], basis, writers, selected_rank)
        rolled_logits, rolled_term_rms = v1.mode_donor_forward(model, tokens, scales["L5H5"], basis, writers, selected_rank, roll=True)
        minimum_term_rms = min(minimum_term_rms, term_rms, rolled_term_rms)
        targets = batch_rows[:, 1:].cuda()
        for name, logits in zip(names_forward, (native_logits, absent_logits, factor_logits, five_logits, mode_logits, rolled_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    behavior = v1.atom_parent.summarize({"native": nll["native"], "absent": nll["absent"], "factor_donor": nll["factor_donor"], "five_donor": nll["five_donor"], "product_donor": nll["mode_donor"], "rolled_donor": nll["rolled_donor"]}, masks)
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
    result = {"schema": "equality_l5h5_m4_contracted_mode_graph_v2_result", "terminal": terminal, "predictions": predictions, "selection": {"qualified": qualified, "rank": selected_rank, "fraction_of_contracted_rank": selected_rank / 512, "natural": natural_reports[selected_rank]}, "natural_rank_curve": {str(rank): report for rank, report in natural_reports.items()}, "code_score": {"versus_five_write_parent": code_report, "versus_native_residual": code_native_report, "rolled_equal_norm_control": rolled_report, "halves": code_halves}, "composition": {"split": "alternating frozen singular rank", "first_rank": len(first_modes), "second_rank": len(second_modes), "additive_effect_prediction": composition}, "behavior": behavior, "comparisons": {"native_atom_selected_width": atom_result["selection"]["width"], "five_write_recovery": parent_recovery, "mode_recovery": recovery, "mode_minus_five": recovery - parent_recovery, "rolled_recovery": rolled_recovery, "mode_minus_rolled": recovery - rolled_recovery}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "v1_float32_bridge_error": invalid["instrument"]["contracted_operator_reconstruction_relative_l2"], "inherited_native_m4_write_replay_error": inherited_exact["maximum_native_m4_write_replay_error"], "inherited_native_five_score_replay_error": inherited_exact["maximum_native_five_score_replay_error"], "minimum_donor_term_rms": minimum_term_rms}, "program_price": {"selected_scalar_bilinear_modes": selected_rank, "derived_basis_values": selected_rank * 4608, "derived_writer_values": selected_rank * 1152, "native_product_evaluations_retained": 4608, "m4_input_state_port": True, "learned_parameters": 0}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Float64-certified weight-only Q/K-contracted SVD compression of M4's product-to-writer map. Deployed dense modes retain native M4 input and all native product evaluations."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_rank": selected_rank, "natural": natural_reports[selected_rank], "code": code_report, "recovery": recovery, "rolled_recovery": rolled_recovery, "composition": composition, "bridge_error": bridge_error, "v1_bridge_error": invalid["instrument"]["contracted_operator_reconstruction_relative_l2"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
