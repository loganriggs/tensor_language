#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;192 executions;180seconds;M4 one-map projection correction.
"""Add one selected Q/K projection of the child-gauge BF16 residual mismatch."""
import json
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
import run_equality_l5h5_m4_contracted_mode_graph_v1 as mode_parent
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
import run_equality_l5h5_m4_shared_projection_gauge_v1 as gauge_parent
import run_equality_l5h5_m4_shared_projection_kernel_v1 as kernel_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_SHARED_PROJECTION_CORRECTION_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_SHARED_PROJECTION_GAUGE_V1_RESULT.json"
DOCUMENTS = 192
MAPS = ("Q1", "K1", "Q2", "K2")


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
    if parent["terminal"] != "valid_equality_l5h5_m4_shared_projection_gauge_null" or parent["selection"]["gauge"] != "child":
        raise ValueError("shared gauge parent changed")
    if not parent["predictions"]["pred_a_lawful_parent_and_bridge"] or parent["predictions"]["pred_c_ood_shared_interaction_fidelity"]:
        raise ValueError("shared gauge null authority changed")
    roles, kernel_result, explicit_parent, manifest = gauge_parent.load_bound()
    return roles, parent, explicit_parent, manifest


def plan():
    roles, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_shared_projection_correction_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 192, "candidate_correction_maps": list(MAPS), "authority_qk_projections": 16, "selected_qk_projections": 13, "complete_behavior_forwards": 0, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def corrected_scores(residuals, weights, cos, sin, correction_index):
    projected = {cell: [F.linear(residuals[cell].float(), weight.float()) for weight in weights] for cell in ("baseline", "remainder", "joint")}
    projected["child"] = gauge_parent.derive(projected, "child")
    derived_child = residuals["joint"].float() - residuals["remainder"].float() + residuals["baseline"].float()
    correction_residual = residuals["child"].float() - derived_child
    projected_correction = F.linear(correction_residual, weights[correction_index].float())
    projected["child"][correction_index] = projected["child"][correction_index] + projected_correction
    scores = {}
    for cell in gauge_parent.CELLS:
        denominator = kernel_parent.rms_denominator(residuals[cell])
        raw = [(value / denominator).to(residuals[cell].dtype) for value in projected[cell]]
        scores[cell] = kernel_parent.score_from_raw(raw, cos, sin)
    scores["interaction"] = scores["joint"] - scores["child"] - scores["remainder"] + scores["baseline"]
    scores["composed"] = ((scores["baseline"] + (scores["child"] - scores["baseline"])) + (scores["remainder"] - scores["baseline"])) + scores["interaction"]
    diagnostics = {"correction_residual_squared_norm": float(correction_residual.double().square().sum()), "projected_correction_squared_norm": float(projected_correction.double().square().sum()), "corrected_child_port_squared_norm": float(projected["child"][correction_index].double().square().sum())}
    return scores, diagnostics


def empty_stats():
    return {name: kernel_parent.empty_stats() for name in MAPS}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, parent, explicit_parent, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = split_parent.compile_basis(model)
    stats = {"natural": empty_stats(), "code": empty_stats(), "half_0": empty_stats(), "half_1": empty_stats()}
    diagnostics = {name: {"correction_residual_squared_norm": 0.0, "projected_correction_squared_norm": 0.0, "corrected_child_port_squared_norm": 0.0} for name in MAPS}
    selected_map = None
    natural_reports = None
    for role, label in (("final_natural", "natural"), ("ood_code", "code")):
        for start in range(0, DOCUMENTS, action_parent.BATCH):
            tokens = roles[role][start:start + action_parent.BATCH, :-1].cuda()
            residual, groups, attention, product, mlp4, scale5 = mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
            residuals = kernel_parent.mode_residuals(groups, product, mlp4, basis, writers, scale5, residual.dtype)
            weights = residual_parent.head_weights(attention)
            cos, sin = residual_parent.rotary_ports(attention, residual)
            authority = kernel_parent.authority_scores(residuals, attention)
            support = induction.induction_fetch_mask(tokens)
            active = range(4) if label == "natural" else (MAPS.index(selected_map),)
            for index in active:
                name = MAPS[index]
                shared, row = corrected_scores(residuals, weights, cos, sin, index)
                kernel_parent.accumulate(stats[label][name], shared, authority, support)
                for key, value in row.items():
                    diagnostics[name][key] += value
                if label == "code":
                    kernel_parent.accumulate(stats["half_0" if start < 96 else "half_1"][name], shared, authority, support)
        if label == "natural":
            natural_reports = {name: kernel_parent.finish(rows) for name, rows in stats["natural"].items()}
            selected_map = min(MAPS, key=lambda name: (natural_reports[name]["interaction"]["relative_l2"], MAPS.index(name)))
    selected_natural = natural_reports[selected_map]
    code_report = kernel_parent.finish(stats["code"][selected_map])
    half_reports = [kernel_parent.finish(stats[name][selected_map]) for name in ("half_0", "half_1")]
    selected_diag = diagnostics[selected_map]
    selected_diag["projected_correction_to_child_port_norm_ratio"] = (selected_diag["projected_correction_squared_norm"] / max(selected_diag["corrected_child_port_squared_norm"], 1e-30)) ** .5
    pred_a = bool(explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5)
    pred_b = bool(selected_natural["interaction"]["relative_l2"] <= .10 and selected_natural["interaction"]["cosine"] >= .95)
    pred_c = bool(code_report["interaction"]["relative_l2"] <= .10 and code_report["interaction"]["cosine"] >= .95 and all(row["interaction"]["relative_l2"] <= .15 for row in half_reports))
    pred_d = bool(code_report["baseline"]["relative_l2"] <= .01 and code_report["baseline"]["cosine"] >= .999 and code_report["joint"]["relative_l2"] <= .01 and code_report["joint"]["cosine"] >= .999)
    pred_e = bool(selected_natural["closure"]["relative_l2"] <= 2e-6 and code_report["closure"]["relative_l2"] <= 2e-6 and all(row["closure"]["relative_l2"] <= 2e-6 for row in half_reports) and planned["selected_qk_projections"] == 13 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_parent_and_bridge": pred_a, "pred_b_natural_one_map_correction": pred_b, "pred_c_ood_corrected_interaction_fidelity": pred_c, "pred_d_baseline_and_joint_fidelity": pred_d, "pred_e_composition_and_projection_price": pred_e}
    terminal = "equality_l5h5_m4_shared_projection_one_map_correction" if all(predictions.values()) else "valid_equality_l5h5_m4_shared_projection_correction_null" if pred_a and pred_d else "invalid"
    result = {"schema": "equality_l5h5_m4_shared_projection_correction_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"map": selected_map, "natural": selected_natural}, "natural_correction_frontier": natural_reports, "code": {"overall": code_report, "halves": half_reports}, "diagnostics": selected_diag, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "parent_selected_gauge": parent["selection"]["gauge"], "parent_code_interaction_error": parent["code"]["overall"]["interaction"]["relative_l2"]}, "program_price": {"authority_qk_projections": 16, "selected_qk_projections": 13, "reduction_fraction": .1875, "learned_parameters": 0}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Natural-selected one-Q/K-map projection of the child-gauge BF16 residual mismatch for the explicit M4 interaction graph."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_map": selected_map, "natural_frontier": natural_reports, "code": code_report, "halves": half_reports, "diagnostics": selected_diag, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
