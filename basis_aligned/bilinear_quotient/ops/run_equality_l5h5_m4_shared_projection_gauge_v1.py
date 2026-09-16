#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;192 executions;180seconds;M4 shared-projection gauges.
"""Select a natural-only twelve-projection arithmetic gauge for the M4 interaction."""
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
import run_equality_l5h5_m4_shared_projection_kernel_v1 as kernel_parent
import run_equality_l5h5_residual_source_graph_v1 as residual_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_SHARED_PROJECTION_GAUGE_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_SHARED_PROJECTION_KERNEL_V1_RESULT.json"
DOCUMENTS = 192
GAUGES = ("baseline", "child", "remainder", "joint")
CELLS = ("baseline", "child", "remainder", "joint")


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
    if parent["terminal"] != "valid_equality_l5h5_m4_shared_projection_kernel_null" or not parent["predictions"]["pred_a_lawful_parent_and_bridge"] or not parent["predictions"]["pred_b_valid_shared_baseline"]:
        raise ValueError("shared-projection parent changed")
    if parent["predictions"]["pred_d_shared_interaction_fidelity"] or parent["reports"]["code"]["interaction"]["relative_l2"] <= .10:
        raise ValueError("shared interaction null authority changed")
    roles, explicit_parent, manifest = kernel_parent.load_bound()
    return roles, parent, explicit_parent, manifest


def plan():
    roles, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_shared_projection_gauge_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "execution_count": 192, "gauges": list(GAUGES), "authority_qk_projections": 16, "selected_qk_projections": 12, "complete_behavior_forwards": 0, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def derive(projected, omitted):
    if omitted == "baseline":
        return [child + remainder - joint for child, remainder, joint in zip(projected["child"], projected["remainder"], projected["joint"])]
    if omitted == "child":
        return [joint - remainder + baseline for joint, remainder, baseline in zip(projected["joint"], projected["remainder"], projected["baseline"])]
    if omitted == "remainder":
        return [joint - child + baseline for joint, child, baseline in zip(projected["joint"], projected["child"], projected["baseline"])]
    return [child + remainder - baseline for child, remainder, baseline in zip(projected["child"], projected["remainder"], projected["baseline"])]


def gauge_scores(residuals, weights, cos, sin, omitted):
    projected = {}
    for cell in CELLS:
        if cell != omitted:
            projected[cell] = [F.linear(residuals[cell].float(), weight.float()) for weight in weights]
    projected[omitted] = derive(projected, omitted)
    scores = {}
    for cell in CELLS:
        denominator = kernel_parent.rms_denominator(residuals[cell])
        raw = [(value / denominator).to(residuals[cell].dtype) for value in projected[cell]]
        scores[cell] = kernel_parent.score_from_raw(raw, cos, sin)
    scores["interaction"] = scores["joint"] - scores["child"] - scores["remainder"] + scores["baseline"]
    scores["composed"] = ((scores["baseline"] + (scores["child"] - scores["baseline"])) + (scores["remainder"] - scores["baseline"])) + scores["interaction"]
    return scores


def empty_stats():
    return {gauge: kernel_parent.empty_stats() for gauge in GAUGES}


def finish(stats):
    return {gauge: kernel_parent.finish(rows) for gauge, rows in stats.items()}


def valid_natural(report):
    return report["baseline"]["relative_l2"] <= .01 and report["joint"]["relative_l2"] <= .01


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
    selected_gauge = None
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
            active_gauges = GAUGES if label == "natural" else (selected_gauge,)
            for gauge in active_gauges:
                shared = gauge_scores(residuals, weights, cos, sin, gauge)
                kernel_parent.accumulate(stats[label][gauge], shared, authority, support)
                if label == "code":
                    kernel_parent.accumulate(stats["half_0" if start < 96 else "half_1"][gauge], shared, authority, support)
        if label == "natural":
            natural_reports = finish(stats["natural"])
            valid = [gauge for gauge in GAUGES if valid_natural(natural_reports[gauge])]
            pool = valid if valid else list(GAUGES)
            selected_gauge = min(pool, key=lambda gauge: (natural_reports[gauge]["interaction"]["relative_l2"], GAUGES.index(gauge)))
    code_report = kernel_parent.finish(stats["code"][selected_gauge])
    half_reports = [kernel_parent.finish(stats[name][selected_gauge]) for name in ("half_0", "half_1")]
    selected_natural = natural_reports[selected_gauge]
    pred_a = bool(explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5)
    pred_b = bool(valid_natural(selected_natural) and selected_natural["interaction"]["relative_l2"] <= .10 and selected_natural["interaction"]["cosine"] >= .95)
    pred_c = bool(code_report["interaction"]["relative_l2"] <= .10 and code_report["interaction"]["cosine"] >= .95 and all(row["interaction"]["relative_l2"] <= .15 for row in half_reports))
    pred_d = bool(code_report["baseline"]["relative_l2"] <= .01 and code_report["baseline"]["cosine"] >= .999 and code_report["joint"]["relative_l2"] <= .01 and code_report["joint"]["cosine"] >= .999)
    pred_e = bool(selected_natural["closure"]["relative_l2"] <= 2e-6 and code_report["closure"]["relative_l2"] <= 2e-6 and all(row["closure"]["relative_l2"] <= 2e-6 for row in half_reports) and planned["selected_qk_projections"] == 12 and manifest["learned_parameters"] == 0)
    predictions = {"pred_a_lawful_parent_and_bridge": pred_a, "pred_b_natural_shared_gauge": pred_b, "pred_c_ood_shared_interaction_fidelity": pred_c, "pred_d_shared_baseline_and_joint_fidelity": pred_d, "pred_e_composition_and_projection_price": pred_e}
    terminal = "equality_l5h5_m4_shared_projection_gauge" if all(predictions.values()) else "valid_equality_l5h5_m4_shared_projection_gauge_null" if pred_a and pred_d else "invalid"
    result = {"schema": "equality_l5h5_m4_shared_projection_gauge_v1_result", "terminal": terminal, "predictions": predictions, "selection": {"gauge": selected_gauge, "natural": selected_natural}, "natural_gauge_frontier": natural_reports, "code": {"overall": code_report, "halves": half_reports}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "parent_omit_joint_code_interaction_error": parent["reports"]["code"]["interaction"]["relative_l2"]}, "program_price": {"authority_qk_projections": 16, "selected_qk_projections": 12, "reduction_fraction": .25, "learned_parameters": 0}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Natural-only selection among four twelve-projection arithmetic gauges for the explicit M4 interaction graph."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_gauge": selected_gauge, "natural_frontier": natural_reports, "code": code_report, "halves": half_reports, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
