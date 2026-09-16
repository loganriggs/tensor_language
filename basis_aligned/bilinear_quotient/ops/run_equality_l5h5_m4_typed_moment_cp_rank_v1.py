#!/usr/bin/env python3
# BQGATE:192 natural and 192 code partial prefixes;180seconds;typed-context M4 moment audit.
"""Test query/key typed activation moments for reader-weighted M4 compression."""
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

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction
import run_equality_l5h5_m4_input_port_factor_graph_v1 as input_parent
import run_equality_l5h5_m4_moment_weighted_cp_rank_v1 as moment_parent
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_TYPED_MOMENT_CP_RANK_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_MOMENT_WEIGHTED_CP_RANK_V1_RESULT.json"
ALL_POSITION_NATURAL_RANK256 = 0.4233676210727503
TYPES = ("query", "key", "union", "off_support")
CAPTURE = {"state": None}


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
    if parent["terminal"] != "valid_equality_l5h5_m4_moment_weighted_cp_null" or not parent["predictions"]["pred_a_moment_and_spectral_instrument"]:
        raise ValueError("all-position moment authority changed")
    roles, _, _, _, _, _ = input_parent.load_bound()
    return roles, parent


def plan():
    roles, _ = load_bound()
    return {"schema": "equality_l5h5_m4_typed_moment_cp_rank_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "partial_prefix_executions": len(roles["final_natural"]) + len(roles["ood_code"]), "context_types": list(TYPES), "spectral_reports": 8, "checkpoint_loads": 1, "gradients": 0, "fits": 0, "parameter_updates": 0, "new_text": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def capture_left_input(_module, arguments):
    CAPTURE["state"] = arguments[0]


def empty_accumulator():
    return {name: {"moment": torch.zeros((1152, 1152), device="cuda"), "energy": 0.0, "state_square": 0.0, "count": 0} for name in TYPES}


@torch.no_grad()
def collect_typed(model, rows, output_basis):
    accumulators = empty_accumulator()
    batch = input_parent.v1.action_parent.BATCH
    for start in range(0, len(rows), batch):
        tokens = rows[start:start + batch, :-1].cuda()
        _, _, _, product, _, _ = input_parent.v1.mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        state = CAPTURE["state"]
        support = induction.induction_fetch_mask(tokens)
        query = support.any(dim=-1)
        key = support.any(dim=-2)
        masks = {"query": query, "key": key, "union": query | key, "off_support": ~(query | key)}
        for name, mask in masks.items():
            selected_state = state[mask].float()
            selected_product = product[mask].float()
            row = accumulators[name]
            row["moment"].add_(selected_state.T @ selected_state)
            coefficients = selected_product @ output_basis.T
            row["energy"] += float(coefficients.double().square().sum())
            row["state_square"] += float(selected_state.double().square().sum())
            row["count"] += selected_state.shape[0]
    return {name: (row["moment"] / row["count"], row["energy"] / row["count"], row["state_square"] / (row["count"] * 1152), row["count"]) for name, row in accumulators.items()}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, parent = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    handle = model.transformer.h[4].mlp.Left.register_forward_pre_hook(capture_left_input)
    started = time.perf_counter()
    basis, writers, bridge_error = split_parent.compile_basis(model)
    qk = torch.cat(split_parent.residual_parent.head_weights(model.transformer.h[5].attn), dim=0).float()
    response_norms = (qk @ writers[:, :256].float()).square().sum(0).sqrt()
    output_basis = basis[:256].float() * response_norms.unsqueeze(1)
    left = model.transformer.h[4].mlp.Left.weight.float()
    right = model.transformer.h[4].mlp.Right.weight.float()
    raw = {label: collect_typed(model, roles[role], output_basis) for label, role in (("natural", "final_natural"), ("code", "ood_code"))}
    handle.remove()
    reports = {label: {name: moment_parent.metric_report(*raw[label][name], output_basis, left, right) for name in TYPES} for label in ("natural", "code")}
    rank256 = {label: {name: reports[label][name]["cp_relative_error_lower_bound"]["256"] for name in TYPES} for label in reports}
    pred_a = bool(bridge_error <= 2e-5 and all(report["tokens"] >= 128 and .98 <= report["mean_state_square"] <= 1.02 and report["moment_negative_to_maximum_ratio"] <= 2e-4 and report["input_mode_total_energy_relative_difference"] <= 2e-4 for panel in reports.values() for report in panel.values()))
    pred_b = bool(all(abs(rank256["natural"][name] - rank256["code"][name]) <= .10 for name in ("query", "key")))
    pred_c = bool(all(rank256["natural"][name] <= .35 for name in ("query", "key")))
    pred_d = bool(all(rank256["natural"][name] <= .8 * ALL_POSITION_NATURAL_RANK256 for name in ("query", "key")))
    pred_e = bool(all(reports["natural"][name]["gaussian_energy_relative_error"] <= .25 for name in ("query", "key")) and max(rank256["natural"]["query"], rank256["natural"]["key"]) + .05 <= rank256["natural"]["off_support"])
    predictions = {"pred_a_typed_moment_instrument": pred_a, "pred_b_query_key_metric_transfer": pred_b, "pred_c_typed_rank256_compression_open": pred_c, "pred_d_typed_improves_all_position_metric": pred_d, "pred_e_gaussian_fidelity_and_context_specificity": pred_e}
    terminal = "equality_l5h5_m4_typed_context_cp_fit_licensed" if all(predictions.values()) else "valid_equality_l5h5_m4_typed_context_moment_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_typed_moment_cp_rank_v1_result", "terminal": terminal, "predictions": predictions, "all_position_natural_rank256_lower_bound": ALL_POSITION_NATURAL_RANK256, "rank256_summary": rank256, "panels": reports, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error}, "price": planned | {"model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Frozen equality-edge query/key/off-support moment audit for the exact L5-reader-weighted MLP4 tensor. Context types are diagnostic structural roles; no CP fit, extraction, removal, or behavioral claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "rank256_summary": rank256, "gaussian_error": {label: {name: reports[label][name]["gaussian_energy_relative_error"] for name in TYPES} for label in reports}, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
