#!/usr/bin/env python3
# BQGATE:two natural and one code partial-prefix passes;180seconds;quadratic M4 context gate.
"""Test a frozen label-free quadratic state gate for M4 tensor compression."""
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
import run_equality_l5h5_m4_input_port_factor_graph_v1 as input_parent
import run_equality_l5h5_m4_moment_weighted_cp_rank_v1 as moment_parent
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_QUADRATIC_CONTEXT_GATE_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_TYPED_MOMENT_CP_RANK_V1_RESULT.json"
ALL_POSITION_NATURAL_RANK256 = 0.4233676210727503
GATE_RANK = 64
ARMS = ("low", "high")
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
    if parent["terminal"] != "valid_equality_l5h5_m4_typed_context_moment_null" or not parent["predictions"]["pred_a_typed_moment_instrument"]:
        raise ValueError("typed-context authority changed")
    roles, _, _, _, _, _ = input_parent.load_bound()
    return roles, parent


def plan():
    roles, _ = load_bound()
    return {"schema": "equality_l5h5_m4_quadratic_context_gate_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "natural_partial_prefix_passes": 2, "code_partial_prefix_passes": 1, "gate_rank": GATE_RANK, "gate_inputs": ["normalized_mlp4_state"], "task_labels": 0, "domain_labels": 0, "equality_support_inputs": 0, "spectral_reports": 4, "checkpoint_loads": 1, "gradients": 0, "fits": 0, "parameter_updates": 0, "new_text": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def capture_left_input(_module, arguments):
    CAPTURE["state"] = arguments[0]
    moment_parent.CAPTURE["state"] = arguments[0]


def empty_accumulator():
    return {name: {"moment": torch.zeros((1152, 1152), device="cuda"), "energy": 0.0, "state_square": 0.0, "count": 0, "score_sum": 0.0} for name in ARMS}


@torch.no_grad()
def collect_gated(model, rows, output_basis, gate_basis, threshold):
    accumulators = empty_accumulator()
    batch = input_parent.v1.action_parent.BATCH
    for start in range(0, len(rows), batch):
        tokens = rows[start:start + batch, :-1].cuda()
        _, _, _, product, _, _ = input_parent.v1.mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        state = CAPTURE["state"]
        flat_state = state.reshape(-1, 1152).float()
        flat_product = product.reshape(-1, 4608).float()
        scores = (flat_state @ gate_basis).square().sum(1)
        masks = {"low": scores < threshold, "high": scores >= threshold}
        for name, mask in masks.items():
            selected_state = flat_state[mask]
            selected_product = flat_product[mask]
            selected_scores = scores[mask]
            row = accumulators[name]
            row["moment"].add_(selected_state.T @ selected_state)
            coefficients = selected_product @ output_basis.T
            row["energy"] += float(coefficients.double().square().sum())
            row["state_square"] += float(selected_state.double().square().sum())
            row["score_sum"] += float(selected_scores.double().sum())
            row["count"] += selected_state.shape[0]
    output = {}
    for name, row in accumulators.items():
        if row["count"] == 0:
            output[name] = None
        else:
            output[name] = {"raw": (row["moment"] / row["count"], row["energy"] / row["count"], row["state_square"] / (row["count"] * 1152), row["count"]), "mean_gate_score": row["score_sum"] / row["count"]}
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
    natural_moment, _, _, natural_count = moment_parent.collect_moment(model, roles["final_natural"], output_basis)
    eigenvalues, eigenvectors = torch.linalg.eigh(((natural_moment + natural_moment.T) * .5).double())
    gate_basis = eigenvectors[:, -GATE_RANK:].float()
    threshold = float(eigenvalues[-GATE_RANK:].sum())
    raw = {label: collect_gated(model, roles[role], output_basis, gate_basis, threshold) for label, role in (("natural", "final_natural"), ("code", "ood_code"))}
    handle.remove()
    collapsed = any(raw[label][arm] is None for label in raw for arm in ARMS)
    reports = {}
    for label in raw:
        reports[label] = {}
        for arm in ARMS:
            if raw[label][arm] is None:
                reports[label][arm] = {"tokens": 0, "mean_gate_score": None}
            else:
                report = moment_parent.metric_report(*raw[label][arm]["raw"], output_basis, left, right)
                report["mean_gate_score"] = raw[label][arm]["mean_gate_score"]
                reports[label][arm] = report
    natural_fraction = {arm: reports["natural"][arm]["tokens"] / natural_count for arm in ARMS}
    instrument_cells = [reports[label][arm] for label in reports for arm in ARMS if reports[label][arm]["tokens"] > 0]
    pred_a_numeric = bool(bridge_error <= 2e-5 and all(.98 <= report["mean_state_square"] <= 1.02 and report["moment_negative_to_maximum_ratio"] <= 2e-4 and report["input_mode_total_energy_relative_difference"] <= 2e-4 for report in instrument_cells))
    populated = bool(not collapsed and all(reports[label][arm]["tokens"] >= 128 for label in reports for arm in ARMS))
    balanced = bool(all(.20 <= natural_fraction[arm] <= .80 for arm in ARMS))
    pred_a = bool(pred_a_numeric and populated and balanced)
    if collapsed:
        rank256 = {label: {arm: None for arm in ARMS} for label in reports}
        pred_b = pred_c = pred_d = pred_e = False
    else:
        rank256 = {label: {arm: reports[label][arm]["cp_relative_error_lower_bound"]["256"] for arm in ARMS} for label in reports}
        pred_b = bool(all(rank256["natural"][arm] <= .35 for arm in ARMS))
        pred_c = bool(all(abs(rank256["code"][arm] - rank256["natural"][arm]) <= .10 for arm in ARMS))
        pred_d = bool(max(rank256["natural"].values()) <= .8 * ALL_POSITION_NATURAL_RANK256)
        pred_e = bool(all(reports[label][arm]["gaussian_energy_relative_error"] <= .25 for label in reports for arm in ARMS))
    predictions = {"pred_a_frozen_gate_instrument_and_population": pred_a, "pred_b_natural_conditional_compression_open": pred_b, "pred_c_frozen_code_gate_transfer": pred_c, "pred_d_conditional_improves_all_position_metric": pred_d, "pred_e_conditional_gaussian_fidelity": pred_e}
    terminal = "equality_l5h5_m4_quadratic_context_cp_fit_licensed" if all(predictions.values()) else "valid_equality_l5h5_m4_quadratic_context_gate_null" if pred_a_numeric else "invalid"
    result = {"schema": "equality_l5h5_m4_quadratic_context_gate_v1_result", "terminal": terminal, "predictions": predictions, "gate": {"formula": "||P64 x||^2 - natural_mean", "rank": GATE_RANK, "threshold": threshold, "discovery_positions": natural_count, "natural_arm_fraction": natural_fraction, "collapsed_frozen_arm": collapsed, "task_labels": 0, "domain_labels": 0, "equality_support_inputs": 0}, "rank256_summary": rank256, "panels": reports, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "numeric_pass": pred_a_numeric, "population_pass": populated, "balance_pass": balanced}, "price": planned | {"model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Frozen natural-moment-derived quadratic gate over normalized MLP4 state, tested unchanged on code. No task/domain/support gate input and no CP fit or behavioral claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "gate": result["gate"], "rank256_summary": rank256, "instrument": result["instrument"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
