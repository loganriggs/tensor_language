#!/usr/bin/env python3
# BQGATE:192 natural and 192 code partial prefixes;180seconds;moment-weighted M4 CP-rank certificate.
"""Test empirical/Gaussian activation moments as a metric for M4 CP compression."""
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

import bilin18_observed_model_facade as facade
import run_equality_l5h5_m4_input_port_factor_graph_v1 as input_parent
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_MOMENT_WEIGHTED_CP_RANK_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_READER_WEIGHTED_CP_RANK_V1_RESULT.json"
RANKS = (64, 128, 256, 512, 1024, 1152)
ISOTROPIC_RANK256 = 0.7588316301712472
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
    if parent["terminal"] != "equality_l5h5_m4_reader_weighted_cp_rank_obstruction" or not all(parent["predictions"].values()):
        raise ValueError("isotropic CP-rank authority changed")
    roles, _, _, _, _, _ = input_parent.load_bound()
    return roles, parent


def plan():
    roles, _ = load_bound()
    return {"schema": "equality_l5h5_m4_moment_weighted_cp_rank_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "partial_prefix_executions": len(roles["final_natural"]) + len(roles["ood_code"]), "checkpoint_loads": 1, "input_width": 1152, "native_product_width": 4608, "selected_output_rank": 256, "candidate_cp_ranks": list(RANKS), "gradients": 0, "fits": 0, "parameter_updates": 0, "new_text": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def capture_left_input(_module, arguments):
    CAPTURE["state"] = arguments[0]


@torch.no_grad()
def collect_moment(model, rows, output_basis):
    moment = torch.zeros((1152, 1152), device="cuda", dtype=torch.float32)
    energy = 0.0
    state_square = 0.0
    count = 0
    batch = input_parent.v1.action_parent.BATCH
    for start in range(0, len(rows), batch):
        tokens = rows[start:start + batch, :-1].cuda()
        _, _, _, product, _, _ = input_parent.v1.mode_parent.atom_parent.fine_parts_with_m4(model, tokens)
        state = CAPTURE["state"]
        if state is None:
            raise RuntimeError("MLP4 normalized input was not captured")
        flat_state = state.reshape(-1, 1152).float()
        flat_product = product.reshape(-1, 4608).float()
        moment.add_(flat_state.T @ flat_state)
        coefficients = flat_product @ output_basis.T
        energy += float(coefficients.double().square().sum())
        state_square += float(flat_state.double().square().sum())
        count += flat_state.shape[0]
    return moment / count, energy / count, state_square / (count * 1152), count


def unfolding_spectrum(first, second, output_gram):
    partner_gram = second @ second.T
    partner_gram.mul_(output_gram)
    gram = first.T @ (partner_gram @ first)
    gram = ((gram + gram.T) * .5).double().cpu()
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = float(eigenvalues.sum())
    cumulative = eigenvalues.flip(0).cumsum(0)
    curve = {}
    for rank in RANKS:
        retained = float(cumulative[rank - 1] / max(total, 1e-30))
        curve[str(rank)] = {"energy_retained": retained, "cp_relative_error_lower_bound": math.sqrt(max(0.0, 1.0 - retained))}
    probabilities = eigenvalues / max(total, 1e-30)
    effective_rank = float(torch.exp(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()))
    return {"total_energy": total, "effective_rank": effective_rank, "curve": curve}


def metric_report(moment, empirical_energy, state_square, count, output_basis, left, right):
    eigenvalues, eigenvectors = torch.linalg.eigh(((moment + moment.T) * .5).double())
    negative_ratio = max(0.0, -float(eigenvalues[0])) / max(float(eigenvalues[-1]), 1e-30)
    root = (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.T
    root = root.float()
    transformed_left = left @ root
    transformed_right = right @ root
    output_gram = output_basis.T @ output_basis
    left_report = unfolding_spectrum(transformed_left, transformed_right, output_gram)
    right_report = unfolding_spectrum(transformed_right, transformed_left, output_gram)
    energy_difference = abs(left_report["total_energy"] - right_report["total_energy"]) / max(left_report["total_energy"], 1e-30)
    cross = transformed_left @ transformed_right.T
    diagonal = (transformed_left * transformed_right).sum(1)
    traces = output_basis @ diagonal
    trace_square = float(traces.double().square().sum())
    transpose_contraction = float((output_gram.double() * cross.double() * cross.T.double()).sum())
    gaussian_energy = trace_square + left_report["total_energy"] + transpose_contraction
    gaussian_error = abs(gaussian_energy - empirical_energy) / max(empirical_energy, 1e-30)
    lower_bound = {str(rank): max(left_report["curve"][str(rank)]["cp_relative_error_lower_bound"], right_report["curve"][str(rank)]["cp_relative_error_lower_bound"]) for rank in RANKS}
    return {"tokens": count, "mean_state_square": state_square, "moment_minimum_eigenvalue": float(eigenvalues[0]), "moment_maximum_eigenvalue": float(eigenvalues[-1]), "moment_negative_to_maximum_ratio": negative_ratio, "left_input_unfolding": left_report, "right_input_unfolding": right_report, "input_mode_total_energy_relative_difference": energy_difference, "cp_relative_error_lower_bound": lower_bound, "empirical_same_state_energy": empirical_energy, "gaussian_same_state_energy": gaussian_energy, "gaussian_energy_relative_error": gaussian_error, "gaussian_terms": {"trace_square": trace_square, "frobenius": left_report["total_energy"], "transpose_contraction": transpose_contraction}}


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
    attention5 = model.transformer.h[5].attn
    qk = torch.cat(split_parent.residual_parent.head_weights(attention5), dim=0).float()
    response_norms = (qk @ writers[:, :256].float()).square().sum(0).sqrt()
    output_basis = basis[:256].float() * response_norms.unsqueeze(1)
    left = model.transformer.h[4].mlp.Left.weight.float()
    right = model.transformer.h[4].mlp.Right.weight.float()
    raw = {}
    for label, role in (("natural", "final_natural"), ("code", "ood_code")):
        raw[label] = collect_moment(model, roles[role], output_basis)
    handle.remove()
    reports = {label: metric_report(*raw[label], output_basis, left, right) for label in ("natural", "code")}
    natural_rank256 = reports["natural"]["cp_relative_error_lower_bound"]["256"]
    code_rank256 = reports["code"]["cp_relative_error_lower_bound"]["256"]
    pred_a = bool(all(report["moment_negative_to_maximum_ratio"] <= 2e-4 and .98 <= report["mean_state_square"] <= 1.02 and report["input_mode_total_energy_relative_difference"] <= 2e-4 for report in reports.values()) and bridge_error <= 2e-5)
    pred_b = bool(natural_rank256 <= .8 * ISOTROPIC_RANK256)
    pred_c = bool(natural_rank256 <= .50)
    pred_d = bool(abs(code_rank256 - natural_rank256) <= .10)
    pred_e = bool(all(report["gaussian_energy_relative_error"] <= .25 for report in reports.values()))
    predictions = {"pred_a_moment_and_spectral_instrument": pred_a, "pred_b_moment_metric_reduces_rank256_bound": pred_b, "pred_c_natural_rank256_compression_open": pred_c, "pred_d_code_metric_transfer": pred_d, "pred_e_gaussian_moment_fidelity": pred_e}
    terminal = "equality_l5h5_m4_moment_weighted_cp_compression_open" if all(predictions.values()) else "valid_equality_l5h5_m4_moment_weighted_cp_null" if pred_a else "invalid"
    result = {"schema": "equality_l5h5_m4_moment_weighted_cp_rank_v1_result", "terminal": terminal, "predictions": predictions, "isotropic_rank256_lower_bound": ISOTROPIC_RANK256, "panels": reports, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error}, "price": planned | {"model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Empirical-second-moment and matched zero-mean Gaussian metric audit for the exact rank-256 L5-reader-weighted MLP4 tensor on frozen natural/code panels. No fitted CP node or behavioral circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "natural": {"rank256": natural_rank256, "gaussian_error": reports["natural"]["gaussian_energy_relative_error"]}, "code": {"rank256": code_rank256, "gaussian_error": reports["code"]["gaussian_energy_relative_error"]}, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
