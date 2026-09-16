#!/usr/bin/env python3
# BQGATE:no prompts;one checkpoint;180seconds;reader-weighted M4 CP-rank obstruction.
"""Certify CP-rank lower bounds from exact reader-weighted tensor unfoldings."""
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
import run_equality_l5h5_m4_most_additive_mode_split_v1 as split_parent
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_READER_WEIGHTED_CP_RANK_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_INPUT_PORT_FACTOR_GRAPH_V1_RESULT.json"
RANKS = (16, 32, 64, 128, 256, 512, 1024, 1152)
SEED = 202609161850


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
    if parent["terminal"] != "equality_l5h5_m4_input_port_factor_graph_extracted_ood" or not all(parent["predictions"].values()):
        raise ValueError("normalized-input authority changed")
    return parent


def plan():
    load_bound()
    return {"schema": "equality_l5h5_m4_reader_weighted_cp_rank_v1_plan", "checkpoint_loads": 1, "prompts": 0, "model_forwards": 0, "gradients": 0, "fits": 0, "parameter_updates": 0, "selected_output_rank": 256, "input_width": 1152, "native_product_width": 4608, "candidate_cp_ranks": list(RANKS), "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def spectrum_report(gram):
    gram = ((gram + gram.T) * .5).double().cpu()
    eigenvalues = torch.linalg.eigvalsh(gram)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    positive = eigenvalues.clamp_min(0)
    total = float(positive.sum())
    descending = positive.flip(0)
    cumulative = descending.cumsum(0)
    curve = {}
    for rank in RANKS:
        retained = float(cumulative[min(rank, len(cumulative)) - 1] / max(total, 1e-30))
        curve[str(rank)] = {"energy_retained": retained, "cp_relative_error_lower_bound": math.sqrt(max(0.0, 1.0 - retained))}
    probabilities = positive / max(total, 1e-30)
    effective_rank = float(torch.exp(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()))
    return {"minimum_eigenvalue": minimum, "maximum_eigenvalue": maximum, "negative_to_maximum_ratio": max(0.0, -minimum) / max(maximum, 1e-30), "total_energy": total, "effective_rank": effective_rank, "curve": curve}


def contraction_errors(gram, output_basis, first, second, generator):
    errors = []
    for _ in range(4):
        vector = torch.randn(first.shape[1], generator=generator, device=first.device, dtype=torch.float32)
        first_projection = first @ vector
        direct = (output_basis * first_projection.unsqueeze(0)) @ second
        observed = float(direct.double().square().sum())
        expected = float(vector.double() @ gram.double() @ vector.double())
        errors.append(abs(observed - expected) / max(abs(observed), 1e-30))
    return errors


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    parent = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = split_parent.compile_basis(model)
    mlp4 = model.transformer.h[4].mlp
    attention5 = model.transformer.h[5].attn
    qk = torch.cat(split_parent.residual_parent.head_weights(attention5), dim=0).float()
    reader_responses = qk @ writers[:, :256].float()
    response_norms = reader_responses.square().sum(0).sqrt()
    normalized_responses = reader_responses / response_norms.clamp_min(1e-30)
    response_gram = normalized_responses.T @ normalized_responses
    response_off_diagonal = response_gram - torch.eye(256, device=response_gram.device, dtype=response_gram.dtype)
    maximum_response_off_diagonal = float(response_off_diagonal.abs().max())
    output_basis = basis[:256].float() * response_norms.unsqueeze(1)
    left = mlp4.Left.weight.float()
    right = mlp4.Right.weight.float()
    output_gram = output_basis.T @ output_basis
    right_gram = right @ right.T
    right_gram.mul_(output_gram)
    gram_left = left.T @ (right_gram @ left)
    del right_gram
    torch.cuda.empty_cache()
    left_gram = left @ left.T
    left_gram.mul_(output_gram)
    gram_right = right.T @ (left_gram @ right)
    del left_gram, output_gram
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    left_contraction_errors = contraction_errors(gram_left, output_basis, left, right, generator)
    right_contraction_errors = contraction_errors(gram_right, output_basis, right, left, generator)
    left_report = spectrum_report(gram_left)
    right_report = spectrum_report(gram_right)
    energy_relative_difference = abs(left_report["total_energy"] - right_report["total_energy"]) / max(left_report["total_energy"], 1e-30)
    lower_bound = {str(rank): max(left_report["curve"][str(rank)]["cp_relative_error_lower_bound"], right_report["curve"][str(rank)]["cp_relative_error_lower_bound"]) for rank in RANKS}
    pred_a = bool(max(left_contraction_errors + right_contraction_errors) <= 5e-4 and energy_relative_difference <= 2e-4 and max(left_report["negative_to_maximum_ratio"], right_report["negative_to_maximum_ratio"]) <= 2e-4)
    pred_b = bool(lower_bound["64"] >= .50)
    pred_c = bool(lower_bound["256"] >= .25)
    pred_d = bool(lower_bound["512"] >= .10)
    pred_e = bool(maximum_response_off_diagonal <= 2e-4 and bridge_error <= 2e-5)
    predictions = {"pred_a_implicit_gram_instrument": pred_a, "pred_b_rank64_cp_obstruction": pred_b, "pred_c_rank256_cp_obstruction": pred_c, "pred_d_rank512_cp_obstruction": pred_d, "pred_e_reader_coordinate_orthogonality": pred_e}
    terminal = "equality_l5h5_m4_reader_weighted_cp_rank_obstruction" if all(predictions.values()) else "valid_equality_l5h5_m4_reader_weighted_cp_compression_open" if pred_a and pred_e else "invalid"
    result = {"schema": "equality_l5h5_m4_reader_weighted_cp_rank_v1_result", "terminal": terminal, "predictions": predictions, "reader": {"selected_output_rank": 256, "maximum_normalized_response_off_diagonal": maximum_response_off_diagonal, "minimum_response_norm": float(response_norms.min()), "maximum_response_norm": float(response_norms.max()), "contracted_operator_reconstruction_relative_l2": bridge_error}, "left_input_unfolding": left_report, "right_input_unfolding": right_report, "cp_relative_error_lower_bound": lower_bound, "instrument": {"left_random_contraction_relative_errors": left_contraction_errors, "right_random_contraction_relative_errors": right_contraction_errors, "total_energy_relative_difference": energy_relative_difference}, "price": planned | {"model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Weight-only CP-rank lower bound for the retained rank-256 MLP4 tensor in the established L5 Q/K reader-weighted Frobenius metric. This is not an activation-moment, context-gated, or behavioral impossibility theorem."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "cp_relative_error_lower_bound": lower_bound, "instrument": result["instrument"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
