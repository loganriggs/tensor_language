"""RUNG 420 -- ATTENTION0 QK GLOBAL TOKEN-FUNCTION CARRIER.

Rung418 found uniform diffuse cross-head Q/K projector overlap but no atomic
shared edge.  Estimate the top-24 eigenspace of each side's average projector,
transport its branch-specific right-coordinate realization to held-out token
IDs, remove it, and test whether private residual overlap falls to the null
floor.  Then compare the carrier with MLP0's complete degree-one token action.

This is gauge-invariant identification, not compression or adoption.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_qk_common_carrier_results.json"
PARENT_RESULT = BQ / "attention0_cross_head_qk_shared_half_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
VOCAB = 50_257
N_HEAD = 9
HD = 128
D = 1152
CARRIER_RANK = 24
MLP_RANK = 64
ENTRIES = tuple((head, branch) for head in range(N_HEAD) for branch in (1, 2))
SIDES = ("q", "k")


def _whiten_full(value, fit_mask):
    fit = value[fit_mask].float()
    mean = fit.mean(0, keepdim=True)
    centered = value.float() - mean
    gram = fit.double().sub(mean.double()).T @ fit.double().sub(mean.double())
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    floor = eigenvalues[-1].clamp_min(1e-30) * 1e-12
    inverse = eigenvectors @ torch.diag(eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
    basis = centered @ inverse.float()
    identity = torch.eye(HD, device=value.device)
    error = float((basis[fit_mask].T @ basis[fit_mask] - identity).abs().max())
    return basis, error


def _orthonormal_basis(value, expected_rank):
    gram = value.double().T @ value.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    threshold = eigenvalues[0].clamp_min(1e-30) * 1e-9
    effective = int((eigenvalues > threshold).sum())
    keep = min(expected_rank, effective)
    transform = (eigenvectors[:, :keep] * eigenvalues[:keep].rsqrt()).float()
    result = value @ transform
    error = float((result.T @ result - torch.eye(keep, device=value.device)).abs().max())
    return result, effective, error


def _overlap(left, right):
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    return float(singular.square().sum() / min(left.shape[1], right.shape[1]))


def _pair_values(bases):
    values = []
    for left_index, left in enumerate(ENTRIES):
        for right in ENTRIES[left_index + 1:]:
            if left[0] != right[0]:
                values.append(_overlap(bases[left], bases[right]))
    return values


def _stats(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "sd": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "p99": float(torch.quantile(tensor, .99)),
    }


def _top_average_projector(bases, seed):
    stack = torch.cat([bases[entry] for entry in ENTRIES], dim=1) / len(ENTRIES) ** .5
    devices = [stack.device.index] if stack.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if stack.is_cuda:
            torch.cuda.manual_seed_all(seed)
        left, singular, _right = torch.pca_lowrank(
            stack, q=2 * CARRIER_RANK, center=False, niter=8)
    carrier = left[:, :CARRIER_RANK]
    rayleigh = singular[:CARRIER_RANK].square()
    residual = stack @ (stack.T @ carrier) - carrier * rayleigh
    relative_residual = float(residual.norm() / (carrier * rayleigh).norm().clamp_min(1e-30))
    return carrier, rayleigh, relative_residual


def _right_projector(carrier, fit_basis):
    coordinates = carrier.T @ fit_basis
    _u, singular, vh = torch.linalg.svd(coordinates, full_matrices=False)
    right = vh[:CARRIER_RANK].T
    projector = right @ right.T
    capture = float(coordinates.square().sum() / CARRIER_RANK)
    return projector, capture, singular


def _component_bases(full_bases, fit_mask, select_mask, carrier, seed, side_index):
    carrier_only = {}
    remainder = {}
    random_remainder = {}
    projectors = {}
    captures = {}
    ranks = {}
    orth_errors = []
    for entry_index, entry in enumerate(ENTRIES):
        basis = full_bases[entry]
        projector, capture, singular = _right_projector(carrier, basis[fit_mask])
        projectors[entry] = projector
        captures[entry] = capture
        carrier_basis, carrier_rank, carrier_error = _orthonormal_basis(
            basis[select_mask] @ projector, CARRIER_RANK)
        residual_basis, residual_rank, residual_error = _orthonormal_basis(
            basis[select_mask] @ (torch.eye(HD, device=basis.device) - projector),
            HD - CARRIER_RANK)
        generator = torch.Generator(device="cpu").manual_seed(
            seed + side_index * 100 + entry_index)
        random = torch.randn(HD, CARRIER_RANK, generator=generator, device="cpu").to(basis.device)
        random_right = torch.linalg.qr(random, mode="reduced").Q
        random_projector = random_right @ random_right.T
        random_basis, random_rank, random_error = _orthonormal_basis(
            basis[select_mask] @ (torch.eye(HD, device=basis.device) - random_projector),
            HD - CARRIER_RANK)
        carrier_only[entry] = carrier_basis
        remainder[entry] = residual_basis
        random_remainder[entry] = random_basis
        ranks[str(entry)] = {
            "carrier": carrier_rank, "remainder": residual_rank,
            "haar_remainder": random_rank,
            "fit_carrier_coordinate_min_singular": float(singular[-1]),
        }
        orth_errors.extend((carrier_error, residual_error, random_error))
    return {
        "carrier": carrier_only,
        "remainder": remainder,
        "haar_remainder": random_remainder,
        "projectors": projectors,
        "captures": captures,
        "ranks": ranks,
        "orth_error": max(orth_errors),
    }


def _permuted_carrier(fit_bases, seed, side_index):
    permuted = {}
    for entry_index, entry in enumerate(ENTRIES):
        generator = torch.Generator(device="cpu").manual_seed(
            seed + side_index * 100 + entry_index)
        permutation = torch.randperm(
            len(fit_bases[entry]), generator=generator, device="cpu").to(
                fit_bases[entry].device)
        permuted[entry] = fit_bases[entry][permutation]
    carrier, eigenvalues, residual = _top_average_projector(permuted, seed + 10_000)
    return carrier, eigenvalues, residual


def _capture_mlp0(model):
    block0 = model.transformer.h[0]
    z_rows = []
    action_rows = []
    for start in range(0, VOCAB, 256):
        token = torch.arange(start, min(start + 256, VOCAB), device="cuda").view(-1, 1)
        raw = F.rms_norm(model.transformer.wte(token), (D,))
        remix = (block0.lambdas[0] + block0.lambdas[1]) * raw
        attention, _value = block0.attn(F.rms_norm(remix, (D,)), None)
        z = F.rms_norm(remix + attention, (D,))
        write = block0.mlp(z)
        action = write - block0.mlp.Down_bias.view(1, 1, D).to(write)
        z_rows.append(z[:, 0].float())
        action_rows.append(action[:, 0].float())
    return torch.cat(z_rows), torch.cat(action_rows)


def _standardize(train, full):
    mean = train.mean(0, keepdim=True)
    scale = (train - mean).square().mean().sqrt().clamp_min(1e-12)
    return (full - mean) / scale, mean, scale


def _degree_one(z, action, fit_mask):
    fit_z = z[fit_mask]
    fit_action = action[fit_mask]
    covariance = fit_z.T @ fit_z / len(fit_z)
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = values[-1] * 1e-6
    inverse = (vectors * values.clamp_min(floor).reciprocal()) @ vectors.T
    coefficient = inverse @ (fit_z.T @ fit_action / len(fit_z))
    return z @ coefficient, {
        "covariance_floor": float(floor),
        "effective_rank": int((values > floor).sum()),
        "coefficient_rank": int(torch.linalg.matrix_rank(coefficient)),
    }


def _leading_token_basis(value, fit_mask, rank):
    fit = value[fit_mask].float()
    fit = fit - fit.mean(0, keepdim=True)
    gram = fit.T @ fit
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    values = values[-rank:]
    vectors = vectors[:, -rank:]
    result = fit @ (vectors * values.clamp_min(values[-1] * 1e-10).rsqrt())
    error = float((result.T @ result - torch.eye(rank, device=result.device)).abs().max())
    return result, values, error


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert VOCAB == 50_257 and len(ENTRIES) == 18
        assert CARRIER_RANK == 24 and MLP_RANK == 64
        assert N_HEAD * HD == D
        print("ATTENTION0 QK COMMON CARRIER | dry run: rank24, heldout transport, MLP0 L")
        return

    started = time.time()
    sys.path.insert(0, str(QK))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    from tier2_model import load_elriggs, reference_forward
    from tier2_folding import branch_factors, scores_from_factors
    import attention0_cross_head_qk_shared_half as parent
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    device = torch.device("cuda")
    model, config = load_elriggs("bilin18", device=device, dtype=torch.float64)
    gate_factors = {
        branch: branch_factors(model, branch, dtype=torch.float64) for branch in (1, 2)}
    factors = {
        branch: branch_factors(model, branch, dtype=torch.float32) for branch in (1, 2)}
    receipt = json.loads(ROWS_RECEIPT.read_text())
    rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    fold_errors = parent._fold_gate(
        model, gate_factors, rows, scores_from_factors, reference_forward)
    del gate_factors

    token_ids = torch.arange(VOCAB, device=device)
    fit_mask = token_ids.remainder(5) != 4
    select_mask = ~fit_mask
    full_bases = {side: {} for side in SIDES}
    whiten_errors = []
    unit_errors = []
    for entry in ENTRIES:
        for side in SIDES:
            value = parent._factor(factors, entry, side)
            unit_errors.append(float((value.square().mean(-1) - 1).abs().max()))
            basis, error = _whiten_full(value, fit_mask)
            full_bases[side][entry] = basis
            whiten_errors.append(error)

    parent_result = json.loads(PARENT_RESULT.read_text())
    parent_means = {
        side: float(torch.tensor([
            pair[f"{side}_centered"]["projector_overlap"]
            for pair in parent_result["pairwise"]], dtype=torch.float64).mean())
        for side in SIDES}

    side_results = {}
    carriers = {}
    max_component_orth_error = 0.0
    for side_index, side in enumerate(SIDES):
        fit_bases = {entry: full_bases[side][entry][fit_mask] for entry in ENTRIES}
        original_fit = _stats(_pair_values(fit_bases))
        original_select_bases = {}
        for entry in ENTRIES:
            original_select_bases[entry], _rank, error = _orthonormal_basis(
                full_bases[side][entry][select_mask], HD)
            max_component_orth_error = max(max_component_orth_error, error)
        original_select = _stats(_pair_values(original_select_bases))

        carrier, spectrum, solver_residual = _top_average_projector(fit_bases, 420)
        carrier_repeat, repeat_spectrum, repeat_residual = _top_average_projector(fit_bases, 421)
        stability = _overlap(carrier, carrier_repeat)
        permuted, permuted_spectrum, permuted_residual = _permuted_carrier(
            fit_bases, 420_000, side_index)
        components = _component_bases(
            full_bases[side], fit_mask, select_mask, carrier, 420_500, side_index)
        permuted_components = _component_bases(
            full_bases[side], fit_mask, select_mask, permuted, 420_700, side_index)
        max_component_orth_error = max(
            max_component_orth_error, components["orth_error"],
            permuted_components["orth_error"])
        carrier_stats = _stats(_pair_values(components["carrier"]))
        remainder_stats = _stats(_pair_values(components["remainder"]))
        haar_remainder_stats = _stats(_pair_values(components["haar_remainder"]))
        permuted_carrier_stats = _stats(_pair_values(permuted_components["carrier"]))
        captures = list(components["captures"].values())
        side_results[side] = {
            "original_fit": original_fit,
            "original_select": original_select,
            "parent_fit_mean": parent_means[side],
            "parent_fit_mean_abs_error": abs(original_fit["mean"] - parent_means[side]),
            "carrier_average_projector_eigenvalues": spectrum.cpu().tolist(),
            "carrier_repeat_eigenvalues": repeat_spectrum.cpu().tolist(),
            "carrier_solver_relative_residual": solver_residual,
            "carrier_repeat_solver_relative_residual": repeat_residual,
            "carrier_repeat_projector_overlap": stability,
            "branch_carrier_capture": {
                "mean": sum(captures) / len(captures),
                "min": min(captures), "max": max(captures)},
            "select_carrier_only": carrier_stats,
            "select_private_remainder": remainder_stats,
            "select_haar_removal_remainder": haar_remainder_stats,
            "select_permuted_carrier_only": permuted_carrier_stats,
            "permuted_average_projector_eigenvalues": permuted_spectrum.cpu().tolist(),
            "permuted_solver_relative_residual": permuted_residual,
            "component_ranks": components["ranks"],
        }
        carriers[side] = carrier

    qk_carrier_overlap = _overlap(carriers["q"], carriers["k"])

    # The factor tables are no longer needed.  Cast the same checkpoint to float32
    # before the exhaustive length-one MLP0 path.
    del factors, full_bases
    model = model.float()
    z_native, action_native = _capture_mlp0(model)
    z, z_mean, z_scale = _standardize(z_native[fit_mask], z_native)
    action, action_mean, action_scale = _standardize(
        action_native[fit_mask], action_native)
    linear_standard, degree_one = _degree_one(z, action, fit_mask)
    linear = linear_standard * action_scale
    mean_action = action_mean.expand_as(action_native)
    quadratic = action_native - mean_action - linear
    mlp_objects = {
        "L": linear,
        "z": z_native,
        "F": action_native,
        "Q": quadratic,
    }
    mlp_bases = {}
    mlp_spectra = {}
    mlp_orth_errors = []
    for name, value in mlp_objects.items():
        basis, spectrum, error = _leading_token_basis(value, fit_mask, MLP_RANK)
        mlp_bases[name] = basis
        mlp_spectra[name] = spectrum.cpu().tolist()
        mlp_orth_errors.append(error)
    generator = torch.Generator(device="cpu").manual_seed(420_900)
    permutation = torch.randperm(
        int(fit_mask.sum()), generator=generator, device="cpu").to(device)
    l_permuted = mlp_bases["L"][permutation]
    mlp_alignment = {}
    for side in SIDES:
        mlp_alignment[side] = {
            name: _overlap(carriers[side], basis)
            for name, basis in mlp_bases.items()}
        mlp_alignment[side]["L_token_permuted"] = _overlap(
            carriers[side], l_permuted)
        mlp_alignment[side]["L_margin_over_permuted"] = (
            mlp_alignment[side]["L"] - mlp_alignment[side]["L_token_permuted"])

    pred_a = (
        max(fold_errors.values()) <= 1e-10
        and max(unit_errors) <= 1e-6
        and max(whiten_errors) <= 2e-4
        and max_component_orth_error <= 2e-4
        and all(side_results[side]["parent_fit_mean_abs_error"] <= .002 for side in SIDES)
        and all(side_results[side]["carrier_repeat_projector_overlap"] >= .95 for side in SIDES)
        and int(fit_mask.sum() + select_mask.sum()) == VOCAB
        and not bool((fit_mask & select_mask).any())
        and all(torch.isfinite(carrier).all() for carrier in carriers.values()))
    pred_b = (
        all(side_results[side]["branch_carrier_capture"]["mean"] >= .65 for side in SIDES)
        and qk_carrier_overlap >= .25)
    pred_c = all(
        side_results[side]["select_carrier_only"]["mean"] >= .70
        and side_results[side]["select_private_remainder"]["mean"] <= .03
        and side_results[side]["select_private_remainder"]["mean"]
            <= .25 * side_results[side]["original_select"]["mean"]
        and side_results[side]["select_haar_removal_remainder"]["mean"] >= .12
        and side_results[side]["select_carrier_only"]["mean"]
            - side_results[side]["select_permuted_carrier_only"]["mean"] >= .40
        for side in SIDES)
    pred_d = (
        all(mlp_alignment[side]["L"] >= .25 for side in SIDES)
        and max(mlp_alignment[side]["L"] for side in SIDES) >= .40
        and all(mlp_alignment[side]["L_margin_over_permuted"] >= .20 for side in SIDES))
    strong_null = (
        not pred_a
        or any(side_results[side]["select_carrier_only"]["mean"] < .30 for side in SIDES)
        or any(side_results[side]["select_private_remainder"]["mean"] > .10 for side in SIDES)
        or any(mlp_alignment[side]["L_margin_over_permuted"] < .10 for side in SIDES))

    result = {
        "status": "attention0_qk_common_carrier_complete",
        "rung": 420,
        "claim_level": "gauge_invariant_token_function_identification_not_compression_or_adoption",
        "definition": {
            "carrier": "top24 token-function eigenspace of the mean of 18 centered branch projectors, separately for q and k",
            "private_remainder": "heldout branch function after removing the FIT carrier-aligned rank24 right-coordinate subspace",
            "mlp0_L": "complete degree-one least-squares component of exact length-one bias-free MLP0 token action",
        },
        "population": {
            "real_tokens": VOCAB, "FIT_mod_not4": int(fit_mask.sum()),
            "SELECT_mod4": int(select_mask.sum()), "FINAL_opened": 0},
        "exactness": {
            "fold_max_abs_by_branch": fold_errors,
            "factor_unit_rms_max_abs": max(unit_errors),
            "fit_whitening_max_abs": max(whiten_errors),
            "component_orthogonality_max_abs": max_component_orth_error,
            "mlp_token_basis_orthogonality_max_abs": max(mlp_orth_errors),
        },
        "rank": {"carrier": CARRIER_RANK, "mlp_comparison": MLP_RANK},
        "sides": side_results,
        "q_vs_k_carrier_projector_overlap": qk_carrier_overlap,
        "mlp0": {
            "degree_one_fit": degree_one,
            "z_mean_shape": list(z_mean.shape), "z_scale": float(z_scale),
            "action_scale": float(action_scale),
            "leading64_eigenvalues": mlp_spectra,
            "carrier_alignment": mlp_alignment,
        },
        'pred_a_exact_reproducible_instrument': bool(pred_a),
        'pred_b_global_fit_carrier': bool(pred_b),
        'pred_c_unseen_carrier_and_private_residual': bool(pred_c),
        'pred_d_mlp0_degree_one_connection': bool(pred_d),
        "strong_null_no_transporting_qk_mlp0_common_carrier": bool(strong_null),
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "next_step": (
            "native_qk_score_and_downstream_carrier_intervention"
            if pred_a and pred_b and pred_c and pred_d and not strong_null
            else "attention_only_carrier_intervention" if pred_a and pred_b and pred_c and not strong_null
            else "continuous_coupled_qk_times_ov_block_term" if pred_a
            else "instrument_repair_only"),
        "config": config,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "exactness": result["exactness"],
        "q_vs_k_carrier_overlap": qk_carrier_overlap,
        "sides": side_results, "mlp_alignment": mlp_alignment,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c, "pred_d": pred_d,
        "strong_null": strong_null, "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("ATTENTION0 QK COMMON CARRIER DONE", flush=True)


if __name__ == "__main__":
    main()
