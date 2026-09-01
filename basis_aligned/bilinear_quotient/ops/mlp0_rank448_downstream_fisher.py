"""RUNG 406 -- DOWNSTREAM CE-FISHER SHARED-INPUT p448 SCREEN.

Retain the full directional suffix-CE gradient at each MLP0 input, form its
Fisher Gram, and build a legal p448 shared-input program in the unchanged
covariance whitening frame.  Compare covariance and eigenvalue-matched
coordinate-shuffled Fisher controls on four frozen waves, then use rung403's
exact 32-arm branch instrument on the fixed Fisher candidate.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "mlp0_rank448_downstream_fisher_results.json"
PARENT_LARGE = BQ / "mlp0_rank448_branch_large_confirmation_results.json"
PARENT_BRANCH = BQ / "mlp0_rank448_branch_error_factorial_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
CONFIRM_RECEIPT = BQ / "mlp0_c512_mlp1_interchange_v1_rows_receipt.json"
CONFIRM_CACHE = BQ / ".rowcache_mlp0_c512_mlp1_interchange_v1/eval_384_source_documents.pt"
FIT_CACHE = BQ / ".rowcache/fineweb_n192_skip11000.pt"
FIT_SLICE = (0, 24)
RANK = 448
D = 1152
H = 4608
FIT_HALF = 12
SOURCE_DOCUMENTS = 384
WAVE_DOCUMENTS = 96
N_WAVES = 4
SHUFFLE_SEED = 406
PROGRAM_NAMES = ("covariance", "Fisher", "Fisher_shuffled")
COMPACT_VALUES = D * RANK + 2 * H * RANK + H * D + D
SAVING_VALUES = 3 * H * D + D - COMPACT_VALUES


def _tensor_sha256(value: torch.Tensor):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _manual_logits(model, index):
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    logits = model.lm_head(F.rms_norm(x, (D,)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


def _fisher_metrics(model, rows, device):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    accumulators = {
        "all": torch.zeros(D, D, device=device),
        "half_a": torch.zeros(D, D, device=device),
        "half_b": torch.zeros(D, D, device=device),
    }
    counts = {name: 0 for name in accumulators}
    document_nonzero = []
    gradient_norms = []
    for start in range(0, len(rows), 2):
        state = {}

        def pre_hook(_module, args):
            leaf = args[0].detach().requires_grad_(True)
            state["leaf"] = leaf
            return (leaf,)

        handle = model.transformer.h[0].mlp.register_forward_pre_hook(pre_hook)
        try:
            batch = rows[start:start + 2]
            index = batch[:, :-1].to(device)
            target = batch[:, 1:].to(device)
            logits = _manual_logits(model, index)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum")
            loss.backward()
        finally:
            handle.remove()
        leaf = state["leaf"]
        if leaf.grad is None:
            raise RuntimeError("MLP0 input leaf received no downstream CE gradient")
        gradient = leaf.grad.detach().reshape(-1, D).float()
        batch_documents = len(batch)
        rows_per_document = gradient.shape[0] // batch_documents
        for offset in range(batch_documents):
            document_index = start + offset
            value = gradient[offset * rows_per_document:(offset + 1) * rows_per_document]
            nonzero = bool(float(value.norm()) > 0 and torch.isfinite(value).all())
            document_nonzero.append(nonzero)
            gradient_norms.append(value.norm(dim=1).cpu())
            gram = value.T @ value
            accumulators["all"] += gram
            counts["all"] += len(value)
            half = "half_a" if document_index < FIT_HALF else "half_b"
            accumulators[half] += gram
            counts[half] += len(value)
        del logits, loss, leaf, gradient, state
    metrics = {}
    symmetry = {}
    for name, accumulator in accumulators.items():
        value = accumulator / counts[name]
        symmetry[name] = float((value - value.T).norm() / value.norm().clamp_min(1e-30))
        metrics[name] = 0.5 * (value + value.T)
    norms = torch.cat(gradient_norms)
    return metrics, {
        "fit_documents": len(rows),
        "positions": counts["all"],
        "half_positions": [counts["half_a"], counts["half_b"]],
        "all_documents_finite_nonzero": all(document_nonzero),
        "nonzero_document_count": sum(document_nonzero),
        "gradient_norm_mean": float(norms.mean()),
        "gradient_norm_median": float(norms.median()),
        "gradient_norm_p95": float(norms.quantile(.95)),
        "raw_metric_symmetry_relative_error": symmetry,
        "metric_hashes": {name: _tensor_sha256(value) for name, value in metrics.items()},
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (
            PARENT_LARGE, PARENT_BRANCH, ROWS_RECEIPT,
            CONFIRM_RECEIPT, CONFIRM_CACHE, FIT_CACHE))
        assert FIT_SLICE == (0, 24) and FIT_HALF == 12
        assert COMPACT_VALUES == 9_954_432 and SAVING_VALUES == 5_971_968
        assert SOURCE_DOCUMENTS == N_WAVES * WAVE_DOCUMENTS
        print("MLP0 p448 DOWNSTREAM FISHER | dry run: rows, metric, controls, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(BQ.parent / "qk_mdl"))
    import bilin18_observed_model_facade as facade
    import mlp0_centered_context_anova_factorial as base
    import mlp0_rank448_branch_error_factorial as rung403
    import mlp0_rank448_token_grammar_active_subspace as rung405
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits as covariance_logits

    device = torch.device("cuda")
    parent_large = json.loads(PARENT_LARGE.read_text())
    parent_branch = json.loads(PARENT_BRANCH.read_text())
    if (not parent_large["null_large_confirmation_fails"]
            or not json.loads((BQ / "mlp0_rank448_token_grammar_active_subspace_results.json").read_text())[
                "null_first_order_token_grammar_metric_fails"]):
        raise RuntimeError("prior nulls do not license downstream Fisher")

    receipt = json.loads(CONFIRM_RECEIPT.read_text())
    current_rows_receipt = json.loads(ROWS_RECEIPT.read_text())
    all_rows = torch.load(CONFIRM_CACHE, map_location="cpu")
    records = receipt["document_provenance"]["sets"]["eval"]
    chunk0 = [index for index, record in enumerate(records) if int(record["chunk_id"]) == 0]
    ordinals = [int(records[index]["source_document_ordinal"]) for index in chunk0]
    document_ids = [records[index]["document_id"] for index in chunk0]
    confirm_rows = all_rows[chunk0, :257].long().contiguous()
    population_exact = (
        tuple(all_rows.shape) == (585, 513)
        and tuple(confirm_rows.shape) == (SOURCE_DOCUMENTS, 257)
        and ordinals == list(range(SOURCE_DOCUMENTS))
        and len(set(document_ids)) == SOURCE_DOCUMENTS
        and _tensor_sha256(all_rows) == receipt["entries"]["eval"]["tensor_raw_sha256"]
        and all(receipt["disjointness_gates"].values())
        and all(current_rows_receipt["disjointness"].values())
    )
    waves = [
        confirm_rows[index * WAVE_DOCUMENTS:(index + 1) * WAVE_DOCUMENTS]
        for index in range(N_WAVES)
    ]
    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows_program = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()

    source_model, source_checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    covariance = _covariance(source_model, fit_rows_program, covariance_logits)
    covariance_program, covariance_basis, covariance_diagnostic = _rrr_program(
        source_model.transformer.h[0].mlp, covariance, rank=RANK)
    with torch.enable_grad():
        fisher_metrics, gradient_diagnostic = _fisher_metrics(
            source_model, fit_rows_program, device)
    covariance_square, covariance_inverse, _covariance_values, _floor = rung405._whitening(
        covariance)
    whitened = {
        name: covariance_square @ value @ covariance_square
        for name, value in fisher_metrics.items()
    }
    whitened = {name: 0.5 * (value + value.T) for name, value in whitened.items()}
    fisher_basis, fisher_values, fisher_orthogonality = rung405._top_basis(whitened["all"])
    half_a_basis, _half_a_values, half_a_orthogonality = rung405._top_basis(
        whitened["half_a"])
    half_b_basis, _half_b_values, half_b_orthogonality = rung405._top_basis(
        whitened["half_b"])
    half_overlap = float((half_a_basis.T @ half_b_basis).square().sum() / RANK)
    shuffle_generator = torch.Generator(device=device)
    shuffle_generator.manual_seed(SHUFFLE_SEED)
    permutation = torch.randperm(D, generator=shuffle_generator, device=device)
    shuffled_basis = fisher_basis[permutation].contiguous()
    shuffled_orthogonality = float((
        shuffled_basis.T @ shuffled_basis - torch.eye(RANK, device=device)
    ).abs().max())
    programs = {
        "covariance": {name: value.detach().clone() for name, value in covariance_program.items()},
        "Fisher": rung405._program_from_basis(
            source_model.transformer.h[0].mlp, fisher_basis,
            covariance_square, covariance_inverse),
        "Fisher_shuffled": rung405._program_from_basis(
            source_model.transformer.h[0].mlp, shuffled_basis,
            covariance_square, covariance_inverse),
    }
    fisher_positive = fisher_values.clamp_min(0)
    fisher_top_energy = float(
        fisher_positive[-RANK:].sum() / fisher_positive.sum().clamp_min(1e-30))
    metric_diagnostic = {
        **gradient_diagnostic,
        "shuffle_seed": SHUFFLE_SEED,
        "fisher_whitened_hash": _tensor_sha256(whitened["all"]),
        "fisher_basis_hash": _tensor_sha256(fisher_basis),
        "shuffled_basis_hash": _tensor_sha256(shuffled_basis),
        "shuffled_metric_eigenvalue_relative_error": 0.0,
        "basis_orthogonality_max_abs_error": {
            "Fisher": fisher_orthogonality,
            "half_a": half_a_orthogonality,
            "half_b": half_b_orthogonality,
            "Fisher_shuffled": shuffled_orthogonality,
        },
        "half_fit_normalized_subspace_overlap": half_overlap,
        "top448_fisher_energy": fisher_top_energy,
    }
    del fisher_metrics, whitened, fisher_values, half_a_basis, half_b_basis
    del covariance, covariance_square, covariance_inverse, source_model
    torch.cuda.empty_cache()

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    native_factors = {
        "left": model.transformer.h[0].mlp.Left.weight.detach().float(),
        "right": model.transformer.h[0].mlp.Right.weight.detach().float(),
        "down": model.transformer.h[0].mlp.Down.weight.detach().float(),
    }
    fit_rows_reference = rows_parent.load_role(current_rows_receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(current_rows_receipt["entries"]["SELECT"])
    token_fit, context_fit, gain_fit = base._capture_inputs(model, fit_rows_reference, device)
    native_reference = rung403._reference_for_factors(
        token_fit, context_fit, gain_fit, native_factors, device)
    references = {
        name: rung403._reference_for_factors(
            token_fit, context_fit, gain_fit, program, device)
        for name, program in programs.items()
    }

    wave_results = []
    for index, rows in enumerate(waves):
        branch, denominators = rung405._branch_errors(
            model, rows, base, native_factors, native_reference,
            programs, references, device)
        physical = rung405._score_wave(model, rows, programs, rung403, device)
        wave_results.append({
            "wave": index,
            "branch_relative_mse": branch,
            "native_branch_squared_energy": denominators,
            **physical,
        })
        print(f"wave {index}: " + " ".join(
            f"{name}={physical['damage'][name]:+.6f}" for name in PROGRAM_NAMES),
            flush=True)
    fisher_factorial = rung403._score_role(
        model, select_rows, device, base, native_factors, programs["Fisher"],
        native_reference, references["Fisher"])

    pooled_damage = {
        name: sum(wave["damage"][name] for wave in wave_results) / N_WAVES
        for name in PROGRAM_NAMES
    }
    pooled_branch = {}
    for name in PROGRAM_NAMES:
        pooled_branch[name] = {}
        for branch in ("T", "I"):
            numerator = sum(
                wave["branch_relative_mse"][name][branch]
                * wave["native_branch_squared_energy"][branch]
                for wave in wave_results)
            denominator = sum(
                wave["native_branch_squared_energy"][branch] for wave in wave_results)
            pooled_branch[name][branch] = numerator / denominator

    expected_shapes = {
        "encoder": (RANK, D), "left": (H, RANK), "right": (H, RANK),
        "down": (D, H), "bias": (D,),
    }
    shapes_exact = all(
        {key: tuple(value.shape) for key, value in program.items()} == expected_shapes
        for program in programs.values())
    baseline_wave_reference = [
        parent_large["waves"][str(index)]["total_compact_ce_damage"]
        for index in range(N_WAVES)
    ]
    baseline_reproduction_error = max(
        abs(wave_results[index]["damage"]["covariance"] - baseline_wave_reference[index])
        for index in range(N_WAVES))
    factorial_diag = fisher_factorial["diagnostics"]
    metric_exact = (
        metric_diagnostic["all_documents_finite_nonzero"]
        and all(math.isfinite(value) for value in (
            metric_diagnostic["gradient_norm_mean"],
            metric_diagnostic["gradient_norm_median"],
            metric_diagnostic["gradient_norm_p95"]))
        and max(metric_diagnostic["raw_metric_symmetry_relative_error"].values()) <= 1e-6
        and max(metric_diagnostic["basis_orthogonality_max_abs_error"].values()) <= 1e-5
        and metric_diagnostic["shuffled_metric_eigenvalue_relative_error"] <= 1e-5
    )
    endpoints_exact = all(
        max(wave["pre_mlp0_state_replay_max_abs_error"].values()) == 0.0
        and max(wave["compact_endpoint_duplicate_max_abs_error"].values()) == 0.0
        and all(value == WAVE_DOCUMENTS // 4 for value in wave["calls"].values())
        and wave["scored_positions"] == WAVE_DOCUMENTS * 192
        for wave in wave_results)
    factorial_exact = (
        factorial_diag["native_analytical_identity_relative_mse"] <= 1e-8
        and factorial_diag["compact_analytical_identity_relative_mse"] <= 1e-8
        and factorial_diag["native_endpoint_state_max_abs_error"] == 0.0
        and factorial_diag["compact_endpoint_state_max_abs_error"] == 0.0
        and factorial_diag["pre_mlp0_state_replay_max_abs_error"] == 0.0
        and factorial_diag["live_census"])
    pred_a = (
        population_exact and shapes_exact and metric_exact and endpoints_exact and factorial_exact
        and int(covariance_diagnostic["rank"]) == RANK
        and COMPACT_VALUES == 9_954_432
        and baseline_reproduction_error <= 1e-6)
    pred_b = (
        half_overlap >= .50
        and pooled_damage["Fisher"] <= pooled_damage["Fisher_shuffled"] - .001)
    wave_improvement = [
        wave["damage"]["covariance"] - wave["damage"]["Fisher"]
        for wave in wave_results
    ]
    pred_c = (
        pooled_damage["Fisher"] <= .85 * pooled_damage["covariance"]
        and sum(value >= .0002 for value in wave_improvement) >= 3
        and min(wave_improvement) >= -.001)
    baseline_shapley = parent_branch["roles"]["SELECT"]["shapley_ce_damage"]
    fisher_shapley = fisher_factorial["shapley_ce_damage"]
    baseline_ti = baseline_shapley["T"] + baseline_shapley["I"]
    fisher_ti = fisher_shapley["T"] + fisher_shapley["I"]
    ti_reduction = baseline_ti - fisher_ti
    baseline_other = sum(baseline_shapley[name] for name in ("C", "S", "A"))
    fisher_other = sum(fisher_shapley[name] for name in ("C", "S", "A"))
    pred_d = (
        fisher_ti <= .75 * baseline_ti
        and fisher_factorial["total_compact_ce_damage"]
        <= parent_branch["roles"]["SELECT"]["total_compact_ce_damage"]
        and abs(fisher_other - baseline_other) <= max(ti_reduction, 0.0))
    strong_null = (
        not pred_a
        or pooled_damage["covariance"] - pooled_damage["Fisher"] < .0002
        or pooled_damage["Fisher_shuffled"] - pooled_damage["Fisher"] < .0002
        or sum(value < 0 for value in wave_improvement) >= 3
        or half_overlap <= .25)

    result = {
        "status": "mlp0_rank448_downstream_fisher_complete",
        "rung": 406,
        "claim_level": "single_site_equal_price_directional_ce_fisher_screen_not_compression",
        "convention": "CE added above native; lower is better",
        "population": {
            "program_fit_cache": str(FIT_CACHE),
            "program_fit_slice": list(FIT_SLICE),
            "program_fit_documents": len(fit_rows_program),
            "evaluation_receipt": str(CONFIRM_RECEIPT),
            "source_documents": SOURCE_DOCUMENTS,
            "waves": N_WAVES,
            "documents_per_wave": WAVE_DOCUMENTS,
            "total_scored_positions": SOURCE_DOCUMENTS * 192,
            "population_exact": population_exact,
        },
        "program": {
            "source_checkpoint": source_checkpoint.__dict__,
            "source_dtype": "float32_as_in_rung328",
            "evaluation_dtype": "bfloat16_as_in_rung401",
            "rank": RANK,
            "literal_mlp0_values": COMPACT_VALUES,
            "saving_values": SAVING_VALUES,
            "shapes_exact": shapes_exact,
            "covariance_basis_hash": _tensor_sha256(covariance_basis),
            "covariance_diagnostic": covariance_diagnostic,
        },
        "metric_diagnostic": metric_diagnostic,
        "wave_results": {str(index): value for index, value in enumerate(wave_results)},
        "pooled_ce_damage": pooled_damage,
        "pooled_branch_relative_mse": pooled_branch,
        "fisher_wave_ce_improvement_over_covariance": wave_improvement,
        "baseline_wave_reproduction_max_abs_error": baseline_reproduction_error,
        "fisher_select_factorial": fisher_factorial,
        "select_shapley_comparison": {
            "baseline": baseline_shapley,
            "Fisher": fisher_shapley,
            "baseline_T_plus_I": baseline_ti,
            "Fisher_T_plus_I": fisher_ti,
            "T_plus_I_reduction": ti_reduction,
            "baseline_C_plus_S_plus_A": baseline_other,
            "Fisher_C_plus_S_plus_A": fisher_other,
        },
        'pred_a_fisher_instrument_and_equal_price_are_exact': bool(pred_a),
        'pred_b_directional_fisher_is_stable_and_beats_shuffle': bool(pred_b),
        'pred_c_fisher_improves_heldout_prediction_robustly': bool(pred_c),
        'pred_d_fisher_repairs_token_grammar_damage': bool(pred_d),
        "null_downstream_fisher_does_not_improve_p448": bool(strong_null),
        "next_object": (
            "whole_model_gate_for_fixed_downstream_fisher_p448"
            if pred_a and pred_b and pred_c and pred_d and not strong_null else None),
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 DOWNSTREAM FISHER DONE", flush=True)


if __name__ == "__main__":
    main()
