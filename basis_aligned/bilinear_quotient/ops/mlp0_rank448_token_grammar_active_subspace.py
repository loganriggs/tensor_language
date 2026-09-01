"""RUNG 405 -- GLOBAL T+I ACTIVE-SUBSPACE p448 SCREEN.

Build equal-price rank-448 MLP0 shared-input programs from exact output-space
Jacobians of the token-main T branch, token-by-context I branch, and their
equal-trace sum.  Compare with the unchanged covariance p448 and a seeded
random whitening-frame subspace on four frozen 96-document waves.

This is a tangent active-subspace screen, not compression/adoption.  See
MLP0_RANK448_TOKEN_GRAMMAR_ACTIVE_SUBSPACE_PREREGISTRATION.md.
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
OUT = BQ / "mlp0_rank448_token_grammar_active_subspace_results.json"
PARENT = BQ / "mlp0_rank448_branch_large_confirmation_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
CONFIRM_RECEIPT = BQ / "mlp0_c512_mlp1_interchange_v1_rows_receipt.json"
CONFIRM_CACHE = BQ / ".rowcache_mlp0_c512_mlp1_interchange_v1/eval_384_source_documents.pt"
FIT_CACHE = BQ / ".rowcache/fineweb_n192_skip11000.pt"
FIT_SLICE = (0, 24)
RANK = 448
D = 1152
H = 4608
SOURCE_DOCUMENTS = 384
WAVE_DOCUMENTS = 96
N_WAVES = 4
PROBES = 2
PROBE_SEED = 405
RANDOM_SEED = 1405
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
PROGRAM_NAMES = ("covariance", "T_active", "I_active", "TI_active", "random")
COMPACT_VALUES = D * RANK + 2 * H * RANK + H * D + D
SAVING_VALUES = 3 * H * D + D - COMPACT_VALUES


def _tensor_sha256(value: torch.Tensor):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _whitening(covariance: torch.Tensor):
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    floor = float(values[0]) * 1e-6
    safe = values.clamp_min(floor)
    square = (vectors * safe.sqrt()) @ vectors.T
    inverse = (vectors * safe.rsqrt()) @ vectors.T
    return square, inverse, values, floor


@torch.no_grad()
def _active_metrics(model, rows, covariance, base, device):
    token_cpu, context_cpu, _gain_cpu = base._capture_inputs(model, rows, device)
    token_mean = token_cpu.mean(0).to(device)
    context_mean = context_cpu.mean(0).to(device)
    total_mean = token_mean + context_mean
    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    lm = F.linear(total_mean, left)
    rm = F.linear(total_mean, right)
    metric_t = torch.zeros(D, D, device=device)
    metric_i = torch.zeros(D, D, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(PROBE_SEED)
    observations = 0
    for start in range(0, len(token_cpu), 128):
        token = token_cpu[start:start + 128].to(device) - token_mean
        context = context_cpu[start:start + 128].to(device) - context_mean
        lt, rt = F.linear(token, left), F.linear(token, right)
        lc, rc = F.linear(context, left), F.linear(context, right)
        batch = len(token)
        for _probe in range(PROBES):
            q = torch.randint(
                0, 2, (batch, D), generator=generator, device=device,
                dtype=torch.int8).float().mul_(2).sub_(1)
            hidden_probe = q @ down
            grad_t = (hidden_probe * (rm + rt)) @ left \
                + (hidden_probe * (lm + lt)) @ right
            grad_it = (hidden_probe * rc) @ left + (hidden_probe * lc) @ right
            grad_ic = (hidden_probe * rt) @ left + (hidden_probe * lt) @ right
            metric_t.addmm_(grad_t.T, grad_t)
            metric_i.addmm_(grad_it.T, grad_it)
            metric_i.addmm_(grad_ic.T, grad_ic)
            observations += batch
    metric_t /= observations
    metric_i /= observations
    symmetry = {
        "T": float((metric_t - metric_t.T).norm() / metric_t.norm().clamp_min(1e-30)),
        "I": float((metric_i - metric_i.T).norm() / metric_i.norm().clamp_min(1e-30)),
    }
    metric_t = 0.5 * (metric_t + metric_t.T)
    metric_i = 0.5 * (metric_i + metric_i.T)
    covariance_square, covariance_inverse, covariance_values, floor = _whitening(covariance)
    whitened_t = covariance_square @ metric_t @ covariance_square
    whitened_i = covariance_square @ metric_i @ covariance_square
    whitened_t = 0.5 * (whitened_t + whitened_t.T)
    whitened_i = 0.5 * (whitened_i + whitened_i.T)
    trace_t = float(torch.trace(whitened_t))
    trace_i = float(torch.trace(whitened_i))
    whitened_ti = whitened_t / trace_t + whitened_i / trace_i
    return {
        "T": whitened_t,
        "I": whitened_i,
        "TI": 0.5 * (whitened_ti + whitened_ti.T),
    }, covariance_square, covariance_inverse, {
        "probe_seed": PROBE_SEED,
        "probes_per_position": PROBES,
        "positions": len(token_cpu),
        "gradient_observations": observations,
        "raw_metric_symmetry_relative_error": symmetry,
        "whitened_trace": {"T": trace_t, "I": trace_i},
        "covariance_floor": floor,
        "covariance_top448_energy": float(
            covariance_values[:RANK].clamp_min(0).sum()
            / covariance_values.clamp_min(0).sum()),
        "metric_hash": {
            "T": _tensor_sha256(whitened_t),
            "I": _tensor_sha256(whitened_i),
            "TI": _tensor_sha256(whitened_ti),
        },
    }


@torch.no_grad()
def _program_from_basis(mlp, basis, covariance_square, covariance_inverse):
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    stacked = torch.cat((left, right), dim=0)
    encoder = basis.T @ covariance_inverse
    decoder = covariance_square @ basis
    coefficient = stacked @ decoder
    return {
        "encoder": encoder.contiguous(),
        "left": coefficient[:H].contiguous(),
        "right": coefficient[H:].contiguous(),
        "down": mlp.Down.weight.detach().float().clone(),
        "bias": mlp.Down_bias.detach().float().clone(),
    }


@torch.no_grad()
def _top_basis(metric):
    values, vectors = torch.linalg.eigh(metric)
    basis = vectors[:, torch.argsort(values, descending=True)[:RANK]].contiguous()
    orthogonality = float((basis.T @ basis - torch.eye(RANK, device=basis.device)).abs().max())
    return basis, values, orthogonality


@torch.no_grad()
def _ti_components(token, context, reference, factors):
    token_delta = token.float() - reference["token_mean"]
    context_delta = context.float() - reference["context_mean"]
    total_mean = reference["token_mean"] + reference["context_mean"]
    encoder = factors.get("encoder")
    if encoder is not None:
        token_delta = F.linear(token_delta, encoder)
        context_delta = F.linear(context_delta, encoder)
        total_mean = F.linear(total_mean, encoder)
    lt, rt = F.linear(token_delta, factors["left"]), F.linear(token_delta, factors["right"])
    lc, rc = F.linear(context_delta, factors["left"]), F.linear(context_delta, factors["right"])
    lm, rm = F.linear(total_mean, factors["left"]), F.linear(total_mean, factors["right"])
    down = factors["down"]
    token_main = F.linear(lt * rm + lm * rt + lt * rt, down)
    token_main -= reference["token_self_mean"]
    interaction = F.linear(lt * rc + lc * rt, down)
    gain_mean = reference["gain_mean"]
    return gain_mean * token_main, gain_mean * interaction


@torch.no_grad()
def _branch_errors(model, rows, base, native_factors, native_reference,
                   programs, references, device):
    token_cpu, context_cpu, _gain_cpu = base._capture_inputs(model, rows, device)
    numerators = {name: {"T": 0.0, "I": 0.0} for name in programs}
    denominators = {"T": 0.0, "I": 0.0}
    for start in range(0, len(token_cpu), 256):
        token = token_cpu[start:start + 256].to(device)
        context = context_cpu[start:start + 256].to(device)
        native_t, native_i = _ti_components(
            token, context, native_reference, native_factors)
        denominators["T"] += float(native_t.square().sum())
        denominators["I"] += float(native_i.square().sum())
        for name, factors in programs.items():
            compact_t, compact_i = _ti_components(
                token, context, references[name], factors)
            numerators[name]["T"] += float((compact_t - native_t).square().sum())
            numerators[name]["I"] += float((compact_i - native_i).square().sum())
    return {
        name: {
            branch: numerators[name][branch] / max(denominators[branch], 1e-30)
            for branch in ("T", "I")
        }
        for name in programs
    }, denominators


def _manual_logits(model, index):
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def _score_wave(model, rows, programs, rung403, device):
    # Match rung403/404 exactly: mean token CE inside each document in float32,
    # convert each document mean to a Python float, then pool in float64.
    document_ce = {"native": [], **{name: [] for name in programs}}
    calls = {name: 0 for name in programs}
    state_error = {name: 0.0 for name in programs}
    endpoint_error = {name: 0.0 for name in programs}
    for start in range(0, len(rows), DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        index = batch[:, :-1].to(device)
        target = batch[:, 1:].to(device)
        baseline_state = {}

        def native_hook(_module, args, output):
            baseline_state["value"] = args[0].detach().clone()
            return output

        handle = model.transformer.h[0].mlp.register_forward_hook(native_hook)
        logits = _manual_logits(model, index)
        handle.remove()
        losses = F.cross_entropy(
            logits[:, SCORING].float().transpose(1, 2),
            target[:, SCORING], reduction="none"
        ).mean(1)
        document_ce["native"].extend(float(loss) for loss in losses)
        for name, program in programs.items():
            def compact_hook(_module, args, output, name=name, program=program):
                calls[name] += 1
                state_error[name] = max(
                    state_error[name], float((args[0] - baseline_state["value"]).abs().max()))
                compact = rung403._compact_deployed(args[0], program, output.dtype)
                duplicate = rung403._compact_deployed(args[0], program, output.dtype)
                endpoint_error[name] = max(
                    endpoint_error[name], float((compact - duplicate).abs().max()))
                return compact

            handle = model.transformer.h[0].mlp.register_forward_hook(compact_hook)
            logits = _manual_logits(model, index)
            handle.remove()
            losses = F.cross_entropy(
                logits[:, SCORING].float().transpose(1, 2),
                target[:, SCORING], reduction="none"
            ).mean(1)
            document_ce[name].extend(float(loss) for loss in losses)
    ce = {
        name: float(torch.tensor(values, dtype=torch.float64).mean())
        for name, values in document_ce.items()
    }
    return {
        "ce": ce,
        "damage": {name: ce[name] - ce["native"] for name in programs},
        "calls": calls,
        "pre_mlp0_state_replay_max_abs_error": state_error,
        "compact_endpoint_duplicate_max_abs_error": endpoint_error,
        "scored_positions": len(rows) * 192,
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (
            PARENT, ROWS_RECEIPT, CONFIRM_RECEIPT, CONFIRM_CACHE, FIT_CACHE))
        assert COMPACT_VALUES == 9_954_432 and SAVING_VALUES == 5_971_968
        assert SOURCE_DOCUMENTS == N_WAVES * WAVE_DOCUMENTS
        assert PROBES == 2 and len(PROGRAM_NAMES) == 5
        print("MLP0 p448 TOKEN-GRAMMAR ACTIVE | dry run: rows, metrics, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(BQ.parent / "qk_mdl"))
    import bilin18_observed_model_facade as facade
    import mlp0_centered_context_anova_factorial as base
    import mlp0_rank448_branch_error_factorial as rung403
    import mlp0_rank448_branch_large_confirmation as rung404
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits as covariance_logits

    device = torch.device("cuda")
    parent = json.loads(PARENT.read_text())
    if not parent["null_large_confirmation_fails"]:
        raise RuntimeError("rung404 did not reject the interaction-only route")

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
    baseline, baseline_basis, baseline_diagnostic = _rrr_program(
        source_model.transformer.h[0].mlp, covariance, rank=RANK)
    metrics, covariance_square, covariance_inverse, metric_diagnostic = _active_metrics(
        source_model, fit_rows_program, covariance, base, device)
    bases = {}
    eigenvalues = {}
    orthogonality = {}
    for name, key in (("T_active", "T"), ("I_active", "I"), ("TI_active", "TI")):
        basis, values, orth = _top_basis(metrics[key])
        bases[name], eigenvalues[name], orthogonality[name] = basis, values, orth
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(RANDOM_SEED)
    random_matrix = torch.randn(D, RANK, device=device, generator=random_generator)
    bases["random"] = torch.linalg.qr(random_matrix, mode="reduced").Q.contiguous()
    orthogonality["random"] = float((
        bases["random"].T @ bases["random"] - torch.eye(RANK, device=device)
    ).abs().max())
    programs = {
        "covariance": {name: value.detach().clone() for name, value in baseline.items()},
        **{
            name: _program_from_basis(
                source_model.transformer.h[0].mlp, basis,
                covariance_square, covariance_inverse)
            for name, basis in bases.items()
        },
    }
    basis_hashes = {
        "covariance": _tensor_sha256(baseline_basis),
        **{name: _tensor_sha256(basis) for name, basis in bases.items()},
    }
    del metrics, eigenvalues, covariance, covariance_square, covariance_inverse, source_model
    torch.cuda.empty_cache()

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    native_factors = {
        "left": model.transformer.h[0].mlp.Left.weight.detach().float(),
        "right": model.transformer.h[0].mlp.Right.weight.detach().float(),
        "down": model.transformer.h[0].mlp.Down.weight.detach().float(),
    }
    fit_rows_reference = rows_parent.load_role(current_rows_receipt["entries"]["FIT"])
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
        branch, denominators = _branch_errors(
            model, rows, base, native_factors, native_reference,
            programs, references, device)
        physical = _score_wave(model, rows, programs, rung403, device)
        wave_results.append({
            "wave": index,
            "branch_relative_mse": branch,
            "native_branch_squared_energy": denominators,
            **physical,
        })
        print(f"wave {index}: " + " ".join(
            f"{name}={physical['damage'][name]:+.6f}" for name in PROGRAM_NAMES),
            flush=True)

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
        parent["waves"][str(index)]["total_compact_ce_damage"]
        for index in range(N_WAVES)
    ]
    baseline_reproduction_error = max(
        abs(wave_results[index]["damage"]["covariance"] - baseline_wave_reference[index])
        for index in range(N_WAVES))
    metrics_exact = (
        all(math.isfinite(value) and value > 0
            for value in metric_diagnostic["whitened_trace"].values())
        and max(metric_diagnostic["raw_metric_symmetry_relative_error"].values()) <= 1e-6
        and max(orthogonality.values()) <= 1e-5
    )
    endpoints_exact = all(
        max(wave["pre_mlp0_state_replay_max_abs_error"].values()) == 0.0
        and max(wave["compact_endpoint_duplicate_max_abs_error"].values()) == 0.0
        and all(value == WAVE_DOCUMENTS // DOCUMENT_BATCH
                for value in wave["calls"].values())
        and wave["scored_positions"] == WAVE_DOCUMENTS * 192
        for wave in wave_results)
    pred_a = (
        population_exact and shapes_exact and metrics_exact and endpoints_exact
        and int(baseline_diagnostic["rank"]) == RANK
        and COMPACT_VALUES == 9_954_432
        and baseline_reproduction_error <= 1e-6
    )
    baseline_t = pooled_branch["covariance"]["T"]
    baseline_i = pooled_branch["covariance"]["I"]
    joint_geomean = math.sqrt(
        pooled_branch["TI_active"]["T"] * pooled_branch["TI_active"]["I"])
    baseline_geomean = math.sqrt(baseline_t * baseline_i)
    random_geomean = math.sqrt(
        pooled_branch["random"]["T"] * pooled_branch["random"]["I"])
    pred_b = (
        pooled_branch["T_active"]["T"] <= .95 * baseline_t
        and pooled_branch["I_active"]["I"] <= .95 * baseline_i
        and joint_geomean <= .95 * baseline_geomean
        and joint_geomean <= .90 * random_geomean
    )
    wave_joint_improvement = [
        wave["damage"]["covariance"] - wave["damage"]["TI_active"]
        for wave in wave_results
    ]
    pred_c = (
        pooled_damage["TI_active"] <= .85 * pooled_damage["covariance"]
        and sum(value >= .0002 for value in wave_joint_improvement) >= 3
        and min(wave_joint_improvement) >= -.001
    )
    best_specialist = min(pooled_damage["T_active"], pooled_damage["I_active"])
    pred_d = (
        pooled_damage["TI_active"] <= best_specialist + .0005
        and pooled_branch["TI_active"]["T"] <= 1.05 * baseline_t
        and pooled_branch["TI_active"]["I"] <= 1.05 * baseline_i
        and 0 <= pooled_damage["TI_active"] <= .020
    )
    specialist_t_gain = 1 - pooled_branch["T_active"]["T"] / baseline_t
    specialist_i_gain = 1 - pooled_branch["I_active"]["I"] / baseline_i
    strong_null = (
        not pred_a
        or pooled_damage["covariance"] - pooled_damage["TI_active"] < .0002
        or sum(value < 0 for value in wave_joint_improvement) >= 3
        or max(specialist_t_gain, specialist_i_gain) < .02
    )

    result = {
        "status": "mlp0_rank448_token_grammar_active_subspace_complete",
        "rung": 405,
        "claim_level": "single_site_equal_price_tangent_active_subspace_screen_not_compression",
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
            "basis_hashes": basis_hashes,
            "orthogonality_max_abs_error": orthogonality,
            "covariance_baseline_diagnostic": baseline_diagnostic,
        },
        "metric_diagnostic": metric_diagnostic,
        "wave_results": {str(index): value for index, value in enumerate(wave_results)},
        "pooled_ce_damage": pooled_damage,
        "pooled_branch_relative_mse": pooled_branch,
        "joint_wave_ce_improvement_over_covariance": wave_joint_improvement,
        "baseline_wave_reproduction_max_abs_error": baseline_reproduction_error,
        "specialist_own_branch_relative_gain": {
            "T": specialist_t_gain, "I": specialist_i_gain,
        },
        'pred_a_instrument_identity_and_price_are_exact': bool(pred_a),
        'pred_b_derivative_metrics_are_branch_specific': bool(pred_b),
        'pred_c_joint_metric_improves_physical_prediction_robustly': bool(pred_c),
        'pred_d_joint_basis_is_balanced_and_competitive': bool(pred_d),
        "null_first_order_token_grammar_metric_fails": bool(strong_null),
        "next_object": (
            "whole_model_gate_for_fixed_TI_active_p448"
            if pred_a and pred_b and pred_c and pred_d and not strong_null else None),
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 TOKEN-GRAMMAR ACTIVE SUBSPACE DONE", flush=True)


if __name__ == "__main__":
    main()
