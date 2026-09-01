"""RUNG 410 -- WEIGHT-DERIVED SHARED OUTPUT QUADRATIC PRODUCER.

Build the selected rank-64 p448 output correction from causal MLP0 input using
quadratic forms contracted directly from native-minus-compact bilinear weights.
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
OUT = BQ / "mlp0_p448_shared_output_quadratic_producer_results.json"
PARENT = BQ / "mlp0_p448_causal_output_interface_oracle_results.json"
LOSS_ARTIFACT = BQ / "mlp0_p448_router_oracle_losses.pt"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
CONFIRM_RECEIPT = BQ / "mlp0_c512_mlp1_interchange_v1_rows_receipt.json"
CONFIRM_CACHE = BQ / ".rowcache_mlp0_c512_mlp1_interchange_v1/eval_384_source_documents.pt"
FIT_CACHE = BQ / ".rowcache/fineweb_n192_skip11000.pt"
LOSS_TENSOR_SHA = "e6d92614ad4fbe5b6e63aa2939e7df6ecb197281c114aae9696baf3bc68ab082"
FIT_SLICE = (0, 24)
D = 1152
H = 4608
OUTPUT_RANK = 64
FORM_RANKS = (8, 16, 24)
SOURCE_DOCUMENTS = 384
TRAIN_DOCUMENTS = 192
EVAL_DOCUMENTS = 192
WAVE_DOCUMENTS = 96
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
P448_VALUES = 9_954_432
P640_VALUES = 11_945_088
P768_VALUES = 13_272_192
ARMS = (
    "NATIVE", "P448", "P640", "P768", "ORACLE64", "FULL_ANALYTIC",
    "AFFINE", "R8", "R16", "R24", "SHUFFLED24",
)


def _producer_price(rank):
    return D * OUTPUT_RANK + OUTPUT_RANK * rank * (D + 1) + OUTPUT_RANK


def _tensor_sha256(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _project(value, basis):
    shape = value.shape
    flat = value.float().reshape(-1, D)
    return ((flat @ basis) @ basis.T).reshape(shape)


def _top_basis(gram, rank):
    gram = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(gram)
    basis = vectors[:, -rank:].contiguous()
    retained = float(values[-rank:].double().sum() / values.clamp_min(0).double().sum())
    orthogonality = float((basis.T @ basis - torch.eye(rank, device=basis.device)).abs().max())
    return basis, retained, orthogonality


def _quadratic_coefficients(state, directions, values, beta, rank):
    shape = state.shape[:-1]
    flat = state.float().reshape(-1, D)
    chosen_directions = directions[:, :rank]
    chosen_values = values[:, :rank]
    projections = torch.einsum("nd,jrd->njr", flat, chosen_directions)
    coefficients = (projections.square() * chosen_values[None]).sum(-1) + beta[None]
    return coefficients.reshape(*shape, OUTPUT_RANK)


def _relative_mse(left, right):
    numerator = float((left.double() - right.double()).square().sum())
    denominator = float(right.double().square().sum())
    return numerator / max(denominator, 1e-30)


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (
            PARENT, LOSS_ARTIFACT, ROWS_RECEIPT, CONFIRM_RECEIPT, CONFIRM_CACHE, FIT_CACHE))
        assert TRAIN_DOCUMENTS + EVAL_DOCUMENTS == SOURCE_DOCUMENTS
        assert [_producer_price(rank) for rank in FORM_RANKS] == [664_128, 1_254_464, 1_844_800]
        assert P448_VALUES + _producer_price(24) == 11_799_232 < P640_VALUES < P768_VALUES
        print("MLP0 p448 QUADRATIC PRODUCER | dry run: forms, split, arms, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(BQ.parent / "qk_mdl"))
    import bilin18_observed_model_facade as facade
    import mlp0_rank448_branch_error_factorial as rung403
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    device = torch.device("cuda")
    parent = json.loads(PARENT.read_text())
    if (parent["null_rank64_output_repair_has_no_specific_heldout_gain"]
            or parent["next_object"] != "shared_rank64_output_producer"):
        raise RuntimeError("rung409 does not license a shared output producer")

    receipt = json.loads(CONFIRM_RECEIPT.read_text())
    current_rows_receipt = json.loads(ROWS_RECEIPT.read_text())
    all_rows = torch.load(CONFIRM_CACHE, map_location="cpu")
    records = receipt["document_provenance"]["sets"]["eval"]
    chunk0 = [index for index, record in enumerate(records) if int(record["chunk_id"]) == 0]
    ordinals = [int(records[index]["source_document_ordinal"]) for index in chunk0]
    document_ids = [records[index]["document_id"] for index in chunk0]
    rows = all_rows[chunk0, :257].long().contiguous()
    population_exact = (
        tuple(all_rows.shape) == (585, 513)
        and tuple(rows.shape) == (SOURCE_DOCUMENTS, 257)
        and ordinals == list(range(SOURCE_DOCUMENTS))
        and len(set(document_ids)) == SOURCE_DOCUMENTS
        and _tensor_sha256(all_rows) == receipt["entries"]["eval"]["tensor_raw_sha256"]
        and all(receipt["disjointness_gates"].values())
        and all(current_rows_receipt["disjointness"].values()))
    saved = torch.load(LOSS_ARTIFACT, map_location="cpu", weights_only=True)
    saved_losses = saved["losses"].float()
    saved_names = tuple(saved["program_names"])
    loss_authority_exact = (
        tuple(saved_losses.shape) == (8, SOURCE_DOCUMENTS, 192)
        and _tensor_sha256(saved_losses) == LOSS_TENSOR_SHA)

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    program_fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    source_model, source_checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    covariance = _covariance(source_model, program_fit_rows, _manual_logits)
    programs = {}
    program_diagnostics = {}
    for rank in (448, 640, 768):
        program, _basis, diagnostic = _rrr_program(
            source_model.transformer.h[0].mlp, covariance, rank=rank)
        programs[rank] = {name: value.detach().clone() for name, value in program.items()}
        program_diagnostics[rank] = diagnostic
    del covariance, _basis, source_model
    torch.cuda.empty_cache()

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    native_factors = {
        "left": model.transformer.h[0].mlp.Left.weight.detach().float(),
        "right": model.transformer.h[0].mlp.Right.weight.detach().float(),
        "down": model.transformer.h[0].mlp.Down.weight.detach().float(),
    }
    down_bias = model.transformer.h[0].mlp.Down.bias
    native_bias = (
        down_bias.detach().float() if down_bias is not None
        else torch.zeros(D, device=device)
    )
    compact_factors = {name: value.detach().float() for name, value in programs[448].items()}
    compact_bias = programs[448]["bias"].detach().float()

    error_gram = torch.zeros((D, D), device=device)
    state_second = torch.zeros((D, D), device=device)
    sample_state = None
    train_pass1_calls = 0
    for start in range(0, TRAIN_DOCUMENTS, DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)

        def attention(event):
            return event.block.attn(event.state, event.first_value)

        def mlp(event):
            nonlocal train_pass1_calls, sample_state
            native = event.block.mlp(event.state)
            if event.site != 0:
                return native
            train_pass1_calls += 1
            compact = rung403._compact_deployed(event.state, programs[448], native.dtype)
            state = event.state[:, SCORING].reshape(-1, D).float()
            error = (native.float() - compact.float())[:, SCORING].reshape(-1, D)
            state_second.add_(state.T @ state)
            error_gram.add_(error.T @ error)
            if sample_state is None:
                sample_state = state[:8].detach().clone()
            return native

        facade.forward_with_dispatch(model, tokens, attention, mlp)

    output_basis, output_retained, output_orth = _top_basis(error_gram, OUTPUT_RANK)
    state_second /= TRAIN_DOCUMENTS * (SCORING.stop - SCORING.start)
    c_values, c_vectors = torch.linalg.eigh(0.5 * (state_second + state_second.T))
    floor = 1e-6 * float(c_values.max())
    floored = c_values.clamp_min(floor)
    square = (c_vectors * floored.sqrt()[None]) @ c_vectors.T
    inverse_square = (c_vectors * floored.rsqrt()[None]) @ c_vectors.T
    covariance_inverse_error = float(
        (square @ inverse_square - torch.eye(D, device=device)).abs().max())

    native_left = native_factors["left"]
    native_right = native_factors["right"]
    compact_left = compact_factors["left"] @ compact_factors["encoder"]
    compact_right = compact_factors["right"] @ compact_factors["encoder"]
    native_output_weights = output_basis.T @ native_factors["down"]
    compact_output_weights = output_basis.T @ compact_factors["down"]
    beta = (native_bias - compact_bias) @ output_basis
    directions = torch.empty((OUTPUT_RANK, 24, D), device=device)
    signed_values = torch.empty((OUTPUT_RANK, 24), device=device)
    full_form_energy = 0.0
    retained_form_energy = {rank: 0.0 for rank in FORM_RANKS}
    full_sample = torch.empty((len(sample_state), OUTPUT_RANK), device=device)
    truncated_sample = torch.empty_like(full_sample)
    for output in range(OUTPUT_RANK):
        native_form = native_left.T @ (native_output_weights[output, :, None] * native_right)
        compact_form = compact_left.T @ (compact_output_weights[output, :, None] * compact_right)
        form = 0.5 * ((native_form - compact_form) + (native_form - compact_form).T)
        weighted = square @ form @ square
        values, vectors = torch.linalg.eigh(0.5 * (weighted + weighted.T))
        order = torch.argsort(values.abs(), descending=True)
        values = values[order]
        vectors = vectors[:, order]
        directions[output] = (inverse_square @ vectors[:, :24]).T
        signed_values[output] = values[:24]
        energy = values.double().square()
        full_form_energy += float(energy.sum())
        for rank in FORM_RANKS:
            retained_form_energy[rank] += float(energy[:rank].sum())
        whitened_sample = sample_state @ inverse_square
        full_sample[:, output] = (
            (whitened_sample @ vectors).square() * values[None]).sum(-1) + beta[output]
        truncated_sample[:, output] = (
            (sample_state @ directions[output].T).square()
            * signed_values[output][None]).sum(-1) + beta[output]
    retained_form_energy = {
        rank: value / max(full_form_energy, 1e-30)
        for rank, value in retained_form_energy.items()
    }
    direct_sample_native = rung403._bilinear(sample_state, sample_state, native_factors) + native_bias
    direct_sample_compact = rung403._bilinear(sample_state, sample_state, compact_factors) + compact_bias
    direct_sample = (direct_sample_native - direct_sample_compact) @ output_basis
    full_form_sample_relative_mse = _relative_mse(full_sample, direct_sample)
    rank24_sample_relative_mse = _relative_mse(truncated_sample, direct_sample)

    sum_state = torch.zeros(D, device=device, dtype=torch.float64)
    sum_target = torch.zeros(OUTPUT_RANK, device=device, dtype=torch.float64)
    cross = torch.zeros((D, OUTPUT_RANK), device=device, dtype=torch.float64)
    second_center = torch.zeros((D, D), device=device, dtype=torch.float64)
    train_pass2_calls = 0
    affine_count = 0
    for start in range(0, TRAIN_DOCUMENTS, DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)

        def attention(event):
            return event.block.attn(event.state, event.first_value)

        def mlp(event):
            nonlocal train_pass2_calls, affine_count
            native = event.block.mlp(event.state)
            if event.site != 0:
                return native
            train_pass2_calls += 1
            compact = rung403._compact_deployed(event.state, programs[448], native.dtype)
            state = event.state[:, SCORING].reshape(-1, D).double()
            target = ((native.float() - compact.float())[:, SCORING].reshape(-1, D)
                      @ output_basis).double()
            sum_state.add_(state.sum(0))
            sum_target.add_(target.sum(0))
            cross.add_(state.T @ target)
            second_center.add_(state.T @ state)
            affine_count += len(state)
            return native

        facade.forward_with_dispatch(model, tokens, attention, mlp)
    mean_state = sum_state / affine_count
    mean_target = sum_target / affine_count
    centered_second = second_center - affine_count * torch.outer(mean_state, mean_state)
    centered_cross = cross - affine_count * torch.outer(mean_state, mean_target)
    ridge = 1e-3 * float(torch.trace(centered_second)) / D
    affine_weight = torch.linalg.solve(
        centered_second + ridge * torch.eye(D, device=device, dtype=torch.float64),
        centered_cross).float()
    affine_bias = (mean_target - mean_state @ affine_weight.double()).float()

    permutation = torch.randperm(OUTPUT_RANK, generator=torch.Generator(device=device).manual_seed(410), device=device)
    shuffled_directions = directions[permutation]
    shuffled_values = signed_values[permutation]

    loss_parts = {name: [] for name in ARMS}
    calls = {name: {"forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0} for name in ARMS}
    coefficient_errors = {name: [0.0, 0.0] for name in ("AFFINE", "R8", "R16", "R24", "SHUFFLED24")}
    state_replay_max = 0.0
    eval_rows = rows[TRAIN_DOCUMENTS:]
    for start in range(0, EVAL_DOCUMENTS, DOCUMENT_BATCH):
        batch = eval_rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        cache = {}

        for arm in ARMS:
            def attention(event, arm=arm):
                calls[arm]["attention"] += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event, arm=arm):
                nonlocal state_replay_max
                native = event.block.mlp(event.state)
                if event.site != 0:
                    calls[arm]["other_mlp"] += 1
                    return native
                calls[arm]["site0"] += 1
                if not cache:
                    compact = rung403._compact_deployed(event.state, programs[448], native.dtype)
                    direct_native = rung403._bilinear(event.state, event.state, native_factors) + native_bias
                    direct_compact = rung403._bilinear(event.state, event.state, compact_factors) + compact_bias
                    analytic_coeff = (direct_native - direct_compact) @ output_basis
                    deployed_coeff = (native.float() - compact.float()) @ output_basis
                    state = event.state.float()
                    predictions = {
                        "AFFINE": state @ affine_weight + affine_bias,
                        "R8": _quadratic_coefficients(state, directions, signed_values, beta, 8),
                        "R16": _quadratic_coefficients(state, directions, signed_values, beta, 16),
                        "R24": _quadratic_coefficients(state, directions, signed_values, beta, 24),
                        "SHUFFLED24": _quadratic_coefficients(
                            state, shuffled_directions, shuffled_values, beta, 24),
                    }
                    cache.update({
                        "state": event.state.detach().clone(), "native": native.detach().clone(),
                        "compact": compact.detach().clone(), "analytic": analytic_coeff.detach().clone(),
                        "deployed": deployed_coeff.detach().clone(),
                        "predictions": {name: value.detach().clone() for name, value in predictions.items()},
                    })
                    target_scored = deployed_coeff[:, SCORING]
                    for name, value in predictions.items():
                        coefficient_errors[name][0] += float(
                            (value[:, SCORING].double() - target_scored.double()).square().sum())
                        coefficient_errors[name][1] += float(target_scored.double().square().sum())
                else:
                    state_replay_max = max(
                        state_replay_max, float((event.state - cache["state"]).abs().max()))

                if arm == "NATIVE":
                    result = native
                elif arm == "P448":
                    result = cache["compact"]
                elif arm == "P640":
                    result = rung403._compact_deployed(event.state, programs[640], native.dtype)
                elif arm == "P768":
                    result = rung403._compact_deployed(event.state, programs[768], native.dtype)
                elif arm == "ORACLE64":
                    result = cache["compact"].float() + cache["deployed"] @ output_basis.T
                elif arm == "FULL_ANALYTIC":
                    result = cache["compact"].float() + cache["analytic"] @ output_basis.T
                elif arm in cache["predictions"]:
                    result = cache["compact"].float() + cache["predictions"][arm] @ output_basis.T
                else:
                    raise RuntimeError(arm)
                return result.to(native.dtype)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            loss = F.cross_entropy(
                logits[:, SCORING].transpose(1, 2), targets[:, SCORING], reduction="none")
            loss_parts[arm].append(loss.cpu())
            calls[arm]["forwards"] += 1

    losses = torch.stack([torch.cat(loss_parts[name]) for name in ARMS])
    native_ce = float(losses[0].double().mean())
    damage = {name: float(losses[index].double().mean()) - native_ce
              for index, name in enumerate(ARMS)}
    wave_damage = {}
    for index, arm in enumerate(ARMS):
        wave_damage[arm] = []
        for wave in range(2):
            left, right = wave * WAVE_DOCUMENTS, (wave + 1) * WAVE_DOCUMENTS
            wave_native = float(losses[0, left:right].double().mean())
            wave_damage[arm].append(float(losses[index, left:right].double().mean()) - wave_native)
    coefficient_relative_mse = {
        name: numerator / max(denominator, 1e-30)
        for name, (numerator, denominator) in coefficient_errors.items()
    }

    saved_index = {name: index for index, name in enumerate(saved_names)}
    baseline_saved_error = {
        "NATIVE": float((losses[ARMS.index("NATIVE")] - saved_losses[
            saved_index["native"], TRAIN_DOCUMENTS:]).abs().max()),
        "P448": float((losses[ARMS.index("P448")] - saved_losses[
            saved_index["covariance_p448"], TRAIN_DOCUMENTS:]).abs().max()),
        "P640": float((losses[ARMS.index("P640")] - saved_losses[
            saved_index["covariance_p640"], TRAIN_DOCUMENTS:]).abs().max()),
        "P768": float((losses[ARMS.index("P768")] - saved_losses[
            saved_index["covariance_p768"], TRAIN_DOCUMENTS:]).abs().max()),
    }
    expected_calls = {
        "forwards": EVAL_DOCUMENTS // DOCUMENT_BATCH,
        "attention": 18 * EVAL_DOCUMENTS // DOCUMENT_BATCH,
        "site0": EVAL_DOCUMENTS // DOCUMENT_BATCH,
        "other_mlp": 17 * EVAL_DOCUMENTS // DOCUMENT_BATCH,
    }
    calls_live = all(value == expected_calls for value in calls.values())
    programs_exact = all(
        int(program_diagnostics[rank]["rank"]) == rank
        and {name: tuple(value.shape) for name, value in programs[rank].items()} == {
            "encoder": (rank, D), "left": (H, rank), "right": (H, rank),
            "down": (D, H), "bias": (D,)}
        for rank in (448, 640, 768))
    oracle_damage_expected = parent["physical"]["damage"]["TOTAL_ERROR_64"]
    oracle_gain = damage["P448"] - damage["ORACLE64"]
    prices = {str(rank): _producer_price(rank) for rank in FORM_RANKS}
    pred_a = (
        population_exact and loss_authority_exact and programs_exact and calls_live
        and train_pass1_calls == TRAIN_DOCUMENTS // DOCUMENT_BATCH
        and train_pass2_calls == TRAIN_DOCUMENTS // DOCUMENT_BATCH
        and state_replay_max == 0.0 and max(baseline_saved_error.values()) == 0.0
        and abs(output_retained - parent["basis_diagnostics"]["training_retained_energy"]["TOTAL_ERROR_64"]) <= 1e-6
        and output_orth <= 1e-4 and covariance_inverse_error <= 1e-4
        and full_form_sample_relative_mse <= 1e-6
        and abs(damage["FULL_ANALYTIC"] - oracle_damage_expected) <= .0002
        and prices["24"] == 1_844_800
        and P448_VALUES + prices["24"] < P640_VALUES
        and bool(torch.isfinite(losses).all()))
    pred_b = (
        damage["P448"] - damage["R24"] >= .5 * oracle_gain
        and all(wave_damage["P448"][wave] - wave_damage["R24"][wave] >= .0015
                for wave in range(2))
        and damage["R24"] < .0064)
    rank_ce = [damage[f"R{rank}"] for rank in FORM_RANKS]
    rank_mse = [coefficient_relative_mse[f"R{rank}"] for rank in FORM_RANKS]
    pred_c = (
        all(rank_mse[index + 1] <= rank_mse[index] for index in range(2))
        and all(rank_ce[index + 1] <= rank_ce[index] for index in range(2))
        and retained_form_energy[24] >= .70
        and damage["R8"] - damage["R24"] >= .0005)
    pred_d = (
        damage["SHUFFLED24"] - damage["R24"] >= .001
        and damage["AFFINE"] - damage["R24"] >= .0005
        and all(wave_damage[arm][wave] - wave_damage["R24"][wave] > 0
                for arm in ("SHUFFLED24", "AFFINE") for wave in range(2)))
    r24_gain = damage["P448"] - damage["R24"]
    affine_gain = damage["P448"] - damage["AFFINE"]
    strong_null = (
        not pred_a or r24_gain < .0002
        or damage["SHUFFLED24"] - damage["R24"] < .0002
        or P448_VALUES + prices["24"] >= P640_VALUES
        or max(r24_gain, affine_gain) < .25 * oracle_gain)
    selected = "R24" if damage["R24"] <= damage["AFFINE"] else "AFFINE"

    result = {
        "status": "mlp0_p448_shared_output_quadratic_producer_complete",
        "rung": 410,
        "claim_level": "heldout_executable_shared_output_producer_screen_not_adoption",
        "convention": "CE added above native; lower is better",
        "authority": {
            "parent": str(PARENT), "source_documents": SOURCE_DOCUMENTS,
            "train_documents_half_open": [0, TRAIN_DOCUMENTS],
            "evaluation_documents_half_open": [TRAIN_DOCUMENTS, SOURCE_DOCUMENTS],
            "evaluation_waves": [[192, 288], [288, 384]],
            "scoring_positions_half_open": [SCORING.start, SCORING.stop],
            "population_exact": population_exact,
            "loss_tensor_sha256": _tensor_sha256(saved_losses),
            "loss_authority_exact": loss_authority_exact,
            "baseline_saved_loss_max_abs_error": baseline_saved_error,
            "FINAL_opened": 0,
        },
        "programs": {
            "source_checkpoint": source_checkpoint.__dict__,
            "evaluation_checkpoint": checkpoint.__dict__,
            "diagnostics": {str(rank): value for rank, value in program_diagnostics.items()},
            "exact": programs_exact,
        },
        "output_basis": {
            "sha256": _tensor_sha256(output_basis),
            "training_retained_error_energy": output_retained,
            "orthogonality_max_abs": output_orth,
        },
        "quadratic_derivation": {
            "uncentered_covariance_floor": floor,
            "square_inverse_max_abs_error": covariance_inverse_error,
            "full_form_sample_relative_mse": full_form_sample_relative_mse,
            "rank24_sample_relative_mse": rank24_sample_relative_mse,
            "data_metric_form_energy_retained": {str(key): value for key, value in retained_form_energy.items()},
            "beta_max_abs": float(beta.abs().max()),
            "directions_sha256": _tensor_sha256(directions),
            "signed_values_sha256": _tensor_sha256(signed_values),
        },
        "affine": {
            "ridge": ridge, "training_positions": affine_count,
            "weight_sha256": _tensor_sha256(affine_weight),
            "bias_sha256": _tensor_sha256(affine_bias),
        },
        "physical": {
            "native_ce": native_ce, "damage": damage, "wave_damage": wave_damage,
            "coefficient_relative_mse_vs_deployed_oracle": coefficient_relative_mse,
            "calls": calls, "calls_live": calls_live,
            "training_pass1_site0_calls": train_pass1_calls,
            "training_pass2_site0_calls": train_pass2_calls,
            "state_replay_max_abs_error": state_replay_max,
        },
        "literal_price": {
            "p448_values": P448_VALUES,
            "quadratic_producer_interface_values": prices,
            "p448_plus_quadratic_values": {
                str(rank): P448_VALUES + prices[str(rank)] for rank in FORM_RANKS},
            "affine_producer_interface_values": 147_520,
            "p448_plus_affine_values": P448_VALUES + 147_520,
            "p640_values": P640_VALUES, "p768_values": P768_VALUES,
            "rank24_component_count": OUTPUT_RANK * 24,
            "rank24_projection_multiplications_per_token": OUTPUT_RANK * 24 * D,
            "rank24_output_multiplications_per_token": OUTPUT_RANK * D,
            "fit_dense_forms_not_runtime_storage": True,
        },
        "oracle_gain": oracle_gain,
        "rank24_oracle_gain_recovery": r24_gain / oracle_gain if oracle_gain > 0 else 0.0,
        "selected_screen_producer": selected,
        'pred_a_weight_derivation_authority_and_price_are_exact': bool(pred_a),
        'pred_b_priced_rank24_recovers_half_the_output_oracle': bool(pred_b),
        'pred_c_quadratic_spectrum_is_live_and_ordered': bool(pred_c),
        'pred_d_quadratic_producer_beats_shuffle_and_affine': bool(pred_d),
        "null_no_specific_priced_executable_output_producer": bool(strong_null),
        "next_object": (
            f"fresh_ood_signed_composition_gate_{selected}" if pred_a and pred_b and not strong_null
            else "mathematical_screen_only" if pred_a and not strong_null else None),
        "adoption_licensed": False,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 SHARED OUTPUT QUADRATIC PRODUCER DONE", flush=True)


if __name__ == "__main__":
    main()
