"""RUNG 411 -- JOINT TUCKER PRODUCER FOR THE EXACT MLP0 P448 ERROR TENSOR."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
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
OUT = BQ / "mlp0_p448_joint_tucker_quadratic_producer_results.json"
PARENT = BQ / "mlp0_p448_shared_output_quadratic_producer_results.json"
ORACLE_PARENT = BQ / "mlp0_p448_causal_output_interface_oracle_results.json"
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
TUCKER_RANKS = (96, 160, 226)
MATCHED_RANKS = {96: 494, 160: 552, 226: 638}
PROGRAM_RANKS = (448, 494, 552, 638, 640)
SOURCE_DOCUMENTS = 384
TRAIN_DOCUMENTS = 192
EVAL_DOCUMENTS = 192
WAVE_DOCUMENTS = 96
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
P448_VALUES = 9_954_432
P640_VALUES = 11_945_088
R410_R16_DAMAGE = 0.007159569030501078
R410_R24_DAMAGE = 0.006928635183113929
ARMS = (
    "NATIVE", "P448", "P494", "P552", "P638", "P640", "ORACLE64",
    "FULL_ANALYTIC", "TUCKER96", "TUCKER160", "TUCKER226", "HAAR226", "R24",
)


def _producer_price(rank):
    return D * OUTPUT_RANK + D * rank + OUTPUT_RANK * rank * (rank + 1) // 2 + OUTPUT_RANK


def _program_price(rank):
    return P448_VALUES + (rank - 448) * (2 * H + D)


def _tensor_sha256(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _top_basis(gram, rank):
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
    basis = vectors[:, -rank:].contiguous()
    retained = float(values[-rank:].double().sum() / values.clamp_min(0).double().sum())
    orthogonality = float((basis.T @ basis - torch.eye(rank, device=basis.device)).abs().max())
    return basis, retained, orthogonality


def _relative_mse(left, right):
    numerator = float((left.double() - right.double()).square().sum())
    denominator = float(right.double().square().sum())
    return numerator / max(denominator, 1e-30)


def _tucker_coefficients(state, directions, core, beta, rank):
    features = state.float() @ directions[:, :rank]
    coefficients = torch.einsum(
        "...p,jpq,...q->...j", features, core[:, :rank, :rank], features)
    return coefficients + beta


def _independent_coefficients(state, directions, values, beta):
    flat = state.float().reshape(-1, D)
    projections = torch.einsum("nd,jrd->njr", flat, directions)
    result = (projections.square() * values[None]).sum(-1) + beta[None]
    return result.reshape(*state.shape[:-1], OUTPUT_RANK)


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (
            PARENT, ORACLE_PARENT, LOSS_ARTIFACT, ROWS_RECEIPT, CONFIRM_RECEIPT,
            CONFIRM_CACHE, FIT_CACHE))
        assert [_producer_price(rank) for rank in TUCKER_RANKS] == [482_368, 1_082_432, 1_975_808]
        assert [_program_price(MATCHED_RANKS[rank]) for rank in TUCKER_RANKS] == [
            10_431_360, 11_032_704, 11_924_352]
        assert all(
            _program_price(MATCHED_RANKS[rank]) < P448_VALUES + _producer_price(rank)
            for rank in TUCKER_RANKS)
        assert P448_VALUES + _producer_price(226) < P640_VALUES
        assert P448_VALUES + _producer_price(227) > P640_VALUES
        print("MLP0 p448 JOINT TUCKER | dry run: tensor, ranks, prices, controls, bars valid")
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
    oracle_parent = json.loads(ORACLE_PARENT.read_text())
    if parent["null_no_specific_priced_executable_output_producer"]:
        raise RuntimeError("rung410 exact tensor was not licensed for a joint factorization screen")

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
    for rank in PROGRAM_RANKS:
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
    native_bias = down_bias.detach().float() if down_bias is not None else torch.zeros(D, device=device)
    compact_factors = {name: value.detach().float() for name, value in programs[448].items()}
    compact_bias = programs[448]["bias"].detach().float()

    error_gram = torch.zeros((D, D), device=device)
    state_second = torch.zeros((D, D), device=device)
    sample_state = None
    train_calls = 0
    for start in range(0, TRAIN_DOCUMENTS, DOCUMENT_BATCH):
        tokens = rows[start:start + DOCUMENT_BATCH, :-1].to(device)

        def attention(event):
            return event.block.attn(event.state, event.first_value)

        def mlp(event):
            nonlocal train_calls, sample_state
            native = event.block.mlp(event.state)
            if event.site != 0:
                return native
            train_calls += 1
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
    forms = torch.empty((OUTPUT_RANK, D, D), device=device)
    input_gram = torch.zeros((D, D), device=device)
    full_energy = 0.0
    r24_directions = torch.empty((OUTPUT_RANK, 24, D), device=device)
    r24_values = torch.empty((OUTPUT_RANK, 24), device=device)
    for output in range(OUTPUT_RANK):
        native_form = native_left.T @ (native_output_weights[output, :, None] * native_right)
        compact_form = compact_left.T @ (compact_output_weights[output, :, None] * compact_right)
        form = 0.5 * ((native_form - compact_form) + (native_form - compact_form).T)
        weighted = square @ form @ square
        weighted = 0.5 * (weighted + weighted.T)
        forms[output] = weighted
        input_gram.add_(weighted @ weighted)
        full_energy += float(weighted.double().square().sum())
        values, vectors = torch.linalg.eigh(weighted)
        order = torch.argsort(values.abs(), descending=True)
        values, vectors = values[order], vectors[:, order]
        r24_directions[output] = (inverse_square @ vectors[:, :24]).T
        r24_values[output] = values[:24]

    input_values, input_vectors = torch.linalg.eigh(0.5 * (input_gram + input_gram.T))
    input_vectors = input_vectors[:, torch.argsort(input_values, descending=True)[:226]].contiguous()
    shared_directions = inverse_square @ input_vectors
    core = torch.einsum("dp,jde,eq->jpq", input_vectors, forms, input_vectors)
    generator = torch.Generator(device=device).manual_seed(411)
    random_vectors = torch.linalg.qr(torch.randn(D, 226, generator=generator, device=device)).Q
    random_directions = inverse_square @ random_vectors
    random_core = torch.einsum("dp,jde,eq->jpq", random_vectors, forms, random_vectors)
    tied_energy = {
        rank: float(core[:, :rank, :rank].double().square().sum()) / max(full_energy, 1e-30)
        for rank in TUCKER_RANKS
    }
    one_mode_total = float(input_values.clamp_min(0).double().sum())
    one_mode_energy = {
        rank: float(input_values[-rank:].clamp_min(0).double().sum()) / max(one_mode_total, 1e-30)
        for rank in TUCKER_RANKS
    }

    direct_sample_native = rung403._bilinear(sample_state, sample_state, native_factors) + native_bias
    direct_sample_compact = rung403._bilinear(sample_state, sample_state, compact_factors) + compact_bias
    direct_sample = (direct_sample_native - direct_sample_compact) @ output_basis
    full_sample = torch.einsum(
        "nd,jde,ne->nj", sample_state @ inverse_square, forms, sample_state @ inverse_square) + beta
    full_form_sample_relative_mse = _relative_mse(full_sample, direct_sample)

    loss_parts = {name: [] for name in ARMS}
    calls = {name: {"forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0} for name in ARMS}
    coefficient_errors = {name: [0.0, 0.0] for name in (
        "TUCKER96", "TUCKER160", "TUCKER226", "HAAR226", "R24")}
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
                    analytic = (direct_native - direct_compact) @ output_basis
                    deployed = (native.float() - compact.float()) @ output_basis
                    predictions = {
                        "TUCKER96": _tucker_coefficients(
                            event.state, shared_directions, core, beta, 96),
                        "TUCKER160": _tucker_coefficients(
                            event.state, shared_directions, core, beta, 160),
                        "TUCKER226": _tucker_coefficients(
                            event.state, shared_directions, core, beta, 226),
                        "HAAR226": _tucker_coefficients(
                            event.state, random_directions, random_core, beta, 226),
                        "R24": _independent_coefficients(
                            event.state, r24_directions, r24_values, beta),
                    }
                    cache.update({
                        "state": event.state.detach().clone(), "native": native.detach().clone(),
                        "compact": compact.detach().clone(), "analytic": analytic.detach().clone(),
                        "deployed": deployed.detach().clone(),
                        "predictions": {name: value.detach().clone() for name, value in predictions.items()},
                    })
                    target_scored = deployed[:, SCORING]
                    for name, value in predictions.items():
                        coefficient_errors[name][0] += float(
                            (value[:, SCORING].double() - target_scored.double()).square().sum())
                        coefficient_errors[name][1] += float(target_scored.double().square().sum())
                else:
                    state_replay_max = max(
                        state_replay_max, float((event.state - cache["state"]).abs().max()))
                if arm == "NATIVE":
                    result = native
                elif arm.startswith("P"):
                    result = rung403._compact_deployed(event.state, programs[int(arm[1:])], native.dtype)
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
        for rank in PROGRAM_RANKS)
    prices = {str(rank): _producer_price(rank) for rank in TUCKER_RANKS}
    matched_prices = {str(rank): _program_price(MATCHED_RANKS[rank]) for rank in TUCKER_RANKS}
    oracle_damage_expected = oracle_parent["physical"]["damage"]["TOTAL_ERROR_64"]
    oracle_gain = damage["P448"] - damage["ORACLE64"]
    pred_a = (
        population_exact and loss_authority_exact and programs_exact and calls_live
        and train_calls == TRAIN_DOCUMENTS // DOCUMENT_BATCH
        and state_replay_max == 0.0 and max(baseline_saved_error.values()) == 0.0
        and abs(output_retained - parent["output_basis"]["training_retained_error_energy"]) <= 1e-6
        and output_orth <= 1e-4 and covariance_inverse_error <= 1e-4
        and full_form_sample_relative_mse <= 1e-6
        and abs(damage["FULL_ANALYTIC"] - oracle_damage_expected) <= .0002
        and prices == {"96": 482_368, "160": 1_082_432, "226": 1_975_808}
        and matched_prices == {"96": 10_431_360, "160": 11_032_704, "226": 11_924_352}
        and bool(torch.isfinite(losses).all()))
    pred_b = (
        damage["P448"] - damage["TUCKER226"] >= .5 * oracle_gain
        and all(wave_damage["P448"][wave] - wave_damage["TUCKER226"][wave] >= .0015
                for wave in range(2))
        and damage["P638"] - damage["TUCKER226"] >= .0002)
    mse_order = [coefficient_relative_mse[f"TUCKER{rank}"] for rank in TUCKER_RANKS]
    pred_c = (
        all(mse_order[index + 1] <= mse_order[index] for index in range(2))
        and all(tied_energy[TUCKER_RANKS[index + 1]] >= tied_energy[TUCKER_RANKS[index]]
                for index in range(2))
        and damage["TUCKER160"] <= R410_R16_DAMAGE
        and tied_energy[226] >= .60)
    pred_d = (
        damage["HAAR226"] - damage["TUCKER226"] >= .001
        and damage["R24"] - damage["TUCKER226"] >= .0005
        and all(wave_damage[arm][wave] - wave_damage["TUCKER226"][wave] > 0
                for arm in ("HAAR226", "R24") for wave in range(2)))
    matched_margins = {
        str(rank): damage[f"P{MATCHED_RANKS[rank]}"] - damage[f"TUCKER{rank}"]
        for rank in TUCKER_RANKS
    }
    recoveries = {
        str(rank): (damage["P448"] - damage[f"TUCKER{rank}"]) / oracle_gain
        for rank in TUCKER_RANKS
    }
    strong_null = (
        not pred_a or max(matched_margins.values()) < .0002
        or damage["HAAR226"] - damage["TUCKER226"] < .0002
        or max(recoveries.values()) < .25)
    passing = [rank for rank in TUCKER_RANKS if matched_margins[str(rank)] >= .0002]
    selected = min(passing, key=_producer_price) if passing and pred_a and not strong_null else None

    result = {
        "status": "mlp0_p448_joint_tucker_quadratic_producer_complete",
        "rung": 411,
        "claim_level": "heldout_executable_joint_tensor_price_screen_not_adoption",
        "convention": "CE added above native; lower is better",
        "authority": {
            "source_documents": SOURCE_DOCUMENTS,
            "train_documents_half_open": [0, TRAIN_DOCUMENTS],
            "evaluation_documents_half_open": [TRAIN_DOCUMENTS, SOURCE_DOCUMENTS],
            "evaluation_waves": [[192, 288], [288, 384]],
            "scoring_positions_half_open": [SCORING.start, SCORING.stop],
            "population_exact": population_exact, "loss_authority_exact": loss_authority_exact,
            "loss_tensor_sha256": _tensor_sha256(saved_losses),
            "baseline_saved_loss_max_abs_error": baseline_saved_error, "FINAL_opened": 0,
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
        "joint_tensor": {
            "uncentered_covariance_floor": floor,
            "square_inverse_max_abs_error": covariance_inverse_error,
            "full_form_sample_relative_mse": full_form_sample_relative_mse,
            "one_mode_energy_retained": {str(key): value for key, value in one_mode_energy.items()},
            "tied_two_mode_energy_retained": {str(key): value for key, value in tied_energy.items()},
            "shared_directions_sha256": _tensor_sha256(shared_directions),
            "core_sha256": _tensor_sha256(core),
            "haar_directions_sha256": _tensor_sha256(random_directions),
        },
        "physical": {
            "native_ce": native_ce, "damage": damage, "wave_damage": wave_damage,
            "coefficient_relative_mse_vs_deployed_oracle": coefficient_relative_mse,
            "matched_covariance_minus_tucker_damage": matched_margins,
            "oracle_gain_recovery": recoveries,
            "calls": calls, "calls_live": calls_live,
            "training_site0_calls": train_calls,
            "state_replay_max_abs_error": state_replay_max,
            "r410_r24_damage_reproduction_error": abs(damage["R24"] - R410_R24_DAMAGE),
        },
        "literal_price": {
            "p448_values": P448_VALUES,
            "tucker_producer_values": prices,
            "p448_plus_tucker_values": {
                str(rank): P448_VALUES + prices[str(rank)] for rank in TUCKER_RANKS},
            "matched_covariance_total_values": matched_prices,
            "matched_covariance_ranks": {str(key): value for key, value in MATCHED_RANKS.items()},
            "p640_values": P640_VALUES,
            "rank227_exceeds_p640_by_values": P448_VALUES + _producer_price(227) - P640_VALUES,
            "symmetric_core_storage": True,
        },
        "oracle_gain": oracle_gain,
        "selected_screen_rank": selected,
        'pred_a_authority_tensor_identity_and_prices_are_exact': bool(pred_a),
        'pred_b_tucker226_enters_matched_price_contention': bool(pred_b),
        'pred_c_shared_input_structure_is_more_efficient': bool(pred_c),
        'pred_d_joint_tucker_is_specific_vs_haar_and_independent': bool(pred_d),
        "null_no_joint_tucker_producer_beats_matched_covariance_rank": bool(strong_null),
        "next_object": (
            f"fresh_ood_signed_composition_gate_tucker{selected}" if selected is not None
            else "close_low_rank_u64_quadratic_producer_family" if pred_a else None),
        "adoption_licensed": False,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 JOINT TUCKER QUADRATIC PRODUCER DONE", flush=True)


if __name__ == "__main__":
    main()
