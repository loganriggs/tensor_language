"""RUNG 407 -- PRICE-AWARE ORACLE CEILING FOR TWO p448 EXPERTS.

Rebuild the five outcome-frozen p448 programs from rungs 403/405/406, score
physical per-document/per-position losses on the 384-document authority, and
enumerate every two-expert future-loss oracle.  Compare the optimistic oracle
against cheaper single p640/p768 programs before fitting any router state.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import itertools
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
OUT = BQ / "mlp0_p448_router_oracle_ceiling_results.json"
LOSS_ARTIFACT = BQ / "mlp0_p448_router_oracle_losses.pt"
PARENT_405 = BQ / "mlp0_rank448_token_grammar_active_subspace_results.json"
PARENT_406 = BQ / "mlp0_rank448_downstream_fisher_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
CONFIRM_RECEIPT = BQ / "mlp0_c512_mlp1_interchange_v1_rows_receipt.json"
CONFIRM_CACHE = BQ / ".rowcache_mlp0_c512_mlp1_interchange_v1/eval_384_source_documents.pt"
FIT_CACHE = BQ / ".rowcache/fineweb_n192_skip11000.pt"
FIT_SLICE = (0, 24)
RANKS = (448, 640, 768)
D = 1152
H = 4608
SOURCE_DOCUMENTS = 384
WAVE_DOCUMENTS = 96
N_WAVES = 4
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
P448_EXPERTS = ("covariance_p448", "T_active_p448", "I_active_p448",
                "TI_active_p448", "Fisher_p448")
ALL_PROGRAMS = P448_EXPERTS + ("covariance_p640", "covariance_p768")


def _price(rank):
    return D * rank + 2 * H * rank + H * D + D


P448_VALUES = _price(448)
P640_VALUES = _price(640)
P768_VALUES = _price(768)
TWO_P448_VALUES = 2 * (D * 448 + 2 * H * 448) + H * D + D
NATIVE_VALUES = 3 * H * D + D


def _tensor_sha256(value: torch.Tensor):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manual_logits(model, index):
    x = F.rms_norm(model.transformer.wte(index), (D,))
    x0 = x
    value0 = None
    for block in model.transformer.h:
        x, value0 = block(x, value0, x0)
    logits = model.lm_head(F.rms_norm(x, (D,)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


@torch.no_grad()
def _score_losses(model, rows, programs, rung403, device):
    names = ("native",) + tuple(programs)
    loss_parts = {name: [] for name in names}
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
        native_loss = F.cross_entropy(
            logits[:, SCORING].transpose(1, 2), target[:, SCORING], reduction="none")
        loss_parts["native"].append(native_loss.cpu())

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
            loss = F.cross_entropy(
                logits[:, SCORING].transpose(1, 2), target[:, SCORING], reduction="none")
            loss_parts[name].append(loss.cpu())
    losses = torch.stack([torch.cat(loss_parts[name]) for name in names])
    return names, losses, {
        "calls": calls,
        "pre_mlp0_state_replay_max_abs_error": state_error,
        "compact_endpoint_duplicate_max_abs_error": endpoint_error,
    }


def _oracle_summary(losses, names):
    index = {name: position for position, name in enumerate(names)}
    native = losses[index["native"]]
    native_ce = float(native.double().mean())
    damage = {
        name: float(losses[index[name]].double().mean()) - native_ce
        for name in ALL_PROGRAMS
    }
    document_loss = losses.double().mean(2)
    pairs = {}
    for left, right in itertools.combinations(P448_EXPERTS, 2):
        left_document = document_loss[index[left]]
        right_document = document_loss[index[right]]
        winner_left = left_document <= right_document
        document_oracle = torch.minimum(left_document, right_document)
        position_oracle = torch.minimum(losses[index[left]], losses[index[right]])
        wave_damage = []
        for wave in range(N_WAVES):
            start, end = wave * WAVE_DOCUMENTS, (wave + 1) * WAVE_DOCUMENTS
            wave_native = float(document_loss[index["native"], start:end].mean())
            wave_damage.append(float(document_oracle[start:end].mean()) - wave_native)
        key = f"{left}+{right}"
        pairs[key] = {
            "experts": [left, right],
            "document_oracle_damage": float(document_oracle.mean()) - native_ce,
            "position_oracle_damage": float(position_oracle.double().mean()) - native_ce,
            "gain_over_covariance_p448": (
                damage["covariance_p448"] - (float(document_oracle.mean()) - native_ce)),
            "document_winner_fraction": {
                left: float(winner_left.double().mean()),
                right: float((~winner_left).double().mean()),
            },
            "wave_document_oracle_damage": wave_damage,
        }
    best_key = min(pairs, key=lambda key: pairs[key]["document_oracle_damage"])
    p448_indices = torch.tensor([index[name] for name in P448_EXPERTS])
    all_five_document = document_loss[p448_indices].min(0).values
    all_five_position = losses[p448_indices].min(0).values
    return {
        "native_ce": native_ce,
        "program_damage": damage,
        "pairs": pairs,
        "best_pair": best_key,
        "all_five_document_oracle_damage": float(all_five_document.mean()) - native_ce,
        "all_five_position_oracle_damage": float(all_five_position.double().mean()) - native_ce,
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (
            PARENT_405, PARENT_406, ROWS_RECEIPT,
            CONFIRM_RECEIPT, CONFIRM_CACHE, FIT_CACHE))
        assert len(tuple(itertools.combinations(P448_EXPERTS, 2))) == 10
        assert (P448_VALUES, P640_VALUES, P768_VALUES, TWO_P448_VALUES, NATIVE_VALUES) == (
            9_954_432, 11_945_088, 13_272_192, 14_599_296, 15_926_400)
        print("MLP0 p448 ROUTER ORACLE | dry run: programs, pairs, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(BQ.parent / "qk_mdl"))
    import bilin18_observed_model_facade as facade
    import mlp0_centered_context_anova_factorial as base
    import mlp0_rank448_branch_error_factorial as rung403
    import mlp0_rank448_token_grammar_active_subspace as rung405
    import mlp0_rank448_downstream_fisher as rung406
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits as covariance_logits

    device = torch.device("cuda")
    parent405 = json.loads(PARENT_405.read_text())
    parent406 = json.loads(PARENT_406.read_text())
    if (not parent405["null_first_order_token_grammar_metric_fails"]
            or not parent406["null_downstream_fisher_does_not_improve_p448"]):
        raise RuntimeError("prior nulls do not license router oracle")

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
        and all(current_rows_receipt["disjointness"].values()))
    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()

    source_model, source_checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    covariance = _covariance(source_model, fit_rows, covariance_logits)
    covariance_programs = {}
    covariance_bases = {}
    covariance_diagnostics = {}
    for rank in RANKS:
        program, basis, diagnostic = _rrr_program(
            source_model.transformer.h[0].mlp, covariance, rank=rank)
        covariance_programs[rank] = {name: value.detach().clone() for name, value in program.items()}
        covariance_bases[rank] = basis.detach().clone()
        covariance_diagnostics[rank] = diagnostic

    active_metrics, covariance_square, covariance_inverse, active_diagnostic = \
        rung405._active_metrics(source_model, fit_rows, covariance, base, device)
    active_bases = {}
    for name, key in (("T_active_p448", "T"), ("I_active_p448", "I"),
                      ("TI_active_p448", "TI")):
        basis, _values, _orthogonality = rung405._top_basis(active_metrics[key])
        active_bases[name] = basis
    with torch.enable_grad():
        fisher_metrics, fisher_gradient_diagnostic = rung406._fisher_metrics(
            source_model, fit_rows, device)
    fisher_whitened = covariance_square @ fisher_metrics["all"] @ covariance_square
    fisher_whitened = 0.5 * (fisher_whitened + fisher_whitened.T)
    fisher_basis, _fisher_values, _fisher_orthogonality = rung405._top_basis(fisher_whitened)

    programs = {
        "covariance_p448": covariance_programs[448],
        "T_active_p448": rung405._program_from_basis(
            source_model.transformer.h[0].mlp, active_bases["T_active_p448"],
            covariance_square, covariance_inverse),
        "I_active_p448": rung405._program_from_basis(
            source_model.transformer.h[0].mlp, active_bases["I_active_p448"],
            covariance_square, covariance_inverse),
        "TI_active_p448": rung405._program_from_basis(
            source_model.transformer.h[0].mlp, active_bases["TI_active_p448"],
            covariance_square, covariance_inverse),
        "Fisher_p448": rung405._program_from_basis(
            source_model.transformer.h[0].mlp, fisher_basis,
            covariance_square, covariance_inverse),
        "covariance_p640": covariance_programs[640],
        "covariance_p768": covariance_programs[768],
    }
    basis_hashes = {
        "covariance_p448": _tensor_sha256(covariance_bases[448]),
        **{name: _tensor_sha256(basis) for name, basis in active_bases.items()},
        "Fisher_p448": _tensor_sha256(fisher_basis),
        "covariance_p640": _tensor_sha256(covariance_bases[640]),
        "covariance_p768": _tensor_sha256(covariance_bases[768]),
    }
    del active_metrics, fisher_metrics, fisher_whitened, covariance
    del covariance_square, covariance_inverse, source_model
    torch.cuda.empty_cache()

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    names, losses, physical = _score_losses(model, confirm_rows, programs, rung403, device)
    finite_losses = bool(torch.isfinite(losses).all())
    oracle = _oracle_summary(losses, names)
    artifact = {
        "program_names": names,
        "document_ordinals": ordinals,
        "scoring_positions_half_open": [SCORING.start, SCORING.stop],
        "losses": losses,
    }
    torch.save(artifact, LOSS_ARTIFACT)
    artifact_tensor_hash = _tensor_sha256(losses)
    artifact_file_hash = _file_sha256(LOSS_ARTIFACT)

    expected_shapes = {
        rank: {"encoder": (rank, D), "left": (H, rank), "right": (H, rank),
               "down": (D, H), "bias": (D,)}
        for rank in RANKS
    }
    shapes_exact = all(
        {key: tuple(value.shape) for key, value in programs[name].items()}
        == expected_shapes[448 if "p448" in name else 640 if "p640" in name else 768]
        for name in programs)
    parent_wave_errors = []
    for wave in range(N_WAVES):
        start, end = wave * WAVE_DOCUMENTS, (wave + 1) * WAVE_DOCUMENTS
        native_wave = float(losses[names.index("native"), start:end].double().mean())
        comparisons = {
            "covariance_p448": parent405["wave_results"][str(wave)]["damage"]["covariance"],
            "T_active_p448": parent405["wave_results"][str(wave)]["damage"]["T_active"],
            "I_active_p448": parent405["wave_results"][str(wave)]["damage"]["I_active"],
            "TI_active_p448": parent405["wave_results"][str(wave)]["damage"]["TI_active"],
            "Fisher_p448": parent406["wave_results"][str(wave)]["damage"]["Fisher"],
        }
        for name, expected in comparisons.items():
            observed = float(losses[names.index(name), start:end].double().mean()) - native_wave
            parent_wave_errors.append(abs(observed - expected))
    parent_wave_max_error = max(parent_wave_errors)
    expected_calls = SOURCE_DOCUMENTS // DOCUMENT_BATCH
    calls_exact = all(value == expected_calls for value in physical["calls"].values())
    states_exact = (
        max(physical["pre_mlp0_state_replay_max_abs_error"].values()) == 0.0
        and max(physical["compact_endpoint_duplicate_max_abs_error"].values()) == 0.0)
    parent_hash_exact = (
        basis_hashes["covariance_p448"] == parent405["program"]["basis_hashes"]["covariance"]
        and basis_hashes["T_active_p448"] == parent405["program"]["basis_hashes"]["T_active"]
        and basis_hashes["I_active_p448"] == parent405["program"]["basis_hashes"]["I_active"]
        and basis_hashes["TI_active_p448"] == parent405["program"]["basis_hashes"]["TI_active"]
        and basis_hashes["Fisher_p448"] == parent406["metric_diagnostic"]["fisher_basis_hash"])
    prices_exact = (
        (P448_VALUES, P640_VALUES, P768_VALUES, TWO_P448_VALUES, NATIVE_VALUES)
        == (9_954_432, 11_945_088, 13_272_192, 14_599_296, 15_926_400))
    pred_a = (
        population_exact and shapes_exact and finite_losses and calls_exact and states_exact
        and parent_hash_exact and parent_wave_max_error <= 1e-6 and prices_exact
        and abs(covariance_diagnostics[448]["context_cov_retained_energy"]
                - .9011108875274658) <= 2e-6)

    pairs = oracle["pairs"]
    qualifying_b = [key for key, value in pairs.items()
                    if value["document_oracle_damage"]
                    <= .70 * oracle["program_damage"]["covariance_p448"]
                    and value["position_oracle_damage"]
                    <= .50 * oracle["program_damage"]["covariance_p448"]]
    pred_b = bool(qualifying_b)
    best = pairs[oracle["best_pair"]]
    p768_damage = oracle["program_damage"]["covariance_p768"]
    pred_c = (
        best["document_oracle_damage"] <= p768_damage - .0005
        and best["position_oracle_damage"] < p768_damage)
    winner_fractions = list(best["document_winner_fraction"].values())
    covariance_wave_damage = []
    for wave in range(N_WAVES):
        start, end = wave * WAVE_DOCUMENTS, (wave + 1) * WAVE_DOCUMENTS
        native_wave = float(losses[names.index("native"), start:end].double().mean())
        covariance_wave_damage.append(
            float(losses[names.index("covariance_p448"), start:end].double().mean())
            - native_wave)
    pred_d = (
        min(winner_fractions) >= .20
        and all(best["wave_document_oracle_damage"][wave] < covariance_wave_damage[wave]
                for wave in range(N_WAVES)))
    strong_null = (
        not pred_a
        or oracle["all_five_position_oracle_damage"] > p768_damage - .0002
        or best["document_oracle_damage"] >= p768_damage
        or oracle["program_damage"]["covariance_p448"]
        - best["document_oracle_damage"] < .0002)

    result = {
        "status": "mlp0_p448_router_oracle_ceiling_complete",
        "rung": 407,
        "claim_level": "future_loss_oracle_upper_bound_not_router_or_compression",
        "convention": "CE added above native; lower is better",
        "population": {
            "fit_cache": str(FIT_CACHE),
            "fit_slice": list(FIT_SLICE),
            "fit_documents": len(fit_rows),
            "evaluation_receipt": str(CONFIRM_RECEIPT),
            "source_documents": SOURCE_DOCUMENTS,
            "waves": N_WAVES,
            "documents_per_wave": WAVE_DOCUMENTS,
            "scored_positions": SOURCE_DOCUMENTS * 192,
            "population_exact": population_exact,
        },
        "programs": {
            "source_checkpoint": source_checkpoint.__dict__,
            "source_dtype": "float32_as_in_rungs403_406",
            "evaluation_dtype": "bfloat16_as_in_rung404",
            "basis_hashes": basis_hashes,
            "active_metric_diagnostic": active_diagnostic,
            "fisher_gradient_diagnostic": fisher_gradient_diagnostic,
            "covariance_diagnostics": {str(rank): value for rank, value in covariance_diagnostics.items()},
            "shapes_exact": shapes_exact,
        },
        "literal_prices": {
            "one_p448_values": P448_VALUES,
            "one_p640_values": P640_VALUES,
            "one_p768_values": P768_VALUES,
            "two_p448_shared_down_bias_lower_bound_values": TWO_P448_VALUES,
            "native_mlp0_values": NATIVE_VALUES,
            "router_parameters_state_compute_included": False,
        },
        "loss_artifact": {
            "path": str(LOSS_ARTIFACT),
            "shape": list(losses.shape),
            "tensor_sha256": artifact_tensor_hash,
            "file_sha256": artifact_file_hash,
            "finite": finite_losses,
        },
        "physical_checks": {
            **physical,
            "parent_basis_hashes_exact": parent_hash_exact,
            "parent_wave_damage_max_abs_error": parent_wave_max_error,
            "calls_exact": calls_exact,
            "states_endpoints_exact": states_exact,
        },
        "oracle": oracle,
        "covariance_p448_wave_damage": covariance_wave_damage,
        "pairs_satisfying_material_headroom_bar": qualifying_b,
        'pred_a_fixed_programs_population_and_measurement_are_exact': bool(pred_a),
        'pred_b_some_two_expert_oracle_has_material_headroom': bool(pred_b),
        'pred_c_best_pair_oracle_survives_cheaper_p768_control': bool(pred_c),
        'pred_d_best_pair_is_balanced_and_all_wave': bool(pred_d),
        "null_router_oracle_has_no_adoption_headroom": bool(strong_null),
        "next_object": (
            "heldout_prefix_observable_two_state_router"
            if pred_a and pred_b and pred_c and pred_d and not strong_null else None),
        "router_or_compression_licensed": False,
        "FINAL_opened": 0,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 ROUTER ORACLE CEILING DONE", flush=True)


if __name__ == "__main__":
    main()
