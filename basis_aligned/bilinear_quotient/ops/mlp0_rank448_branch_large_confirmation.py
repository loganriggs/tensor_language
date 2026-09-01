"""RUNG 404 -- LARGE-DOCUMENT CONFIRMATION OF p448 INTERACTION DAMAGE.

Use one frozen chunk from each of 384 disjoint FineWeb source documents, split
into four fixed 96-document waves.  Keep rung403's exact float32-source p448
program, BF16 grammar model, FIT reference, five factors, and all 32 arms.

Frozen predictions
------------------
pred_a: row receipt/hash/disjointness/one-row-per-document/four waves are exact;
    p448 energy/rank/shapes/price are exact; all wave identities<=1e-8,
    endpoints/state replay exact, and all calls live.
pred_b: I is top pooled and in every wave; pooled I exceeds T by>=.0015 nat.
pred_c: pooled I is positive and >=60% of positive named Shapley damage; named
    rho with rung403 SELECT>=.80; I is positive in every wave.
pred_d: pooled total damage is in[0,.020] and every wave is positive; pooled and
    per-wave |AUX|<=.002; wave-I range<=.008.

Strong null: pred_a fails; pooled I<=0 or is not top; at least two waves have
I<=T; named rho<=0; or pooled |AUX|>=sum absolute named Shapleys. Full pass
confirms an equal-rank/equal-price global interaction-weighted projection
screen. No compressor, adoption, rank tuning, FINAL, or head-label commitment.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "mlp0_rank448_branch_large_confirmation_results.json"
PARENT = BQ / "mlp0_rank448_branch_error_factorial_results.json"
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
NAMED = ("T", "C", "I", "S")
COMPACT_VALUES = D * RANK + 2 * H * RANK + H * D + D
RUNG328_RETAINED_ENERGY = 0.9011108875274658


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor):
    return hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()


def _positive_top_share(values):
    positive = torch.as_tensor(values, dtype=torch.float64).clamp_min(0)
    total = float(positive.sum())
    return float(positive.max()) / total if total > 0 else 0.0


def _combine(parent_module, waves):
    labels = {subset: parent_module._arm_name(subset) for subset in parent_module.ARMS}
    pooled_ce = {
        label: sum(wave["pooled_ce"][label] for wave in waves) / len(waves)
        for label in labels.values()
    }
    performance = {subset: pooled_ce[label] for subset, label in labels.items()}
    shapley = parent_module._shapley(performance)
    order = sorted(NAMED, key=lambda name: shapley[name], reverse=True)
    return {
        "pooled_ce": pooled_ce,
        "total_compact_ce_damage": pooled_ce["T+C+I+S+A"] - pooled_ce["EMPTY"],
        "shapley_ce_damage": shapley,
        "mobius_ce_damage": parent_module._mobius(performance),
        "named_rank_order": order,
        "named_positive_top_share": _positive_top_share([shapley[name] for name in NAMED]),
    }


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (
            PARENT, ROWS_RECEIPT, CONFIRM_RECEIPT, CONFIRM_CACHE, FIT_CACHE))
        receipt = json.loads(CONFIRM_RECEIPT.read_text())
        assert receipt["selection"]["n_source_documents"] == SOURCE_DOCUMENTS
        assert N_WAVES * WAVE_DOCUMENTS == SOURCE_DOCUMENTS
        assert COMPACT_VALUES == 9_954_432
        print("MLP0 p448 LARGE CONFIRM | dry run: rows, 4 waves, 32 arms, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(BQ.parent / "qk_mdl"))
    import bilin18_observed_model_facade as facade
    import mlp0_centered_context_anova_factorial as base
    import mlp0_rank448_branch_error_factorial as rung403
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    device = torch.device("cuda")
    parent = json.loads(PARENT.read_text())
    if (parent["null_branch_audit_is_invalid_or_uninformative"]
            or parent["select_top_named_branch"] != "I"):
        raise RuntimeError("rung403 does not license the I-led confirmation")

    confirm_receipt = json.loads(CONFIRM_RECEIPT.read_text())
    current_rows_receipt = json.loads(ROWS_RECEIPT.read_text())
    all_rows = torch.load(CONFIRM_CACHE, map_location="cpu")
    records = confirm_receipt["document_provenance"]["sets"]["eval"]
    chunk0_indices = [index for index, record in enumerate(records)
                      if int(record["chunk_id"]) == 0]
    document_ids = [records[index]["document_id"] for index in chunk0_indices]
    ordinals = [int(records[index]["source_document_ordinal"])
                for index in chunk0_indices]
    confirm_rows = all_rows[chunk0_indices, :257].long().contiguous()
    cache_file_sha = _file_sha256(CONFIRM_CACHE)
    cache_tensor_sha = _tensor_sha256(all_rows)
    row_population_exact = (
        tuple(all_rows.shape) == (585, 513)
        and tuple(confirm_rows.shape) == (SOURCE_DOCUMENTS, 257)
        and len(chunk0_indices) == SOURCE_DOCUMENTS
        and len(set(document_ids)) == SOURCE_DOCUMENTS
        and ordinals == list(range(SOURCE_DOCUMENTS))
        and cache_tensor_sha == confirm_receipt["entries"]["eval"]["tensor_raw_sha256"]
        and cache_file_sha == current_rows_receipt["prior_tensor_hashes"][str(CONFIRM_CACHE)]
        and all(confirm_receipt["disjointness_gates"].values())
        and all(current_rows_receipt["disjointness"].values())
    )
    waves = [
        confirm_rows[index * WAVE_DOCUMENTS:(index + 1) * WAVE_DOCUMENTS]
        for index in range(N_WAVES)
    ]

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    program_fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    program_model, program_checkpoint = facade.load_bilin18(
        device=device, dtype=torch.float32)
    covariance = _covariance(program_model, program_fit_rows, _manual_logits)
    program, _basis, program_diagnostic = _rrr_program(
        program_model.transformer.h[0].mlp, covariance, rank=RANK)
    program = {name: value.detach().clone() for name, value in program.items()}
    del covariance, _basis, program_model
    torch.cuda.empty_cache()

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    native_factors = {
        "left": model.transformer.h[0].mlp.Left.weight.detach().float(),
        "right": model.transformer.h[0].mlp.Right.weight.detach().float(),
        "down": model.transformer.h[0].mlp.Down.weight.detach().float(),
    }
    compact_factors = {name: value.detach().float() for name, value in program.items()}
    expected_shapes = {
        "encoder": (RANK, D), "left": (H, RANK), "right": (H, RANK),
        "down": (D, H), "bias": (D,),
    }
    observed_shapes = {name: tuple(value.shape) for name, value in compact_factors.items()}

    fit_rows = rows_parent.load_role(current_rows_receipt["entries"]["FIT"])
    token_cpu, context_cpu, gain_cpu = base._capture_inputs(model, fit_rows, device)
    native_reference = rung403._reference_for_factors(
        token_cpu, context_cpu, gain_cpu, native_factors, device)
    compact_reference = rung403._reference_for_factors(
        token_cpu, context_cpu, gain_cpu, compact_factors, device)
    wave_results = [
        rung403._score_role(
            model, rows, device, base, native_factors, compact_factors,
            native_reference, compact_reference)
        for rows in waves
    ]
    combined = _combine(rung403, wave_results)

    exact_program = (
        observed_shapes == expected_shapes
        and int(program_diagnostic["rank"]) == RANK
        and abs(program_diagnostic["context_cov_retained_energy"]
                - RUNG328_RETAINED_ENERGY) <= 2e-6
    )
    exact_waves = all(
        wave["diagnostics"]["native_analytical_identity_relative_mse"] <= 1e-8
        and wave["diagnostics"]["compact_analytical_identity_relative_mse"] <= 1e-8
        and wave["diagnostics"]["native_endpoint_state_max_abs_error"] == 0.0
        and wave["diagnostics"]["compact_endpoint_state_max_abs_error"] == 0.0
        and wave["diagnostics"]["pre_mlp0_state_replay_max_abs_error"] == 0.0
        and wave["diagnostics"]["live_census"]
        for wave in wave_results
    )
    pred_a = row_population_exact and exact_program and exact_waves
    combined_shapley = combined["shapley_ce_damage"]
    wave_shapley = [wave["shapley_ce_damage"] for wave in wave_results]
    pred_b = (
        max(NAMED, key=lambda name: combined_shapley[name]) == "I"
        and all(max(NAMED, key=lambda name: values[name]) == "I"
                for values in wave_shapley)
        and combined_shapley["I"] >= combined_shapley["T"] + .0015
    )
    parent_select = parent["roles"]["SELECT"]["shapley_ce_damage"]
    named_spearman = rung403._spearman(
        [parent_select[name] for name in NAMED],
        [combined_shapley[name] for name in NAMED])
    pred_c = (
        combined_shapley["I"] > 0
        and combined["named_positive_top_share"] >= .60
        and named_spearman >= .80
        and all(values["I"] > 0 for values in wave_shapley)
    )
    wave_total = [wave["total_compact_ce_damage"] for wave in wave_results]
    wave_i = [values["I"] for values in wave_shapley]
    pred_d = (
        0 <= combined["total_compact_ce_damage"] <= .020
        and all(value > 0 for value in wave_total)
        and abs(combined_shapley["A"]) <= .002
        and all(abs(values["A"]) <= .002 for values in wave_shapley)
        and max(wave_i) - min(wave_i) <= .008
    )
    auxiliary_dominates = (
        abs(combined_shapley["A"])
        >= sum(abs(combined_shapley[name]) for name in NAMED)
    )
    waves_i_not_above_t = sum(values["I"] <= values["T"] for values in wave_shapley)
    strong_null = (
        not pred_a or combined_shapley["I"] <= 0
        or max(NAMED, key=lambda name: combined_shapley[name]) != "I"
        or waves_i_not_above_t >= 2 or named_spearman <= 0
        or auxiliary_dominates
    )

    result = {
        "status": "mlp0_rank448_branch_large_confirmation_complete",
        "rung": 404,
        "claim_level": "large_document_fixed_program_branch_confirmation_not_compression",
        "convention": "CE added above native; lower is better",
        "population": {
            "receipt": str(CONFIRM_RECEIPT),
            "cache": str(CONFIRM_CACHE),
            "cache_file_sha256": cache_file_sha,
            "cache_tensor_sha256": cache_tensor_sha,
            "source_documents": SOURCE_DOCUMENTS,
            "selected_chunk_id": 0,
            "rows": list(confirm_rows.shape),
            "waves": N_WAVES,
            "documents_per_wave": WAVE_DOCUMENTS,
            "scored_positions_per_document": 192,
            "total_scored_positions": SOURCE_DOCUMENTS * 192,
            "row_population_exact": row_population_exact,
        },
        "program_identity": {
            "source_dtype": "float32_as_in_rung328",
            "evaluation_dtype": "bfloat16_as_in_rung401",
            "source_checkpoint": program_checkpoint.__dict__,
            "rank": RANK,
            "shapes": {name: list(shape) for name, shape in observed_shapes.items()},
            "fit_diagnostic": program_diagnostic,
            "literal_mlp0_values": COMPACT_VALUES,
        },
        "parent_rung403_result": str(PARENT),
        "waves": {str(index): value for index, value in enumerate(wave_results)},
        "combined": combined,
        "wave_total_damage": wave_total,
        "wave_i_shapley": wave_i,
        "wave_i_range": max(wave_i) - min(wave_i),
        "waves_i_not_above_t": waves_i_not_above_t,
        "named_spearman_vs_rung403_select": named_spearman,
        "auxiliary_dominates_named_magnitude": auxiliary_dominates,
        'pred_a_population_program_and_instrument_are_exact': bool(pred_a),
        'pred_b_interaction_damage_dominates_at_larger_scale': bool(pred_b),
        'pred_c_rung403_interaction_route_transports': bool(pred_c),
        'pred_d_not_auxiliary_or_one_unstable_wave': bool(pred_d),
        "null_large_confirmation_fails": bool(strong_null),
        "next_object": "global_interaction_weighted_projection" if not strong_null and pred_b and pred_c else None,
        "compression_or_adoption_licensed": False,
        "FINAL_opened": 0,
        "checkpoint": checkpoint.__dict__,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 LARGE BRANCH CONFIRMATION DONE", flush=True)


if __name__ == "__main__":
    main()
