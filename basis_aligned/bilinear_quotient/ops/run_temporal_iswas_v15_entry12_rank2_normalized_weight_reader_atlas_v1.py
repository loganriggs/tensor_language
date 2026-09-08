#!/usr/bin/env python3
"""Normalization-aware literal weight atlas for the causal entry12 rank-two span."""

# BQGATE: EXPERIMENT pred_a_authority_geometry_finiteness_and_price pred_b_raw_and_exact_normalized_rankings_differ pred_c_rms_tangent_tracks_exact_finite_response pred_d_exact_reader_nominations_are_fold_stable pred_e_a_shared_exact_reader_is_nominated pred_f_known_oracle_sources_write_into_the_span
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_response_basis_contract as basis_contract
import normalized_weight_reader_contract as reader
import residual_state_mediation_executor as state_executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v1.json"
BASIS_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json"
OOD_AUDIT = ROOT / "circuits/followups/temporal_iswas_v15_token_router_v16_signature_coverage_audit_v1_result.json"
TOKEN_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_unordered_token_pair_router_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v1"
EXPECTED = {
    "prior": "7fe1e420a2c3493bd8c5e5916e242b1b9af314c8709d7a45dbf5da73a5db2688",
    "basis_result": "ffaec3f0bede415f7b9b1664ae7ddf1667978dc26f42cfa2f81e9c82f0f77467",
    "ood_audit": "92abdb1d7f17ee253288c5dd92cc2369dd3950e21a582b00c95d0b04816a8054",
    "token_result": "582c19b22017fd0bb1070260b4737b8b046c54c3838a0a646a85254d77135be0",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "reader": "bf467b46477b41ae8a56099ea6ebf93a6018241cf26f58ea1e5b7e378922b6ff",
    "basis_contract": "cc05ed73baf02c7ec6f0ec2b9e64a831a1fff6462ab4d8a5ba1f39bc5f70dab1",
    "state_executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "dependency": "4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
}
FILES = {
    "prior": PRIOR, "basis_result": BASIS_RESULT, "ood_audit": OOD_AUDIT,
    "token_result": TOKEN_RESULT, "oracles": ORACLES,
    "reader": ROOT / "ops/normalized_weight_reader_contract.py",
    "basis_contract": ROOT / "ops/entry12_response_basis_contract.py",
    "state_executor": ROOT / "ops/residual_state_mediation_executor.py",
    "dependency": ROOT / "ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
}
PRICE_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 6,
             "transformer_backward_forwards": 0, "model_updates": 0,
             "example_evaluations": 1400, "fit_parameters": 0, "checkpoint_loads": 1}
EXPERTS = {"A1": "A1_oracle", "A2": "A2_oracle"}
KNOWN_WRITERS = {"L8H1", "L9H1", "L9H4", "L11H3"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def stack_prefix(torch, tensor, context, panel):
    indices = context["panel_indices"][panel].detach().cpu().tolist()
    return torch.cat([tensor[index, :int(context["base_batch"].semantic_positions[index]) + 1].float()
                      for index in indices], dim=0)


def relative_geometry_error(observed, expected):
    left = observed["union_singular_values"]
    right = expected["union_singular_values"]
    if expected["ranks"]["construction_union_rank_le_2"] != 2 or len(left) != len(right):
        return float("inf")
    return max(abs(float(a) - float(b)) / max(abs(float(b)), 1e-30) for a, b in zip(left, right))


def coordinate_strengths(torch, matrix, state, basis):
    samples, rank, width = state.shape[0], basis.shape[1], state.shape[1]
    directions = basis.T[None].expand(samples, rank, width).reshape(samples * rank, width)
    states = state[:, None].expand(samples, rank, width).reshape(samples * rank, width)
    tangent = reader.rms_jvp(torch, states, directions)
    normalizer = matrix.norm().clamp_min(1e-30) * directions.norm().clamp_min(1e-30)
    return {
        "raw_coordinate_strength": float((directions @ matrix.T).norm() / normalizer),
        "tangent_coordinate_strength": float((tangent @ matrix.T).norm() /
                                               (matrix.norm().clamp_min(1e-30) * tangent.norm().clamp_min(1e-30))),
    }


def reader_records(torch, model, x0, off_states, delta, context, panel):
    records = []
    for layer in range(12, 18):
        block = model.transformer.h[layer]
        boundary = off_states["entry12"] if layer == 12 else off_states[f"post_mlp{layer - 1}"]
        attention_state = block.lambdas[0].detach() * boundary + block.lambdas[1].detach() * x0
        attention_state = stack_prefix(torch, attention_state, context, panel)
        attention_delta = block.lambdas[0].detach() * stack_prefix(torch, delta, context, panel)
        for head in range(9):
            for attr in ("c_q", "c_k", "c_q2", "c_k2", "c_v"):
                matrix = getattr(block.attn, attr).weight.detach().float()[head * 128:(head + 1) * 128]
                item = reader.reader_response(torch, matrix, attention_state, attention_delta)
                records.append({"label": f"L{layer}H{head}:{attr[2:]}", **item})
        mlp_state = stack_prefix(torch, off_states[f"post_attn{layer}"], context, panel)
        mlp_delta = stack_prefix(torch, delta, context, panel)
        for attr in ("Left", "Right"):
            matrix = getattr(block.mlp, attr).weight.detach().float()
            item = reader.reader_response(torch, matrix, mlp_state, mlp_delta)
            records.append({"label": f"MLP{layer}:{attr.lower()}", **item})
    return records


def writer_records(torch, model, basis):
    records = []
    normalizer_u = basis.norm().clamp_min(1e-30)
    for layer in range(12):
        block = model.transformer.h[layer]
        output = block.attn.c_proj.weight.detach().float()
        for head in range(9):
            matrix = output[:, head * 128:(head + 1) * 128]
            strength = float((basis.T @ matrix).norm() /
                             (normalizer_u * matrix.norm().clamp_min(1e-30)))
            records.append({"label": f"L{layer}H{head}", "writer_strength": strength})
        matrix = block.mlp.Down.weight.detach().float()
        strength = float((basis.T @ matrix).norm() /
                         (normalizer_u * matrix.norm().clamp_min(1e-30)))
        records.append({"label": f"MLP{layer}", "writer_strength": strength})
    return reader.ranked(records, "writer_strength")


def overlap(left, right, n=10):
    a = {row["label"] for row in left[:n]}
    b = {row["label"] for row in right[:n]}
    return len(a & b) / n


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    basis_result = json.loads(BASIS_RESULT.read_text())
    ood = json.loads(OOD_AUDIT.read_text())
    token = json.loads(TOKEN_RESULT.read_text())
    authority = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and basis_result.get("predictions", {}).get("pred_b_construction_union_is_target_sufficient") is True
        and ood.get("terminal") == "unseen_signature_default_off"
        and token.get("terminal") == "token_pair_router_candidate")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "reader_interfaces_per_cell": 282, "held_cells": 4,
           "expected_differentiable_forwards": 6, "price_max": PRICE_MAX}
    if not authority:
        raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)

    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    config = {key: getattr(backend.model.config, key) for key in (
        "vocab_size", "n_layer", "n_head", "n_embd", "squared_mlp",
        "bilinear", "expansion_factor", "gated", "squared_attn", "bilinear_attn")}
    expected_config = {"vocab_size": 50304, "n_layer": 18, "n_head": 9, "n_embd": 1152,
                       "squared_mlp": False, "bilinear": True, "expansion_factor": 4,
                       "gated": False, "squared_attn": True, "bilinear_attn": True}
    counters = {name: 0 for name in PRICE_MAX}
    counters["checkpoint_loads"] = 1
    native = backend.native
    def counted(batch, *, capture):
        counters["native_capture_forwards"] += 1
        counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)
    backend.native = counted

    rows = v15.build_rows()
    bank = parent.capture_bank(backend, rows, counters, factors=False)
    contexts = {parity: parent.attach_references(
        backend, parent.subset_context(bank, parent.row_indices(rows, parity=parity)), counters)
        for parity in (0, 1)}
    off_cache = {}
    for parity, context in contexts.items():
        off_cache[parity] = state_executor.execute(parent, backend, context, counters, {}, capture=True)
    oracle_data = json.loads(ORACLES.read_text())
    on_cache = {panel: {} for panel in EXPERTS}
    for panel, expert in EXPERTS.items():
        bases = dependency.bases_for_evaluation(backend.torch, backend.device, oracle_data, expert)
        for parity, context in contexts.items():
            on_cache[panel][parity] = state_executor.execute(
                parent, backend, context, counters, bases[parity], capture=True)

    records, writers, geometry = {}, {}, {}
    orthogonality_error = geometry_error = 0.0
    for held in (0, 1):
        train = 1 - held
        train_context = contexts[train]
        train_off = off_cache[train][1]["entry12"]
        train_deltas = {panel: on_cache[panel][train][1]["entry12"].float() - train_off.float()
                        for panel in EXPERTS}
        target_a1 = stack_prefix(backend.torch, train_deltas["A1"], train_context, "A1").mean(0)
        target_a2 = stack_prefix(backend.torch, train_deltas["A2"], train_context, "A2").mean(0)
        p_rows = backend.torch.cat([stack_prefix(backend.torch, train_deltas[panel], train_context, "P")
                                    for panel in EXPERTS], dim=0)
        bases, basis_geometry = basis_contract.fit_bases(
            backend.torch, target_a1, target_a2, p_rows, relative_threshold=1e-6)
        union = bases["construction_union_rank_le_2"]
        orthogonality_error = max(orthogonality_error, float(
            (union.T @ union - backend.torch.eye(2, device=union.device)).abs().max()))
        geometry_error = max(geometry_error, relative_geometry_error(
            basis_geometry, basis_result["fits"][str(held)]))
        geometry[str(held)] = {"rank": int(union.shape[1]), **basis_geometry}
        writers[str(held)] = writer_records(backend.torch, backend.model, union)

        context = contexts[held]
        tokens, _lengths = backend._tensor_batch(context["base_batch"])
        x0 = backend.F.rms_norm(backend.model.transformer.wte(tokens), (backend.model.config.n_embd,))
        off_states = off_cache[held][1]
        records[str(held)] = {}
        for panel in EXPERTS:
            source = on_cache[panel][held][1]["entry12"]
            raw_delta = source.float() - off_states["entry12"].float()
            delta = (raw_delta @ union) @ union.T
            panel_records = reader_records(
                backend.torch, backend.model, x0, off_states, delta, context, panel)
            for item in panel_records:
                if not finite(item):
                    raise RuntimeError("nonfinite reader response")
            records[str(held)][panel] = {
                "raw_ranked": reader.ranked([dict(item) for item in panel_records], "raw_strength"),
                "tangent_ranked": reader.ranked([dict(item) for item in panel_records], "tangent_strength"),
                "exact_ranked": reader.ranked([dict(item) for item in panel_records], "exact_strength"),
            }

    raw_exact_overlap = {str(held): {panel: overlap(
        records[str(held)][panel]["raw_ranked"], records[str(held)][panel]["exact_ranked"])
        for panel in EXPERTS} for held in (0, 1)}
    tangent_fidelity = {str(held): {panel: {
        "passing_fraction": sum(
            row["tangent_exact_cosine"] >= .95 and row["tangent_exact_relative_l2"] <= .25
            for row in records[str(held)][panel]["exact_ranked"]) / 282,
        "min_cosine": min(row["tangent_exact_cosine"] for row in records[str(held)][panel]["exact_ranked"]),
        "max_relative_l2": max(row["tangent_exact_relative_l2"] for row in records[str(held)][panel]["exact_ranked"]),
    } for panel in EXPERTS} for held in (0, 1)}
    fold_stability = {panel: {
        "spearman": reader.spearman(records["0"][panel]["exact_ranked"],
                                    records["1"][panel]["exact_ranked"], "exact_strength"),
        "top10_overlap": overlap(records["0"][panel]["exact_ranked"],
                                 records["1"][panel]["exact_ranked"]),
    } for panel in EXPERTS}
    shared_exact = sorted(set.intersection(*(
        {row["label"] for row in records[str(held)][panel]["exact_ranked"][:10]}
        for held in (0, 1) for panel in EXPERTS)))
    writer_known = {fold: {row["label"]: row["writer_strength_percentile"]
                           for row in items if row["label"] in KNOWN_WRITERS}
                    for fold, items in writers.items()}

    counters["fit_parameters"] = 0
    native_closure = max(context["manual_native_max_abs_error"] for context in contexts.values())
    disjoint = set(contexts[0]["base_batch"].row_ids).isdisjoint(contexts[1]["base_batch"].row_ids)
    A = bool(authority and config == expected_config and disjoint and native_closure <= 1e-4
             and orthogonality_error <= 1e-5 and geometry_error <= 1e-5
             and finite({"records": records, "writers": writers, "geometry": geometry})
             and counters["differentiable_transformer_forwards"] == 6
             and all(counters[name] <= PRICE_MAX[name] for name in PRICE_MAX))
    B = any(value <= .8 for fold in raw_exact_overlap.values() for value in fold.values())
    C = all(item["passing_fraction"] >= .9 for fold in tangent_fidelity.values() for item in fold.values())
    D = all(item["spearman"] >= .5 and item["top10_overlap"] >= .4 for item in fold_stability.values())
    E = bool(shared_exact)
    F = all(sum(value >= .75 for value in fold.values()) >= 3 for fold in writer_known.values())
    predictions = dict(zip((
        "pred_a_authority_geometry_finiteness_and_price",
        "pred_b_raw_and_exact_normalized_rankings_differ",
        "pred_c_rms_tangent_tracks_exact_finite_response",
        "pred_d_exact_reader_nominations_are_fold_stable",
        "pred_e_a_shared_exact_reader_is_nominated",
        "pred_f_known_oracle_sources_write_into_the_span"), map(bool, (A, B, C, D, E, F))))
    terminal = ("invalid" if not A else "normalized_reader_candidate" if D and E
                else "unstable_or_private_weight_readers")
    result = {
        "schema": "temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_result_v1",
        "candidate_id": CANDIDATE_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "geometry": geometry,
        "instrument": {"manual_native_max_abs_error": native_closure,
                       "orthogonality_max_abs_error": orthogonality_error,
                       "geometry_relative_max_error": geometry_error,
                       "train_held_disjoint": disjoint, "model_config": config},
        "raw_exact_top10_overlap": raw_exact_overlap,
        "tangent_exact_fidelity": tangent_fidelity,
        "exact_fold_stability": fold_stability,
        "shared_exact_top10_interfaces": shared_exact,
        "known_writer_percentiles": writer_known,
        "upstream_writers": writers,
        "downstream_readers": records,
        "scope": "normalization_aware_weight_geometry_not_causal_identification",
        "predictions": predictions,
        "terminal": terminal,
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "instrument", "raw_exact_top10_overlap", "tangent_exact_fidelity",
        "exact_fold_stability", "shared_exact_top10_interfaces", "known_writer_percentiles",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
