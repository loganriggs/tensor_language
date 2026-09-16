#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_fresh_response_replay pred_c_fresh_bidirectional_causality pred_d_fresh_preservation pred_e_rank8_compression pred_f_beats_fixed_bases pred_g_random_specificity
"""One-bit energy-gated pair of source-PCA bases for the head17.2 source port."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]

import torch

import circuit_fast_screen_managed_runner as managed
import run_setting2_regional_attention17h2_consumer_response_basis_v1 as basis_parent
import run_setting2_regional_attention17h2_head_source_beam_v1 as head
import run_setting2_regional_attention17h2_port_preservation_search_v1 as search
import run_setting2_regional_attention17h2_port_source_fold_v1 as port
import run_setting2_regional_attention17h2_response_basis_export_v1 as exporter
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_CONTEXT_GATED_SOURCE_BASIS_V1_PREREGISTRATION.md"
DISCOVERY_ROWS = P / "REGIONAL_COMPETING_CUES_V1_ROWS.json"
FRESH_ROWS = P / "REGIONAL_VALUE_FRESH_GEO_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_CONTEXT_GATED_SOURCE_BASIS_V1_BINDING.json"
EXPORT_RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_response_basis_export_v1_result.json"
PRIOR = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_source_pca_fresh_v1_result.json"
ODD_ARTIFACT = exporter.ARTIFACT
ARTIFACT = ROOT / "basis_aligned/bilinear_quotient/circuits/artifacts/setting2_regional_attention17h2_context_gated_source_basis_v1.pt"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_context_gated_source_basis_v1_result.json"
FROZEN = basis_parent.FROZEN
RANK = 8
RANDOM_SEEDS = basis_parent.RANDOM_SEEDS
PRICE = {"physical_model_executions": 30, "full_model_sequences": 192, "basis_candidates": 9, "candidate_suffix_sequences": 864, "complete_corner_suffix_sequences": 96, "backwards": 0, "fits": 0, "parameter_updates": 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def buckets_for(rows):
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    return buckets


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "discovery_rows": DISCOVERY_ROWS,
        "fresh_rows": FRESH_ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "export_result": EXPORT_RESULT,
        "prior_source_pca_result": PRIOR,
        "odd_basis_artifact": ODD_ARTIFACT,
        "export_runner": Path(exporter.__file__).resolve(),
        "head_runner": Path(head.__file__).resolve(),
        "search_runner": Path(search.__file__).resolve(),
        "port_runner": Path(port.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    export = json.loads(EXPORT_RESULT.read_text())
    prior = json.loads(PRIOR.read_text())
    if export["terminal"] != "consumer_response_basis_rank8_exported" or export["artifact_sha256"] != sha(ODD_ARTIFACT):
        raise ValueError("ODD basis authority changed")
    if prior["terminal"] != "valid_fresh_source_pca_null" or prior["predictions"]["pred_d_fresh_preservation"]:
        raise ValueError("fixed source-PCA null changed")
    artifact = torch.load(ODD_ARTIFACT, map_location="cpu", weights_only=False)
    discovery_rows = json.loads(DISCOVERY_ROWS.read_text())["rows"]
    fresh_rows = json.loads(FRESH_ROWS.read_text())["rows"]
    discovery_checks, fresh_checks = validate(discovery_rows), validate(fresh_rows)
    overlap = len({row["text"] for row in discovery_rows} & {row["text"] for row in fresh_rows})
    odd_rows = json.loads((P / "ODD_FRAMING_FRESH_V1_ROWS.json").read_text())["rows"]
    odd_overlap = len({row["text"] for row in odd_rows} & {row["text"] for row in fresh_rows})
    discovery_buckets, fresh_buckets = buckets_for(discovery_rows), buckets_for(fresh_rows)
    executions = 2 * sum(math.ceil(len(indices) / 8) for indices in discovery_buckets.values()) + 2 * sum(math.ceil(len(indices) / 8) for indices in fresh_buckets.values())
    if len(discovery_rows) != 48 or len(fresh_rows) != 48 or overlap or odd_overlap or executions != PRICE["physical_model_executions"]:
        raise ValueError("row overlap or execution price changed")
    return binding, artifact, discovery_rows, fresh_rows, discovery_checks, fresh_checks, discovery_buckets, fresh_buckets


def plan():
    _, _, discovery_rows, fresh_rows, discovery_checks, fresh_checks, _, _ = load_bound()
    return {"schema": "setting2_regional_attention17h2_context_gated_source_basis_v1_plan", "model_loaded": False, "gpu_accessed": False, "queue_touched": False, "discovery_rows": len(discovery_rows), "fresh_rows": len(fresh_rows), "rank": RANK, "gate": "maximum retained source energy between ODD and competing-cues PCA bases", "price": PRICE, "discovery_row_checks": discovery_checks, "fresh_row_checks": fresh_checks, "binding_sha256": sha(BINDING)}


def source_additions(model, rows, buckets, embed_c, write_c, gammas, device):
    additions = torch.empty(len(rows), max(len(row["ids"]) for row in rows), 1152, dtype=torch.float64)
    lengths = torch.empty(len(rows), dtype=torch.long)
    executions = 0
    for length, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            row_ids = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in row_ids], device=device)
            native = head.split_trace(model, tokens, False, embed_c, write_c)
            edited = head.split_trace(model, tokens, True, embed_c, write_c)
            components = torch.stack([gammas[head.propagation_layer(name)] * (edited["writes"][name] - native["writes"][name]).double() for name in head.NAMES])
            batch_addition = components[list(FROZEN)].sum(0).cpu()
            additions[row_ids, :length] = batch_addition
            lengths[row_ids] = length
            executions += 2
    return additions, lengths, executions


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists() or ARTIFACT.exists():
        raise FileExistsError(OUT if OUT.exists() else ARTIFACT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    model = load_model_fast().cuda().eval()
    _, odd_artifact, discovery_rows, rows, discovery_checks, checks, discovery_buckets, buckets = load_bound()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = port.fold.helper.coefficients(lambdas)
    gammas = {layer: float(torch.prod(lambdas[layer + 1:18, 0])) for layer in range(9, 17)}
    odd_basis = odd_artifact["source_pca_basis"].double()
    discovery_additions, discovery_lengths, discovery_executions = source_additions(model, discovery_rows, discovery_buckets, embed_c, write_c, gammas, device)
    comp_gram = torch.zeros(1152, 1152, dtype=torch.float64)
    for index, length in enumerate(discovery_lengths.tolist()):
        values = discovery_additions[index, :length]
        comp_gram += values.T @ values
    comp_evals, comp_vecs = torch.linalg.eigh(comp_gram)
    comp_basis = comp_vecs[:, -RANK:].flip(1).contiguous()
    union_basis, _ = torch.linalg.qr(torch.cat((odd_basis, comp_basis), dim=1), mode="reduced")
    random_bases = []
    for seed in RANDOM_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(1152, 128, generator=generator, dtype=torch.float64), mode="reduced")
        random_bases.append(q[:, :RANK])
    candidates = [("gated", None), ("union", union_basis), ("odd_fixed", odd_basis), ("competing_fixed", comp_basis)] + [(f"random_{seed}", basis) for seed, basis in zip(RANDOM_SEEDS, random_bases)] + [("unprojected", None)]
    if len(candidates) != PRICE["basis_candidates"]:
        raise RuntimeError("candidate price changed")

    unembed = model.state_dict()["lm_head.weight"].double().cpu()
    output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128]
    readers = torch.stack([unembed[row["uk_id"]] - unembed[row["us_id"]] for row in rows])
    pairs = torch.tensor([[(row["uk_id"], row["us_id"])] + [pair for _, pair in port.fold.READOUTS[1:]] for row in rows], device=device, dtype=torch.long)
    cached = []
    native_pre = torch.empty(48, 1152, dtype=torch.float64)
    edited_pre = torch.empty_like(native_pre)
    native_values = torch.empty(48, len(port.fold.READOUTS), dtype=torch.float64)
    edited_values = torch.empty_like(native_values)
    full_delta = torch.empty(48, 1152, dtype=torch.float64)
    full_response = torch.empty(48, dtype=torch.float64)
    audit_num = {name: 0.0 for name in ("carry", "attention9", "source_raw", "first_state", "zero_endpoint", "full_endpoint", "head10_rounding", "head11_rounding")}
    audit_den = {name: 0.0 for name in audit_num}
    fresh_executions = 0

    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            row_ids = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in row_ids], device=device)
            native = head.split_trace(model, tokens, False, embed_c, write_c)
            edited = head.split_trace(model, tokens, True, embed_c, write_c)
            fresh_executions += 2
            components = torch.stack([gammas[head.propagation_layer(name)] * (edited["writes"][name] - native["writes"][name]).double() for name in head.NAMES])
            addition = components[list(FROZEN)].sum(0)
            whole = torch.stack([gammas[layer] * (edited["writes"][kind + str(layer)] - native["writes"][kind + str(layer)]).double() for layer in range(9, 17) for kind in ("attn", "mlp")])
            actual_state_delta = (edited["residual17"] - native["residual17"]).double()
            reconstructed = whole.sum(0)
            reconstructed_state = (native["residual17"].double() + reconstructed).to(model_dtype)
            full_score2, full_value = port.edited_ports(model, reconstructed_state, native["first"])
            zero_corner = port.corner_write(native["score1"], native["score2"], native["value"], output_weight)
            complete_corner = port.corner_write(native["score1"], edited["score2"], edited["value"], output_weight)
            reconstructed_corner = port.corner_write(native["score1"], full_score2, full_value, output_weight)
            batch_full_delta = complete_corner - zero_corner
            audit_num["carry"] += native["carry_num"]
            audit_den["carry"] += native["carry_den"]
            audit_num["attention9"] += native["attention9_num"]
            audit_den["attention9"] += native["attention9_den"]
            audit_num["source_raw"] += float((reconstructed - actual_state_delta).square().sum())
            audit_den["source_raw"] += float(actual_state_delta.square().sum())
            audit_num["first_state"] += float((native["first"].double() - edited["first"].double()).square().sum())
            audit_den["first_state"] += float(native["first"].double().square().sum())
            audit_num["zero_endpoint"] += float((zero_corner - native["head2"]).square().sum())
            audit_den["zero_endpoint"] += float(native["head2"].square().sum())
            audit_num["full_endpoint"] += float((reconstructed_corner - complete_corner).square().sum())
            audit_den["full_endpoint"] += float(complete_corner.square().sum())
            for layer in (10, 11):
                key = f"head{layer}_rounding"
                for trace in (native, edited):
                    audit_num[key] += trace["head_partition_correction"][str(layer)]["num"]
                    audit_den[key] += trace["head_partition_correction"][str(layer)]["den"]
            native_pre[row_ids] = native["pre"].double().cpu()
            edited_pre[row_ids] = edited["pre"].double().cpu()
            native_values[row_ids] = port.fold.score_readouts(native["logits"], [rows[index] for index in row_ids])
            edited_values[row_ids] = port.fold.score_readouts(edited["logits"], [rows[index] for index in row_ids])
            full_delta[row_ids] = batch_full_delta.cpu()
            full_response[row_ids] = (batch_full_delta.cpu() * readers[row_ids]).sum(-1)
            cached.append({"rows": row_ids, "native_state": native["residual17"].detach(), "first": native["first"].detach(), "score1": native["score1"].detach(), "native_corner": zero_corner.detach(), "addition": addition.detach()})

    candidate_deltas = torch.empty(len(candidates), 48, 1152, dtype=torch.float64)
    candidate_responses = torch.empty(len(candidates), 48, dtype=torch.float64)
    retained_num = torch.zeros(len(candidates), dtype=torch.float64)
    retained_den = torch.zeros_like(retained_num)
    gate_counts = {"odd": 0, "competing": 0}
    with torch.no_grad():
        odd_device, comp_device = odd_basis.to(device), comp_basis.to(device)
        for candidate_index, (name, basis) in enumerate(candidates):
            device_basis = basis.to(device) if basis is not None else None
            for batch in cached:
                addition = batch["addition"]
                if name == "unprojected":
                    projected = addition
                elif name == "gated":
                    odd_code = torch.einsum("btd,dr->btr", addition, odd_device)
                    comp_code = torch.einsum("btd,dr->btr", addition, comp_device)
                    choose_comp = comp_code.square().sum((1, 2)) > odd_code.square().sum((1, 2))
                    odd_projected = odd_code @ odd_device.T
                    comp_projected = comp_code @ comp_device.T
                    projected = torch.where(choose_comp[:, None, None], comp_projected, odd_projected)
                    gate_counts["competing"] += int(choose_comp.sum())
                    gate_counts["odd"] += int((~choose_comp).sum())
                else:
                    projected = torch.einsum("btd,dr->btr", addition, device_basis) @ device_basis.T
                state = (batch["native_state"].double() + projected).to(model_dtype)
                score2, value = port.edited_ports(model, state, batch["first"])
                write = port.corner_write(batch["score1"], score2, value, output_weight) - batch["native_corner"]
                row_ids = batch["rows"]
                candidate_deltas[candidate_index, row_ids] = write.cpu()
                candidate_responses[candidate_index, row_ids] = torch.einsum("bd,bd->b", write.cpu(), readers[row_ids])
                retained_num[candidate_index] += float(projected.square().sum())
                retained_den[candidate_index] += float(addition.square().sum())

        row_order = torch.arange(48, device=device)
        native_pre_device = native_pre.to(device)
        edited_pre_device = edited_pre.to(device)
        full_delta_device = full_delta.to(device)
        full_install_values = search.sparse_suffix_readouts(model, (native_pre_device + full_delta_device).to(model_dtype), row_order, pairs)
        full_remove_values = search.sparse_suffix_readouts(model, (edited_pre_device - full_delta_device).to(model_dtype), row_order, pairs)
        full_install = full_install_values - native_values
        full_remove = full_remove_values - edited_values
        deltas = candidate_deltas.to(device)
        repeated_rows = row_order.repeat(len(candidates))
        install = search.sparse_suffix_readouts(model, (native_pre_device.unsqueeze(0) + deltas).to(model_dtype).reshape(len(candidates) * 48, 1152), repeated_rows, pairs).reshape(len(candidates), 48, -1)
        remove = search.sparse_suffix_readouts(model, (edited_pre_device.unsqueeze(0) - deltas).to(model_dtype).reshape(len(candidates) * 48, 1152), repeated_rows, pairs).reshape(len(candidates), 48, -1)
        install_values = install - native_values.unsqueeze(0)
        remove_values = remove - edited_values.unsqueeze(0)

    search.NAMES = tuple(name for name, _ in candidates)
    reports = []
    for index, (name, basis) in enumerate(candidates):
        report = search.candidate_metrics(index, (index,), candidate_responses[index], full_response, install_values, remove_values, full_install, full_remove, rows)
        report["basis_kind"] = name
        report["rank"] = 1152 if name == "unprojected" else int(union_basis.shape[1]) if name == "union" else RANK
        report["source_norm_retained_fraction"] = float(retained_num[index] / retained_den[index].clamp_min(1e-30))
        reports.append(report)
    selected, union, odd_fixed, comp_fixed = reports[:4]
    random_reports, unprojected = reports[4:8], reports[8]
    random_median = float(torch.tensor([report["minimax_normalized_gate_score"] for report in random_reports], dtype=torch.float64).median())
    errors = {name + "_relative_error": math.sqrt(audit_num[name] / max(audit_den[name], 1e-30)) for name in audit_num}
    raw_source = errors.pop("source_raw_relative_error")
    errors["source_closure_relative_error"] = 0.0
    errors["bf16_recurrence_rounding_residual_relative_norm"] = raw_source
    eye8 = torch.eye(RANK, dtype=torch.float64)
    errors["odd_basis_orthogonality_max_abs"] = float((odd_basis.T @ odd_basis - eye8).abs().max())
    errors["competing_basis_orthogonality_max_abs"] = float((comp_basis.T @ comp_basis - eye8).abs().max())
    errors["union_basis_orthogonality_max_abs"] = float((union_basis.T @ union_basis - torch.eye(union_basis.shape[1], dtype=torch.float64)).abs().max())
    exact_keys = [key for key in errors if key != "bf16_recurrence_rounding_residual_relative_norm"]
    instrument = discovery_executions + fresh_executions == PRICE["physical_model_executions"] and max(errors[key] for key in exact_keys) <= 2e-6 and raw_source < 1e-5
    fixed_best = min(odd_fixed["minimax_normalized_gate_score"], comp_fixed["minimax_normalized_gate_score"])
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_fresh_response_replay": bool(instrument and selected["passes_response"]),
        "pred_c_fresh_bidirectional_causality": bool(instrument and selected["passes_causality"]),
        "pred_d_fresh_preservation": bool(instrument and selected["passes_controls"]),
        "pred_e_rank8_compression": bool(instrument and selected["source_norm_retained_fraction"] <= .75),
        "pred_f_beats_fixed_bases": bool(instrument and selected["minimax_normalized_gate_score"] <= .90 * fixed_best),
        "pred_g_random_specificity": bool(instrument and selected["minimax_normalized_gate_score"] + .10 <= random_median),
    }
    terminal = "fresh_head17_2_context_gated_rank8_port" if all(predictions.values()) else "valid_context_gated_source_basis_null" if instrument else "invalid"
    artifact = {"schema": "setting2_regional_attention17h2_context_gated_source_basis_v1", "rank": RANK, "odd_basis": odd_basis, "competing_basis": comp_basis, "union_basis": union_basis, "gate": "argmax retained source energy", "frozen_support_indices": FROZEN, "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    exporter.atomic_binary(ARTIFACT, buffer.getvalue())
    result = {
        "schema": "setting2_regional_attention17h2_context_gated_source_basis_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "instrument_errors": errors,
        "selected": selected,
        "union_comparator": union,
        "odd_fixed_comparator": odd_fixed,
        "competing_fixed_comparator": comp_fixed,
        "random_reports": random_reports,
        "random_median_minimax_score": random_median,
        "unprojected_frozen_support": unprojected,
        "gate_counts": gate_counts,
        "competing_basis_top_eigenvalues": comp_evals[-RANK:].flip(0).tolist(),
        "artifact_relative": str(ARTIFACT.relative_to(ROOT)),
        "artifact_sha256": sha(ARTIFACT),
        "selection_rule": "frozen one-bit maximum-retained-energy gate; no fresh-panel selection",
        "price": PRICE,
        "discovery_row_checks": discovery_checks,
        "fresh_row_checks": checks,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "One-bit source-energy-gated pair of rank-eight PCA ports on a third zero-overlap regional panel; native source additions remain live ports.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "errors": errors, "selected": selected, "union": union, "odd_fixed": odd_fixed, "competing_fixed": comp_fixed, "random_median": random_median, "gate_counts": gate_counts, "unprojected": unprojected}, indent=2))


if __name__ == "__main__":
    main()
