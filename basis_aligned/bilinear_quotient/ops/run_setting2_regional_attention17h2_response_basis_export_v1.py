#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_parent_rank8_authority pred_b_finite_vjps pred_c_orthonormal_export
"""Materialize the frozen rank-eight response and source-PCA projectors."""
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
import run_setting2_regional_attention17h2_consumer_response_basis_v1 as parent_runner
import run_setting2_regional_attention17h2_head_source_beam_v1 as head
import run_setting2_regional_attention17h2_port_preservation_search_v1 as search
import run_setting2_regional_attention17h2_port_source_fold_v1 as port
from regional_cue_row_check_v1 import validate


RUNNER = Path(__file__).resolve()
PREREG = P / "SETTING2_REGIONAL_ATTENTION17H2_RESPONSE_BASIS_EXPORT_V1_PREREGISTRATION.md"
ROWS = P / "ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_RESPONSE_BASIS_EXPORT_V1_BINDING.json"
PARENT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_consumer_response_basis_v1_result.json"
ARTIFACT = ROOT / "basis_aligned/bilinear_quotient/circuits/artifacts/setting2_regional_attention17h2_response_basis_rank8_v1.pt"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_response_basis_export_v1_result.json"
FROZEN = parent_runner.FROZEN
RANK = 8
PRICE = {"physical_model_executions": 12, "full_model_sequences": 96, "vjp_calls": 72, "fresh_panel_sequences": 0, "fits": 0, "parameter_updates": 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "rows": ROWS,
        "row_check": P / "regional_cue_row_check_v1.py",
        "parent_result": PARENT,
        "parent_runner": Path(parent_runner.__file__).resolve(),
        "head_runner": Path(head.__file__).resolve(),
        "search_runner": Path(search.__file__).resolve(),
        "port_runner": Path(port.__file__).resolve(),
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    parent = json.loads(PARENT.read_text())
    rank8 = next(report for report in parent["response_rank_frontier"] if report["rank"] == RANK)
    authority = rank8["passes_response"] and rank8["passes_causality"] and rank8["passes_controls"]
    if parent["terminal"] != "valid_consumer_response_basis_null" or not authority:
        raise ValueError("parent rank-eight authority changed")
    rows = json.loads(ROWS.read_text())["rows"]
    checks = validate(rows)
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(len(row["ids"]), []).append(index)
    batches = sum(math.ceil(len(indices) / 8) for indices in buckets.values())
    if len(rows) != 48 or 2 * batches != PRICE["physical_model_executions"] or 2 * batches * len(port.fold.READOUTS) != PRICE["vjp_calls"]:
        raise ValueError("row or execution price changed")
    return binding, parent, rank8, rows, checks, buckets


def plan():
    _, _, rank8, rows, checks, _ = load_bound()
    return {"schema": "setting2_regional_attention17h2_response_basis_export_v1_plan", "model_loaded": False, "gpu_accessed": False, "queue_touched": False, "rows": len(rows), "rank": RANK, "parent_rank8_score": rank8["minimax_normalized_gate_score"], "price": PRICE, "row_checks": checks, "binding_sha256": sha(BINDING)}


def atomic_binary(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o664)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


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
    _, _, rank8, rows, checks, buckets = load_bound()
    device = next(model.parameters()).device
    lambdas = torch.stack([block.lambdas.detach().double().cpu() for block in model.transformer.h])
    embed_c, write_c = port.fold.helper.coefficients(lambdas)
    gammas = {layer: float(torch.prod(lambdas[layer + 1:18, 0])) for layer in range(9, 17)}
    output_weight = model.transformer.h[17].attn.c_proj.weight[:, 2 * 128:3 * 128]
    pairs = torch.tensor([[(row["uk_id"], row["us_id"])] + [pair for _, pair in port.fold.READOUTS[1:]] for row in rows], device=device, dtype=torch.long)
    response_gram = torch.zeros(1152, 1152, dtype=torch.float64)
    source_gram = torch.zeros_like(response_gram)
    vjp_calls = 0
    executions = 0
    gradient_min_norm = math.inf
    gradient_max_norm = 0.0

    for _, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 8):
            row_ids = indices[offset:offset + 8]
            tokens = torch.tensor([rows[index]["ids"] for index in row_ids], device=device)
            native = head.split_trace(model, tokens, False, embed_c, write_c)
            edited = head.split_trace(model, tokens, True, embed_c, write_c)
            executions += 2
            components = torch.stack([gammas[head.propagation_layer(name)] * (edited["writes"][name] - native["writes"][name]).double() for name in head.NAMES])
            addition = components[list(FROZEN)].sum(0)
            flat = addition.cpu().reshape(-1, 1152)
            source_gram += flat.T @ flat
            zero_corner = port.corner_write(native["score1"], native["score2"], native["value"], output_weight)
            for arm, endpoint in (("installation", native["residual17"]), ("removal", edited["residual17"])):
                with torch.enable_grad():
                    state = endpoint.detach().clone().requires_grad_(True)
                    score2, value = port.edited_ports(model, state, native["first"].detach())
                    delta = port.corner_write(native["score1"].detach(), score2, value, output_weight) - zero_corner.detach()
                    pre = native["pre"].detach() + delta if arm == "installation" else edited["pre"].detach() - delta
                    values = search.sparse_suffix_readouts(model, pre, torch.tensor(row_ids, device=device), pairs)
                    for reader_index in range(len(port.fold.READOUTS)):
                        gradient = torch.autograd.grad(values[:, reader_index].sum(), state, retain_graph=reader_index + 1 < len(port.fold.READOUTS))[0]
                        norms = gradient.flatten(1).norm(dim=1)
                        gradient_min_norm = min(gradient_min_norm, float(norms.min()))
                        gradient_max_norm = max(gradient_max_norm, float(norms.max()))
                        normalized = (gradient / norms.clamp_min(1e-30)[:, None, None]).double().cpu().reshape(-1, 1152)
                        response_gram += normalized.T @ normalized
                        vjp_calls += 1

    response_evals, response_vecs = torch.linalg.eigh(response_gram)
    source_evals, source_vecs = torch.linalg.eigh(source_gram)
    response_basis = response_vecs[:, -RANK:].flip(1).contiguous()
    source_basis = source_vecs[:, -RANK:].flip(1).contiguous()
    eye = torch.eye(RANK, dtype=torch.float64)
    response_orth = float((response_basis.T @ response_basis - eye).abs().max())
    source_orth = float((source_basis.T @ source_basis - eye).abs().max())
    predicates = {
        "pred_a_parent_rank8_authority": bool(rank8["passes_response"] and rank8["passes_causality"] and rank8["passes_controls"]),
        "pred_b_finite_vjps": bool(executions == PRICE["physical_model_executions"] and vjp_calls == PRICE["vjp_calls"] and gradient_min_norm > 0 and math.isfinite(gradient_max_norm) and torch.isfinite(response_gram).all()),
        "pred_c_orthonormal_export": bool(response_orth <= 2e-6 and source_orth <= 2e-6),
    }
    terminal = "consumer_response_basis_rank8_exported" if all(predicates.values()) else "invalid"
    artifact = {
        "schema": "setting2_regional_attention17h2_response_basis_rank8_v1",
        "rank": RANK,
        "response_basis": response_basis,
        "source_pca_basis": source_basis,
        "frozen_support_indices": FROZEN,
        "frozen_support_names": [head.NAMES[index] for index in FROZEN],
        "discovery_rows_sha256": sha(ROWS),
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    }
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_binary(ARTIFACT, buffer.getvalue())
    result = {
        "schema": "setting2_regional_attention17h2_response_basis_export_v1_result",
        "terminal": terminal,
        "predictions": predicates,
        "rank": RANK,
        "gradient_min_norm": gradient_min_norm,
        "gradient_max_norm": gradient_max_norm,
        "response_basis_orthogonality_max_abs": response_orth,
        "source_pca_basis_orthogonality_max_abs": source_orth,
        "response_top_eigenvalues": response_evals[-RANK:].flip(0).tolist(),
        "source_top_eigenvalues": source_evals[-RANK:].flip(0).tolist(),
        "artifact_relative": str(ARTIFACT.relative_to(ROOT)),
        "artifact_sha256": sha(ARTIFACT),
        "price": PRICE,
        "row_checks": checks,
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Materialized discovery-derived rank-eight response and source-PCA projectors; no fresh-panel evaluation.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
