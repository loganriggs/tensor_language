#!/usr/bin/env python3
"""Exact rank witnesses for block-10 readers of the head-9.8 write.

This is a weight-only boundary test.  It certifies ranks of W @ O over the
reals by reducing the exact, uniformly scaled float32 matrices modulo a prime.
It does not test activation reachability, approximation, or causal fidelity.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(
    "/workspace/.hf_home/hub/"
    "models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/"
    "snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/"
    "pytorch_model.bin"
)
OUTPUT = ROOT / "HEAD9H8_BLOCK10_CONSUMER_RANK_CERTIFICATE_2026-09-19_1528.json"
PRIME = 65_521


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaled_float32_mod(matrix: np.ndarray) -> np.ndarray:
    """Return (matrix * 2**149) mod PRIME, exactly for finite float32."""
    values = np.ascontiguousarray(matrix, dtype=np.float32)
    bits = values.view(np.uint32)
    signs = np.where(bits >> 31, -1, 1).astype(np.int64)
    exponents = ((bits >> 23) & 0xFF).astype(np.int64)
    fractions = (bits & 0x7FFFFF).astype(np.int64)
    if np.any(exponents == 0xFF):
        raise ValueError("non-finite float32 weight")
    powers = np.array([pow(2, exponent - 1, PRIME) for exponent in range(1, 255)])
    normal = exponents != 0
    integers = fractions.copy()
    integers[normal] = (
        (0x800000 + fractions[normal]) * powers[exponents[normal] - 1]
    ) % PRIME
    return (signs * integers) % PRIME


def rank_and_determinant_mod(matrix: np.ndarray) -> tuple[int, int]:
    """Exact rank and determinant of a square matrix over F_PRIME."""
    work = np.asarray(matrix, dtype=np.int64).copy() % PRIME
    size = work.shape[0]
    determinant = 1
    rank = 0
    for column in range(size):
        candidates = np.flatnonzero(work[rank:, column])
        if candidates.size == 0:
            continue
        pivot = rank + int(candidates[0])
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
            determinant = (-determinant) % PRIME
        pivot_value = int(work[rank, column])
        determinant = determinant * pivot_value % PRIME
        inverse = pow(pivot_value, PRIME - 2, PRIME)
        if rank + 1 < size:
            factors = work[rank + 1 :, column] * inverse % PRIME
            work[rank + 1 :] = (
                work[rank + 1 :] - factors[:, None] * work[rank]
            ) % PRIME
        rank += 1
        if rank == size:
            break
    return rank, determinant if rank == size else 0


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)
    started = time.perf_counter()
    state = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location="cpu")

    output_map = state["transformer.h.9.attn.c_proj.weight"][:, 8 * 128 : 9 * 128]
    output_mod = scaled_float32_mod(output_map.numpy())
    reader_keys = [
        "transformer.h.10.attn.c_q.weight",
        "transformer.h.10.attn.c_k.weight",
        "transformer.h.10.attn.c_q2.weight",
        "transformer.h.10.attn.c_k2.weight",
        "transformer.h.10.attn.c_v.weight",
        "transformer.h.10.mlp.Left.weight",
        "transformer.h.10.mlp.Right.weight",
    ]

    witnesses = {}
    for key in reader_keys:
        reader = state[key]
        witness_reader = reader[:128].contiguous()
        # Each source matrix is scaled by 2**149.  This modular product is
        # therefore exactly (witness_reader @ output_map) * 2**298.
        composed_mod = scaled_float32_mod(witness_reader.numpy()) @ output_mod
        composed_mod %= PRIME
        rank, determinant = rank_and_determinant_mod(composed_mod)
        numerical = witness_reader.double() @ output_map.double()
        singular_values = torch.linalg.svdvals(numerical)
        witnesses[key] = {
            "reader_shape": list(reader.shape),
            "witness_reader_rows": [0, 127],
            "composed_witness_shape": [128, 128],
            "rank_mod_prime": rank,
            "determinant_mod_prime": determinant,
            "certifies_full_column_rank_128_over_reals": rank == 128,
            "numerical_singular_value_min": float(singular_values[-1]),
            "numerical_condition_number": float(
                singular_values[0] / singular_values[-1]
            ),
        }

    all_full = all(
        record["certifies_full_column_rank_128_over_reals"]
        for record in witnesses.values()
    )
    result = {
        "schema": "head9h8_block10_consumer_rank_certificate_v1",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "producer": "transformer.h.9.attn.c_proj.weight[:,1024:1152]",
        "producer_shape": [1152, 128],
        "consumer_boundary": "first 128 rows of each native block-10 attention/MLP input map",
        "common_exact_float32_scale_per_factor": "2**149",
        "composed_exact_scale": "2**298",
        "finite_field_prime": PRIME,
        "witnesses": witnesses,
        "predictions": {
            "pred_a_all_five_attention_input_maps_retain_rank_128": all(
                witnesses[key]["rank_mod_prime"] == 128
                for key in reader_keys[:5]
            ),
            "pred_b_both_mlp_input_maps_retain_rank_128": all(
                witnesses[key]["rank_mod_prime"] == 128
                for key in reader_keys[5:]
            ),
            "pred_c_no_exact_linear_dimension_below_128_at_tested_boundary": all_full,
        },
        "consequence": (
            "The head-9.8 output image remains 128-dimensional under every tested "
            "immediate native block-10 attention/MLP reader. No exact linear "
            "consumer quotient below width 128 exists for these maps. Any smaller "
            "regional program must declare an approximate/task-specific quotient, "
            "exploit the attainable activation manifold, or move to another boundary."
        ),
        "scope": (
            "Exact weight-only rank witnesses for frozen float32 maps. First-128-row "
            "nonsingular minors certify the full composed maps have column rank 128. "
            "This does not establish that all directions are reachable or behaviorally "
            "used, nor does it establish OOD prediction, extraction, selective causal "
            "manipulation, composition, reuse, or a favorable literal program price."
        ),
        "runtime_seconds": time.perf_counter() - started,
        "source_sha256": sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
