#!/usr/bin/env python3
"""Exact dyadic-rank certificate for the layer-9 head-8 output map.

This prices only the independent-port trilinear attention contraction.  It does
not assert independence or reachability of native residual-derived fields.
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
OUTPUT = ROOT / "HEAD9H8_TRILINEAR_RANK_CERTIFICATE_2026-09-19_1224.json"
PRIME = 2_305_843_009_213_693_951  # 2**61 - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dyadic_integer_mod(bits: int) -> int:
    """Return (float32 value * 2**149) modulo PRIME exactly."""
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("non-finite float32 weight")
    if exponent == 0:
        integer = fraction
    else:
        integer = (0x800000 | fraction) * pow(2, exponent - 1, PRIME)
    return (sign * integer) % PRIME


def modular_determinant(matrix: np.ndarray) -> tuple[int, int]:
    """Return rank and determinant modulo PRIME using exact field arithmetic."""
    bits = matrix.astype(np.float32, copy=False).view(np.uint32)
    work = [[dyadic_integer_mod(int(value)) for value in row] for row in bits]
    size = len(work)
    determinant = 1
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column]), None
        )
        if pivot is None:
            return column, 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = (-determinant) % PRIME
        pivot_value = work[column][column]
        determinant = determinant * pivot_value % PRIME
        inverse = pow(pivot_value, PRIME - 2, PRIME)
        for row in range(column + 1, size):
            if not work[row][column]:
                continue
            quotient = work[row][column] * inverse % PRIME
            for inner in range(column, size):
                work[row][inner] = (
                    work[row][inner] - quotient * work[column][inner]
                ) % PRIME
    return size, determinant


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)
    started = time.perf_counter()
    state = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location="cpu")
    full_output = state["transformer.h.9.attn.c_proj.weight"]
    head_output = full_output[:, 8 * 128 : 9 * 128].contiguous()
    witness = head_output[:128].contiguous().numpy()
    exact_rank, determinant_mod_prime = modular_determinant(witness)
    singular_values = torch.linalg.svdvals(head_output.double())

    head_dim = 128
    sequence_length = 32
    final_position_cells = sequence_length
    all_causal_cells = sequence_length * (sequence_length + 1) // 2
    result = {
        "schema": "head9h8_trilinear_rank_certificate_v1",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "weight_key": "transformer.h.9.attn.c_proj.weight[:,1024:1152]",
        "output_map_shape": list(head_output.shape),
        "witness_rows": [0, 127],
        "common_exact_float32_scale": "2**149",
        "finite_field_prime": PRIME,
        "witness_rank_mod_prime": exact_rank,
        "witness_determinant_mod_prime": determinant_mod_prime,
        "exact_full_column_rank_over_reals": exact_rank == head_dim,
        "numerical_singular_value_min": float(singular_values[-1]),
        "numerical_singular_value_max": float(singular_values[0]),
        "numerical_condition_number": float(
            singular_values[0] / singular_values[-1]
        ),
        "independent_port_cp_rank": {
            "final_position_T32": final_position_cells * exact_rank,
            "all_causal_cells_T32": all_causal_cells * exact_rank,
            "formula": "number_of_live_(t,s)_cells * rank(O_head)",
        },
        "literal_native_trilinear_products_T32": {
            "final_position": final_position_cells * head_dim,
            "all_positions": all_causal_cells * head_dim,
        },
        "predictions": {
            "pred_a_head_output_map_has_exact_rank_128": exact_rank == head_dim,
            "pred_b_final_position_cp_rank_is_4096": (
                final_position_cells * exact_rank == 4096
            ),
            "pred_c_all_position_cp_rank_is_67584": (
                all_causal_cells * exact_rank == 67584
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "scope": (
            "Exact float32/dyadic rank witness for the frozen layer-9 head-8 "
            "output map and the induced independent-field trilinear tensor "
            "(p1,p2,v)->write. The CP-rank identity applies when p1, p2, and v "
            "are independent formal ports and counts rank-one trilinear terms. "
            "Native fields share residual/RMS sources, and deeper tensor-network "
            "executions may reuse intermediates; this is not a whole-path, runtime, "
            "causal-fidelity, extraction, selectivity, OOD, or composition proof."
        ),
        "source_sha256": sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
