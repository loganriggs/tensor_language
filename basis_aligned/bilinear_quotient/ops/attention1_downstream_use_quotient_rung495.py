#!/usr/bin/env python3
"""RUNG495 -- exact below-head attention1 pieces grouped by downstream use.

This is a discovery screen.  It constructs the seven finite A/B/V Möbius terms
inside every attention1 head, carries them exactly through MLP1's quadratic
polarization, and compares their 62-circuit response signatures.  Native head
identity is provenance, not the assumed circuit basis.
"""

# BQGATE: EXPERIMENT
# pred_a exact live factor, polarization, branch, mask, call, and gradient instrument
# pred_b one cross-head downstream-use pair survives fixed discovery halves and controls
# pred_c the frozen pair predicts held-out documents and circuit tags
# pred_d at least one native head contains two downstream-distinct material pieces
# pred_e selected pair is only a candidate for a separately registered physical interchange

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mlp0_TI_site_graded_merge_intervention_rung493 as branch_parent
import mlp0_attention1_finite_path_factorial_rung484 as factor_parent
import mlp1_finite_secant_factor_interchange_rung487 as secant_parent
import mlp0_branch_circuit_response_rung481 as circuit_parent


PREREG = POLY / "ATTENTION1_DOWNSTREAM_USE_QUOTIENT_RUNG495_PREREGISTRATION.md"
R494_RESULT = ROOT / "equality_query_scaled_single_index_causal_rung494_results.json"
R493_SOURCE = ROOT / "ops/mlp0_TI_site_graded_merge_intervention_rung493.py"
R493_RESULT = ROOT / "mlp0_TI_site_graded_merge_intervention_rung493_results.json"
R484_SOURCE = ROOT / "ops/mlp0_attention1_finite_path_factorial_rung484.py"
R487_SOURCE = ROOT / "ops/mlp1_finite_secant_factor_interchange_rung487.py"
R481_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
OUT = ROOT / "attention1_downstream_use_quotient_rung495_results.json"
BUNDLE = ROOT / "attention1_downstream_use_quotient_rung495_bundle.pt"
HASHES = {
    PREREG: "5ad987c869cda249520e7b811555c2cfb5eea2ca1fadd543b877a9ad7620d69c",
    R494_RESULT: "8b384663af5fe6b9291c4180f1ea6147a40835cc5e64a172a72f73087ddad261",
    R493_SOURCE: "4f77c3898d8237373a7a35439dabc590882eda47490aaed33a797ddec2cfe08b",
    R493_RESULT: "1131a0dc61f94ca2dba92073eed1a21c2f46a46ac18004be146e15a78161339d",
    R484_SOURCE: "42f66fba01361c976660554197fef7aa66cb20d80eb5b6351b01a1f6e3bf9d54",
    R487_SOURCE: "0339a0e24189eb0eff4ef73940cbe617c3ff6ede75e1e18c4477875893343ffc",
    R481_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
}
BRANCHES = ("T", "C", "I", "S")
FACTORS = ("QK1", "QK2", "OV")
FACTOR_MASKS = tuple(range(1, 8))
HEADS = 9
HEAD_DIM = 128
D = 1152
TOKENS = 256
BATCH = 4
DISCOVERY_RANGE = (0, 500)
DISCOVERY_SPLIT = 250
VALIDATION_RANGE = (500, 1000)
VALIDATION_SPLIT = 750
MASK_TYPES = ("member", "slice_control")
POSITION_SHIFTS = tuple(range(1, 17))
CIRCUIT_PERMUTATION_SEEDS = tuple(range(20260902950, 20260902966))
EXPECTED_DISCOVERY_FORWARDS = (DISCOVERY_RANGE[1] // BATCH) * (2 + len(BRANCHES))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_name(mask: int) -> str:
    return "x".join(name for index, name in enumerate(FACTORS) if mask & (1 << index))


PIECE_NAMES = tuple(
    f"h{head}.{factor_name(mask)}"
    for head in range(HEADS) for mask in FACTOR_MASKS
)
CROSS_HEAD_PAIRS = tuple(
    (left, right)
    for left, right in itertools.combinations(range(len(PIECE_NAMES)), 2)
    if left // len(FACTOR_MASKS) != right // len(FACTOR_MASKS)
)
WITHIN_HEAD_PAIRS = tuple(
    (left, right)
    for left, right in itertools.combinations(range(len(PIECE_NAMES)), 2)
    if left // len(FACTOR_MASKS) == right // len(FACTOR_MASKS)
)


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    r494 = json.loads(R494_RESULT.read_text())
    if r494.get("rung") != 494 \
            or r494.get("pred_a_exact_live_scaled_intervention") is not True \
            or r494.get("pred_b_half_strength_causal_interpolation") is not False \
            or r494.get("pred_c_one_and_half_strength_causal_transfer") is not True \
            or r494.get("pred_d_document_half_stability") is not False \
            or r494.get("strong_null") is not True \
            or r494.get("next_step") != "attention1_exact_QK1_QK2_OV_downstream_use_decomposition":
        raise RuntimeError("rung494 did not license the below-head attention1 route")
    rows, fit_rows, branch_metadata = branch_parent.validate_inputs()
    circuit_rows, circuit_masks, discovery_tags, validation_tags, _, circuit_metadata = \
        circuit_parent.validate_inputs()
    if not torch.equal(rows, circuit_rows):
        raise RuntimeError("branch and 62-circuit row authorities differ")
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("frozen 32/30 circuit split changed")
    return rows, fit_rows, circuit_masks, discovery_tags, validation_tags, {
        "branch": branch_metadata,
        "circuits": circuit_metadata,
        "piece_names": list(PIECE_NAMES),
        "cross_head_pairs": len(CROSS_HEAD_PAIRS),
        "within_head_pairs": len(WITHIN_HEAD_PAIRS),
    }


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def _per_head_attention_writes(attention, parts):
    """Return [batch, query, head, residual] before summing native heads."""
    score_a, score_b, value = parts
    length = score_a.shape[-1]
    pattern = score_a * score_b
    causal = torch.tril(torch.ones(
        length, length, dtype=torch.bool, device=pattern.device))
    pattern = pattern.masked_fill(~causal, 0)
    head_values = torch.einsum("bhqk,bkhu->bhqu", pattern, value)
    output_weight = attention.c_proj.weight.to(
        device=head_values.device, dtype=head_values.dtype).reshape(D, HEADS, HEAD_DIM)
    return torch.einsum("bhqu,ohu->bqho", head_values, output_weight)


def exact_factor_pieces(attention, normal_parts, absent_parts):
    """Return 63 finite pieces and exact endpoint/closure diagnostics.

    Arms choose each of QK1/QK2/OV from absent (bit 0) or normal (bit 1).
    The output piece axis is ordered head-major according to PIECE_NAMES.
    """
    arms = []
    for mask in range(8):
        parts = tuple(
            normal_parts[index] if mask & (1 << index) else absent_parts[index]
            for index in range(3)
        )
        arms.append(_per_head_attention_writes(attention, parts))
    arms = torch.stack(arms, dim=0)  # [arm,batch,query,head,residual]
    effects = torch.zeros_like(arms)
    for mask in range(8):
        for child in range(8):
            if child & ~mask:
                continue
            sign = -1.0 if ((mask.bit_count() - child.bit_count()) % 2) else 1.0
            effects[mask] += sign * arms[child]
    pieces = effects[1:].permute(1, 2, 3, 0, 4).contiguous().reshape(
        arms.shape[1], arms.shape[2], HEADS * len(FACTOR_MASKS), D)
    normal_write = arms[7].sum(2)
    absent_write = arms[0].sum(2)
    reconstructed_delta = pieces.sum(2)
    return pieces, {
        "normal_write": normal_write,
        "absent_write": absent_write,
        "factor_delta": normal_write - absent_write,
        "reconstructed_delta": reconstructed_delta,
        "arms": arms,
    }


def exact_mlp1_piece_responses(mlp1, pieces, direct_absent, normal_attention,
                               absent_attention):
    """Carry every attention piece through the exact quadratic MLP1 secant."""
    midpoint = direct_absent.float() + (
        normal_attention.float() + absent_attention.float()) / 2
    responses = secant_parent._secant(
        mlp1, pieces.float(), midpoint.unsqueeze(2))
    complete = secant_parent._secant(
        mlp1, pieces.float().sum(2), midpoint)
    return responses, complete


def _relative_squared(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    return float((left - right).square().sum() / right.square().sum().clamp_min(1e-30))


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(PIECE_NAMES) == 63
        assert len(CROSS_HEAD_PAIRS) == 1764
        assert len(WITHIN_HEAD_PAIRS) == 189
        print(json.dumps({
            "status": "implementation_in_progress_dry_run",
            "rung": 495,
            "model_loaded": False,
            "downstream_use_outcomes_opened": False,
            "validation_documents_or_tags_opened": False,
            "pieces_per_branch": len(PIECE_NAMES),
            "branches": list(BRANCHES),
            "discovery_forward_price": EXPECTED_DISCOVERY_FORWARDS,
            "implemented_now": [
                "frozen_authority_validation",
                "exact_per_head_factor_arms",
                "seven_term_mobius_decomposition",
                "exact_mlp1_piece_polarization",
            ],
            "remaining_before_enqueue": [
                "branch_absent_gradient_capture",
                "62_circuit_signature_accumulation",
                "permutation_and_position_controls",
                "registered_scorer_and_receipt",
            ],
        }, indent=2, sort_keys=True))
        return
    raise RuntimeError("rung495 implementation is not yet complete; do not enqueue")


if __name__ == "__main__":
    main()
