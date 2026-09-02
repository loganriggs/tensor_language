#!/usr/bin/env python3
"""RUNG496 -- exact Q1/K1/Q2/K2/V allocations grouped by downstream use.

Implementation state: the exact five-factor/Shapley algebra and frozen parent
authority are complete.  Collection, controls, scorer, and receipt remain to be
implemented before this file is eligible for the managed runner.
"""

# BQGATE: EXPERIMENT
# pred_a exact five-factor arms, Mobius/Shapley closure, calls, masks, and liveness
# pred_b one cross-head query/query or key/key side confirms under three allocations
# pred_c the frozen side relation predicts held-out documents and circuit families
# pred_d the shared side is more specific than opposite-side or whole-head similarity
# pred_e candidate only; finite input-side interchange remains separately required

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import attention1_downstream_use_quotient_rung495 as parent


PREREG = POLY / "ATTENTION1_QUERY_KEY_DOWNSTREAM_SHAPLEY_RUNG496_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/attention1_downstream_use_quotient_rung495.py"
PARENT_RESULT = ROOT / "attention1_downstream_use_quotient_rung495b_results.json"
PARENT_BUNDLE = ROOT / "attention1_downstream_use_quotient_rung495b_bundle.pt"
OUT = ROOT / "attention1_query_key_downstream_shapley_rung496_results.json"
BUNDLE = ROOT / "attention1_query_key_downstream_shapley_rung496_bundle.pt"
HASHES = {
    PREREG: "603d427f83d603b43647eeed3b05147282f92482976cd2c537a7d0ae64ef15a7",
    PARENT_SOURCE: "5385ad0c540f9cbfef153bc6e545d7d5daaf0916129bb6e4e1a99dc355cae74d",
    PARENT_RESULT: "f06b6098380883a51260fa6646a6fa8d8e1ee7e1f9d784b428e95d8e94161eee",
    PARENT_BUNDLE: "295bf8048ff5b8ad5b29acf9d3f25e91c0b18ef1711aca6c5b4b845f6b447f9a",
}
FACTOR_NAMES = ("Q1", "K1", "Q2", "K2", "V")
HEADS = parent.HEADS
HEAD_DIM = parent.HEAD_DIM
D = parent.D
FULL_MASK = (1 << len(FACTOR_NAMES)) - 1
PIECE_NAMES = tuple(
    f"h{head}.{factor}" for head in range(HEADS) for factor in FACTOR_NAMES)
SIDE_INDICES = tuple(
    index for index, name in enumerate(PIECE_NAMES) if not name.endswith(".V"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear(value, weight):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def attention_factors(attention, state, first_value):
    """Return float32 Q1,K1,Q2,K2,V factors from the real attention state."""
    batch, length, width = state.shape
    if width != D or first_value.shape != (batch, length, HEADS, HEAD_DIM):
        raise RuntimeError("attention1 five-factor interface changed")
    state = state.float()
    first_value = first_value.float()
    q1 = _linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_DIM)
    k1 = _linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_DIM)
    q2 = _linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_DIM)
    k2 = _linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_DIM)
    raw_value = _linear(state, attention.c_v.weight).view(
        batch, length, HEADS, HEAD_DIM)
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value
    cos, sin = attention.rotary(q1)
    module = sys.modules[type(attention).__module__]
    q1 = module.apply_rotary_emb(F.rms_norm(q1, (HEAD_DIM,)), cos, sin)
    k1 = module.apply_rotary_emb(F.rms_norm(k1, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    return q1, k1, q2, k2, value


def _per_head_factor_writes(attention, factors):
    """Evaluate one five-factor arm as [batch,query,head,residual]."""
    q1, k1, q2, k2, value = (factor.float() for factor in factors)
    length = q1.shape[1]
    score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1) / HEAD_DIM
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    pattern = score1 * score2
    causal = torch.tril(torch.ones(
        length, length, dtype=torch.bool, device=pattern.device))
    pattern = pattern.masked_fill(~causal, 0)
    head_values = torch.einsum("bhqk,bkhu->bhqu", pattern, value)
    output_weight = attention.c_proj.weight.to(
        device=head_values.device, dtype=torch.float32).reshape(D, HEADS, HEAD_DIM)
    return torch.einsum("bhqu,ohu->bqho", head_values, output_weight)


def exact_factor_allocations(attention, normal_factors, absent_factors):
    """Return exact Shapley, factor-first, and factor-last raw-write pieces.

    Output views have shape [batch,query,45,residual] in head-major order.
    The Shapley pieces exactly sum to the full normal-minus-absent write.
    """
    arms = []
    for mask in range(1 << len(FACTOR_NAMES)):
        factors = tuple(
            normal_factors[index] if mask & (1 << index)
            else absent_factors[index]
            for index in range(len(FACTOR_NAMES)))
        arms.append(_per_head_factor_writes(attention, factors))
    arms = torch.stack(arms, dim=0)

    effects = torch.zeros_like(arms)
    for mask in range(1 << len(FACTOR_NAMES)):
        for child in range(1 << len(FACTOR_NAMES)):
            if child & ~mask:
                continue
            sign = -1.0 if ((mask.bit_count() - child.bit_count()) % 2) else 1.0
            effects[mask] += sign * arms[child]

    shapley = [torch.zeros_like(arms[0]) for _ in FACTOR_NAMES]
    for mask in range(1, 1 << len(FACTOR_NAMES)):
        share = effects[mask] / mask.bit_count()
        for factor in range(len(FACTOR_NAMES)):
            if mask & (1 << factor):
                shapley[factor] += share
    first = [effects[1 << factor] for factor in range(len(FACTOR_NAMES))]
    last = [
        arms[FULL_MASK] - arms[FULL_MASK ^ (1 << factor)]
        for factor in range(len(FACTOR_NAMES))]

    def flatten(values):
        stacked = torch.stack(values, dim=3)  # [batch,query,head,factor,residual]
        return stacked.reshape(stacked.shape[0], stacked.shape[1], -1, D)

    normal_write = arms[FULL_MASK].sum(2)
    absent_write = arms[0].sum(2)
    return {
        "shapley": flatten(shapley),
        "first": flatten(first),
        "last": flatten(last),
    }, {
        "normal_write": normal_write,
        "absent_write": absent_write,
        "factor_delta": normal_write - absent_write,
        "mobius_reconstruction": effects[1:].sum(0).sum(2),
        "shapley_reconstruction": flatten(shapley).sum(2),
    }


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 495 \
            or result.get("repair_id") != "495b_float32_factor_arithmetic" \
            or result.get("pred_a_exact_live_instrument") is not True \
            or result.get("pred_b_cross_head_downstream_equivalence") is not False \
            or result.get("validation_documents_and_tags_opened") is not False \
            or result.get("strong_null") is not True \
            or result.get("next_step") != \
            "split_QK_score_sides_into_query_and_key_downstream_use":
        raise RuntimeError("lawful rung495b null did not license rung496")
    rows, fit_rows, circuit_masks, discovery_tags, validation_tags, metadata = \
        parent.validate_inputs()
    return rows, fit_rows, circuit_masks, discovery_tags, validation_tags, {
        **metadata,
        "factor_names": list(FACTOR_NAMES),
        "piece_names": list(PIECE_NAMES),
        "eligible_side_indices": list(SIDE_INDICES),
    }


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(PIECE_NAMES) == 45 and len(SIDE_INDICES) == 36
        print(json.dumps({
            "status": "implementation_in_progress",
            "rung": 496,
            "model_loaded": False,
            "downstream_use_outcomes_opened": False,
            "five_factor_arms": 32,
            "shapley_pieces": len(PIECE_NAMES),
            "eligible_query_key_sides": len(SIDE_INDICES),
            "implemented_now": [
                "frozen_parent_authority",
                "float32_five_factor_reconstruction",
                "mobius_interactions",
                "exact_shapley_allocation",
                "factor_first_and_factor_last_controls",
            ],
            "remaining_before_gpu": [
                "downstream_collection",
                "selection_and_controls",
                "scorer_and_receipt",
                "full_cpu_gates",
            ],
        }, indent=2, sort_keys=True))
        return
    raise RuntimeError("rung496 implementation is incomplete; do not enqueue")


if __name__ == "__main__":
    main()
