#!/usr/bin/env python3
"""Engineering-only F.linear orientation repair for direct residual route v1."""

# BQGATE: EXPERIMENT pred_a_authority_exact_identity_coverage_and_price pred_b_weight_route_matches_dynamic_all_module_clamp pred_c_direct_route_retains_rank8_behavior pred_d_skip_gain_is_nonzero_and_frozen
import hashlib
from pathlib import Path

import run_temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v1 as impl

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v2.json"
V1_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_weight_direct_residual_route_v2"
EXPECTED_PRIOR = "39e9cadfc8dff6972ec9e43e35e43df04559c993560881bf711c07546de4529d"
EXPECTED_V1_RESULT = "b9417541013b2db19a7c5239621719467fd76e8aab9c3716070b49b07d083959"
PREDICTION_KEYS = (
    "pred_a_authority_exact_identity_coverage_and_price",
    "pred_b_weight_route_matches_dynamic_all_module_clamp",
    "pred_c_direct_route_retains_rank8_behavior",
    "pred_d_skip_gain_is_nonzero_and_frozen",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_cache(backend, batch, base_output, base11, writer11, q, gain):
    weight = backend.model.transformer.h[11].attn.c_proj.weight
    head_count = int(backend.model.config.n_head)
    head_width = int(backend.model.config.n_embd // head_count)
    cache, writes = {}, []
    for index, (row_id, query) in enumerate(zip(batch.row_ids, batch.semantic_positions)):
        delta = (writer11["head_output"][index, int(query), 3].float()
                 - base11["head_output"][index, int(query), 3].float())
        projected = (delta @ q) @ q.T
        flattened = backend.torch.zeros(head_count * head_width, device=weight.device, dtype=weight.dtype)
        flattened[3*head_width:4*head_width] = projected.to(flattened)
        write = backend.F.linear(flattened, weight) * backend.torch.as_tensor(
            gain, device=weight.device, dtype=weight.dtype)
        base18 = base_output.captured[(row_id, "resid:18")]
        cache[(row_id, "resid:18")] = base18 + write.to(base18)
        writes.append(write.detach().clone())
    return cache, writes


def main():
    if len(PREDICTION_KEYS) != 4:
        raise RuntimeError("prediction inventory changed")
    if sha(PRIOR) != EXPECTED_PRIOR or sha(V1_RESULT) != EXPECTED_V1_RESULT:
        raise RuntimeError("v2 repair authority changed")
    impl.PRIOR = PRIOR
    impl.OUT = OUT
    impl.CANDIDATE_ID = CANDIDATE_ID
    impl.EXPECTED = dict(impl.EXPECTED, prior=EXPECTED_PRIOR)
    impl.direct_cache = direct_cache
    impl.main()


if __name__ == "__main__":
    main()
