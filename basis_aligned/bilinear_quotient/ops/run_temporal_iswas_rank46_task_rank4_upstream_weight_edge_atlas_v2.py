#!/usr/bin/env python3
"""Scorer-corrected v2 entry point for the frozen rank46 task-rank4 edge atlas."""
# BQGATE: EXPERIMENT pred_a_authority_alignment_self_patch_finiteness_and_price pred_b_sparse_causal_incidence pred_c_task_typed_edge_tensor pred_d_promoted_edges_close_through_exact_weights pred_e_additive_atlas_replays_response_order
import hashlib
from pathlib import Path
import run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1 as impl

# Static predicate declaration for the preflight gate; implementation and scoring live in v1.
PREDICATE_SCHEMA = {
    "pred_a_authority_alignment_self_patch_finiteness_and_price": None,
    "pred_b_sparse_causal_incidence": None,
    "pred_c_task_typed_edge_tensor": None,
    "pred_d_promoted_edges_close_through_exact_weights": None,
    "pred_e_additive_atlas_replays_response_order": None,
}

ROOT = Path(__file__).resolve().parents[1]
impl.PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v2.json"
impl.OUT = ROOT / "circuits/followups/temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v2_result.json"
impl.CANDIDATE_ID = "temporal_auxiliary.iswas_rank46_task_rank4_upstream_weight_edge_atlas_v2"
impl.SCHEMA = "temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_result_v2"
impl.EXPECTED = dict(impl.EXPECTED)
impl.EXPECTED["prior"] = "1ac64433c10743a23d0742fa3e83003986065c31c8250667a2298f8de38f317d"

if hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest() != "341fcf3d9b0fe1df029f2da6a3ddcb90e57cfc46ea21893a085c1883dfa095b6":
    raise RuntimeError("v2 weight-edge implementation changed")

if __name__ == "__main__": impl.main()
