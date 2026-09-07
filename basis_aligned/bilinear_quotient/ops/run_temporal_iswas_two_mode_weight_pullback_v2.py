#!/usr/bin/env python3
"""Sole projector-normalization repair of the two-mode weight pullback."""

# BQGATE: EXPERIMENT pred_a_authority_factor_replay_finiteness_and_price pred_b_orthogonal_gauge_preserves_physical_modes pred_c_known_writer_and_readers_are_enriched pred_d_modes_split_weight_rankings pred_e_complete_weight_inventory
import hashlib,json,os
from pathlib import Path
import run_temporal_iswas_two_mode_weight_pullback_v1 as v1

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_two_mode_weight_pullback_v2.json"
PARENT_RESULT=ROOT/"circuits/followups/temporal_iswas_two_mode_weight_pullback_v1_result.json"
IMPLEMENTATION=ROOT/"ops/run_temporal_iswas_two_mode_weight_pullback_v1.py"
OUT=ROOT/"circuits/followups/temporal_iswas_two_mode_weight_pullback_v2_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_two_mode_weight_pullback_v2"
# Static registration for the queue gate; the parent implementation performs the
# corresponding predicate evaluations and writes them into the v2 result.
PREDICTIONS={
    "pred_a_authority_factor_replay_finiteness_and_price":True,
    "pred_b_orthogonal_gauge_preserves_physical_modes":True,
    "pred_c_known_writer_and_readers_are_enriched":True,
    "pred_d_modes_split_weight_rankings":True,
    "pred_e_complete_weight_inventory":True,
}
EXPECTED={"prior":"8687970615f01cdad3836b6eed3422a06f4bac5ce9e98a4d738e76c67eefcd52","parent_result":"1c31dc04a0d5f7ac9c330ec09267e54ba4e0ddffd52e7ccfb15b590e835b1e16","implementation":"abfedf5b2347bc49a13d7008c4e815a33d6c1629f300d9e720f0876c4e983788"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if {"prior":sha(PRIOR),"parent_result":sha(PARENT_RESULT),"implementation":sha(IMPLEMENTATION)}!=EXPECTED:raise RuntimeError("gauge repair authority changed")
    prior,parent=json.loads(PRIOR.read_text()),json.loads(PARENT_RESULT.read_text())
    if prior.get("candidate_id")!=CANDIDATE_ID or parent.get("terminal")!="invalid" or parent["predictions"]["pred_b_orthogonal_gauge_preserves_physical_modes"]:raise RuntimeError("invalid parent disposition changed")
    v1.main(candidate_id=CANDIDATE_ID,out=OUT,normalized_gauge=True,repair_authority=EXPECTED)
if __name__=="__main__":main()
