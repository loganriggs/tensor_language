#!/usr/bin/env python3
"""Float64-only gauge diagnostic repair of the two-mode weight pullback."""

# BQGATE: EXPERIMENT pred_a_authority_factor_replay_finiteness_and_price pred_b_orthogonal_gauge_preserves_physical_modes pred_c_known_writer_and_readers_are_enriched pred_d_modes_split_weight_rankings pred_e_complete_weight_inventory
import hashlib,json
from pathlib import Path
import run_temporal_iswas_two_mode_weight_pullback_v1 as v1

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_two_mode_weight_pullback_v3.json"
PARENT_RESULT=ROOT/"circuits/followups/temporal_iswas_two_mode_weight_pullback_v2_result.json"
IMPLEMENTATION=ROOT/"ops/run_temporal_iswas_two_mode_weight_pullback_v1.py"
OUT=ROOT/"circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
CANDIDATE_ID="cross_task.temporal_iswas_two_mode_weight_pullback_v3"
PREDICTIONS={
    "pred_a_authority_factor_replay_finiteness_and_price":True,
    "pred_b_orthogonal_gauge_preserves_physical_modes":True,
    "pred_c_known_writer_and_readers_are_enriched":True,
    "pred_d_modes_split_weight_rankings":True,
    "pred_e_complete_weight_inventory":True,
}
EXPECTED={"prior":"6a0fa8fa7c6898d9f9c04f86314b0cb725cda2f85c8e4695935d31b5317bb5cd","parent_result":"5a18ef55fd02051523f2a9ff58ee20fdb4f29211d42aa9eccf1d6fc69b2994de","implementation":"abfedf5b2347bc49a13d7008c4e815a33d6c1629f300d9e720f0876c4e983788"}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    observed={"prior":sha(PRIOR),"parent_result":sha(PARENT_RESULT),"implementation":sha(IMPLEMENTATION)}
    if observed!=EXPECTED:raise RuntimeError(f"float64 gauge repair authority changed: {observed}")
    prior,parent=json.loads(PRIOR.read_text()),json.loads(PARENT_RESULT.read_text())
    if prior.get("candidate_id")!=CANDIDATE_ID or parent.get("terminal")!="invalid" or parent["predictions"]["pred_b_orthogonal_gauge_preserves_physical_modes"]:raise RuntimeError("invalid v2 disposition changed")
    v1.main(candidate_id=CANDIDATE_ID,out=OUT,normalized_gauge=True,gauge_float64=True,repair_authority=EXPECTED)
if __name__=="__main__":main()
