#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linearization_remainder_small pred_c_direct_terms_carry_most pred_d_downstream_net_is_small pred_e_largest_responder_is_an_mlp
"""Selection family DoD (v88): RESPONSE CENSUS of the run_selection_dod_battery_v85 set removal on its fresh rows (better_circuits §3.3). Body: v74's (exact
lambda-recurrence split of the resid18 change from the set's lowest block; linear margin attribution; remainder reported). DIRECT =
the attention modules of the set's own blocks (block deltas also contain other heads' responses to earlier edits; stated).
PREDICTIONS: pred_a closure <= 1e-3; pred_b |remainder| <= 0.10 x |exact|; pred_c direct >= 0.80 of linear (registered from keep-only
retention ~1); pred_d |downstream net| <= 0.25 x direct; pred_e largest downstream responder is an MLP. Prior for c-e: unsure -- the
selection sets start at block 6/7, leaving more suffix for MLP responses than the pronoun sets had.
PRICE (registered maximum): 3 batches x (native + edited trace) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
import run_pronoun_gender_dod_response_census_v74 as v74
import run_selection_dod_battery_v85 as line

PREDICTIONS = {"pred_a_recurrence_closure": "<= 1e-3", "pred_b_linearization_remainder_small": "<= 0.10", "pred_c_direct_terms_carry_most": ">= 0.80",
               "pred_d_downstream_net_is_small": "<= 0.25 x direct", "pred_e_largest_responder_is_an_mlp": "mlp:*"}
v74.OUT = v74.ROOT / "circuits/followups/selection_dod_response_census_v88_result.json"
v74.CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_response_census_v88"
v74.FORWARDS_MAX = 8
v74.FROM = min(l for l, _ in line.HEADS)
v74.MODULES = [f"{kind}:{layer:02d}" for layer in range(v74.FROM, 18) for kind in ("attn", "mlp")]
v74.DIRECT = [f"attn:{l:02d}" for l in sorted({l for l, _ in line.HEADS})]
v74.v71 = type("Line", (), {"build": staticmethod(lambda: line.build()[:3]), "HEADS": line.HEADS})

if __name__ == "__main__":
    v74.main()
