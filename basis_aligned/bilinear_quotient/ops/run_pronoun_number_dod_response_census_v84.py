#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linearization_remainder_small pred_c_direct_terms_carry_most pred_d_downstream_net_is_small pred_e_largest_responder_is_an_mlp
"""Pronoun number they/he DoD (v84): RESPONSE CENSUS of the set removal on the v76 fresh rows (better_circuits §3.3). Body: v74's
(exact lambda-recurrence split of the resid18 change, blocks 9-17; linear margin attribution; remainder reported), with the number
set {9.6, 12.4, 15.1, 10.5} and rows. DIRECT = attn:09, attn:10, attn:12, attn:15 (block deltas also contain other heads' responses to
the earlier edits; stated).
PREDICTIONS: pred_a closure <= 1e-3; pred_b |remainder| <= 0.10 x |exact|; pred_c direct >= 0.80 of linear (registered from v76's
keep-only retention 1.53); pred_d |downstream net| <= 0.25 x direct; pred_e largest downstream responder is an MLP. Prior for e: unsure.
PRICE (registered maximum): 3 batches x (native + edited trace) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
import run_pronoun_gender_dod_response_census_v74 as v74
import run_pronoun_number_dod_battery_v76 as v76

PREDICTIONS = {"pred_a_recurrence_closure": "<= 1e-3", "pred_b_linearization_remainder_small": "<= 0.10", "pred_c_direct_terms_carry_most": ">= 0.80",
               "pred_d_downstream_net_is_small": "<= 0.25 x direct", "pred_e_largest_responder_is_an_mlp": "mlp:*"}
v74.OUT = v74.ROOT / "circuits/followups/pronoun_number_dod_response_census_v84_result.json"
v74.CANDIDATE_ID = "pronoun_number.they_vs_he.dod_response_census_v84"
v74.FORWARDS_MAX = 8
v74.v71 = type("Line", (), {"build": staticmethod(lambda: v76.build()[:3]), "HEADS": v76.HEADS})
v74.DIRECT = [f"attn:{l:02d}" for l in sorted({l for l, _ in v76.HEADS})]

if __name__ == "__main__":
    v74.main()
