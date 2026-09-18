#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linearization_remainder_small pred_c_direct_terms_carry_most pred_d_downstream_net_is_small pred_e_largest_responder_is_an_mlp
"""Lexical number were/was DoD (v142): RESPONSE CENSUS of the TENSE-READER change (has-had) under the number-set removal on the v138 natural rows.
Parents: v138 (the number removal moves has-had 0.59 vs null 0.26 on natural text), v140 (mostly 11.3), v141 (orthogonalizing the readout directions
does not remove it). v74's body: the resid18 change under the were-was removal at {11.3, 5.7, 7.8, 9.7} is split exactly by the lambda recurrence from
block 5 and attributed linearly to the has-had margin (every row oriented had-has; the direct share is sign-invariant). DIRECT = attn:05/07/09/11.
REGISTERED READING (v74's keys, scored as written): pred_c "direct >= 0.80" is expected FALSE and pred_e "largest responder is an MLP" expected TRUE --
the tense change should be MLP-borne (a number -> tense dependency in the suffix), which would make the number family's natural-text selectivity
failure a structural limit rather than a readout-direction artifact. pred_d "downstream <= 0.25 x direct" expected FALSE for the same reason.
PRICE (registered maximum): 2 batches x (native + edited trace) = 4 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
import aspectual_dod_lib as L
import run_pronoun_gender_dod_response_census_v74 as v74
import run_number_dod_battery_v55 as v55
import dod_natural_line as N

PREDICTIONS = {"pred_a_recurrence_closure": "<= 1e-3", "pred_b_linearization_remainder_small": "<= 0.10", "pred_c_direct_terms_carry_most": ">= 0.80 (expected false)",
               "pred_d_downstream_net_is_small": "<= 0.25 x direct (expected false)", "pred_e_largest_responder_is_an_mlp": "mlp:* (expected true)"}
ROWS = v74.ROOT / "circuits/followups/lexical_number_dod_natural_rows_v138.json"
CONGRUENT = ("natural_plural_were", "natural_singular_was")


def build():
    rows, sha, had, has = N.rows_from_receipt(ROWS, "__none__", " had", " has", lambda r: f"natural_{r['cue']}_{r['label']}")   # every row scored had - has
    rows = [r for r in rows if r.construction in CONGRUENT]
    return rows, L._single(" were"), L._single(" was")   # removal directions stay the number contrast


v74.OUT = v74.ROOT / "circuits/followups/lexical_number_dod_tense_census_v142_result.json"
v74.CANDIDATE_ID = "lexical_number.pp_intervener.dod_tense_census_v142"
v74.FORWARDS_MAX = 8
v74.FROM = 5
v74.MODULES = [f"{kind}:{layer:02d}" for layer in range(5, 18) for kind in ("attn", "mlp")]
v74.DIRECT = ["attn:05", "attn:07", "attn:09", "attn:11"]
v74.v71 = type("Line", (), {"build": staticmethod(build), "HEADS": ((11, 3), (5, 7), (7, 8), (9, 7))})

if __name__ == "__main__":
    v74.main()
