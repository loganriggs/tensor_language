#!/usr/bin/env python3

import run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as v2
import run_attn8_h3_h7_cross_behavior_factor_interchange_v3 as v3
import test_run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as fixtures


def test_v3_keeps_v2_arms_bars_splits_and_price():
    old, new = v2.compile_plan(), v3.compile_plan()
    for key in ("target_arms", "control_arms", "fixed_heads", "splits", "price", "bars"):
        assert new[key] == old[key]
    assert new["authority_sha256"] == v3.authority.EXPECTED_ROWS_SHA256


def test_incapable_control_makes_pred_a_and_all_claims_false():
    evidence, controls, capability, exactness = fixtures._fixture()
    for row in controls:
        if row["split"] == "FIT" and row["control_id"] == "step_two":
            row["native_preference_margin"] = -1.
    predictions = v3.score(evidence, controls, capability, exactness,
                           v3.compile_plan()["bars"])["predictions"]
    assert predictions == {"pred_a_instrument_live": False,
                           "pred_b_shared_payload_private_router": False,
                           "pred_c_shared_score_and_payload": False,
                           "pred_d_generic_numeral_or_copy_bus": False}
