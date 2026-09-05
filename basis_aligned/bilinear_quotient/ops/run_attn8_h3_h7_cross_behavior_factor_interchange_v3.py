#!/usr/bin/env python3
"""Create-only v3 repair of cross-behavior factor control capability."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_shared_payload_private_router pred_c_shared_score_and_payload pred_d_generic_numeral_or_copy_bus

from __future__ import annotations

import json
import os
import sys

import torch

import circuit_fast_screen_candidate_attn8_h3_h7_cross_behavior_factor_interchange_v3 as authority
import run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as v2


ROOT = v2.ROOT
OUT = ROOT / "circuits/fast_screens/attn8_h3_h7_cross_behavior_factor_interchange_v3_control_repair_result.json"
PRIOR_ART_SHA256 = "d0760a70470388dd4cd1d7ed24c3aacbd489f4c5389f5ceeef71f92c6811f055"
ARMS, CONTROL_ARMS, HEADS = v2.ARMS, v2.CONTROL_ARMS, v2.HEADS


def build_rows():
    return authority.build_rows()


def compile_plan():
    plan = v2.compile_plan()
    plan.update(schema="attn8_h3_h7_cross_behavior_factor_interchange_plan_v3_control_repair",
                candidate_id="numeric_successor.attn8_h3_h7_cross_behavior_factor_interchange_v3_control_repair",
                authority_sha256=authority.validate_rows(build_rows()),
                repair="step_two_preference_and_control_capability_only")
    return plan


def score(evidence, control_evidence, capability, exactness, bars):
    scored = v2.score(evidence, control_evidence, capability, exactness, bars)
    control_capability = all(
        item["native_preference_accuracy"] >= bars["minimum_native_accuracy"]
        for split in scored["splits"].values()
        for control in split["controls"].values()
        for item in control.values())
    old = scored["predictions"]
    instrument = old["pred_a_instrument_live"] and control_capability
    scored["control_native_capability_live"] = control_capability
    scored["predictions"] = {
        "pred_a_instrument_live": instrument,
        "pred_b_shared_payload_private_router": instrument and old["pred_b_shared_payload_private_router"],
        "pred_c_shared_score_and_payload": instrument and old["pred_c_shared_score_and_payload"],
        "pred_d_generic_numeral_or_copy_bus": instrument and old["pred_d_generic_numeral_or_copy_bus"],
    }
    return scored


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    model, checkpoint = v2.r573.facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    evidence, controls, capability = [], [], {}
    exactness = {key: 0. for key in ("native_replay_relative_squared_error",
        "head_source_sum_relative_squared_error", "value_split_relative_squared_error",
        "installed_term_max_absolute_error")}
    rows = build_rows()
    for split in ("FIT", "SELECT"):
        split_e, split_c, split_cap, split_exact = v2.evaluate_split(model, rows, split, torch)
        evidence += split_e; controls += split_c; capability[split] = split_cap
        exactness = {key: max(exactness[key], split_exact[key]) for key in exactness}
    scored = score(evidence, controls, capability, exactness, plan["bars"])
    predictions = scored["predictions"]
    terminal = ("invalid" if not predictions["pred_a_instrument_live"] else
                "shared_score_and_payload" if predictions["pred_c_shared_score_and_payload"] else
                "shared_payload_private_router" if predictions["pred_b_shared_payload_private_router"] else
                "generic_numeral_or_copy_bus" if predictions["pred_d_generic_numeral_or_copy_bus"] else
                "location_only_null")
    result = {"schema": "attn8_h3_h7_cross_behavior_factor_interchange_result_v3_control_repair",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "terminal": terminal,
        "score": scored, "evidence": evidence, "control_evidence": controls,
        "evaluated_splits": ["FIT", "SELECT"], "forbidden_splits_opened": [],
        "model_forwards": 6}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=1)+"\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__": main()
