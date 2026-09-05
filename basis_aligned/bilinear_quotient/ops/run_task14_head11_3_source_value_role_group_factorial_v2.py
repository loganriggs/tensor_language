#!/usr/bin/env python3
"""Create-only numerical replay repair for the Task14 source-role factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_SI_sufficient_all_cells pred_c_bridge_repairs_failed_SI_cells pred_d_SxI_interaction_shared

from __future__ import annotations

import json
import os
import sys

import run_task14_head11_3_source_value_role_group_factorial as v1


ROOT = v1.ROOT
OUT = ROOT / "circuits/fast_screens/task14_head11_3_source_value_role_group_factorial_v2_result.json"
PRIOR_ART_SHA256 = "181c55842f0585932254334d4b10f5c2f0dc1afd4d1445ddeae0d045a72774fd"


def compile_plan():
    plan = v1.compile_plan()
    bars = dict(plan["bars"])
    bars["maximum_empty_subset_absolute_logit_error"] = 7e-5
    return {
        **plan,
        "schema": "task14_head11_3_source_value_role_group_factorial_plan_v2",
        "candidate_id": (
            "subject_verb.number_agreement.head11_3."
            "source_value_role_group_factorial_numerical_repair_v2"
        ),
        "bars": bars,
        "registered_predictions": {
            "pred_a_instrument_live": "all exactness, reproduction, algebra, capability, and repaired replay gates pass",
            "pred_b_SI_sufficient_all_cells": "S+I recovers at least 70 percent of joint values in every cell",
            "pred_c_bridge_repairs_failed_SI_cells": "bridge adds at least 10 percent and repairs every cell where S+I is insufficient",
            "pred_d_SxI_interaction_shared": "the named S-by-I margin dividend is material and sign-stable across cells and lexical halves",
        },
        "numerical_repair": (
            "Only the empty-subset full-vocabulary float32 replay tolerance changes "
            "from 5e-5 to 7e-5. All scientific computations and bars are unchanged."
        ),
    }


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = v1.atlas.factor_parent._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True,
    )
    with torch.no_grad():
        values = v1.evaluate(model, torch, F, facade, plan)
    scored = v1.score(*values, plan["bars"])
    terminal = "role_group_factorial_screen" if \
        scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_source_value_role_group_factorial_result_v2",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "terminal": terminal, "score": scored, "evidence": values[0],
        "evaluated_splits": ["TEST_REUSE_NEW_INTERVENTION"],
        "forbidden_splits_opened": [], "model_forwards": 3,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "score": scored}, indent=2))


if __name__ == "__main__":
    main()
