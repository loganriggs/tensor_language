#!/usr/bin/env python3
"""Instrument-only replay-tolerance repair for the direct L13H8 readout fold."""

from __future__ import annotations

import copy
import json

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_direct_readout_fold as parent


CANDIDATE_ID = "bracket.pending_opener.l13h8_mu_delta_direct_readout_fold_v2"
PRIOR_ART_SHA256 = "ca99d6974cbbe99e137ce8943fc9aa0090d3dab82e38f7ccfbf1f0c0b701b0fd"
ROWS, ROWS_SHA256 = parent.ROWS, parent.ROWS_SHA256
FAMILIES = parent.FAMILIES
TARGET_FAMILIES, STABILITY_FAMILIES = parent.TARGET_FAMILIES, parent.STABILITY_FAMILIES
PATCH_LAYER, PATCH_HEAD, WRITE_BANK = parent.PATCH_LAYER, parent.PATCH_HEAD, parent.WRITE_BANK


def compile_plan() -> dict:
    plan = copy.deepcopy(parent.compile_plan())
    plan.update({
        "schema": "bracket_l13h8_mu_delta_direct_readout_fold_plan_v2",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "invalid_parent_result_sha256": "05b13e9bc0de8c59122a05ae5a423951cae604629b9a56b0f791e9cf743f182d",
        "instrument_only_correction": (
            "softcap replay tolerance 2e-5 -> 4e-5 after v1 observed a 3.0518e-5 "
            "float32 floor; all causal conditions, scientific bars, and predictions unchanged"
        ),
    })
    plan["bars"]["softcap_output_replay_max_absolute_error"] = 4e-5
    return plan


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
