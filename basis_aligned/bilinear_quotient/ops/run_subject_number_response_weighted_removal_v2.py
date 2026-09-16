#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_replay_and_instrument pred_b_candidate_removes_native_effect pred_c_candidate_beats_law_and_null pred_d_unrelated_reader_selectivity
"""Correct the number-confounded collateral reader in selective removal V1."""
from __future__ import annotations

import json
from pathlib import Path

import run_subject_number_response_weighted_removal_v1 as base


RUNNER = Path(__file__).resolve()
POLY = base.POLY
ORIGINAL_PREREG = base.PREREG
CORRECTION = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_REMOVAL_V2_CORRECTION.md"
INVALID_RUNNER = Path(base.__file__).resolve()
INVALID_RESULT = base.OUT
BINDING = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_REMOVAL_V2_BINDING.json"
OUT = base.ROOT / "circuits/fast_screens/subject_number_response_weighted_removal_v2_result.json"
READOUTS = (("green_black", (4077, 2042)), ("cat_dog", (3797, 3290)),
            ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
            ("apple_orange", (17180, 10912)))
ORIGINAL_PLAN = base.plan
PREDICTION_REGISTRY = {"pred_a_replay_and_instrument": None,
                       "pred_b_candidate_removes_native_effect": None,
                       "pred_c_candidate_beats_law_and_null": None,
                       "pred_d_unrelated_reader_selectivity": None}


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": ORIGINAL_PREREG, "correction": CORRECTION,
             "invalid_runner": INVALID_RUNNER, "invalid_result": INVALID_RESULT,
             "artifact": base.ARTIFACT, "law": base.LAW, "fresh_result": base.FRESH,
             "authority": Path(base.authority.__file__),
             "capability_result": base.capability.RESULT,
             "capability_license": base.capability.LICENSE}
    if binding["files"] != {key: base.sha(path) for key, path in paths.items()} \
            or binding["price"] != base.PRICE or binding["methods"] != list(base.METHODS) \
            or binding["readouts"] != [[name, list(pair)] for name, pair in READOUTS] \
            or binding["nulls"] != base.NULLS or binding["seed"] != base.SEED:
        raise ValueError("binding changed")
    artifact, law, fresh, invalid = (json.loads(path.read_text()) for path in
                                     (base.ARTIFACT, base.LAW, base.FRESH, INVALID_RESULT))
    if artifact["terminal"] != "response_weighted_prototypes_frozen_opened_only" \
            or law["terminal"] != "bilinear_scalar_law_frozen_weights_only" \
            or fresh["terminal"] != "response_weighted_fresh_causal_null" \
            or invalid["terminal"] != "response_weighted_selective_removal_null" \
            or invalid["predictions"]["pred_d_unrelated_reader_selectivity"] is not False:
        raise ValueError("parent status changed")
    base.licensing.validate_causal_preflight(base.capability.build_gate(),
        base.capability.RESULT, base.capability.LICENSE,
        expected_license_sha256=binding["files"]["capability_license"],
        causal_candidate_id=base.authority.CAUSAL_CANDIDATE_ID)
    return binding, artifact, law, fresh


def plan():
    value = ORIGINAL_PLAN()
    value.update(schema="subject_number_response_weighted_removal_v2_plan",
                 readouts=[name for name, _ in READOUTS],
                 binding_sha256=base.sha(BINDING),
                 correction_sha256=base.sha(CORRECTION),
                 invalid_result_sha256=base.sha(INVALID_RESULT))
    return value


def main():
    original_atomic = base.managed.atomic_create_json

    def corrected_atomic(path, value):
        value["schema"] = "subject_number_response_weighted_removal_v2_result"
        value["correction_sha256"] = base.sha(CORRECTION)
        value["invalid_parent_sha256"] = base.sha(INVALID_RESULT)
        value["scope"] = ("Corrected prospective-on-opened-fourth-corpus selective removal of the frozen "
                          "response-weighted L11H3 write against symbolic-law, equal-norm same-site, and "
                          "five genuinely non-number collateral readers.")
        return original_atomic(path, value)

    base.RUNNER = RUNNER
    base.PREREG = ORIGINAL_PREREG
    base.BINDING = BINDING
    base.OUT = OUT
    base.READOUTS = READOUTS
    base.load_bound = load_bound
    base.plan = plan
    base.managed.atomic_create_json = corrected_atomic
    base.main()


if __name__ == "__main__":
    main()
