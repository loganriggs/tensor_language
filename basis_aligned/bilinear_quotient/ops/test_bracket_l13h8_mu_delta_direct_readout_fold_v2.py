#!/usr/bin/env python3

from __future__ import annotations

import inspect

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_direct_readout_fold as v1
import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_direct_readout_fold_v2 as v2
import run_bracket_l13h8_mu_delta_direct_readout_fold_v2 as runner


def test_only_technical_replay_bar_changes():
    p1, p2 = v1.compile_plan(), v2.compile_plan()
    assert p2["bars"]["softcap_output_replay_max_absolute_error"] == 4e-5
    p1["bars"]["softcap_output_replay_max_absolute_error"] = 4e-5
    for key in ("schema", "candidate_id", "prior_art_sha256"):
        p1[key] = p2[key]
    p1["invalid_parent_result_sha256"] = p2["invalid_parent_result_sha256"]
    p1["instrument_only_correction"] = p2["instrument_only_correction"]
    assert p1 == p2


def test_successor_reuses_frozen_science_and_has_new_output():
    source = inspect.getsource(runner)
    assert "parent.evaluate" in source and "parent.score" in source
    assert "model_forwards\": 3" in source
    assert "direct_readout_fold_v2_result.json" in source
    assert "TEST" not in source and "OOD" not in source
