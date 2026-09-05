"""Focused CPU tests for the exact narrative-tense H3 source-route screen."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import torch

import run_narrative_tense_attn11_head3_source_route_cross_task_payload as run


def test_plan_price_pairing_and_partition_are_frozen():
    plan = run.compile_plan()
    assert plan["price"] == {
        "model_forwards": 14, "example_evaluations": 1600,
        "backwards": 0, "parameter_updates": 0,
    }
    assert plan["pairing_sha256"] == run.PAIRING_SHA256
    rows = run.build_rows()
    assert len(run.build_pairing(rows)) == 64
    for row in rows:
        family = row["transform_id"]
        if family == "A1":
            assert row["T_positions"] == (0, 4)
        elif family == "A2":
            assert row["T_positions"] == (3, 5, 7)
        assert set(row["S_positions"]).isdisjoint(row["T_positions"])
        assert set(row["S_positions"]) | set(row["T_positions"]) | set(row["R_positions"]) \
            == set(range(len(row["base_ids"])))


def test_no_model_environment_returns_plan_before_loader():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], check=True, capture_output=True,
        text=True, env=env,
    )
    assert '"model_loaded": false' in completed.stdout


def test_group_head_uses_exact_score_value_and_joint_terms():
    base = {
        "p": torch.tensor([[2.0, 3.0]]),
        "u": torch.tensor([[[5.0, 7.0], [11.0, 13.0]]]),
    }
    donor = {
        "p": torch.tensor([[17.0, 19.0]]),
        "u": torch.tensor([[[23.0, 29.0], [31.0, 37.0]]]),
    }
    base["head"] = torch.einsum("bk,bkd->bd", base["p"], base["u"])
    donor["head"] = torch.einsum("bk,bkd->bd", donor["p"], donor["u"])
    native_other = base["p"][:, 1, None] * base["u"][:, 1]
    assert torch.equal(
        run._group_head(base, donor, (0,), "score", torch),
        native_other + donor["p"][:, 0, None] * base["u"][:, 0],
    )
    assert torch.equal(
        run._group_head(base, donor, (0,), "value", torch),
        native_other + base["p"][:, 0, None] * donor["u"][:, 0],
    )
    assert torch.equal(
        run._group_head(base, donor, (0,), "joint", torch),
        native_other + donor["p"][:, 0, None] * donor["u"][:, 0],
    )


def _synthetic(*, was_ce_gain: float):
    cell_specs = (
        ("A1/direct/past_to_present", "is"),
        ("A2/relative/past_to_present", "is"),
        ("A1/direct/present_to_past", "was"),
        ("A2/relative/present_to_past", "was"),
    )
    values = {
        "native_noop": (0.0, 0.0), "complete_head": (10.0, 10.0),
        "S_score": (5.0, 5.0), "S_value": (6.0, 6.0), "S_joint": (7.0, 7.0),
        "T_score": (1.5, 1.5), "T_value": (1.5, 1.5), "T_joint": (2.0, 2.0),
        "R_joint": (1.0, 1.0),
    }
    evidence = []
    for cell, state in cell_specs:
        family = cell[:2]
        for row_index in range(4):
            for arm, (margin, ce) in values.items():
                evidence.append({"row_id": f"{cell}:{row_index}", "family": family,
                                 "cell_id": cell, "arm": arm, "margin_delta": margin,
                                 "donor_ce_gain": ce, "target_state": state})
            if state == "is":
                cross = {"task14_singular_value": (4.0, 4.0),
                         "task14_plural_value": (-1.0, -1.0)}
            else:
                cross = {"task14_singular_value": (-1.0, -1.0),
                         "task14_plural_value": (4.0, was_ce_gain)}
            for arm, (margin, ce) in cross.items():
                evidence.append({"row_id": f"{cell}:{row_index}", "family": family,
                                 "cell_id": cell, "arm": arm, "margin_delta": margin,
                                 "donor_ce_gain": ce, "target_state": state})
    for family in run.CONTROL_FAMILIES:
        for row_index in range(4):
            for arm in run.WITHIN_ARMS:
                movement = .10 if arm == "complete_head" else .05
                evidence.append({"row_id": f"{family}:{row_index}", "family": family,
                                 "cell_id": f"{family}/control", "arm": arm,
                                 "margin_delta": 0.0, "donor_ce_gain": 0.0,
                                 "normalized_control_movement": movement,
                                 "target_state": None})
    capability = {cell: {"base": 1.0, "donor": 1.0} for cell, _ in cell_specs}
    exactness = {"native_replay_max_absolute_error": 0.0,
                 "source_sum_max_absolute_error": 0.0,
                 "installed_noop_max_absolute_error": 0.0,
                 "complete_head_endpoint_reproduction_max_absolute_error": 0.0,
                 "task14_source_sum_max_absolute_error": 0.0}
    return evidence, capability, exactness


def test_synthetic_bidirectional_cross_task_semantic_reuse():
    scored = run.score(*_synthetic(was_ce_gain=4.0))
    pred = scored["predictions"]
    assert pred["pred_a_instrument_live"]
    assert pred["pred_b_self_route"]
    assert not pred["pred_c_tense_cue_route"]
    assert pred["pred_e_cross_task_semantic_reuse"]
    assert not pred["pred_f_generic_output_token_confound"]
    assert scored["factor_readout"]["S"]["classification"] \
        == "score_and_value_redundant_or_interactive"


def test_synthetic_same_is_or_are_margin_is_output_token_confound():
    scored = run.score(*_synthetic(was_ce_gain=-1.0))
    pred = scored["predictions"]
    assert pred["pred_a_instrument_live"]
    assert pred["pred_b_self_route"]
    assert scored["cross_task"]["is"]["passed"]
    assert not scored["cross_task"]["was"]["passed"]
    assert not pred["pred_e_cross_task_semantic_reuse"]
    assert pred["pred_f_generic_output_token_confound"]
    assert run._terminal(pred) == "self_route_generic_output_token_confound"


def test_complete_head_endpoint_corruption_invalidates_instrument():
    evidence, capability, exactness = _synthetic(was_ce_gain=4.0)
    exactness["complete_head_endpoint_reproduction_max_absolute_error"] = 5.1e-5
    scored = run.score(evidence, capability, exactness)
    assert not scored["predictions"]["pred_a_instrument_live"]
    assert run._terminal(scored["predictions"]) == "invalid"
