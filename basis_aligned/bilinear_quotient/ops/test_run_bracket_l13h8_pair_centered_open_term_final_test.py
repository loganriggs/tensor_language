"""Focused CPU tests for the R545 FINAL_TEST pair-centered opener experiment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import circuit_fast_screen_candidate_bracket_l13h8_pair_centered_open_term_final_test as authority
import run_bracket_l13h8_pair_centered_open_term_final_test as run


def _native(correct=True):
    evidence = []
    for row in authority.ROWS:
        for side in ("base", "donor"):
            evidence.append({
                "row_id": row["row_id"], "family_id": row["family_id"],
                "role": row["role"], "side": side,
                "cell_id": run._capability_cell(row, side),
                "correct": correct, "answer_margin": 2.0 if correct else -1.0,
                "full_vocab_CE": .2,
            })
    return evidence


def _causal(control_effect=0.0):
    records = []
    for row in authority.ROWS:
        for side in ("base", "donor"):
            direction = "base_to_donor" if side == "base" else "donor_to_base"
            donor_side = "donor" if side == "base" else "base"
            target = row["role"] == "target"
            effect = 1.2 if target else control_effect
            records.append({
                "row_id": row["row_id"], "family_id": row["family_id"],
                "role": row["role"], "direction": direction,
                "ordered_pair": f"{row[f'{side}_answer_id']}->{row[f'{donor_side}_answer_id']}",
                "open_term_norm": 1.0,
                "complete_donor_margin_effect": 2.0 if target else 0.0,
                "complete_donor_CE_improvement": 2.0 if target else 0.0,
                "complete_recipient_margin_damage": 2.0 if target else 0.0,
                "complete_recipient_CE_damage": 2.0 if target else 0.0,
                "complete_recipient_correct": True,
                "open_swap_donor_margin_effect": effect,
                "open_swap_donor_CE_improvement": effect,
                "open_swap_recipient_margin_damage": effect,
                "open_swap_recipient_CE_damage": effect,
                "open_swap_recipient_correct": True,
                "midpoint_donor_margin_effect": effect,
                "midpoint_donor_CE_improvement": effect,
                "midpoint_recipient_margin_damage": effect,
                "midpoint_recipient_CE_damage": effect,
                "midpoint_recipient_correct": True,
            })
    return records


def test_frozen_final_test_authority_and_price():
    plan = authority.compile_plan()
    assert len(authority.ROWS) == 180
    assert {row["split"] for row in authority.ROWS} == {"FINAL_TEST"}
    assert plan["closed_splits"] == ["OOD"]
    assert plan["price"] == {"model_forwards": 5, "example_evaluations": 1800,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["active_price_if_capability_fails"]["model_forwards"] == 1
    assert len({tuple(row[f"{side}_ids"]) for row in authority.ROWS
                for side in ("base", "donor")}) == 360


def test_native_capability_has_exact_frozen_cells():
    report = run.score_native(_native())
    assert report["passed"]
    assert len(report["cells"]) == 30
    assert sorted(cell["n"] for cell in report["cells"].values()) == [6] * 24 + [36] * 6
    bad = _native()
    cell = bad[0]["cell_id"]
    for item in [item for item in bad if item["cell_id"] == cell][:2]:
        item["correct"], item["answer_margin"] = False, -1.0
    assert not run.score_native(bad)["passed"]


def test_capability_failure_never_opens_causal_arms(monkeypatch):
    bad = _native()
    # Two failures in the same six-example cell make 4/6 < .75.
    cell = bad[0]["cell_id"]
    changed = 0
    for item in bad:
        if item["cell_id"] == cell and changed < 2:
            item["correct"], item["answer_margin"] = False, -1.0
            changed += 1
    monkeypatch.setattr(run, "evaluate_native",
                        lambda *_: (None, None, None, None, bad))
    monkeypatch.setattr(run, "evaluate_causal",
                        lambda *_: (_ for _ in ()).throw(AssertionError("causal arms opened")))
    capability, evidence, records, screen, forwards = run.run_staged(None, None, None, None)
    assert not capability["passed"]
    assert evidence == bad and records == [] and screen is None and forwards == 1


def test_synthetic_selective_pair_centered_outcome_and_control_confound():
    capability = run.score_native(_native())
    held = run.score_causal(_causal(), 0.0, capability)
    assert held["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_pair_centered_open_term_held": True,
        "pred_c_transfer_without_selective_necessity": False,
        "pred_d_no_heldout_open_term_circuit": False,
    }
    confounded = run.score_causal(_causal(control_effect=.4), 0.0, capability)
    assert not confounded["predictions"]["pred_b_pair_centered_open_term_held"]
    assert confounded["predictions"]["pred_c_transfer_without_selective_necessity"]


def test_no_model_dry_run_is_deterministic_and_outcome_blind():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(Path(run.__file__))]
    first = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    second = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    plan = json.loads(first.stdout)
    assert plan == authority.compile_plan()
    assert plan["outcome_reads"] == []
    assert plan["price"]["model_forwards"] == 5
