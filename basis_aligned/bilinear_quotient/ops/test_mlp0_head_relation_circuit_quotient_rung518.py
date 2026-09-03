import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("mlp0_head_relation_circuit_quotient_rung518.py")
SPEC = importlib.util.spec_from_file_location("r518", PATH)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_atom_vocabulary_round_trip():
    assert len(R.ATOM_NAMES) == 45
    assert len(set(R.ATOM_NAMES)) == 45
    for head in range(9):
        for group in range(5):
            assert R.atom_parts(R.atom_index(head, group)) == (head, group)


def test_exact_proportional_pair_passes_both_backgrounds():
    responses, _expected = R.planted_problem(51800)
    left, right = R.PLANTED_PAIRS[0]
    metrics = R.pair_metrics(responses, left, right)
    assert metrics["holds"]
    assert abs(metrics["beta_left_from_right"] - 0.5) < 1e-12
    for half in metrics["halves"].values():
        for background in half.values():
            for kind in background.values():
                assert abs(kind["signed_cosine"] - 1) < 1e-12
                assert kind["left_from_right_relative_residual"] < 1e-12
                assert kind["right_from_left_relative_residual"] < 1e-12


def test_all_eight_planted_relations_recover_without_false_pairs():
    result = R.planted_suite()
    assert len(result["cases"]) == 8
    assert result["all_eight_exact"]


def test_random_unrelated_pair_fails():
    responses, _expected = R.planted_problem(51801)
    metrics = R.pair_metrics(responses, 1, 2)
    assert not metrics["holds"]


def test_dry_run_opens_no_model_or_outcome():
    result = R.dry_run()
    assert result["model_loaded"] is False
    assert result["model_outcomes_opened"] is False
    assert result["atoms"] == 45
    assert result["unordered_pairs"] == 990
