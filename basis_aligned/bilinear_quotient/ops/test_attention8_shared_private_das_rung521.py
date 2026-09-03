"""CPU-only focused checks for the rung-521 Stage-A executable."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


PATH = Path(__file__).with_name("attention8_shared_private_das_rung521.py")
SPEC = importlib.util.spec_from_file_location("attention8_shared_private_das_rung521", PATH)
RUNG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNG
SPEC.loader.exec_module(RUNG)


def test_control_hierarchy_keeps_original_stages_then_falls_back_deterministically():
    assert RUNG.CONTROL_STAGES[:4] == RUNG.daslib.DEFAULT_MATCH_STAGES
    assert RUNG.CONTROL_STAGES[-1] == ("all",)
    descriptors = {
        "token": torch.tensor([17]),
        "position_bin": torch.tensor([3]),
        "ce_decile": torch.tensor([8]),
        "token_class": torch.tensor([5]),
        "all": torch.tensor([0]),
    }
    assert RUNG._match_key(descriptors, 0, 0) == (17, 3, 8)
    assert RUNG._match_key(descriptors, 0, 4) == (5, 3)
    assert RUNG._match_key(descriptors, 0, 5) == (3, 8)
    assert RUNG._match_key(descriptors, 0, 9) == (0,)


def test_response_transfer_metrics_have_literal_expected_values():
    response = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    same = RUNG._cosine_residual_recovery(response, response)
    assert same["signed_cosine"] == pytest.approx(1.0)
    assert same["relative_residual"] == pytest.approx(0.0)
    assert same["aligned_recovery"] == pytest.approx(1.0)
    opposite = RUNG._cosine_residual_recovery(response, -response)
    assert opposite["signed_cosine"] == pytest.approx(-1.0)
    assert opposite["aligned_recovery"] == pytest.approx(-1.0)


def test_permutation_is_deterministic_and_does_not_touch_unregistered_positions():
    values = torch.arange(12, dtype=torch.float64)
    groups = [torch.tensor([1, 3, 5]), torch.tensor([6, 8, 10])]
    first = RUNG._permuted_effect(values, groups, seed=521)
    second = RUNG._permuted_effect(values, groups, seed=521)
    assert torch.equal(first, second)
    assert sorted(first[groups[0]].tolist()) == sorted(values[groups[0]].tolist())
    assert sorted(first[groups[1]].tolist()) == sorted(values[groups[1]].tolist())
    untouched = torch.tensor([0, 2, 4, 7, 9, 11])
    assert torch.equal(first[untouched], values[untouched])


def test_row_donors_are_exact_decile_different_document_permutations():
    data = RUNG._load_cpu_inputs()
    for split, row_mask in data["row_masks"].items():
        result = RUNG._construct_donors_for_split(
            split, row_mask, data["base_ce"], data["docids"]
        )
        recipient = result["recipient"]
        recipient_rows = row_mask.nonzero().flatten()
        stacked = []
        for donor_map, summary in zip(
            result["maps"], result["identity"]["row_CE_decile_distance_by_map"], strict=True
        ):
            donors = donor_map[recipient]
            assert torch.equal(donors % RUNG.TOKENS, recipient % RUNG.TOKENS)
            donor_rows = donors.view(-1, RUNG.TOKENS)[:, 0] // RUNG.TOKENS
            assert torch.equal(donor_rows.sort().values, recipient_rows)
            assert bool((data["docids"][donor_rows] != data["docids"][recipient_rows]).all())
            assert summary["mean_absolute_decile_distance"] == 0.0
            stacked.append(donor_rows)
        donor_rows_by_map = torch.stack(stacked)
        assert all(
            donor_rows_by_map[:, column].unique().numel() == 8
            for column in range(donor_rows_by_map.shape[1])
        )


def test_public_runtime_diagnostics_do_not_misreport_one_partial_call_count():
    public = RUNG._public_runtime_diag({
        "fit_rows": torch.tensor([1]),
        "row_to_local": torch.tensor([0]),
        "forward_calls": 99,
        "capture_forward_calls": 6,
        "swap_forward_calls": 7,
    })
    assert public == {"capture_forward_calls": 6, "swap_forward_calls": 7}
