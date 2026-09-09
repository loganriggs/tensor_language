#!/usr/bin/env python3

import pytest
import torch

import attention_source_destination_cell_primitive as cells


def fixture():
    native = {
        "q": torch.tensor([[[1.0, 2.0], [2.0, 1.0]]]),
        "k": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "q2": torch.tensor([[[2.0, -1.0], [1.0, 3.0]]]),
        "k2": torch.tensor([[[1.0, 1.0], [-1.0, 2.0]]]),
        "u": torch.tensor([[[1.0, 3.0, 2.0], [4.0, -1.0, 2.0]]]),
    }
    donor = {name: value + (index + 1) * .25
             for index, (name, value) in enumerate(native.items())}
    source_groups = torch.tensor([[[True, False], [False, True], [False, False]]])
    destination_groups = torch.tensor([[[True, False], [False, True], [False, False]]])
    return native, donor, source_groups, destination_groups


def test_cells_close_exact_full_write_on_every_covered_destination():
    native, donor, source_groups, destination_groups = fixture()
    result = cells.source_destination_cell_deltas(
        native, donor, source_groups, destination_groups, torch)
    reconstructed = result["cells"].sum((2, 3))
    assert result["cells"].shape == (1, 2, 3, 3, 3)
    assert torch.allclose(reconstructed, result["donor"] - result["base"],
                          atol=1e-6, rtol=1e-6)
    assert result["shape"] == {
        "batch": 1, "queries": 2, "destination_groups": 3,
        "source_groups": 3, "output": 3,
    }


def test_sufficiency_and_reset_are_reciprocal_for_the_same_cell():
    native, donor, source_groups, destination_groups = fixture()
    result = cells.source_destination_cell_deltas(
        native, donor, source_groups, destination_groups, torch)
    selected = torch.zeros(3, 3, dtype=torch.bool)
    selected[1, 0] = True
    sufficient = cells.compose_cell_intervention(result, selected, "sufficiency", torch)
    reset = cells.compose_cell_intervention(result, selected, "reset", torch)
    assert torch.allclose(sufficient - result["base"], result["donor"] - reset)
    full = torch.ones(3, 3, dtype=torch.bool)
    assert torch.allclose(
        cells.compose_cell_intervention(result, full, "sufficiency", torch), result["donor"])
    assert torch.allclose(
        cells.compose_cell_intervention(result, full, "reset", torch), result["base"])


def test_cells_respect_causality_before_destination_partitioning():
    native, donor, source_groups, destination_groups = fixture()
    result = cells.source_destination_cell_deltas(
        native, donor, source_groups, destination_groups, torch)
    # At query zero, the source-one role cannot contribute.
    assert torch.equal(result["cells"][0, 0, :, 1],
                       torch.zeros_like(result["cells"][0, 0, :, 1]))


def test_destination_groups_reject_overlap_and_bad_selection():
    native, donor, source_groups, destination_groups = fixture()
    overlap = destination_groups.clone()
    overlap[:, 2] = overlap[:, 0]
    with pytest.raises(ValueError, match="overlap"):
        cells.source_destination_cell_deltas(
            native, donor, source_groups, overlap, torch)
    result = cells.source_destination_cell_deltas(
        native, donor, source_groups, destination_groups, torch)
    with pytest.raises(ValueError, match="boolean"):
        cells.compose_cell_intervention(result, torch.ones(3, 3), "reset", torch)
    with pytest.raises(ValueError, match="mode"):
        cells.compose_cell_intervention(
            result, torch.ones(3, 3, dtype=torch.bool), "blend", torch)
