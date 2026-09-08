import pytest
import torch

import final_rank2_manipulation_contract as contract


def test_removal_and_sufficiency_split_parallel_delta():
    off = torch.zeros(2, 3)
    on = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    basis = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    result = contract.removal_and_sufficiency(off, on, basis)
    assert torch.equal(result["sufficient"], torch.tensor([[1., 2., 0.], [4., 5., 0.]]))
    assert torch.equal(result["removed"], torch.tensor([[0., 0., 3.], [0., 0., 6.]]))


def test_swap_exchanges_whole_payload_and_leaves_controls_off():
    rows = [{"group_number": 0, "transform_id": name} for name in ("A1", "A2", "P", "C")]
    off = torch.zeros(4, 2)
    parallel = torch.tensor([[1., 0.], [0., 2.], [7., 7.], [8., 8.]])
    swapped = contract.paired_payload_swap(torch, off, parallel, rows)
    assert torch.equal(swapped, torch.tensor([[0., 2.], [1., 0.], [0., 0.], [0., 0.]]))


def test_bad_group_inventory_fails_closed():
    rows = [{"group_number": 0, "transform_id": name} for name in ("A1", "A2", "P")]
    with pytest.raises(contract.FinalManipulationError):
        contract.paired_payload_swap(torch, torch.zeros(3, 2), torch.zeros(3, 2), rows)
