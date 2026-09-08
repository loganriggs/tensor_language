import pytest
import torch

import head_response_mediation_contract as contract


def test_capture_and_absolute_execution_hooks_are_single_use_and_removed():
    projection = torch.nn.Linear(6, 6, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(6))
    source = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    output, captured = contract.capture_preprojection(
        torch, projection, lambda: projection(source), n_heads=3)
    assert torch.equal(output, source)
    assert torch.equal(captured, source.reshape(1, 2, 3, 2))
    absolute = torch.full((1, 2, 3, 2), 7.0)
    changed = contract.execute_with_absolute_preprojection(
        projection, lambda: projection(source), absolute)
    assert torch.equal(changed, torch.full_like(source, 7.0))
    assert torch.equal(projection(source), source)


def test_capture_and_execution_hooks_fail_closed_on_shape_or_call_count():
    projection = torch.nn.Linear(6, 6, bias=False)
    source = torch.zeros(1, 2, 6)
    with pytest.raises(contract.MediationContractError):
        contract.capture_preprojection(torch, projection, lambda: source, n_heads=3)
    with pytest.raises(contract.MediationContractError):
        contract.capture_preprojection(
            torch, projection, lambda: projection(source), n_heads=4)
    with pytest.raises(contract.MediationContractError):
        contract.execute_with_absolute_preprojection(
            projection, lambda: projection(source), torch.zeros(1, 1, 2, 2))


def test_absolute_cells_have_reset_and_rescue_semantics_only_on_prefix():
    off = torch.zeros(2, 4, 3, 2)
    on = torch.arange(off.numel(), dtype=torch.float32).reshape_as(off) + 1
    cells = contract.build_absolute_cells(
        torch, off, on, head=1, semantic_positions=(1, 2))
    assert torch.equal(cells["00"], off)
    assert torch.equal(cells["11"], on)
    assert torch.equal(cells["01"][0, :2, 1], on[0, :2, 1])
    assert torch.equal(cells["01"][0, 2:, 1], off[0, 2:, 1])
    assert torch.equal(cells["10"][1, :3, 1], off[1, :3, 1])
    assert torch.equal(cells["10"][1, 3:, 1], on[1, 3:, 1])
    assert torch.equal(cells["01"][:, :, 0], off[:, :, 0])
    assert torch.equal(cells["10"][:, :, 2], on[:, :, 2])


def test_module_cells_share_the_singleton_semantics_for_all_heads():
    off = torch.zeros(1, 3, 3, 2)
    on = torch.ones_like(off)
    cells = contract.build_absolute_set_cells(
        torch, off, on, heads=(0, 1, 2), semantic_positions=(1,))
    assert torch.equal(cells["01"][:, :2], on[:, :2])
    assert torch.equal(cells["01"][:, 2:], off[:, 2:])
    assert torch.equal(cells["10"][:, :2], off[:, :2])
    assert torch.equal(cells["10"][:, 2:], on[:, 2:])


def test_decomposition_separates_rescue_bypass_and_interaction_exactly():
    result = contract.decompose_margin_cells({
        "00": [0.0, 1.0], "01": [2.0, 4.0],
        "10": [5.0, 7.0], "11": [8.0, 13.0],
    })
    assert result["head_rescue"] == [2.0, 3.0]
    assert result["upstream_bypass_with_head_reset"] == [5.0, 6.0]
    assert result["head_reset_loss"] == [3.0, 6.0]
    assert result["mobius_interaction"] == [1.0, 3.0]
    assert result["full_upstream_effect"] == [8.0, 12.0]
    assert result["closure_max_abs_error"] == 0.0


@pytest.mark.parametrize("bad", [-1, 3])
def test_head_range_fails_closed(bad):
    value = torch.zeros(1, 2, 3, 4)
    with pytest.raises(contract.MediationContractError):
        contract.build_absolute_cells(torch, value, value, head=bad, semantic_positions=(1,))


def test_shape_position_and_cell_schema_fail_closed():
    value = torch.zeros(1, 2, 3, 4)
    with pytest.raises(contract.MediationContractError):
        contract.build_absolute_cells(torch, value, value[:, :1], head=1, semantic_positions=(0,))
    with pytest.raises(contract.MediationContractError):
        contract.build_absolute_cells(torch, value, value, head=1, semantic_positions=(2,))
    with pytest.raises(contract.MediationContractError):
        contract.decompose_margin_cells({"00": [0.0]})
    with pytest.raises(contract.MediationContractError):
        contract.build_absolute_set_cells(
            torch, value, value, heads=(), semantic_positions=(1,))
    with pytest.raises(contract.MediationContractError):
        contract.build_absolute_set_cells(
            torch, value, value, heads=(1, 1), semantic_positions=(1,))


def test_nonfinite_and_length_mismatch_fail_closed():
    with pytest.raises(contract.MediationContractError):
        contract.decompose_margin_cells({
            "00": [0.0], "01": [1.0], "10": [2.0], "11": [float("nan")]})
    with pytest.raises(contract.MediationContractError):
        contract.decompose_margin_cells({
            "00": [0.0], "01": [1.0, 2.0], "10": [2.0], "11": [3.0]})
