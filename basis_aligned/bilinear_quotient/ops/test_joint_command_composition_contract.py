import torch

import joint_command_composition_contract as joint


def test_additive_cells_have_zero_interaction_and_exact_closure():
    base = torch.tensor([2., -1.]); temporal = torch.tensor([1., 3.]); iswas = torch.tensor([-2., 4.])
    result = joint.decompose(torch, {"00": base, "10": base + temporal,
                                     "01": base + iswas, "11": base + temporal + iswas})
    assert result["closure_max_abs_error"] == 0
    assert result["interaction_rms_over_both_rms"] == 0
    assert torch.equal(result["temporal_shapley"], temporal)
    assert torch.equal(result["iswas_shapley"], iswas)


def test_interaction_and_shapley_reconstruct_both_effect():
    cells = {"00": torch.zeros(3), "10": torch.tensor([1., 0, 0]),
             "01": torch.tensor([0., 2, 0]), "11": torch.tensor([2., 3., 4.])}
    result = joint.decompose(torch, cells)
    interaction = torch.tensor([1., 1., 4.])
    assert torch.equal(result["components"]["interaction"], interaction)
    assert torch.allclose(result["temporal_shapley"] + result["iswas_shapley"], cells["11"])
    assert result["closure_max_abs_error"] == 0


def test_missing_or_mismatched_cells_fail_closed():
    for cells in ({"00": [0.]}, {"00": [0.], "10": [1.], "01": [1.], "11": [1., 2.]}):
        try:
            joint.decompose(torch, cells)
        except joint.JointCompositionError:
            pass
        else:
            raise AssertionError("invalid cells accepted")


def test_serializable_removes_tensors():
    result = joint.serializable(joint.decompose(
        torch, {"00": [0.], "10": [1.], "01": [2.], "11": [3.]}))
    assert result["components"]["temporal"] == [1.0]
