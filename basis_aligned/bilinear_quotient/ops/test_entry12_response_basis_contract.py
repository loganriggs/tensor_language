import torch

import entry12_response_basis_contract as contract


def test_shared_union_and_p_complement_are_exact():
    e1 = torch.tensor([1., 0., 0.])
    e2 = torch.tensor([0., 1., 0.])
    bases, report = contract.fit_bases(torch, e1, e2, torch.tensor([[1., 0., 0.]]))
    assert bases["shared_dim_rank1"].shape == (3, 1)
    assert bases["construction_union_rank_le_2"].shape == (3, 2)
    assert bases["p_complement_union_rank_le_2"].shape == (3, 1)
    assert report["p_rank"] == 1
    for basis in bases.values():
        assert torch.allclose(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-6)


def test_sign_gauge_does_not_cancel_shared_dim():
    target = torch.tensor([1., 2., 0.])
    bases, _report = contract.fit_bases(torch, target, -target, torch.zeros(1, 3))
    direction = bases["shared_dim_rank1"][:, 0]
    assert torch.allclose(direction.abs(), target.div(target.norm()).abs(), atol=1e-6)


def test_projected_absolute_changes_only_semantic_prefix():
    off = torch.zeros(1, 3, 2)
    on = torch.ones_like(off)
    basis = torch.tensor([[1.], [0.]])
    expected = torch.tensor([[[1., 0.], [1., 0.], [0., 0.]]])
    assert torch.equal(contract.projected_absolute(torch, off, on, basis, (1,)), expected)
