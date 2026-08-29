import pytest
import torch

import hybrid_telescoping_certificate as hybrid


def test_exact_telescope_and_ce_bound_for_nonlinear_arbitrary_logits():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(6, 4, 3, 11, generator=generator, dtype=torch.float64)
    targets = torch.randint(0, 11, (4, 3), generator=generator)
    result = hybrid.certify(logits, targets)
    assert result.cuts == 5
    assert result.samples == 12
    assert result.telescope_max_abs_error < 1e-12
    assert result.maximum_ce_bound_violation < 1e-12


def test_margin_certificate_implies_observed_top1_stability():
    logits = torch.tensor([
        [[10.0, 0.0, -1.0], [2.0, 1.9, 0.0]],
        [[9.0, 0.5, -1.0], [1.9, 2.0, 0.0]],
    ])
    result = hybrid.certify(logits, torch.tensor([0, 0]))
    assert result.top1_certified_fraction == 0.5
    assert result.top1_observed_unchanged_fraction == 0.5


def test_cancellation_ratio_detects_large_canceling_steps():
    native = torch.tensor([[2.0, 0.0]])
    first = native + torch.tensor([[100.0, -100.0]])
    final = native + torch.tensor([[0.1, -0.1]])
    result = hybrid.certify(torch.stack((native, first, final)), torch.tensor([0]))
    assert result.median_cancellation_ratio > 1000
    assert result.max_end_to_end_logit_error == pytest.approx(0.1)


@pytest.mark.parametrize("bad_logits,bad_targets", [
    (torch.zeros(1, 2, 3), torch.zeros(2, dtype=torch.long)),
    (torch.zeros(2, 2, 1), torch.zeros(2, dtype=torch.long)),
    (torch.zeros(2, 2, 3), torch.tensor([0, 3])),
])
def test_invalid_contract_fails_closed(bad_logits, bad_targets):
    with pytest.raises(ValueError):
        hybrid.certify(bad_logits, bad_targets)
