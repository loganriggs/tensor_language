import pytest
import torch

import attention_factor_mediation_executor as executor


def test_mediator_site_enumeration_is_complete_and_deterministic():
    assert len(executor.mediator_sites(12)) == 45
    assert executor.mediator_sites(12, 4) == tuple(
        f"L12H4:{factor}" for factor in executor.FACTORS)
    assert executor.mediator_sites(13, 2, "q") == ("L13H2:q",)


def test_factor_hybrid_changes_prefix_only():
    background = torch.zeros(2, 4, 3)
    source = torch.ones_like(background)
    result = executor.prefix_hybrid(background, source, (1, 2))
    assert torch.equal(result[0, :2], torch.ones(2, 3))
    assert torch.equal(result[0, 2:], torch.zeros(2, 3))
    assert torch.equal(result[1, :3], torch.ones(3, 3))
    assert torch.equal(result[1, 3:], torch.zeros(1, 3))


def test_invalid_factor_and_shape_fail_closed():
    with pytest.raises(executor.AttentionFactorError):
        executor.site(12, 0, "output")
    with pytest.raises(executor.AttentionFactorError):
        executor.mediator_sites(12, factor="q")
    with pytest.raises(executor.AttentionFactorError):
        executor.prefix_hybrid(torch.zeros(2, 3, 4), torch.zeros(2, 4, 4), (1, 1))
