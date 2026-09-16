import torch

import run_equality_l8h4_equal_norm_removal_null_v1 as run


def test_matched_random_direction_is_deterministic_and_position_norm_matched():
    torch.manual_seed(3)
    term = torch.randn(2, 5, 17, dtype=torch.float32)
    term[0, 0] = 0
    first, first_error = run.matched_random_direction(term, 123)
    second, second_error = run.matched_random_direction(term, 123)
    other, _ = run.matched_random_direction(term, 124)
    assert torch.equal(first, second)
    assert not torch.equal(first[1:], other[1:])
    torch.testing.assert_close(first.norm(dim=-1), term.norm(dim=-1), rtol=1e-6, atol=1e-6)
    assert first_error <= 1e-6 and second_error <= 1e-6
    assert torch.count_nonzero(first[0, 0]) == 0


def test_registered_price_and_seeds_are_fixed():
    assert len(run.CONTROL_SEEDS) == len(set(run.CONTROL_SEEDS)) == 16
    assert run.PRICE == {"forward_calls": 864, "ood_documents": 192, "matched_directions": 16,
                         "checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0}
