import torch

import run_temporal_auxiliary_will_had_block11h3_regularized_cdas_v1 as subject


def test_tangent_pair_is_symmetric_unit_and_nontrivial():
    coordinate = torch.tensor([[1.0], [2.0], [-1.0]])
    noise = torch.tensor([[0.5], [-0.25], [1.5]])
    plus, minus = subject.tangent_pair(torch, coordinate, 0.05, noise)
    center = coordinate / coordinate.norm()
    assert torch.allclose(plus.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(minus.norm(), torch.tensor(1.0), atol=1e-6)
    assert float((plus - minus).norm()) > 0
    assert torch.allclose((plus + minus) / 2,
                          center * float(((plus + minus) / 2).norm()), atol=1e-6)


def test_registered_price_matches_optimizer_schedule():
    checkpoints_per_start = 1 + subject.STEPS // 10
    kl = 3 * (subject.STEPS * 2 + checkpoints_per_start * 2)
    noisy = 3 * (subject.STEPS * 4 + checkpoints_per_start * 2)
    optimizer_forwards = kl + noisy + noisy
    assert optimizer_forwards == 3198
    assert subject.FORWARDS == 6 + 3 + optimizer_forwards + 26
    assert subject.EVALUATIONS == 128 + 48 + optimizer_forwards * 16 + 624
    assert subject.BACKWARD_FORWARDS == 3000
    assert subject.UPDATES == 900
