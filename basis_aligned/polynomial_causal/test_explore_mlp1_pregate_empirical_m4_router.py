import torch

import explore_mlp1_pregate_empirical_m4_router as subject
import explore_mlp1_pregate_quadratic_router as control


def test_sample_score_loss_equals_explicit_fourth_moment_contraction():
    torch.manual_seed(3)
    states = torch.randn(31, 4, dtype=torch.float64)
    errors = torch.randn(3, 4, 4, dtype=torch.float64)
    errors = 0.5 * (errors + errors.transpose(1, 2))
    direct_scores = torch.einsum("ni,aij,nj->na", states, errors, states)
    direct = direct_scores.square().mean()
    explicit = subject.explicit_fourth_moment_loss(states, errors)
    assert torch.allclose(direct, explicit, atol=1e-10, rtol=1e-10)


def test_frobenius_rank_can_be_wrong_on_reachable_support():
    torch.manual_seed(8)
    states = torch.zeros(2_000, 5)
    states[:, :2] = torch.randn(2_000, 2)
    target_q = torch.diag(torch.tensor([1.0, -1.0, 0.0, 0.0, 10.0]))
    # Largest-|eigenvalue| rank two spends one mode on an unreachable coefficient.
    frobenius_q = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0, 10.0]))
    empirical_q = torch.diag(torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0]))
    target = torch.einsum("ni,ij,nj->n", states, target_q, states)
    frobenius = torch.einsum("ni,ij,nj->n", states, frobenius_q, states)
    empirical = torch.einsum("ni,ij,nj->n", states, empirical_q, states)
    denominator = target.square().mean()
    assert (frobenius - target).square().mean() / denominator > 0.4
    assert torch.equal(empirical, target)


def test_canonicalization_preserves_indefinite_scores_and_fixes_gauge():
    torch.manual_seed(11)
    factors = torch.randn(4, 9, 3)
    values = torch.tensor([
        [3.0, -2.0, 0.7], [1.0, -4.0, 2.0],
        [-3.0, 0.2, 5.0], [2.0, -1.0, -0.5],
    ])
    states = torch.randn(17, 9)
    before = control.quadratic_scores(states, factors, values, 3)
    canonical, canonical_values = subject.canonicalize_signed_squares(factors, values)
    after = control.quadratic_scores(states, canonical, canonical_values, 3)
    assert torch.allclose(before, after, atol=2e-4, rtol=2e-5)
    gram = canonical.transpose(1, 2) @ canonical
    identity = torch.eye(3)[None]
    assert torch.allclose(gram, identity, atol=2e-5, rtol=2e-5)
    pivots = canonical.gather(
        1, canonical.abs().argmax(dim=1, keepdim=True),
    ).squeeze(1)
    assert bool((pivots >= 0).all())


def test_empirical_optimizer_recovers_planted_non_gaussian_quadratics():
    torch.manual_seed(19)
    documents, positions, dimension, atoms, rank = 24, 48, 8, 4, 2
    centers = torch.randn(documents, 3) * torch.tensor([2.5, 0.7, 0.2])
    local = torch.randn(documents, positions, 3)
    local[..., 0] = torch.sign(local[..., 0]) * local[..., 0].abs().square()
    support = (centers[:, None, :] + local).reshape(-1, 3)
    states = torch.zeros(documents * positions, dimension)
    states[:, :3] = support
    states[:, 3:] = 0.01 * torch.randn(len(states), dimension - 3)

    planted = torch.zeros(atoms, dimension, rank)
    planted[:, :3] = torch.randn(atoms, 3, rank)
    planted = torch.linalg.qr(planted, mode="reduced").Q
    planted_values = torch.tensor([
        [2.0, -1.2], [1.3, -2.4], [-1.8, 0.8], [2.7, -0.9],
    ])
    targets = control.quadratic_scores(states, planted, planted_values, rank)
    initial = {
        rank: (
            planted + 0.08 * torch.randn_like(planted),
            planted_values * 0.75,
        )
    }
    fitted, curve = subject.fit_empirical_factors(
        states, targets, initial, device=torch.device("cpu"), steps=400,
        batch_size=192, learning_rate=0.02, final_learning_rate=0.001,
        seed=41, monitor_positions=512, monitor_every=100,
    )
    factors, values = fitted[rank]
    relative = float(subject.empirical_score_loss(states, targets, factors, values))
    assert relative < 2e-4
    assert curve[-1]["canonicalization_relative_score_mse"][str(rank)] < 1e-10


def test_bootstrap_interval_is_deterministic_and_paired():
    values = torch.linspace(-0.2, 0.1, 96)
    first = subject.bootstrap_mean_interval(values, draws=2_000, seed=7)
    second = subject.bootstrap_mean_interval(values, draws=2_000, seed=7)
    assert first == second
    assert first["bootstrap_95_high"] < 0
