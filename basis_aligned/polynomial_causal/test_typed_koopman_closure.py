import pytest
import torch

import typed_koopman_closure as closure


def test_reduced_rank_fit_recovers_exact_small_transition():
    generator = torch.Generator().manual_seed(19)
    source = torch.randn(80, 5, generator=generator, dtype=torch.float64)
    left = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    right = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    target = source @ left @ right
    rank2 = closure.fit_reduced_rank_transition(source, target, 2)
    rank1 = closure.fit_reduced_rank_transition(source, target, 1)
    assert rank2.normalized_closure_defect < 1e-24
    assert rank1.normalized_closure_defect > 1e-5
    assert torch.allclose(source @ rank2.coefficient, target, atol=1e-11, rtol=1e-11)


def test_metric_support_ignores_registered_null_output_direction():
    generator = torch.Generator().manual_seed(20)
    source = torch.randn(40, 2, generator=generator, dtype=torch.float64)
    target = torch.cat((source, torch.randn(40, 1, generator=generator, dtype=torch.float64)), dim=1)
    metric = torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64))
    fit = closure.fit_reduced_rank_transition(source, target, 2, target_metric=metric)
    assert fit.metric_support_rank == 2
    assert fit.normalized_closure_defect < 1e-24


def test_weighted_defect_is_invariant_under_orthogonal_target_gauge():
    generator = torch.Generator().manual_seed(21)
    source = torch.randn(50, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(50, 3, generator=generator, dtype=torch.float64)
    raw = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    metric = raw.T @ raw + 0.2 * torch.eye(3, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    original = closure.fit_reduced_rank_transition(source, target, 2, target_metric=metric)
    replay = closure.fit_reduced_rank_transition(
        source, target @ q, 2, target_metric=q.T @ metric @ q,
    )
    assert original.normalized_closure_defect == pytest.approx(
        replay.normalized_closure_defect, abs=1e-12,
    )


def test_two_step_identity_and_triangle_certificate_hold_with_correlated_errors():
    generator = torch.Generator().manual_seed(22)
    source = torch.randn(30, 3, generator=generator, dtype=torch.float64)
    first = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    second = torch.randn(4, 2, generator=generator, dtype=torch.float64)
    shared_error = torch.randn(30, 4, generator=generator, dtype=torch.float64) * 0.1
    middle = source @ first + shared_error
    target = middle @ second - 0.5 * shared_error @ second
    direct = torch.linalg.lstsq(source, target).solution
    report = closure.two_step_closure_report(
        source, middle, target, first, second, direct_coefficient=direct,
    )
    assert report.identity_maximum_absolute_error < 1e-12
    assert report.composed_norm <= report.triangle_upper_bound + 1e-12
    assert report.direct_norm is not None
    assert report.composition_to_direct_ratio is not None


def test_affine_constant_is_explicit_and_validation_fails_closed():
    features = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
    augmented = closure.augment_constant(features)
    assert augmented.dtype == torch.float64
    assert torch.equal(augmented[:, 0], torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="positive semidefinite"):
        closure.fit_reduced_rank_transition(
            augmented, augmented, 1,
            target_metric=torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64)),
        )
    with pytest.raises(ValueError, match="finite"):
        closure.augment_constant(torch.tensor([[float("nan")]], dtype=torch.float64))
    with pytest.raises(ValueError, match="rank exceeds"):
        closure.fit_reduced_rank_transition(augmented, augmented, 3)
