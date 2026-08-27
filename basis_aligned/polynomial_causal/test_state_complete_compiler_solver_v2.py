from __future__ import annotations

import torch

import state_complete_compiler_solver_v2 as solver


def _synthetic(causal: bool = False):
    generator = torch.Generator().manual_seed(19)
    n, products, coefficients = 700, 12, 4
    phi = torch.randn(n, products, generator=generator, dtype=torch.float64)
    q = torch.randn(products, coefficients, generator=generator, dtype=torch.float64)
    true = torch.zeros(products, dtype=torch.float64)
    true[[1, 7]] = torch.tensor([1.5, -0.8], dtype=torch.float64)
    bias = torch.randn(coefficients, generator=generator, dtype=torch.float64)
    target = (phi * true) @ q + bias
    adjoint = None
    if causal:
        adjoint = torch.randn(n, coefficients, generator=generator, dtype=torch.float64)
    return phi, q, target, adjoint, true


def test_euclidean_statistics_and_refit_recover_sparse_native_program() -> None:
    phi, q, target, _, true = _synthetic()
    hessian, linear, intercept, offset = solver.native_quadratic_statistics(phi, q, target)
    amplitudes = solver.refit_support(hessian, linear, torch.tensor([1, 7]))
    assert torch.allclose(amplitudes, true[[1, 7]], atol=2e-7, rtol=2e-7)
    assert torch.allclose(intercept, target.mean(dim=0))
    serialized = solver.materialize_native_intercept(
        intercept, offset, q, torch.tensor([1, 7]), amplitudes
    )
    assert torch.allclose(serialized, target[0] - (phi[0] * true) @ q,
                          atol=2e-7, rtol=2e-7)


def test_causal_constant_is_optimal_against_small_perturbations() -> None:
    _, _, target, adjoint, _ = _synthetic(causal=True)
    constant = solver.causal_constant(target, adjoint)

    def loss(value):
        error = value - target
        directional = (adjoint * error).sum(dim=1).square().mean()
        directional /= adjoint.square().sum(dim=1).mean()
        isotropic = solver.CAUSAL_FLOOR * error.square().mean() / target.shape[1]
        return directional + isotropic

    base = loss(constant)
    for coordinate in range(target.shape[1]):
        moved = constant.clone()
        moved[coordinate] += 1e-3
        assert loss(moved) >= base - 1e-10


def test_causal_statistics_are_symmetric_finite_and_support_refittable() -> None:
    phi, q, target, adjoint, _ = _synthetic(causal=True)
    hessian, linear, intercept, offset = solver.native_quadratic_statistics(
        phi, q, target, adjoint=adjoint
    )
    assert torch.allclose(hessian, hessian.T)
    assert torch.isfinite(hessian).all() and torch.isfinite(linear).all()
    assert intercept.shape == (target.shape[1],)
    assert torch.equal(offset, torch.zeros_like(offset))
    amplitudes = solver.refit_support(hessian, linear, torch.tensor([1, 7]))
    assert torch.isfinite(amplitudes).all()


def test_fista_path_and_fit_only_frontier_find_planted_support() -> None:
    phi, q, target, _, _ = _synthetic()
    hessian, linear, _, _ = solver.native_quadratic_statistics(phi, q, target)
    path = solver.fista_l1_path(
        hessian, linear, ratios=(0.1, 0.01, 0.0), iterations=800
    )
    assert [row["lambda_ratio"] for row in path] == [0.1, 0.01, 0.0]
    frontier = solver.select_refit_frontier(hessian, linear, path, (2, 4))
    assert set(frontier) == {2, 4}
    assert set(frontier[2]["support"].tolist()) == {1, 7}
    assert torch.allclose(
        frontier[2]["amplitudes"],
        torch.tensor([1.5, -0.8], dtype=torch.float64)[
            torch.tensor([0 if i == 1 else 1 for i in frontier[2]["support"]])
        ],
        atol=2e-6, rtol=2e-6,
    )


def test_solver_rejects_zero_causal_gradient_and_bad_support() -> None:
    phi, q, target, _, _ = _synthetic()
    try:
        solver.native_quadratic_statistics(
            phi, q, target, adjoint=torch.zeros_like(target)
        )
    except ValueError as error:
        assert "zero energy" in str(error)
    else:
        raise AssertionError("zero causal adjoint was accepted")
    hessian, linear, _, _ = solver.native_quadratic_statistics(phi, q, target)
    try:
        solver.refit_support(hessian, linear, torch.tensor([1, 1]))
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate native support was accepted")
