from __future__ import annotations

import torch

import mlp_global_gate_response as gate


def test_trajectory_complete_contraction_equals_shared_alpha_autograd() -> None:
    generator = torch.Generator().manual_seed(17)
    contexts, positions, width, hidden, probes = 2, 4, 3, 5, 3
    state = torch.randn(contexts, positions, width, generator=generator, dtype=torch.float64)
    left = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    down = torch.randn(width, hidden, generator=generator, dtype=torch.float64)
    gradients = torch.randn(
        probes, contexts, positions, width, generator=generator, dtype=torch.float64,
    )
    products = (state @ left.T) * (state @ right.T)
    expected = gate.trajectory_complete_response(products, gradients, down)
    separated = torch.empty_like(expected)
    for probe in range(probes):
        for context in range(contexts):
            alpha = torch.ones(hidden, dtype=torch.float64, requires_grad=True)
            write = (products[context] * alpha) @ down.T
            score = (write * gradients[probe, context]).sum()
            separated[probe, context] = torch.autograd.grad(score, alpha)[0]
    assert torch.allclose(expected, separated, atol=1e-12, rtol=1e-12)
    local_only = torch.einsum(
        "cn,pco,on->pcn", products[:, 0], gradients[:, :, 0], down,
    )
    assert not torch.allclose(expected, local_only)


def test_response_is_invariant_to_gate_scale_gauge_and_equivariant_to_permutation() -> None:
    generator = torch.Generator().manual_seed(23)
    state = torch.randn(2, 3, 4, generator=generator, dtype=torch.float64)
    left = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    right = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    down = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    gradients = torch.randn(2, 2, 3, 4, generator=generator, dtype=torch.float64)
    products = (state @ left.T) * (state @ right.T)
    base = gate.trajectory_complete_response(products, gradients, down)
    scales = torch.linspace(0.5, 2.0, 6, dtype=torch.float64)
    gauged_products = (state @ (left * scales[:, None]).T) * (state @ right.T)
    gauged = gate.trajectory_complete_response(gauged_products, gradients, down / scales)
    assert torch.allclose(base, gauged, atol=1e-11, rtol=1e-11)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    permuted = gate.trajectory_complete_response(
        products[:, :, permutation], gradients, down[:, permutation],
    )
    assert torch.allclose(permuted, base[:, :, permutation])


def test_ridge_selector_finds_stable_response_span() -> None:
    generator = torch.Generator().manual_seed(31)
    core = torch.randn(4, 5, 3, generator=generator, dtype=torch.float64)
    mixing = torch.tensor([
        [1.0, 0.0, 0.0, 0.7, 0.0, 0.2, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.2, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.2],
    ], dtype=torch.float64)
    first = core @ mixing
    second = first + 1e-4 * torch.randn(first.shape, generator=generator, dtype=torch.float64)
    report = gate.paired_selector_report(
        first, second, budgets=(3, 5), target_rank=3, random_seed=20260828,
    )
    assert report["status"] == "paired_gate_response_selector_complete"
    assert report["rows"]["3"]["ridge"]["jaccard"] == 1.0
    assert report["rows"]["3"]["ridge"]["first_to_second_capture"] > 0.999999
    assert report["rows"]["3"]["ridge"]["first_to_second_all_on_relative_error"] < 1e-3
    assert "finite-removal" in report["claim_boundary"]


def test_selector_validation_is_fail_closed() -> None:
    response = torch.ones(2, 3, 4)
    try:
        gate.trajectory_complete_response(response, torch.ones(1, 2, 2, 3), torch.ones(3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("position mismatch was accepted")
    try:
        gate.paired_selector_report(
            response, torch.ones(2, 2, 4), budgets=(2,), target_rank=1, random_seed=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired response halves were accepted")
