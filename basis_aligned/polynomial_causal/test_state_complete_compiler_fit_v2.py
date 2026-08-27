from __future__ import annotations

import torch

import state_complete_compiler_fit_v2 as fit


def test_fit_shuffle_is_document_blocked_deterministic_and_train_expandable() -> None:
    documents = ["a", "a", "b", "b", "c", "d", "e", "f"]
    first = fit.document_block_permutation(documents, 17)
    second = fit.document_block_permutation(documents, 17)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(len(documents)))
    assert abs(int(first[0]) - int(first[1])) == 1
    expanded = fit.expand_capture_permutation(first)
    assert expanded.shape == (len(documents) * 64,)


def test_fit_adjoint_clip_freezes_quantile_and_preserves_directions() -> None:
    adjoint = torch.tensor([[3.0, 4.0], [0.3, 0.4], [30.0, 40.0]])
    clipped, threshold = fit.clip_fit_adjoints(adjoint, quantile=0.5)
    assert threshold == 5.0
    assert torch.allclose(clipped.norm(dim=1), torch.tensor([5.0, 0.5, 5.0],
                                                           dtype=torch.float64))
    cosine = torch.nn.functional.cosine_similarity(clipped, adjoint.double())
    assert torch.allclose(cosine, torch.ones_like(cosine))


def test_euclidean_affine_cells_recover_low_rank_targets_and_keep_interfaces() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(500, 10, generator=generator, dtype=torch.float64)
    y = x @ torch.randn(10, 2, generator=generator, dtype=torch.float64) @ torch.randn(
        2, 5, generator=generator, dtype=torch.float64
    ) + torch.randn(5, generator=generator, dtype=torch.float64)
    a = fit.euclidean_affine_states(
        x, y, lambdas=(0.0,), ranks=(2,), interface="z_only_c", family="A"
    )[(0.0, 2)]
    b = fit.euclidean_affine_states(
        x, y, lambdas=(0.0,), ranks=(2,), interface="state_complete_p", family="B"
    )[(0.0, 2)]
    for state in (a, b):
        normalized = (x.float() - state["mean"]) / state["scale"]
        prediction = (normalized @ state["left"]) @ state["right"] + state["bias"]
        assert torch.mean((prediction.double() - y).square()) < 1e-10
    assert a["interface"] == "z_only_c"
    assert b["interface"] == "state_complete_p"


def test_causal_affine_fit_reduces_registered_loss_on_tiny_problem() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(300, 9, generator=generator, dtype=torch.float64)
    y = x @ torch.randn(9, 3, generator=generator, dtype=torch.float64) @ torch.randn(
        3, 4, generator=generator, dtype=torch.float64
    )
    g = torch.randn_like(y)
    states, diagnostics = fit.causal_affine_states(
        x, y, g, lambdas=(0.0,), ranks=(3,), epochs=3,
        token_batch=100, learning_rate=0.003, seed=3,
    )
    state = states[(0.0, 3)]
    normalized = (x.float() - state["mean"]) / state["scale"]
    prediction = (normalized @ state["left"]) @ state["right"] + state["bias"]
    assert torch.isfinite(prediction).all()
    curve = diagnostics[0]["full_fit_loss_initial_then_epochs"]
    assert diagnostics[0]["selected_fit_loss"] <= curve[0]
    assert state["selected_fit_epoch"] == diagnostics[0]["selected_fit_epoch"]


def test_native_feature_and_state_frontier_serializes_independent_sparse_terms() -> None:
    generator = torch.Generator().manual_seed(23)
    n, d, products, coefficients = 600, 7, 12, 4
    z = torch.randn(n, d, generator=generator)
    left = torch.randn(products, d, generator=generator)
    right = torch.randn(products, d, generator=generator)
    q = torch.randn(products, coefficients, generator=generator)
    phi = fit.native_features(z, left, right, token_batch=100)
    amplitudes = torch.zeros(products)
    amplitudes[[2, 9]] = torch.tensor([1.2, -0.7])
    target = (phi * amplitudes) @ q + torch.randn(coefficients, generator=generator)
    states, diagnostics = fit.native_states(
        phi, target, left, right, q, adjoint=None,
        family="D_state_complete_native_euclidean", k_grid=(2, 4),
    )
    assert set(states) == {2, 4}
    assert set(states[2]["indices"].tolist()) == {2, 9}
    prediction = fit.compiler.native_projected_output(z.double(), states[2])
    assert torch.mean((prediction - target.double()).square()) < 1e-9
    assert diagnostics["objective"] == "euclidean"


def test_constant_control_contains_no_examples_or_indices() -> None:
    state = fit.constant_state(torch.randn(20, 64))
    assert state["grammar"] == "constant"
    assert state["bias"].shape == (64,)
    assert not any("row" in key or "label" in key or "index" in key for key in state)
