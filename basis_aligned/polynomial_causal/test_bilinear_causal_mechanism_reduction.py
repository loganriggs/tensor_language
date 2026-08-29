import torch

import bilinear_causal_mechanism_reduction as cmr


def fixture(seed: int = 19):
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(257, 7, generator=generator, dtype=torch.float64)
    left = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    right = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    down = torch.randn(5, 11, generator=generator, dtype=torch.float64)
    bias = torch.randn(5, generator=generator, dtype=torch.float64)
    return states, left, right, down, bias


def test_bilinear_cmr_score_and_map_are_exactly_gauge_invariant():
    states, left, right, down, bias = fixture()
    products = cmr.product_activations(states, left, right)
    original = cmr.bilinear_write(states, left, right, down, bias)
    scales_l = torch.logspace(-3, 3, left.shape[0], dtype=torch.float64)
    scales_r = torch.logspace(2, -2, left.shape[0], dtype=torch.float64)
    left_g, right_g, down_g = cmr.apply_channel_gauge(
        left, right, down, scales_l, scales_r,
    )
    products_g = cmr.product_activations(states, left_g, right_g)
    gauged = cmr.bilinear_write(states, left_g, right_g, down_g, bias)
    score = cmr.cmr_logit_scores(products, down)
    score_g = cmr.cmr_logit_scores(products_g, down_g)
    assert torch.allclose(gauged, original, atol=1e-10, rtol=1e-10)
    assert torch.allclose(score_g, score, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(products_g.var(dim=0), products.var(dim=0))


def test_constant_replacement_compiles_exactly_without_runtime_mask():
    states, left, right, down, bias = fixture()
    products = cmr.product_activations(states, left, right)
    replaced = torch.tensor([1, 4, 9])
    constants = products[:, replaced].mean(dim=0)
    folded = cmr.compile_constant_replacement(down, bias, replaced, constants)
    intervened = products.clone()
    intervened[:, replaced] = constants
    reference = intervened @ down.T + bias
    compiled = products[:, folded.kept] @ folded.down.T + folded.bias
    assert torch.allclose(compiled, reference, atol=1e-11, rtol=1e-11)


def test_affine_replacement_compiles_exactly_without_runtime_mask():
    states, left, right, down, bias = fixture()
    products = cmr.product_activations(states, left, right)
    replaced = torch.tensor([0, 3, 8])
    kept = torch.tensor([i for i in range(products.shape[1]) if i not in replaced])
    generator = torch.Generator().manual_seed(23)
    intercept = torch.randn(3, generator=generator, dtype=torch.float64)
    coefficients = torch.randn(3, kept.numel(), generator=generator, dtype=torch.float64)
    folded = cmr.compile_affine_replacement(
        down, bias, replaced, intercept, coefficients,
    )
    intervened = products.clone()
    intervened[:, replaced] = intercept + products[:, kept] @ coefficients.T
    reference = intervened @ down.T + bias
    compiled = products[:, folded.kept] @ folded.down.T + folded.bias
    assert torch.equal(folded.kept, kept)
    assert torch.allclose(compiled, reference, atol=1e-11, rtol=1e-11)


def test_joint_score_exposes_off_diagonal_terms_and_additive_special_case():
    generator = torch.Generator().manual_seed(29)
    base = torch.randn(4096, 1, generator=generator, dtype=torch.float64)
    correlated = torch.cat((base, base + 0.01 * torch.randn(
        4096, 1, generator=generator, dtype=torch.float64,
    )), dim=1)
    outgoing = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    assert float(cmr.off_diagonal_fraction(correlated, outgoing)) > 0.45

    independent = torch.randn(200_000, 2, generator=generator, dtype=torch.float64)
    fraction = float(cmr.off_diagonal_fraction(independent, outgoing))
    assert abs(fraction) < 0.01


def test_margin_certificate_is_a_valid_empirical_lower_bound():
    low_logits = torch.tensor([
        [4.0, 0.0, -1.0],
        [0.1, 0.0, -2.0],
        [2.0, 0.0, -1.0],
        [1.0, -1.0, -2.0],
    ], dtype=torch.float64)
    high_logits = low_logits + torch.tensor([
        [-0.1, 0.1, 0.0],
        [-0.2, 0.2, 0.0],
        [-0.3, 0.3, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=torch.float64)
    top2 = torch.topk(low_logits, 2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    squared_error = (high_logits - low_logits).square().sum(dim=1)
    observed = (high_logits.argmax(dim=1) == low_logits.argmax(dim=1)).double().mean()
    bound = cmr.certified_iia_lower_bound(margins, squared_error, epsilon=0.5)
    assert float(observed) >= float(bound)

