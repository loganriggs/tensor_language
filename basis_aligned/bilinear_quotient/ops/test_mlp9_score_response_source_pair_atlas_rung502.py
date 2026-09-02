from types import SimpleNamespace

import torch

import mlp9_score_response_source_pair_atlas_rung502 as rung


def test_registered_source_pairs_cover_symmetric_expansion():
    assert len(rung.SOURCES) == 20
    assert len(rung.SOURCE_PAIRS) == 210
    left = torch.randn(2, 3, 20, 7)
    right = torch.randn(2, 3, 20, 7)
    total = torch.zeros(2, 3, 7)
    for i, j in rung.SOURCE_PAIRS:
        total += left[:, :, i] * right[:, :, j]
        if i != j:
            total += left[:, :, j] * right[:, :, i]
    torch.testing.assert_close(total, left.sum(2) * right.sum(2))


def test_residual_coefficients_match_direct_scalar_recurrence():
    blocks = []
    for site in range(10):
        blocks.append(SimpleNamespace(lambdas=torch.tensor([.91 + site / 1000, .03])))
    model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))
    embedding, writes = rung._source_coefficients(model)
    coefficients = {"E": torch.tensor(1.0)}
    for site, block in enumerate(blocks):
        coefficients = {key: block.lambdas[0] * value for key, value in coefficients.items()}
        coefficients["E"] += block.lambdas[1]
        coefficients[f"A{site}"] = torch.tensor(1.0)
        if site < 9:
            coefficients[f"M{site}"] = torch.tensor(1.0)
    torch.testing.assert_close(embedding, coefficients["E"])
    for site in range(10):
        torch.testing.assert_close(writes[site], coefficients[f"A{site}"])
        if site < 9:
            torch.testing.assert_close(writes[site], coefficients[f"M{site}"])


def test_unordered_contractions_match_explicit_pairs():
    factors = {"left": torch.randn(2, 3, 20, 7),
               "right": torch.randn(2, 3, 20, 7)}
    weight = torch.randn(2, 3, 7)
    observed = rung._unordered_contraction(weight, factors)
    expected = []
    for i, j in rung.SOURCE_PAIRS:
        hidden = factors["left"][:, :, i] * factors["right"][:, :, j]
        if i != j:
            hidden += factors["left"][:, :, j] * factors["right"][:, :, i]
        expected.append((weight * hidden).sum())
    # einsum and the explicit loop sum the same float32 products in different orders.
    torch.testing.assert_close(
        observed, torch.stack(expected).double(), rtol=1e-4, atol=1e-5)


def test_selection_keeps_all_and_only_registered_passers():
    stats = rung._empty_stats(3)
    for background in range(2):
        stats["denominators"][background, :2, 0] = 100
        stats["denominators"][background, :2, 5] = 10
        stats["pair_response_num"][background, :2, 0] = 2
        stats["pair_payload_num"][background, :2, 0] = .4
        stats["pair_gradient_num"][background, :2, 0] = .2
        # Pair 1 misses the score-versus-payload ratio.
        stats["pair_response_num"][background, :2, 1] = 2
        stats["pair_payload_num"][background, :2, 1] = 1.5
        stats["pair_gradient_num"][background, :2, 1] = .2
    selected, detail = rung._select_pairs(stats)
    assert selected == [0]
    assert detail[rung.PAIR_NAMES[0]]["selected"] is True
    assert detail[rung.PAIR_NAMES[1]]["selected"] is False


def test_pre_outcome_batch_aligned_intervals_are_exact():
    assert rung.DOC_QUARTERS == ((0, 124), (124, 248), (248, 374), (374, 500))
    covered = [doc for left, right in rung.DOC_QUARTERS for doc in range(left, right)]
    assert covered == list(range(500))
