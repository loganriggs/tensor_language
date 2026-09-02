from types import SimpleNamespace

import torch
import torch.nn.functional as F

import mlp9_score_response_source_pair_atlas_rung502b as rung


def _fake_model():
    blocks = [SimpleNamespace(lambdas=torch.tensor([.8 + site / 100, .2]))
              for site in range(10)]
    return SimpleNamespace(transformer=SimpleNamespace(h=blocks))


def test_norm_weights_partition_each_token_complement():
    sources = torch.randn(2, 3, 20, 7)
    weights = rung._norm_weights(sources)
    torch.testing.assert_close(weights.sum(2), torch.ones_like(weights.sum(2)))
    assert bool((weights >= 0).all())


def test_both_gauges_sum_to_exact_deployed_state():
    torch.manual_seed(502)
    model = _fake_model()
    x0 = torch.randn(2, 3, 7)
    attentions = [torch.randn_like(x0) for _ in range(10)]
    mlps = [torch.randn_like(x0) for _ in range(9)]
    embedding_coefficient, write_coefficients = rung.first._source_coefficients(model)
    analytic = [embedding_coefficient * x0]
    analytic.extend(write_coefficients[i] * attentions[i] for i in range(10))
    analytic.extend(write_coefficients[i] * mlps[i] for i in range(9))
    raw = torch.stack(analytic, dim=2).sum(2) + .01 * torch.randn_like(x0)
    z = F.rms_norm(raw, (7,))
    gauges, original, diagnostics = rung.exact_source_gauges(
        model, x0, attentions, mlps, raw, z)
    assert set(gauges) == set(rung.GAUGES)
    for sources in gauges.values():
        torch.testing.assert_close(sources.sum(2), z, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(original[0].sum(2) + original[1], z)
    assert all(value < 1e-12 for value in diagnostics["state_closure"].values())


def test_repaired_intervals_and_price_are_literal():
    assert rung.DOC_QUARTERS == ((0, 124), (124, 248), (248, 374), (374, 500))
    assert 125 * (1 + 1 + 3 + 3) == 1000
    assert len(rung.SOURCES) == 20 and len(rung.SOURCE_PAIRS) == 210


def test_first_invalid_receipt_and_authority_are_pinned():
    rows, masks, tags, metadata = rung.validate_inputs()
    assert tuple(rows.shape) == (1000, 257)
    assert len(masks) == 62 and len(tags) == 32
    assert metadata["first_outcomes_reused_for_scoring"] is False
