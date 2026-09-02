from types import SimpleNamespace

import torch

import mlp9_attention8_finite_partner_screen_rung503 as rung


def test_partner_vocabulary_excludes_carrier_and_rounding_complement():
    assert len(rung.PARTNERS) == 18
    assert "E" not in rung.PARTNERS
    assert "A8" not in rung.PARTNERS
    assert rung.PARTNER_SOURCE_INDICES == (
        1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)


def test_finite_partner_contribution_is_difference_of_finite_responses():
    torch.manual_seed(503)
    absent = torch.randn(2, 3, 7)
    score = torch.randn(2, 3, 7)
    absent_removed = torch.randn(18, 2, 3, 7)
    score_removed = torch.randn(18, 2, 3, 7)
    observed = rung._partner_contributions(
        absent, score, absent_removed, score_removed)
    expected = (absent - score).unsqueeze(0) - (absent_removed - score_removed)
    torch.testing.assert_close(observed, expected)


def test_input_liveness_measures_the_deployed_bf16_edit():
    raw = torch.ones(2, 3, rung.D, dtype=torch.bfloat16)
    sources = torch.zeros(2, 3, 18, rung.D)
    sources[:, :, 0] = .25
    values = rung._singleton_input_edit_rms(raw, sources)
    assert float(values[0]) > 0
    assert bool((values[1:] == 0).all())


def test_group_removal_is_simultaneous_not_sum_of_singletons():
    class ToyMLP(torch.nn.Module):
        def forward(self, value):
            return value.square()

    raw = torch.randn(2, 3, rung.D, dtype=torch.bfloat16)
    sources = torch.randn(2, 3, 18, rung.D)
    observed = rung._group_removed_write(ToyMLP(), raw, sources, [0, 1])
    edited = (raw.float() - sources[:, :, [0, 1]].sum(2)).to(torch.bfloat16)
    expected = torch.nn.functional.rms_norm(edited, (rung.D,)).square()
    torch.testing.assert_close(observed, expected)


def test_selection_rule_retains_every_passing_source_without_topk():
    stats = rung._empty_stats([f"tag{i}" for i in range(32)])
    stats["denominators"][:, :, 0] = 100.0
    stats["denominators"][:, :, 1] = 100.0
    stats["denominators"][:, :, 5] = 100.0
    for source in (1, 4, 12):
        stats["pair_response_num"][:, :, source] = 2.0
        stats["pair_payload_num"][:, :, source] = .2
        stats["pair_gradient_num"][:, :, source] = 2.0
    selected, details = rung._select_partners(stats, (0, 1))
    assert selected == [1, 4, 12]
    assert [name for name, row in details.items() if row["selected"]] == [
        rung.PARTNERS[index] for index in selected]


def test_registered_prices_and_intervals_are_literal():
    assert rung.DOC_QUARTERS == ((0, 124), (124, 248), (248, 374), (374, 500))
    assert 62 * 8 == 496
    assert 62 * 2 * 3 * 18 == 6696
    assert 63 * 8 + 496 == 1000
    assert 63 * 2 * 3 * 19 + 6696 == 13878


def test_authority_and_prior_verdict_are_pinned():
    rows, masks, tags, metadata = rung.validate_inputs()
    assert tuple(rows.shape) == (1000, 257)
    assert len(masks) == 62 and len(tags) == 32
    assert metadata["rung502b_outcomes_loaded_for_selection"] is False
