import torch

import mlp9_finite_two_source_interaction_rung504 as rung


def test_pair_vocabulary_is_complete_unordered_and_source_stable():
    assert rung.PARTNERS == (
        "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A9",
        "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
    assert len(rung.PAIR_INDICES) == 153
    assert len(set(rung.PAIR_INDICES)) == 153
    assert all(left < right for left, right in rung.PAIR_INDICES)


def test_finite_effect_is_difference_of_finite_state_differences():
    torch.manual_seed(504)
    absent = torch.randn(2, 3)
    score = torch.randn(2, 3)
    absent_removed = torch.randn(7, 2, 3)
    score_removed = torch.randn(7, 2, 3)
    complete, contribution = rung.finite_effect(
        absent, score, absent_removed, score_removed)
    torch.testing.assert_close(complete, absent - score)
    torch.testing.assert_close(
        contribution,
        (absent - score).unsqueeze(0) - (absent_removed - score_removed))


def test_mixed_difference_recovers_injected_pair_synergy():
    torch.manual_seed(505)
    singleton = torch.randn(rung.SINGLETON_COUNT, 2, 3, 4)
    synergy = torch.randn(rung.PAIR_COUNT, 2, 3, 4)
    pair = torch.stack([
        singleton[left] + singleton[right]
        for left, right in rung.PAIR_INDICES]) + synergy
    torch.testing.assert_close(rung.finite_mixed(pair, singleton), synergy)


def test_source_sums_match_literal_singleton_and_pair_removals():
    torch.manual_seed(506)
    sources = torch.randn(2, 3, rung.SINGLETON_COUNT, 5)
    sums = rung.source_sums(sources)
    torch.testing.assert_close(sums[0], sources[:, :, 0])
    left, right = rung.PAIR_INDICES[0]
    torch.testing.assert_close(
        sums[rung.SINGLETON_COUNT], sources[:, :, left] + sources[:, :, right])


def test_literal_registered_prices_include_shifted_confirmation_controls():
    assert rung.ORDINARY_EVALUATIONS_SELECTION == 63612
    assert rung.ORDINARY_EVALUATIONS_TOTAL == 128250
    assert rung.POSITION_EVALUATIONS_PER_SELECTED_PAIR == 6048
    assert rung.expected_price(0, False) == {
        "full_model_forwards": 496,
        "mlp9_plus_suffix_evaluations": 63612,
        "backwards": 0,
    }
    assert rung.expected_price(3, True)["mlp9_plus_suffix_evaluations"] == 146394


def test_parent_verdict_and_hash_authority_are_pinned():
    rows, masks, tags, metadata = rung.validate_inputs()
    assert tuple(rows.shape) == (1000, 257)
    assert len(masks) == 62 and len(tags) == 32
    assert metadata["rung503_pair_outcomes_loaded_for_selection"] is False
