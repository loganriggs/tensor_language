import torch

import compilation_mask_cut_rank_v1 as cut


def test_registry_is_complete_disjoint_connected_and_challenge_balanced():
    cut.validate_registry()
    assert len(cut.ANCHOR_CELLS) == 15
    assert len(cut.TRAIN_CELLS) == 28
    assert len(cut.VALIDATION_CELLS) == 10
    assert len(cut.HELDOUT_CELLS) == 11
    assert cut.inhomogeneous_tt_parameter_count(1) == 52
    assert cut.inhomogeneous_tt_parameter_count(2) == 192
    assert cut.inhomogeneous_tt_parameter_count(4) == 736


def test_anchoring_removes_all_additive_prefix_and_suffix_effects():
    prefix = torch.arange(8, dtype=torch.float64)[:, None]
    suffix = torch.arange(8, dtype=torch.float64)[None, :].square()
    interaction = cut.anchored_interaction(3.0 + prefix + suffix)
    assert torch.equal(interaction, torch.zeros_like(interaction))
    assert cut.spectral_tail_nre(interaction, 0) == 0.0


def test_rank_two_cut_passes_and_rank_three_cut_falsifies_rank_two():
    generator = torch.Generator().manual_seed(20260828)
    left = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    left[0].zero_()
    right[:, 0].zero_()
    rank_two = left[:, :2] @ right[:2]
    rank_three = left @ right
    assert cut.spectral_tail_nre(cut.anchored_interaction(rank_two), 2) < 1e-12
    assert cut.spectral_tail_nre(cut.anchored_interaction(rank_three), 2) > 1e-3
