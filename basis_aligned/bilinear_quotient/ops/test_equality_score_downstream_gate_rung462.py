import torch

import equality_score_downstream_gate_rung462 as rung


STAKES = {
    "near_positive": .05,
    "far_positive": .20,
    "one_predecessor_positive": .25,
    "multiple_predecessor_positive": .07,
    "all_positive": .10,
    "off_target": .002,
}

PATCH = {
    "near_positive": .02,
    "far_positive": .08,
    "one_predecessor_positive": .10,
    "multiple_predecessor_positive": .03,
    "all_positive": .05,
    "off_target": .001,
}


def make_fit():
    arms = torch.ones(len(rung.ARMS), rung.FIT_DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    patches = torch.ones(
        len(rung.CANDIDATES), rung.FIT_DOCUMENTS, len(rung.CELLS), dtype=torch.float64,
    )
    counts = torch.ones(rung.FIT_DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    for ci, cell in enumerate(rung.CELLS):
        arms[rung.ARMS.index("reference"), :, ci] -= STAKES[cell]
        arms[rung.ARMS.index("score"), :, ci] -= .9 * STAKES[cell]
        patches[0, :, ci] -= PATCH[cell]
        patches[1:, :, ci] -= .005
    response = rung._empty_response_stats()
    response["tokens"].fill_(rung.FIT_DOCUMENTS)
    response["ref2"].fill_(100.0)
    response["hyb2"].fill_(81.0)
    response["cross"].fill_(81.0)
    return arms, patches, counts, response


def make_validation():
    arms = torch.ones(
        len(rung.ARMS), rung.VALIDATION_DOCUMENTS, len(rung.CELLS), dtype=torch.float64,
    )
    patches = torch.ones(
        len(rung.PATCH_MODES), rung.VALIDATION_DOCUMENTS, len(rung.CELLS),
        dtype=torch.float64,
    )
    counts = torch.ones(rung.VALIDATION_DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    for ci, cell in enumerate(rung.CELLS):
        arms[rung.ARMS.index("reference"), :, ci] -= STAKES[cell]
        arms[rung.ARMS.index("score"), :, ci] -= .9 * STAKES[cell]
        patches[rung.PATCH_MODES.index("reference_patch"), :, ci] -= PATCH[cell]
        patches[rung.PATCH_MODES.index("hybrid_patch"), :, ci] -= .9 * PATCH[cell]
        patches[rung.PATCH_MODES.index("permuted_patch"), :, ci] -= .01
    return arms, patches, counts


def test_screen_freezes_best_qualifying_candidate():
    screen = rung.screen_candidates(*make_fit())
    assert screen["qualified_count"] == 1
    assert screen["selected"]["candidate"] == "m8"
    assert abs(screen["selected"]["all_positive_recovery"] - .5) < 1e-12


def test_validation_positive_pattern_passes():
    result = rung.analyze_validation(*make_validation())
    assert result["pred_c_heldout_mediation"]
    assert result["pred_d_context_law"]
    assert result["pred_e_alignment_and_transplant"]
    assert not result["strong_science_null"]


def test_context_reversal_fails_context_law():
    arms, patches, counts = make_validation()
    near = rung.CELLS.index("near_positive")
    far = rung.CELLS.index("far_positive")
    patches[:, :, [near, far]] = patches[:, :, [far, near]].clone()
    result = rung.analyze_validation(arms, patches, counts)
    assert not result["pred_d_context_law"]


def test_large_permuted_effect_fails_alignment_control():
    arms, patches, counts = make_validation()
    patches[rung.PATCH_MODES.index("permuted_patch")] = patches[
        rung.PATCH_MODES.index("reference_patch")
    ]
    result = rung.analyze_validation(arms, patches, counts)
    assert not result["pred_e_alignment_and_transplant"]
    assert result["strong_science_null"]
