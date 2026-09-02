import torch

import equality_score_path_factorial_rung463 as rung


STAKES = {
    "near_positive": .05,
    "far_positive": .20,
    "one_predecessor_positive": .25,
    "multiple_predecessor_positive": .07,
    "all_positive": .10,
    "off_target": .002,
}


def synthetic_paths(direct_recovery=.70, mediated_recovery=.30):
    base_ref = torch.ones(
        len(rung.BASE_ARMS), rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64,
    )
    direct = torch.ones(rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    mlp = torch.ones_like(direct)
    attention = torch.ones_like(direct)
    suffix = torch.ones(
        len(rung.CANDIDATES), rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64,
    )
    counts = torch.ones(rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    for ci, cell in enumerate(rung.CELLS):
        stake = STAKES[cell]
        base_ref[rung.BASE_ARMS.index("reference"), :, ci] -= stake
        direct[:, ci] -= direct_recovery * stake
        mlp[:, ci] -= .20 * stake
        attention[:, ci] -= .05 * stake
        for i in range(len(rung.CANDIDATES)):
            patched_count = len(rung.CANDIDATES) - i
            recovery = .01 + (mediated_recovery - .01) * (
                (patched_count - 1) / (len(rung.CANDIDATES) - 1)
            )
            suffix[i, :, ci] -= recovery * stake
    return base_ref, direct, mlp, attention, suffix, counts


def test_full_path_pattern_passes_registered_predictions():
    result = rung.analyze(*synthetic_paths())
    assert result["pred_b_direct_route"]
    assert result["pred_c_cumulative_suffix"]
    assert result["pred_d_mlp_over_attention"]
    assert result["pred_e_dominant_context_law"]
    assert result["route_classification"] == "both_direct_and_distributed"
    assert not result["strong_science_null"]


def test_suffix_curve_is_ranked_by_patched_write_count():
    result = rung.analyze(*synthetic_paths())
    curve = result["suffix_curve"]
    assert curve["patched_count_vs_recovery_spearman"] > .99
    assert all(row["recovery_increment"] > 0 for row in curve["ascending_adjacent_increments"])


def test_direct_only_branch_is_classified_without_forcing_suffix_pass():
    result = rung.analyze(*synthetic_paths(direct_recovery=.75, mediated_recovery=.05))
    assert result["pred_b_direct_route"]
    assert not result["pred_c_cumulative_suffix"]
    assert result["route_classification"] == "mainly_direct_residual"
    assert not result["strong_science_null"]


def test_two_weak_routes_fire_strong_null():
    result = rung.analyze(*synthetic_paths(direct_recovery=.05, mediated_recovery=.04))
    assert result["strong_science_null"]
    assert not result["pred_b_direct_route"]
    assert not result["pred_c_cumulative_suffix"]
