import numpy as np

import score_mlp0_native_down_hierarchy_v1 as score


def synthetic_ledgers(arms, n_documents=12, n_cells=2):
    ledgers = {}
    for arm, effect in arms.items():
        ledgers[arm] = {}
        for consumer, margin in score.MARGINS.items():
            counts = np.ones((n_documents, n_cells))
            sums = counts * margin * effect
            ledgers[arm][consumer] = {"sums": sums.tolist(), "counts": counts.tolist()}
    return ledgers


def test_familywise_bounds_penalize_all_arms_with_one_common_error():
    coordinates = np.array([[.2], [.4]])
    bootstrap = np.array([[[.3], [.3]], [[.1], [.7]], [[.2], [.4]]])
    upper, lower, upper_error, lower_error = score.familywise_bounds(coordinates, bootstrap)
    assert np.isclose(upper_error, .3)
    assert np.isclose(lower_error, .1)
    assert np.allclose(upper, [.5, .7])
    assert np.allclose(lower, [.1, .3])


def test_comparison_uses_paired_maxima_and_pointwise_no_free_rider(monkeypatch):
    monkeypatch.setattr(score, "MINIMUM_DOCUMENTS_PER_CELL", 1)
    ledgers = synthetic_ledgers({"base": .6, "good": .2, "bad": .8})
    scope = score.score_scope(
        ledgers, np.arange(12), {"good_vs_base": ("base", "good"),
                                "bad_vs_base": ("base", "bad")},
        n_bootstrap=20, seed=1,
    )
    good = scope["superiority"]["comparisons"]["good_vs_base"]
    bad = scope["superiority"]["comparisons"]["bad_vs_base"]
    assert good["point_max_reduction"] > 0 and good["candidate_pointwise_no_worse"]
    assert bad["point_max_reduction"] < 0 and not bad["candidate_pointwise_no_worse"]


def test_arm_points_requires_document_cell_support(monkeypatch):
    monkeypatch.setattr(score, "MINIMUM_DOCUMENTS_PER_CELL", 10)
    ledgers = synthetic_ledgers({"arm": .2}, n_documents=8)
    _, _, reports = score.arm_points(ledgers, ["arm"], np.arange(8))
    assert not reports["arm"]["support_passes"]


def test_coordinatewise_centering_catches_switching_near_tie():
    # Arm-level max centering would see errors [0,0].  Coordinatewise centering
    # sees a +0.09 excursion in the cell that was not the point-estimate maximum.
    coordinates = np.array([[.50, .49]])
    bootstrap = np.array([[[.50, .58]], [[.50, .58]], [[.50, .58]]])
    upper, _, correction, _ = score.familywise_bounds(coordinates, bootstrap)
    assert np.isclose(correction, .09)
    assert np.isclose(upper[0], .59)
