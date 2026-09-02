import torch

import equality_score_code_context_rung461 as rung


def synthetic_inputs():
    cell_values = {
        "near_positive": (.10, .08, 1.0),
        "far_positive": (.20, .18, 2.0),
        "one_predecessor_positive": (.30, .27, 3.0),
        "multiple_predecessor_positive": (.15, .12, 1.5),
        "all_positive": (.12, .10, 1.2),
        "off_target": (.01, .005, .5),
    }
    losses = torch.zeros(len(rung.ARMS), rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    counts = torch.ones(rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    response = rung._empty_response_stats()
    halves = [rung._empty_response_stats(), rung._empty_response_stats()]
    for ci, cell in enumerate(rung.CELLS):
        stake, effect, size = cell_values[cell]
        losses[rung.ARMS.index("base"), :, ci] = 1.0
        losses[rung.ARMS.index("reference"), :, ci] = 1.0 - stake
        losses[rung.ARMS.index("score"), :, ci] = 1.0 - effect
        for stats, tokens in ((response, rung.DOCUMENTS), (halves[0], 96), (halves[1], 96)):
            ref2 = size ** 2 * tokens * rung.D_MODEL
            hyb2 = (.9 * size) ** 2 * tokens * rung.D_MODEL
            stats["ref2"][ci] = ref2
            stats["hyb2"][ci] = hyb2
            stats["cross"][ci] = .9 * (ref2 * hyb2) ** .5
            stats["tokens"][ci] = tokens
    return response, halves, losses, counts


def test_registered_positive_pattern_passes_analysis():
    analysis = rung.analyze(*synthetic_inputs())
    assert analysis["pred_b_context_order"]
    assert analysis["pred_c_causal_tracking"]
    assert analysis["pred_d_shared_direction"]
    assert analysis["pred_e_amplitude_explanation"]
    assert not analysis["strong_science_null"]


def test_raw_coordinate_rms_has_registered_units():
    response, _, _, _ = synthetic_inputs()
    report = rung.response_report(response)
    assert abs(report["one_predecessor_positive"]["reference_raw_coordinate_rms"] - 3.0) < 1e-12
    assert abs(report["far_positive"]["hybrid_raw_coordinate_rms"] - 1.8) < 1e-12


def test_wrong_context_order_fails_primary_prediction():
    response, halves, losses, counts = synthetic_inputs()
    near = rung.CELLS.index("near_positive")
    far = rung.CELLS.index("far_positive")
    losses[:, :, [near, far]] = losses[:, :, [far, near]].clone()
    analysis = rung.analyze(response, halves, losses, counts)
    assert not analysis["pred_b_context_order"]


def test_nonpositive_subgroup_effect_fires_science_null():
    response, halves, losses, counts = synthetic_inputs()
    ci = rung.CELLS.index("far_positive")
    losses[rung.ARMS.index("score"), :, ci] = 1.05
    analysis = rung.analyze(response, halves, losses, counts)
    assert analysis["strong_science_null"]
    assert not analysis["pred_c_causal_tracking"]
