import torch

import equality_score_correction_interchange_rung464 as rung


STAKE = {
    "near_positive": .04, "far_positive": .20,
    "one_predecessor_positive": .25, "multiple_predecessor_positive": .06,
    "all_positive": .11, "off_target": .002,
}
CORRECTION = {
    "near_positive": -.03, "far_positive": .02,
    "one_predecessor_positive": .03, "multiple_predecessor_positive": -.025,
    "all_positive": -.005, "off_target": .001,
}


def synthetic(interchange=True):
    losses = torch.ones(
        len(rung.SOURCES), len(rung.SOURCES), rung.DOCUMENTS, len(rung.CELLS),
        dtype=torch.float64,
    )
    counts = torch.ones(rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    for ci, cell in enumerate(rung.CELLS):
        stake = STAKE[cell]
        correction = CORRECTION[cell]
        direct = stake - correction
        for si, source in enumerate(rung.SOURCES):
            for wi, donor in enumerate(rung.SOURCES):
                source_effect = 0 if source == "0" else direct * (1 if source == "N" else .95)
                if donor == "0":
                    donor_effect = 0
                elif interchange:
                    donor_effect = correction * (1 if donor == "N" else .95)
                else:
                    donor_effect = correction if donor == source else -correction
                losses[si, wi, :, ci] -= source_effect + donor_effect
    return losses, counts


def test_interchange_pattern_passes_registered_predictions():
    result = rung.analyze(*synthetic())
    assert result["pred_b_common_matched_correction"]
    assert result["pred_c_correction_interchange"]
    assert result["pred_d_crossed_complete_circuits"]
    assert result["pred_e_correction_not_standalone"]
    assert not result["strong_science_null"]


def test_source_specific_corrections_fail_interchange_clause():
    result = rung.analyze(*synthetic(interchange=False))
    assert not result["pred_c_correction_interchange"]


def test_nonpositive_matched_circuit_fires_strong_null():
    losses, counts = synthetic()
    all_i = rung.CELLS.index("all_positive")
    losses[rung.SOURCES.index("H"), rung.SOURCES.index("H"), :, all_i] = 1.1
    result = rung.analyze(losses, counts)
    assert result["strong_science_null"]


def test_correction_vector_order_is_explicit():
    result = rung.analyze(*synthetic())
    vector = result["pooled"]["correction_vectors"]["N"]["N"]
    assert vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0
