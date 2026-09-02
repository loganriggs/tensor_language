import torch

import equality_correction_paired_necessity_rung465 as rung


STAKE = {
    "near_positive": .04, "far_positive": .20,
    "one_predecessor_positive": .25, "multiple_predecessor_positive": .06,
    "all_positive": .11, "off_target": .002,
}
CORRECTION = {
    "near_positive": -.04, "far_positive": .03,
    "one_predecessor_positive": .05, "multiple_predecessor_positive": -.035,
    "all_positive": -.01, "off_target": .003,
}


def synthetic():
    full = torch.ones(len(rung.ALL_SOURCES), rung.DOCUMENTS, len(rung.CELLS),
                      dtype=torch.float64)
    removed = torch.ones(len(rung.SOURCES), len(rung.CANDIDATES), rung.DOCUMENTS,
                         len(rung.CELLS), dtype=torch.float64)
    direct = torch.ones(len(rung.SOURCES), rung.DOCUMENTS, len(rung.CELLS),
                        dtype=torch.float64)
    counts = torch.ones(rung.DOCUMENTS, len(rung.CELLS), dtype=torch.float64)
    response = {}
    for ci, cell in enumerate(rung.CELLS):
        correction = CORRECTION[cell]
        for si, source in enumerate(rung.SOURCES):
            source_scale = 1 if source == "N" else .9
            full[rung.ALL_SOURCES.index(source), :, ci] -= STAKE[cell] * source_scale
            direct[si, :, ci] -= (STAKE[cell] - correction) * source_scale
    for ji, site in enumerate(rung.CANDIDATES):
        strength = .35 if site == rung.PRIMARY_SITE else .15 + .005 * ji
        response[site] = {cell: {
            "cosine": .95,
            "native_raw_coordinate_rms": .2 + .01 * (ji % 5),
            "hybrid_raw_coordinate_rms": .21 + .01 * (ji % 5),
            "tokens": 100,
        } for cell in rung.CELLS}
        for ci, cell in enumerate(rung.CELLS):
            correction = CORRECTION[cell]
            for si, source in enumerate(rung.SOURCES):
                source_scale = 1 if source == "N" else .9
                removed[si, ji, :, ci] -= (
                    STAKE[cell] * source_scale - correction * strength * source_scale
                )
    return full, removed, direct, counts, response


def test_shared_primary_and_multisite_pattern_passes():
    result = rung.analyze(*synthetic())
    assert result["pred_b_mlp17_shared_role"]
    assert result["pred_c_mlp17_context_correction"]
    assert result["pred_d_shared_multisite_program"]
    assert result["pred_e_not_raw_amplitude"]
    assert not result["strong_science_null"]


def test_opposite_hybrid_primary_fires_null():
    full, removed, direct, counts, response = synthetic()
    si = rung.SOURCES.index("H")
    ji = rung.CANDIDATES.index(rung.PRIMARY_SITE)
    base = full[rung.ALL_SOURCES.index("0")]
    removed[si, ji] = 2 * full[rung.ALL_SOURCES.index("H")] - removed[si, ji]
    result = rung.analyze(full, removed, direct, counts, response)
    assert not result["pred_b_mlp17_shared_role"]
    assert result["strong_science_null"]


def test_context_vector_order_is_explicit():
    result = rung.analyze(*synthetic())
    vector = result["pooled"]["necessity_vectors"]["N"][rung.PRIMARY_SITE]
    assert vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0


def test_small_primary_fails_registered_floor():
    full, removed, direct, counts, response = synthetic()
    ji = rung.CANDIDATES.index(rung.PRIMARY_SITE)
    for si, source in enumerate(rung.SOURCES):
        removed[si, ji] = full[rung.ALL_SOURCES.index(source)]
    result = rung.analyze(full, removed, direct, counts, response)
    assert not result["pred_b_mlp17_shared_role"]
    assert result["strong_science_null"]
