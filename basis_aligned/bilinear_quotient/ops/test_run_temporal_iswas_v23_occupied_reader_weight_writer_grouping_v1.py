import numpy as np

import run_temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1 as experiment


def test_projector_overlap_extremes():
    a = np.eye(5)[:, :2]
    b = np.eye(5)[:, 2:4]
    assert experiment.projector_overlap(a, a) == 1
    assert experiment.projector_overlap(a, b) == 0


def test_pairwise_is_symmetric_and_excludes_diagonal_from_pairs():
    matrix, pairs = experiment.pairwise([np.array([1., 0.]), np.array([0., 1.])], lambda a, b: a @ b)
    assert np.array_equal(matrix, np.eye(2))
    assert pairs == [0]


def test_zero_forward_price():
    assert set(experiment.PRICE.values()) == {0}
