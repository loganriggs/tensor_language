import numpy as np

import equality_task_reader_commutant_rung479 as subject


def test_commutant_finds_hidden_two_block_direction():
    rng = np.random.default_rng(479)
    rotation = np.linalg.qr(rng.standard_normal((subject.OBSERVATION_DIM,
                                                 subject.OBSERVATION_DIM)))[0]
    matrices = []
    for _ in range(8):
        raw = np.zeros((subject.OBSERVATION_DIM, subject.OBSERVATION_DIM))
        left = rng.standard_normal((11, 11)); left = (left + left.T) / 2
        right = rng.standard_normal((21, 21)); right = (right + right.T) / 2
        raw[:11, :11], raw[11:, 11:] = left, right
        matrices.append(rotation @ raw @ rotation.T)
    result = subject.approximate_commutant(np.stack(matrices))
    assert sorted(result["block_sizes"]) == [11, 21]
    offblock = subject.offblock_summary(matrices, result["projectors"])
    assert offblock["maximum"] < 1e-8


def test_generic_family_has_larger_second_commutant_value():
    rng = np.random.default_rng(480)
    generic = rng.standard_normal((20, subject.OBSERVATION_DIM,
                                   subject.OBSERVATION_DIM))
    generic = (generic + generic.transpose(0, 2, 1)) / 2
    result = subject.approximate_commutant(generic)
    assert result["lambda2"] > 1e-5
    assert abs(result["scalar_residual"]) < 1e-8


def test_profile_report_is_invariant_to_within_block_rotation():
    rng = np.random.default_rng(481)
    n = subject.OBSERVATION_DIM
    projector = np.zeros((n, n)); projector[:10, :10] = np.eye(10)
    matrices = rng.standard_normal((2, 2, 2, n, n, n))
    matrices = (matrices + matrices.transpose(0, 1, 2, 3, 5, 4)) / 2
    tags = [f"r.{2 * (index % 6)}.x{index}" for index in range(n)]
    before = subject.profile_report(matrices, (projector, np.eye(n) - projector), tags)
    local = np.linalg.qr(rng.standard_normal((10, 10)))[0]
    rotation = np.eye(n); rotation[:10, :10] = local
    rotated = np.einsum("ab,hskcbd,de->hskcae", rotation.T, matrices, rotation)
    after = subject.profile_report(rotated, (projector, np.eye(n) - projector), tags)
    assert np.allclose([row["minimum_view_cosine"] for row in before],
                       [row["minimum_view_cosine"] for row in after])


def test_cosine_identity():
    vector = np.arange(1, 9, dtype=float)
    assert subject._cosine(vector, 3 * vector) > .999999
