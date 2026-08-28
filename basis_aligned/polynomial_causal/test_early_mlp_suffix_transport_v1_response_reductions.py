import pytest
import torch

import early_mlp_suffix_transport_v1_response_reductions as reductions


UNIT = "a" * 64


def _triplet(base, positive, negative):
    return reductions.ResponseTriplet(
        baseline=torch.tensor(base, dtype=torch.float64),
        positive=torch.tensor(positive, dtype=torch.float64),
        negative=torch.tensor(negative, dtype=torch.float64),
    )


def _broadcast_triplet(base, positive, negative, *, width):
    def expand(value):
        return torch.tensor(value, dtype=torch.float64).reshape(1, 1, width).expand(
            4, 192, width,
        ).clone()
    return reductions.ResponseTriplet(expand(base), expand(positive), expand(negative))


def test_vector_reduction_uses_both_edited_minus_own_baseline_occurrences():
    teacher = _broadcast_triplet(
        [10.0, 20.0], [11.0, 22.0], [7.0, 24.0], width=2,
    )
    student = _broadcast_triplet(
        [-5.0, 8.0], [-3.0, 9.0], [-9.0, 10.0], width=2,
    )
    result = reductions._reduce_vector_response(
        teacher=teacher, student=student, unit_identity=UNIT,
    )
    # teacher responses [1,2],[-3,4]; student [2,1],[-4,2]
    scale = 192
    assert result.teacher_sum.tolist() == [30.0 * scale] * 4
    assert result.student_sum.tolist() == [25.0 * scale] * 4
    assert result.dot_sum.tolist() == [24.0 * scale] * 4
    assert result.error_sum.tolist() == [7.0 * scale] * 4


def test_centered_logit_reduction_removes_only_per_occurrence_constant_shift():
    teacher = _broadcast_triplet(
        [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], width=3,
    )
    student = _broadcast_triplet(
        [9.0, 9.0, 9.0], [15.0, 16.0, 17.0], [3.0, 2.0, 1.0], width=3,
    )
    centered = reductions.reduce_centered_logit_response(
        teacher=teacher, student=student, unit_identity=UNIT,
    )
    assert centered.error_sum.tolist() == pytest.approx([0.0] * 4, abs=1e-12)
    assert centered.teacher_sum[0].item() == pytest.approx(4.0 * 192)
    uncentered = reductions._reduce_vector_response(
        teacher=teacher, student=student, unit_identity=UNIT,
    )
    assert bool((uncentered.error_sum > 100.0).all())


def test_output_kl_matches_registered_reference_and_sums_both_signs():
    teacher = _broadcast_triplet(
        [0.0, 0.0], [2.0, 0.0], [0.0, 2.0], width=2,
    )
    student = _broadcast_triplet(
        [100.0, -100.0], [1.0, 0.0], [0.0, 1.0], width=2,
    )
    result = reductions.reduce_output_kl_response(
        teacher=teacher, student=student, unit_identity=UNIT,
    )
    def kl(reference, candidate):
        lp = torch.log_softmax(torch.tensor(reference, dtype=torch.float64), -1)
        lq = torch.log_softmax(torch.tensor(candidate, dtype=torch.float64), -1)
        return torch.sum(lp.exp() * (lp - lq))
    expected_numerator = kl([2.0, 0.0], [1.0, 0.0]) + kl([0.0, 2.0], [0.0, 1.0])
    expected_denominator = kl([2.0, 0.0], [0.0, 0.0]) + kl([0.0, 2.0], [0.0, 0.0])
    assert result.numerator_sum[0].item() == pytest.approx(float(expected_numerator) * 192)
    assert result.denominator_sum[0].item() == pytest.approx(float(expected_denominator) * 192)


def test_reduction_rejects_graphs_shape_mismatch_and_bad_identity():
    graph = torch.ones(1, 2, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        reductions.ResponseTriplet(graph, graph.detach(), graph.detach())
    teacher = _triplet([[0.0, 0.0]], [[1.0, 0.0]], [[-1.0, 0.0]])
    student = _triplet([[0.0]], [[1.0]], [[-1.0]])
    with pytest.raises(ValueError, match="supports differ"):
        reductions._reduce_vector_response(
            teacher=teacher, student=student, unit_identity=UNIT,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        reductions._reduce_vector_response(
            teacher=teacher, student=teacher, unit_identity="bad",
        )


def test_reduction_outputs_are_cpu_float64_copies():
    teacher = _broadcast_triplet(
        [0.0] * 64, [1.0] + [0.0] * 63, [-1.0] + [0.0] * 63, width=64,
    )
    result = reductions.reduce_code_response(
        teacher=teacher, student=teacher, unit_identity=UNIT,
    )
    assert result.teacher_sum.device.type == "cpu"
    assert result.teacher_sum.dtype == torch.float64
    teacher.positive.fill_(99.0)
    assert result.teacher_sum[0].item() == pytest.approx(384.0)


def test_public_reducers_freeze_all_scored_positions_and_modalities():
    wrong = _triplet([[0.0] * 64], [[1.0] * 64], [[-1.0] * 64])
    with pytest.raises(ValueError, match=r"\[4,192,64\]"):
        reductions.reduce_code_response(
            teacher=wrong, student=wrong, unit_identity=UNIT,
        )
    with pytest.raises(ValueError, match=r"\[4,192,vocab\]"):
        reductions.reduce_centered_logit_response(
            teacher=wrong, student=wrong, unit_identity=UNIT,
        )
