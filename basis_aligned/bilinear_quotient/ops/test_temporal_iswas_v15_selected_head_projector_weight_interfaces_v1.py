import torch

import run_temporal_iswas_v15_selected_head_projector_weight_interfaces_v1 as runner


def test_writer_score_is_vector_norm_and_sign_invariant():
    covector = torch.tensor([3.0, 4.0])
    matrix = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
    expected = float((covector @ matrix).norm()) / (
        float(covector.norm()) * float(torch.linalg.matrix_norm(matrix))
    )
    assert runner.writer_score(torch, covector, matrix) == expected
    assert runner.writer_score(torch, -covector, matrix) == expected


def test_normalized_score_is_vector_norm_and_sign_invariant():
    matrix = torch.tensor([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
    vector = torch.tensor([3.0, 4.0])
    assert runner.normalized_score(torch, vector=vector, matrix=matrix) == \
        runner.normalized_score(torch, vector=-vector, matrix=matrix)
