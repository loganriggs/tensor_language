import torch

import explore_mlp1_sparse_capacity_frontier as subject


def test_price_frontier_exact_and_storage_bounded():
    assert subject.price(512)["stored_reals"] == 2_950_272
    assert subject.price(768)["stored_reals"] == 4_424_832
    assert subject.price(768)["full_mlp_storage_saved_reals"] == 883_584
    assert subject.price(768)["executed_down_multiplies_per_token"] == 3_575_808
    assert subject.price(896)["stored_reals"] == 5_162_112
    assert subject.price(921)["full_mlp_storage_saved_reals"] > 0
    assert subject.price(922)["full_mlp_storage_saved_reals"] < 0


def test_topk_relu_keeps_only_largest_positive_scores():
    scores = torch.arange(-40.0, 40.0).reshape(2, 40)
    result = subject.topk_relu(scores, 4)
    assert (result != 0).sum(dim=1).tolist() == [0, 4]
    assert result[1].nonzero().flatten().tolist() == [36, 37, 38, 39]


def test_program_shapes_and_intercept():
    encoder = torch.zeros(40, subject.GATE_DIM)
    decoder = torch.zeros(subject.OUTPUT_DIM, 40)
    intercept = torch.arange(subject.OUTPUT_DIM, dtype=torch.float32)
    program = subject.SparseProgram(encoder, decoder, intercept)
    output = program(torch.zeros(2, 3, subject.GATE_DIM))
    assert output.shape == (2, 3, subject.OUTPUT_DIM)
    assert torch.equal(output[1, 2], intercept)
