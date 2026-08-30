import pytest
import torch

import mlp1_sparse_down_program_v1 as subject


def normalized_state():
    generator = torch.Generator().manual_seed(7)
    encoder = torch.randn(
        subject.DICTIONARY_SIZE, subject.GATE_DIM, generator=generator,
    )
    encoder /= encoder.norm(dim=1, keepdim=True)
    return {
        "encoder": encoder.float(),
        "decoder": torch.randn(
            subject.OUTPUT_DIM, subject.DICTIONARY_SIZE, generator=generator,
        ).float(),
        "intercept": torch.randn(subject.OUTPUT_DIM, generator=generator).float(),
    }


def test_topk_relu_has_exact_support_and_drops_negative_values():
    scores = torch.tensor([[4.0, -9.0, 3.0, 2.0], [-1.0, -2.0, -3.0, -4.0]])
    output = subject.topk_relu(scores, 2)
    assert torch.equal(output[0], torch.tensor([4.0, 0.0, 3.0, 0.0]))
    assert torch.count_nonzero(output[1]) == 0


def test_program_matches_literal_computation_and_preserves_dtype():
    state = normalized_state()
    program = subject.SparseDownProgram(state, "cpu")
    gate = torch.randn(2, 3, subject.GATE_DIM).bfloat16()
    observed = program(gate)
    flat = gate.float().reshape(-1, subject.GATE_DIM)
    expected = (
        subject.topk_relu(flat @ state["encoder"].T) @ state["decoder"].T
        + state["intercept"]
    ).reshape(2, 3, subject.OUTPUT_DIM).bfloat16()
    assert observed.dtype == torch.bfloat16
    assert torch.equal(observed, expected)


def test_state_validation_and_price_are_literal():
    state = normalized_state()
    state["encoder"][0] *= 2
    with pytest.raises(RuntimeError, match="unit norm"):
        subject.validate_state(state)
    price = subject.SparseDownProgram.price()
    assert price["stored_float32_reals"] == 2_950_272
    assert price["native_down_reals"] == 5_308_416
    assert 0.14 < price["fraction_of_native_full_mlp1_storage_saved"] < 0.15
