from __future__ import annotations

import torch

from tensor_bilin18_program import LAYERS, TensorBilin18Program
from tensor_preserving_attention import (
    PROJECTION_NAMES, StoredLinear, TensorAttentionBank,
    TensorPreservingSquaredAttention,
)
from tensor_preserving_mlp import TensorMLPBank, TensorPreservingBilinearMLP


def dense(weight: torch.Tensor) -> StoredLinear:
    return StoredLinear(weight=weight)


def tiny_program() -> TensorBilin18Program:
    width = 2
    eye = torch.eye(width)
    attentions = []
    mlps = []
    for site in range(LAYERS):
        scale = 0.04 + site * 0.0005
        projections = {
            name: dense(eye * (scale if name in {"v", "proj"} else 1.0))
            for name in PROJECTION_NAMES
        }
        attentions.append(TensorPreservingSquaredAttention(
            projections, lamb=torch.tensor(0.25), inv_freq=torch.tensor([1.0]),
            n_head=1,
        ))
        mlps.append(TensorPreservingBilinearMLP(
            dense(eye * 0.02), dense(eye * 0.03), dense(eye * 0.04),
            torch.tensor([0.001, -0.001]),
        ))
    embedding = torch.tensor([
        [0.7, -0.2], [-0.3, 0.9], [0.4, 0.5], [-0.8, -0.1],
    ])
    unembedding = torch.tensor([
        [0.2, -0.3], [0.4, 0.1], [-0.5, 0.7], [0.8, -0.2], [0.1, 0.6],
    ])
    lambdas = torch.tensor([[1.0, 0.01 * (site % 3)] for site in range(LAYERS)])
    return TensorBilin18Program(
        token_embedding=embedding,
        residual_lambdas=lambdas,
        unembedding=unembedding,
        attention_bank=TensorAttentionBank(attentions),
        mlp_bank=TensorMLPBank(mlps),
    )


def test_complete_program_has_context_and_total_support() -> None:
    program = tiny_program()
    first = torch.tensor([[0, 1, 2]])
    second = torch.tensor([[3, 1, 2]])
    first_logits = program(first)
    second_logits = program(second)
    assert first_logits.shape == (1, 3, 5)
    assert torch.isfinite(first_logits).all()
    assert torch.equal(first[:, 1:], second[:, 1:])
    assert float((first_logits[:, 1:] - second_logits[:, 1:]).abs().max()) > 0
    receipt = program.cost_receipt()
    assert receipt["native_calls_per_forward"] == 0
    assert receipt["fitted_lookup_table_values"] == 0
    assert receipt["total_input_support"] is True
    assert "causal" in receipt["sequence_primitive"]


def test_shell_is_cloned_and_costed_completely() -> None:
    program = tiny_program()
    receipt = program.cost_receipt()
    # attention: 18 * (6 matrices of 4 values + lambda + inv_freq) = 468
    # MLP: 18 * (3 matrices of 4 values + 2 bias values) = 252
    # shell: embedding 8 + lambdas 36 + unembedding 10 = 54
    assert receipt["attention"]["total_stored_values"] == 468
    assert receipt["mlp"]["total_stored_values"] == 252
    assert receipt["shell"]["total_shell_stored_values"] == 54
    assert receipt["total_stored_values"] == 774
    assert receipt["shell"]["parameter_free_rmsnorm_calls"] == 38
    assert receipt["shell"]["parameter_free_softcap_calls"] == 1


def test_operation_receipt_prices_shell_and_tensor_core() -> None:
    program = tiny_program()
    receipt = program.operation_receipt(batch=2, sequence=3)
    assert receipt["embedding_lookups"] == 6
    assert receipt["unembedding_multiply_adds"] == 6 * 10
    assert receipt["residual_scalar_multiplies"] == 2 * 18 * 6 * 2
    assert receipt["residual_additions"] == 3 * 18 * 6 * 2
    assert receipt["whole_state_rmsnorm_elements"] == 38 * 6 * 2
    assert receipt["softcap_elements"] == 6 * 5
    assert receipt["attention_multiply_adds"] > 0
    assert receipt["mlp_bilinear_multiplies"] > 0


def test_token_contract_rejects_wrong_dtype_device_or_support() -> None:
    program = tiny_program()
    for malformed in (
        torch.tensor([1, 2]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[0, 4]]),
    ):
        try:
            program(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed token tensor was accepted")
