from __future__ import annotations

import pytest
import torch

import tensor_bilin18_global_gate_intervention as gate
from test_tensor_bilin18_program import tiny_program


def tiny_transaction(program=None, tokens=None):
    program = tiny_program() if program is None else program
    tokens = torch.tensor([[0, 1, 2], [1, 2, 3]]) if tokens is None else tokens
    return gate.GlobalGateResponseTransaction(
        program=program, tokens=tokens, row_ids=("r0", "r1"),
        first_probe_seeds=(11, 12), second_probe_seeds=(21, 22),
        score_start=1, score_stop=3, source_site=1, production=False,
    )


def test_all_one_global_gate_scale_exactly_replays_program() -> None:
    program = tiny_program()
    tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])
    expected = program(tokens)
    observed, alpha, receipt = gate.forward_with_global_gate_scale_leaf(
        program, tokens, source_site=1,
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert alpha.shape == (2, 2) and alpha.is_leaf and alpha.requires_grad
    assert receipt["scale_shared_across_positions"] is True
    assert receipt["context_scales_independent"] is True


def test_batched_gate_responses_equal_separate_context_transactions() -> None:
    program = tiny_program()
    tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])
    batch = tiny_transaction(program, tokens).consume()
    separate = []
    for index, row in enumerate(("r0", "r1")):
        transaction = gate.GlobalGateResponseTransaction(
            program=program, tokens=tokens[index:index + 1], row_ids=(row,),
            first_probe_seeds=(11, 12), second_probe_seeds=(21, 22),
            score_start=1, score_stop=3, source_site=1, production=False,
        )
        separate.append(transaction.consume())
    for index in range(2):
        torch.testing.assert_close(batch.first[index], separate[index].first[0])
        torch.testing.assert_close(batch.second[index], separate[index].second[0])
    assert batch.receipt["response_shape_per_half"] == [2, 2, 2]
    assert batch.receipt["all_token_positions_share_each_gate_scale"] is True


def test_shared_gate_derivative_equals_sum_of_position_specific_derivatives() -> None:
    program = tiny_program()
    tokens = torch.tensor([[0, 1, 2]])
    shared_logits, shared_alpha, _ = gate.forward_with_global_gate_scale_leaf(
        program, tokens, source_site=1,
    )
    shared_score = shared_logits[:, 1:, 0].sum()
    shared_gradient = torch.autograd.grad(shared_score, shared_alpha)[0]

    positions = []
    for position in range(tokens.shape[1]):
        alpha = torch.ones(1, 2, requires_grad=True)
        if position:
            marker = torch.cat((
                torch.ones(1, position, 2), alpha[:, None, :],
                torch.ones(1, tokens.shape[1] - position - 1, 2),
            ), dim=1)
        else:
            marker = torch.cat((
                alpha[:, None, :], torch.ones(1, tokens.shape[1] - 1, 2),
            ), dim=1)

        original = program.mlp_bank.programs[1]
        state = torch.nn.functional.embedding(tokens, program.token_embedding)
        state = torch.nn.functional.rms_norm(state, (program.width,))
        initial = state
        first_value = None
        for site in range(18):
            lambdas = program.residual_lambdas[site].to(state.dtype)
            state = lambdas[0] * state + lambdas[1] * initial
            attention_state = torch.nn.functional.rms_norm(state, (program.width,))
            attention_write, first_value = program.attention_bank.programs[site](
                attention_state, first_value,
            )
            state = state + attention_write
            mlp_state = torch.nn.functional.rms_norm(state, (program.width,))
            if site == 1:
                product = original.left(mlp_state) * original.right(mlp_state)
                write = original.down(product * marker) + original.down_bias
            else:
                write = program.mlp_bank.programs[site](mlp_state)
            state = state + write
        state = torch.nn.functional.rms_norm(state, (program.width,))
        logits = torch.nn.functional.linear(state, program.unembedding)
        logits = (30.0 * torch.tanh(logits / 30.0)).float()
        positions.append(torch.autograd.grad(logits[:, 1:, 0].sum(), alpha)[0])
    torch.testing.assert_close(shared_gradient, torch.stack(positions).sum(dim=0))


def test_transaction_revokes_aliases_and_is_one_use() -> None:
    transaction = tiny_transaction()
    result = transaction.consume()
    assert transaction.aliases_revoked and result.receipt["graph_aliases_revoked"]
    with pytest.raises(RuntimeError, match="spent"):
        transaction.consume()


def test_transaction_rejects_overlapping_probe_halves() -> None:
    program = tiny_program()
    with pytest.raises(ValueError, match="disjoint"):
        gate.GlobalGateResponseTransaction(
            program=program, tokens=torch.tensor([[0, 1, 2]]), row_ids=("r0",),
            first_probe_seeds=(11, 12), second_probe_seeds=(12, 22),
            score_start=1, score_stop=3, source_site=1, production=False,
        )


def test_global_gate_forward_rejects_invalid_site_before_indexing_bank() -> None:
    with pytest.raises(ValueError, match="outside"):
        gate.forward_with_global_gate_scale_leaf(
            tiny_program(), torch.tensor([[0, 1, 2]]), source_site=18,
        )
