from __future__ import annotations

import dataclasses

import pytest
import torch

import predictive_quotient_v1_interface_proof as proof


def make_transaction(*, malicious_physical: bool = False):
    torch.manual_seed(20260828)
    producer = torch.nn.Linear(5, 4, bias=True).double()
    inputs = torch.randn(2, 3, 5, dtype=torch.float64)
    code = producer(inputs)
    physical_matrix = torch.randn(4, 4, dtype=torch.float64)
    parent_matrix = torch.randn(4, 4, dtype=torch.float64)
    captured_code = code

    def physical(interface: torch.Tensor) -> torch.Tensor:
        source = captured_code if malicious_physical else interface
        return source @ physical_matrix

    def parent(interface: torch.Tensor) -> torch.Tensor:
        return torch.roll(interface, shifts=1, dims=1) @ parent_matrix

    def suffix(physical_write: torch.Tensor, parent_read: torch.Tensor) -> torch.Tensor:
        return torch.tanh(physical_write + 0.7 * parent_read)

    transaction = proof.SealedInterfaceProofTransaction(
        code, expected_shape=(2, 3, 4), physical_consumer=physical,
        parent_consumer=parent, suffix_consumer=suffix,
        protected_parameters=producer.parameters(),
    )
    return producer, code, transaction


def test_post_producer_leaf_is_exact_connected_and_producer_is_untouched() -> None:
    producer, code, transaction = make_transaction()
    producer.weight.grad = torch.randn_like(producer.weight)
    producer.bias.grad = torch.randn_like(producer.bias)
    parameter_snapshots = tuple(
        (parameter.detach().clone(), parameter.grad.detach().clone())
        for parameter in producer.parameters()
    )
    receipt = transaction.consume()
    assert receipt.code_sha256 == receipt.interface_sha256
    assert receipt.exact_numerical_identity and receipt.interface_is_leaf
    assert receipt.producer_graph_disconnected
    assert receipt.physical_path_gradient_norm > 0
    assert receipt.parent_path_gradient_norm > 0
    assert receipt.suffix_gradient_norm > 0
    assert receipt.protected_parameter_count == 2
    assert receipt.protected_parameters_unchanged
    assert receipt.graph_aliases_revoked and transaction.graph_aliases_revoked
    for parameter, (value, gradient) in zip(producer.parameters(), parameter_snapshots):
        torch.testing.assert_close(parameter, value)
        torch.testing.assert_close(parameter.grad, gradient)
    assert not code.retains_grad
    assert all(not torch.is_tensor(value) for value in dataclasses.asdict(receipt).values())


def test_transaction_is_one_use_and_revokes_on_success() -> None:
    _, _, transaction = make_transaction()
    transaction.consume()
    with pytest.raises(RuntimeError, match="spent"):
        transaction.consume()


def test_consumer_that_bypasses_interface_fails_and_revokes() -> None:
    _, _, transaction = make_transaction(malicious_physical=True)
    with pytest.raises(RuntimeError, match="physical and parent reads"):
        transaction.consume()
    assert transaction.closed and transaction.graph_aliases_revoked


def test_wrong_shape_and_non_graph_code_fail_before_transaction_exists() -> None:
    consumer = lambda value: value
    with pytest.raises(ValueError, match="exact shape"):
        proof.SealedInterfaceProofTransaction(
            torch.ones(2, 3, 4, requires_grad=True), expected_shape=(2, 4, 4),
            physical_consumer=consumer, parent_consumer=consumer,
            suffix_consumer=lambda left, right: left + right,
        )
    with pytest.raises(ValueError, match="graph-bearing"):
        proof.SealedInterfaceProofTransaction(
            torch.ones(2, 3, 4), expected_shape=(2, 3, 4),
            physical_consumer=consumer, parent_consumer=consumer,
            suffix_consumer=lambda left, right: left + right,
        )


def test_suffix_failure_revokes_every_alias() -> None:
    _, code, transaction = make_transaction()
    transaction._SealedInterfaceProofTransaction__suffix_consumer = (
        lambda physical, parent: torch.full_like(physical, float("nan"))
    )
    with pytest.raises(RuntimeError, match="disconnected or nonfinite"):
        transaction.consume()
    assert transaction.graph_aliases_revoked
    assert not code.retains_grad
