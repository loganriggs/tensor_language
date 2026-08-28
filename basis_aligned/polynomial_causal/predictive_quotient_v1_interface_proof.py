"""CPU-only proof harness for the predictive-quotient MLP0 interface.

This is not a model adapter, role launcher, or numerical authority.  It proves the
graph semantics required before the source-closed adapter is changed: a numerically
identical post-producer leaf must feed both physical and parent reads, downstream
outputs must differentiate through it, producer parameters must remain untouched,
and graph-bearing aliases must be revoked on every exit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Iterable

import torch


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _gradient_snapshot(parameter: torch.Tensor) -> tuple[str, str | None]:
    return (
        _tensor_sha256(parameter),
        None if parameter.grad is None else _tensor_sha256(parameter.grad),
    )


@dataclass(frozen=True)
class InterfaceProofReceipt:
    """Tensor-free evidence emitted after every graph-bearing alias is revoked."""

    code_sha256: str
    interface_sha256: str
    exact_numerical_identity: bool
    interface_is_leaf: bool
    producer_graph_disconnected: bool
    physical_path_gradient_norm: float
    parent_path_gradient_norm: float
    suffix_gradient_norm: float
    protected_parameter_count: int
    protected_parameters_unchanged: bool
    graph_aliases_revoked: bool

    def __post_init__(self) -> None:
        hashes = (self.code_sha256, self.interface_sha256)
        if any(len(value) != 64 for value in hashes) or not all((
            self.exact_numerical_identity,
            self.interface_is_leaf,
            self.producer_graph_disconnected,
            self.protected_parameters_unchanged,
            self.graph_aliases_revoked,
        )) or min(
            self.physical_path_gradient_norm,
            self.parent_path_gradient_norm,
            self.suffix_gradient_norm,
        ) <= 0 or self.protected_parameter_count < 0:
            raise ValueError("predictive-quotient interface proof did not close")


class SealedInterfaceProofTransaction:
    """One-use fake transaction exercising the proposed production graph boundary."""

    def __init__(
        self,
        predicted_code: torch.Tensor,
        *,
        expected_shape: tuple[int, int, int],
        physical_consumer: Callable[[torch.Tensor], torch.Tensor],
        parent_consumer: Callable[[torch.Tensor], torch.Tensor],
        suffix_consumer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        protected_parameters: Iterable[torch.Tensor] = (),
    ) -> None:
        if not torch.is_tensor(predicted_code) or not predicted_code.requires_grad or (
            tuple(predicted_code.shape) != expected_shape
        ) or not predicted_code.is_floating_point() or not bool(
            torch.isfinite(predicted_code.detach()).all()
        ):
            raise ValueError("predicted code must be a finite graph-bearing tensor of exact shape")
        if any(not callable(value) for value in (
            physical_consumer, parent_consumer, suffix_consumer,
        )):
            raise TypeError("all quotient interface consumers must be callable")
        parameters = tuple(protected_parameters)
        if any(not torch.is_tensor(parameter) for parameter in parameters):
            raise TypeError("protected parameters must be tensors")
        self.__predicted_code = predicted_code
        self.__expected_shape = expected_shape
        self.__physical_consumer = physical_consumer
        self.__parent_consumer = parent_consumer
        self.__suffix_consumer = suffix_consumer
        self.__protected_parameters = parameters
        self.__closed = False

    @property
    def closed(self) -> bool:
        return self.__closed

    @property
    def graph_aliases_revoked(self) -> bool:
        return self.__closed and all(getattr(self, name) is None for name in (
            "_SealedInterfaceProofTransaction__predicted_code",
            "_SealedInterfaceProofTransaction__physical_consumer",
            "_SealedInterfaceProofTransaction__parent_consumer",
            "_SealedInterfaceProofTransaction__suffix_consumer",
            "_SealedInterfaceProofTransaction__protected_parameters",
        ))

    def _revoke(self) -> None:
        self.__predicted_code = None
        self.__physical_consumer = None
        self.__parent_consumer = None
        self.__suffix_consumer = None
        self.__protected_parameters = None
        self.__closed = True

    def consume(self) -> InterfaceProofReceipt:
        if self.__closed:
            raise RuntimeError("predictive-quotient interface transaction is spent")
        code = self.__predicted_code
        physical_consumer = self.__physical_consumer
        parent_consumer = self.__parent_consumer
        suffix_consumer = self.__suffix_consumer
        parameters = self.__protected_parameters
        assert code is not None and physical_consumer is not None and (
            parent_consumer is not None and suffix_consumer is not None and parameters is not None
        )
        before = tuple(_gradient_snapshot(parameter) for parameter in parameters)
        try:
            interface = code.detach().requires_grad_(True)
            code_hash = _tensor_sha256(code)
            interface_hash = _tensor_sha256(interface)
            exact_identity = bool(torch.equal(code.detach(), interface.detach()))
            is_leaf = interface.is_leaf and interface.grad_fn is None

            physical = physical_consumer(interface)
            parent = parent_consumer(interface)
            if any(not torch.is_tensor(value) or not value.requires_grad or not bool(
                torch.isfinite(value).all()
            ) for value in (physical, parent)):
                raise RuntimeError("physical and parent reads must both consume the interface leaf")
            suffix = suffix_consumer(physical, parent)
            if not torch.is_tensor(suffix) or not suffix.requires_grad or not bool(
                torch.isfinite(suffix).all()
            ):
                raise RuntimeError("suffix output is disconnected or nonfinite")

            physical_gradient, = torch.autograd.grad(
                physical.square().sum(), interface, retain_graph=True, allow_unused=True,
            )
            parent_gradient, = torch.autograd.grad(
                parent.square().sum(), interface, retain_graph=True, allow_unused=True,
            )
            suffix_gradient, = torch.autograd.grad(
                suffix.square().sum(), interface, retain_graph=True, allow_unused=True,
            )
            if physical_gradient is None or parent_gradient is None:
                raise RuntimeError("physical and parent reads must both consume the interface leaf")
            if suffix_gradient is None:
                raise RuntimeError("suffix output must consume the interface leaf")
            source_gradient, = torch.autograd.grad(
                suffix.square().sum(), code, allow_unused=True,
            )
            norms = tuple(float(torch.linalg.vector_norm(value)) for value in (
                physical_gradient, parent_gradient, suffix_gradient,
            ))
            after = tuple(_gradient_snapshot(parameter) for parameter in parameters)
            unchanged = before == after
            receipt_arguments = dict(
                code_sha256=code_hash,
                interface_sha256=interface_hash,
                exact_numerical_identity=exact_identity,
                interface_is_leaf=is_leaf,
                producer_graph_disconnected=source_gradient is None,
                physical_path_gradient_norm=norms[0],
                parent_path_gradient_norm=norms[1],
                suffix_gradient_norm=norms[2],
                protected_parameter_count=len(parameters),
                protected_parameters_unchanged=unchanged,
            )
        finally:
            # Production must clear a larger alias set; this proves the one-use pattern
            # and fail-closed behavior without opening any model or role.
            self._revoke()
        return InterfaceProofReceipt(
            **receipt_arguments, graph_aliases_revoked=self.graph_aliases_revoked,
        )
