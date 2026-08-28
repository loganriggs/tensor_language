"""Pure differentiable runtime primitives for suffix-transport v1.

This module deliberately has no row, model-loader, artifact, or scoring access.  It
owns the one affine implementation shared by the L/R/S/T routes, the physical
projected replacement, deterministic fit permutations, and the two registered
training losses.  Model-specific orchestration must call these primitives rather
than defining route-specific forward paths.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import early_mlp_suffix_transport_v1 as contract


D_MODEL = contract.D_MODEL
CODE_DIM = contract.CODE_DIM
LEARNING_RATES = (1e-5, 3e-5, 1e-4)
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_CLIP_NORM = 1.0
SEQUENCE_LENGTH = 256
SCORE_START = 64
SCORE_STOP = 256


def _sha256_text(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def tensor_identity_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor identity requires a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def logical_identity_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


PREREGISTRATION_SHA256 = "11577380d65c813cf9e80e92002de9569928d293747c278c065939b3f3b24193"
IMPLEMENTATION_AMENDMENT_SHA256 = (
    "f4d019352c9443cbbea3f1f78a025fa94e0ba51c5c3a91e33d81a141b0c6e4a7"
)


@dataclass(frozen=True)
class TraceIdentity:
    """Exact fit-step identity shared by student and one named teacher route."""

    schema_version: int
    protocol_sha256: str
    implementation_amendment_sha256: str
    source_commit: str
    inherited_snapshot_sha256: str
    rows_receipt_sha256: str
    fit_role_tensor_sha256: str
    ordered_batch_indices_sha256: str
    ordered_input_tokens_sha256: str
    program_snapshot_sha256: str
    teacher_mapping_sha256: str
    role: str
    phase: str
    route: str
    control: str
    teacher_kind: str
    trial: int
    epoch: int
    optimizer_step: int
    batch_ordinal: int
    student_states: tuple[tuple[int, str], ...]
    batch_rows: int
    sequence_length: int
    score_start: int
    score_stop: int
    code_dim: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.schema_version, self.trial, self.epoch, self.optimizer_step,
            self.batch_ordinal, self.batch_rows, self.sequence_length,
            self.score_start, self.score_stop, self.code_dim,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise ValueError("trace integer identity fields are malformed")
        if self.schema_version != 1 or self.protocol_sha256 != PREREGISTRATION_SHA256 or (
            self.implementation_amendment_sha256 != IMPLEMENTATION_AMENDMENT_SHA256
        ):
            raise ValueError("trace protocol identity changed")
        if not isinstance(self.source_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.source_commit,
        ):
            raise ValueError("trace source commit is malformed")
        for name, value in (
            ("inherited snapshot", self.inherited_snapshot_sha256),
            ("rows receipt", self.rows_receipt_sha256),
            ("fit role", self.fit_role_tensor_sha256),
            ("batch indices", self.ordered_batch_indices_sha256),
            ("input tokens", self.ordered_input_tokens_sha256),
            ("program snapshot", self.program_snapshot_sha256),
            ("teacher mapping", self.teacher_mapping_sha256),
        ):
            if not _sha256_text(value):
                raise ValueError(f"trace {name} hash is malformed")
        if self.role not in {
            "early_mlp_suffix_transport_v1_fit",
            "early_mlp_suffix_transport_v1_validation",
            "early_mlp_suffix_transport_v1_final",
        }:
            raise ValueError("trace role is not licensed for fit, validation, or final")
        if self.phase not in {
            "initial_denominator", "fit", "validation", "final",
        } or self.route not in {
            "Q", "L", "R", "S0", "S1", "T",
        } or self.teacher_kind not in {"coordinate_labels", "oon_logits"}:
            raise ValueError("trace phase/route/teacher identity is unknown")
        allowed_controls = {"true", "document_shuffle", "zero_A"} | {
            f"A_null_{index:02d}" for index in range(20)
        } | {
            "inherited_q", "hybrid_s0_l1", "hybrid_l0_s1",
            "hybrid_r0_l1", "hybrid_l0_r1", "new_fit_mean",
        }
        if self.control not in allowed_controls:
            raise ValueError("trace control identity is unknown")
        legal_fit = (
            self.phase == "initial_denominator" and self.route == "Q"
            and self.control == "true" and self.teacher_kind == "coordinate_labels"
        ) or (
            self.phase == "fit" and self.route == "L"
            and self.control in {"true", "document_shuffle"}
            and self.teacher_kind == "coordinate_labels"
        ) or (
            self.phase == "fit" and self.route in {"R", "S0", "S1"}
            and self.control in {"true", "document_shuffle"}
            and self.teacher_kind == "oon_logits"
        ) or (
            self.phase == "fit" and self.route == "T"
            and (self.control == "true" or self.control.startswith("A_null_"))
            and self.teacher_kind == "oon_logits"
        )
        legal_validation_control = self.control == "true" or (
            self.control == "document_shuffle" and self.route in {"L", "R", "S0", "S1"}
        ) or (self.control.startswith("A_null_") and self.route == "T")
        legal_validation = self.phase == "validation" and self.route in {
            "L", "R", "S0", "S1", "T",
        } and legal_validation_control and self.teacher_kind == (
            "coordinate_labels" if self.route == "L" else "oon_logits"
        )
        legal_final_action_control = legal_validation_control or (
            self.route == "L" and self.control in {"inherited_q", "new_fit_mean"}
        ) or (
            self.route == "T" and self.control == "zero_A"
        ) or (
            self.route == "S0" and self.control == "hybrid_s0_l1"
        ) or (
            self.route == "S1" and self.control == "hybrid_l0_s1"
        ) or (
            self.route == "R" and self.control in {
                "hybrid_r0_l1", "hybrid_l0_r1",
            }
        )
        legal_final = self.phase == "final" and self.route in {
            "L", "R", "S0", "S1", "T",
        } and legal_final_action_control and self.teacher_kind == (
            "coordinate_labels" if self.route == "L" else "oon_logits"
        )
        if not (legal_fit or legal_validation or legal_final):
            raise ValueError("trace phase/route/control/teacher combination is illegal")
        required_role = (
            "early_mlp_suffix_transport_v1_fit"
            if self.phase in {"initial_denominator", "fit"}
            else f"early_mlp_suffix_transport_v1_{self.phase}"
        )
        if self.role != required_role:
            raise ValueError("trace role and execution phase differ")
        if self.trial not in range(3) or self.epoch not in range(3) or (
            self.optimizer_step < 0 or self.batch_ordinal < 0
        ):
            raise ValueError("trace trial/epoch/step identity changed")
        # Every fitted or scored Q/L/R/S/T package physically executes both rank-64
        # replacements; S0/S1 name the trainable subset, not an N/P execution state.
        # Only one-shot final scoring may restore exact MLP2 as the registered E
        # alternate background; fit and validation remain P/P/N.
        allowed_states = {((0, "P"), (1, "P"), (2, "N"))}
        if self.phase == "final":
            allowed_states.add(((0, "P"), (1, "P"), (2, "E")))
        if self.student_states not in allowed_states:
            raise ValueError("trace must execute a registered P/P/N or final P/P/E state")
        if self.batch_rows != BATCH_SIZE or self.sequence_length != SEQUENCE_LENGTH or (
            self.score_start != SCORE_START or self.score_stop != SCORE_STOP
            or self.code_dim != CODE_DIM
        ):
            raise ValueError("trace batch/support dimensions changed")

    @property
    def sha256(self) -> str:
        return logical_identity_sha256({
            field: getattr(self, field) for field in self.__dataclass_fields__
        })

    @property
    def nonce(self) -> str:
        return self.sha256

    @classmethod
    def from_inputs(
        cls, *, inputs: torch.Tensor, ordered_batch_indices: Sequence[int],
        source_commit: str, inherited_snapshot_sha256: str, rows_receipt_sha256: str,
        fit_role_tensor_sha256: str, program_snapshot_sha256: str,
        teacher_mapping_sha256: str, phase: str, route: str, control: str,
        teacher_kind: str, trial: int, epoch: int, optimizer_step: int,
        batch_ordinal: int, student_states: tuple[tuple[int, str], ...],
        role: str = "early_mlp_suffix_transport_v1_fit",
    ) -> "TraceIdentity":
        if not torch.is_tensor(inputs) or inputs.dtype != torch.long or tuple(inputs.shape) != (
            BATCH_SIZE, SEQUENCE_LENGTH,
        ):
            raise ValueError("trace inputs must be int64 [4,256]")
        indices = tuple(ordered_batch_indices)
        if len(indices) != BATCH_SIZE or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError("ordered batch indices are malformed or duplicated")
        return cls(
            schema_version=1, protocol_sha256=PREREGISTRATION_SHA256,
            implementation_amendment_sha256=IMPLEMENTATION_AMENDMENT_SHA256,
            source_commit=source_commit,
            inherited_snapshot_sha256=inherited_snapshot_sha256,
            rows_receipt_sha256=rows_receipt_sha256,
            fit_role_tensor_sha256=fit_role_tensor_sha256,
            ordered_batch_indices_sha256=logical_identity_sha256(list(indices)),
            ordered_input_tokens_sha256=tensor_identity_sha256(inputs),
            program_snapshot_sha256=program_snapshot_sha256,
            teacher_mapping_sha256=teacher_mapping_sha256,
            role=role, phase=phase, route=route,
            control=control, teacher_kind=teacher_kind, trial=trial, epoch=epoch,
            optimizer_step=optimizer_step, batch_ordinal=batch_ordinal,
            student_states=student_states, batch_rows=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH, score_start=SCORE_START,
            score_stop=SCORE_STOP, code_dim=CODE_DIM,
        )

    def require_inputs(self, inputs: torch.Tensor) -> None:
        if not torch.is_tensor(inputs) or inputs.dtype != torch.long or tuple(inputs.shape) != (
            self.batch_rows, self.sequence_length,
        ) or tensor_identity_sha256(inputs) != self.ordered_input_tokens_sha256:
            raise RuntimeError("input tokens differ from the trace identity")

    def require_batch_indices(self, ordered_batch_indices: Sequence[int]) -> None:
        indices = tuple(ordered_batch_indices)
        if len(indices) != self.batch_rows or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in indices
        ) or len(set(indices)) != len(indices) or logical_identity_sha256(
            list(indices)
        ) != self.ordered_batch_indices_sha256:
            raise RuntimeError("ordered batch indices differ from the trace identity")


@dataclass(frozen=True)
class ScopeLease:
    name: str
    serial: int


class ScopeCoordinator:
    """Mutual exclusion for student, coordinate-label, and autonomous-OON scopes."""

    def __init__(self) -> None:
        self._active: ScopeLease | None = None
        self._serial = 0

    @contextmanager
    def enter(self, name: str):
        if name not in {"student", "coordinate", "oon"}:
            raise ValueError("unknown suffix-transport capability scope")
        if self._active is not None:
            raise RuntimeError(
                f"capability scope {name} overlaps active {self._active.name} scope"
            )
        self._serial += 1
        lease = ScopeLease(name=name, serial=self._serial)
        self._active = lease
        try:
            yield lease
        finally:
            if self._active != lease:
                raise RuntimeError("capability scope ownership changed")
            self._active = None

    def require_active(self, lease: ScopeLease) -> None:
        if self._active != lease:
            raise RuntimeError("ephemeral capability lease is inactive")

    @property
    def idle(self) -> bool:
        return self._active is None


class StudentTrace:
    """Sealed, one-use handoff of detached current student states."""

    __slots__ = (
        "__basis_sha256", "__calls", "__consumed", "__identity_sha256",
        "__expected_trace_sha256", "__issuer_id", "__nonce", "__program_sha256",
        "__release", "__sealed", "__site_metadata", "__trace_sha256", "__values",
    )

    def __init__(
        self, *, issuer_id: str, identity: TraceIdentity, program_sha256: str,
        basis_sha256: Mapping[int, str], values: Mapping[int, torch.Tensor],
        calls: Mapping[int, int], release: Any,
    ) -> None:
        object.__setattr__(self, "_StudentTrace__sealed", False)
        if not _sha256_text(issuer_id) or not _sha256_text(program_sha256) or set(
            basis_sha256
        ) != {0, 1} or any(not _sha256_text(value) for value in basis_sha256.values()):
            raise RuntimeError("student trace provenance is malformed")
        if set(values) != {0, 1} or dict(calls) != {0: 1, 1: 1}:
            raise RuntimeError("student trace requires exactly one MLP0/1 state")
        copied: dict[int, torch.Tensor] = {}
        metadata: dict[int, dict[str, Any]] = {}
        for site in (0, 1):
            value = values[site]
            if not torch.is_tensor(value) or tuple(value.shape) != (
                identity.batch_rows, SEQUENCE_LENGTH, D_MODEL,
            ) or not bool(torch.isfinite(value).all()) or value.requires_grad or (
                value.grad_fn is not None
            ):
                raise RuntimeError(f"student trace MLP{site} state is malformed")
            copied[site] = value.detach().cpu().contiguous().clone()
            metadata[site] = {
                "shape": list(copied[site].shape),
                "dtype": str(copied[site].dtype),
                "sha256": tensor_identity_sha256(copied[site]),
            }
        payload = {
            "issuer_id": issuer_id, "identity_sha256": identity.sha256,
            "nonce": identity.nonce, "program_sha256": program_sha256,
            "basis_sha256": {str(site): basis_sha256[site] for site in (0, 1)},
            "site_metadata": {str(site): metadata[site] for site in (0, 1)},
            "student_calls": {str(site): int(calls[site]) for site in (0, 1)},
        }
        self.__issuer_id = issuer_id
        self.__identity_sha256 = identity.sha256
        self.__nonce = identity.nonce
        self.__program_sha256 = program_sha256
        self.__basis_sha256 = dict(basis_sha256)
        self.__site_metadata = metadata
        self.__calls = dict(calls)
        self.__values = copied
        self.__trace_sha256 = logical_identity_sha256(payload)
        self.__expected_trace_sha256 = self.__trace_sha256
        self.__consumed = False
        self.__release = release
        object.__setattr__(self, "_StudentTrace__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_StudentTrace__sealed", False):
            raise AttributeError("student trace is sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("student traces cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("student traces cannot be copied")

    def __reduce__(self):
        raise RuntimeError("student traces cannot be serialized")

    @property
    def issuer_id(self) -> str:
        return self.__issuer_id

    @property
    def identity_sha256(self) -> str:
        return self.__identity_sha256

    @property
    def nonce(self) -> str:
        return self.__nonce

    @property
    def trace_sha256(self) -> str:
        return self.__trace_sha256

    @property
    def site_metadata(self) -> Mapping[int, Mapping[str, Any]]:
        return {
            site: {
                "shape": list(value["shape"]), "dtype": value["dtype"],
                "sha256": value["sha256"],
            }
            for site, value in self.__site_metadata.items()
        }

    @property
    def student_calls(self) -> Mapping[int, int]:
        return dict(self.__calls)

    @property
    def consumed(self) -> bool:
        return self.__consumed

    def _require_integrity(self) -> None:
        for site in (0, 1):
            value = self.__values[site]
            metadata = self.__site_metadata[site]
            if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata[
                "dtype"
            ] or tensor_identity_sha256(value) != metadata["sha256"]:
                raise RuntimeError("student trace tensor mutated after issuance")
        payload = {
            "issuer_id": self.__issuer_id, "identity_sha256": self.__identity_sha256,
            "nonce": self.__nonce, "program_sha256": self.__program_sha256,
            "basis_sha256": {
                str(site): self.__basis_sha256[site] for site in (0, 1)
            },
            "site_metadata": {
                str(site): self.__site_metadata[site] for site in (0, 1)
            },
            "student_calls": {str(site): self.__calls[site] for site in (0, 1)},
        }
        if self.__trace_sha256 != self.__expected_trace_sha256 or (
            logical_identity_sha256(payload) != self.__expected_trace_sha256
        ):
            raise RuntimeError("student trace metadata mutated after issuance")

    def _consume(self, *, issuer_id: str, identity: TraceIdentity) -> dict[int, torch.Tensor]:
        if self.__consumed:
            raise RuntimeError("student trace was already consumed")
        if issuer_id != self.__issuer_id or identity.sha256 != self.__identity_sha256 or (
            identity.nonce != self.__nonce
        ):
            raise RuntimeError("student trace identity or issuer mismatch")
        self._require_integrity()
        object.__setattr__(self, "_StudentTrace__consumed", True)
        release = self.__release
        object.__setattr__(self, "_StudentTrace__release", None)
        if release is None:
            raise RuntimeError("student trace release capability is absent")
        output = {site: value.clone() for site, value in self.__values.items()}
        self.__values.clear()
        release(self.__trace_sha256)
        return output

    def _discard(self, *, issuer_id: str, identity: TraceIdentity) -> None:
        """Spend a failed trace without exposing its captured states."""

        if self.__consumed:
            raise RuntimeError("student trace was already consumed")
        if issuer_id != self.__issuer_id or identity.sha256 != self.__identity_sha256:
            raise RuntimeError("student trace identity or issuer mismatch")
        self._require_integrity()
        object.__setattr__(self, "_StudentTrace__consumed", True)
        self.__values.clear()
        release = self.__release
        object.__setattr__(self, "_StudentTrace__release", None)
        if release is None:
            raise RuntimeError("student trace release capability is absent")
        release(self.__trace_sha256)


def _finite_tensor(name: str, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach().clone().float()


class AffineCodeProgram(nn.Module):
    """The common float32 ``((z-mean)/scale) W + bias`` code predictor."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        scale: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        mean_value = _finite_tensor("mean", mean, (D_MODEL,))
        scale_value = _finite_tensor("scale", scale, (D_MODEL,))
        if bool(torch.any(scale_value <= 0)):
            raise ValueError("scale must be strictly positive")
        self.register_buffer("mean", mean_value)
        self.register_buffer("scale", scale_value)
        self.weight = nn.Parameter(_finite_tensor("weight", weight, (D_MODEL, CODE_DIM)))
        self.bias = nn.Parameter(_finite_tensor("bias", bias, (CODE_DIM,)))

    @classmethod
    def from_v21_state(cls, state: Mapping[str, Any]) -> "AffineCodeProgram":
        required = {"grammar", "interface", "mean", "scale", "left", "right", "bias"}
        if not isinstance(state, Mapping) or not required.issubset(state):
            raise ValueError("v2.1 affine initialization is incomplete")
        if state["grammar"] != "affine" or state["interface"] != "state_complete_p":
            raise ValueError("suffix transport requires a state-complete affine initialization")
        left = _finite_tensor("left", state["left"], (D_MODEL, CODE_DIM))
        right = _finite_tensor("right", state["right"], (CODE_DIM, CODE_DIM))
        return cls(
            mean=state["mean"], scale=state["scale"],
            weight=left @ right, bias=state["bias"],
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(z) or z.shape[-1] != D_MODEL or not torch.isfinite(z).all():
            raise ValueError("z must be finite and end in d_model")
        flat = z.float().reshape(-1, D_MODEL)
        code = ((flat - self.mean) / self.scale) @ self.weight + self.bias
        return code.view(*z.shape[:-1], CODE_DIM)

    def frozen_normalization(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu().clone(), "scale": self.scale.detach().cpu().clone()}


class JointAffineProgram(nn.Module):
    """One shared executable implementation for local, suffix, singleton, and T arms."""

    def __init__(
        self,
        site0: AffineCodeProgram,
        site1: AffineCodeProgram,
        *,
        route: str,
    ) -> None:
        super().__init__()
        if route not in {"L", "R", "S0", "S1", "T"}:
            raise ValueError("unknown suffix-transport route")
        self._topology_sealed = False
        self._route = route
        self.site0 = site0
        self.site1 = site1
        self.cross = nn.Parameter(torch.zeros(CODE_DIM, CODE_DIM, dtype=torch.float32)) \
            if route == "T" else None
        self._topology_sealed = True
        self.set_route_trainability()

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_topology_sealed", False) and name in {"route", "_route", "cross"}:
            raise AttributeError("suffix-transport route/cross topology is immutable")
        super().__setattr__(name, value)

    @property
    def route(self) -> str:
        return self._route

    def _validate_topology(self) -> None:
        route = self._route
        if route not in {"L", "R", "S0", "S1", "T"}:
            raise RuntimeError("suffix-transport route identity changed")
        cross = self._parameters.get("cross")
        if route == "T":
            if not isinstance(cross, nn.Parameter) or tuple(cross.shape) != (
                CODE_DIM, CODE_DIM,
            ) or not torch.isfinite(cross).all():
                raise RuntimeError("T cross topology changed")
        elif cross is not None:
            raise RuntimeError("non-T route acquired a cross parameter")

    @classmethod
    def from_v21_states(
        cls, states: Mapping[int, Mapping[str, Any]], *, route: str,
    ) -> "JointAffineProgram":
        if set(states) != {0, 1}:
            raise ValueError("joint initialization requires exactly sites 0 and 1")
        return cls(
            AffineCodeProgram.from_v21_state(states[0]),
            AffineCodeProgram.from_v21_state(states[1]),
            route=route,
        )

    def independent_clone(self, *, route: str | None = None) -> "JointAffineProgram":
        requested = self.route if route is None else route
        clone = JointAffineProgram(
            copy.deepcopy(self.site0), copy.deepcopy(self.site1), route=requested,
        )
        if requested == "T" and self.cross is not None:
            with torch.no_grad():
                clone.cross.copy_(self.cross)
        return clone

    def site0_code(self, z0: torch.Tensor) -> torch.Tensor:
        self._validate_topology()
        return self.site0(z0)

    def set_route_trainability(self) -> tuple[nn.Parameter, ...]:
        """Freeze the exact registered parameter set and return its stable order."""

        self._validate_topology()
        if self.cross is not None and not torch.count_nonzero(self.cross.detach()) == 0:
            raise RuntimeError("T must enter training from exactly zero A")
        site0_on = self.route in {"L", "R", "S0"}
        site1_on = self.route in {"L", "R", "S1"}
        for parameter in self.site0.parameters():
            parameter.requires_grad_(site0_on)
        for parameter in self.site1.parameters():
            parameter.requires_grad_(site1_on)
        if self.cross is not None:
            self.cross.requires_grad_(self.route == "T")
        ordered = (
            self.site0.weight, self.site0.bias, self.site1.weight, self.site1.bias,
            *((self.cross,) if self.cross is not None else ()),
        )
        return tuple(parameter for parameter in ordered if parameter.requires_grad)

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        self._validate_topology()
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    @property
    def expected_trainable_parameter_names(self) -> tuple[str, ...]:
        expected = {
            "L": ("site0.weight", "site0.bias", "site1.weight", "site1.bias"),
            "R": ("site0.weight", "site0.bias", "site1.weight", "site1.bias"),
            "S0": ("site0.weight", "site0.bias"),
            "S1": ("site1.weight", "site1.bias"),
            "T": ("cross",),
        }
        return expected[self.route]

    def require_exact_trainability(self) -> None:
        if self.trainable_parameter_names != self.expected_trainable_parameter_names:
            raise RuntimeError(
                "suffix-transport route trainable tensor set changed: "
                f"{self.trainable_parameter_names} != {self.expected_trainable_parameter_names}"
            )

    def site1_code(self, z1: torch.Tensor, parent_code: torch.Tensor | None = None) -> torch.Tensor:
        self._validate_topology()
        local = self.site1(z1)
        if self.route != "T":
            if parent_code is not None:
                raise ValueError("L/R/S programs cannot consume a parent code")
            return local
        if self.cross is None:
            raise RuntimeError("T route lacks its dense cross map")
        if parent_code is None or parent_code.shape != local.shape:
            raise ValueError("transport requires same-shaped executable parent code")
        if not torch.isfinite(parent_code).all():
            raise ValueError("parent code must be finite")
        return local + parent_code.float() @ self.cross

    @staticmethod
    def projected_replacement(
        deployed_output: torch.Tensor, predicted_code: torch.Tensor, basis: torch.Tensor,
    ) -> torch.Tensor:
        if deployed_output.shape[:-1] != predicted_code.shape[:-1] or (
            deployed_output.shape[-1] != D_MODEL or predicted_code.shape[-1] != CODE_DIM
        ):
            raise ValueError("deployed output and code shapes are incompatible")
        contract.validate_orthonormal_basis("basis", basis)
        basis_value = basis.to(device=deployed_output.device, dtype=torch.float32)
        deployed = deployed_output.float()
        live_code = deployed.reshape(-1, D_MODEL) @ basis_value
        replacement = (predicted_code.float().reshape(-1, CODE_DIM) - live_code) @ basis_value.T
        return deployed_output + replacement.view_as(deployed_output).to(deployed_output.dtype)


class DeployedNWrite:
    """One-use typed handle for a live frozen-ship ``N`` write.

    This type prevents the physical ``P_B[N]`` operation from silently accepting an
    arbitrary tensor or a native-original ``O`` write.  Construction alone is not a
    model-provenance authority: the observed model adapter must mint it exactly once
    from each live frozen-surrogate site and separately prove zero native calls.
    """

    __slots__ = (
        "__consumed", "__forward_nonce", "__issuer_id", "__sealed", "__site",
        "__state", "__state_snapshot", "__state_version", "__value",
        "__value_version",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("deployed N write type cannot be subclassed")

    def __init__(
        self, *, site: int, state: torch.Tensor, value: torch.Tensor,
        forward_nonce: str, issuer_id: str,
    ) -> None:
        object.__setattr__(self, "_DeployedNWrite__sealed", False)
        if site not in (0, 1) or not isinstance(state, torch.Tensor) or not isinstance(
            value, torch.Tensor,
        ) or state.shape != value.shape or state.shape[-1] != D_MODEL:
            raise ValueError("deployed N write site/state/value is malformed")
        if not _sha256_text(forward_nonce) or not _sha256_text(issuer_id):
            raise ValueError("deployed N write identity is malformed")
        if not bool(torch.isfinite(state.detach()).all()) or not bool(
            torch.isfinite(value.detach()).all()
        ):
            raise ValueError("deployed N write contains nonfinite values")
        # Own the write tensor so a caller cannot mutate the handle indirectly
        # through the tensor object it passed to mint. ``clone`` deliberately
        # preserves autograd connectivity; ``detach`` would sever the live suffix.
        owned_value = value.clone()
        self.__site = site
        self.__state = state
        self.__state_snapshot = state.detach().clone()
        self.__state_version = state._version
        self.__value = owned_value
        self.__value_version = owned_value._version
        self.__forward_nonce = forward_nonce
        self.__issuer_id = issuer_id
        self.__consumed = False
        object.__setattr__(self, "_DeployedNWrite__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_DeployedNWrite__sealed", False):
            raise AttributeError("deployed N writes are sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("deployed N writes cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("deployed N writes cannot be copied")

    def __reduce__(self):
        raise RuntimeError("deployed N writes cannot be serialized")

    def _consume(
        self, *, site: int, state: torch.Tensor, forward_nonce: str, issuer_id: str,
    ) -> torch.Tensor:
        if self.__consumed:
            raise RuntimeError("deployed N write was already consumed")
        if site != self.__site or state is not self.__state or (
            forward_nonce != self.__forward_nonce or issuer_id != self.__issuer_id
        ):
            raise RuntimeError("deployed N write site/state/forward identity changed")
        if state._version != self.__state_version or self.__value._version != (
            self.__value_version
        ):
            raise RuntimeError("deployed N write tensor mutated after mint")
        if state.shape != self.__state_snapshot.shape or state.dtype != (
            self.__state_snapshot.dtype
        ) or state.device != self.__state_snapshot.device or not torch.equal(
            state.detach(), self.__state_snapshot,
        ):
            raise RuntimeError("deployed N write state content mutated after mint")
        if self.__value.shape != state.shape or not bool(
            torch.isfinite(state.detach()).all()
        ) or not bool(torch.isfinite(self.__value.detach()).all()):
            raise RuntimeError("deployed N write tensor became malformed after mint")
        object.__setattr__(self, "_DeployedNWrite__consumed", True)
        value = self.__value
        object.__setattr__(self, "_DeployedNWrite__state", None)
        object.__setattr__(self, "_DeployedNWrite__state_snapshot", None)
        object.__setattr__(self, "_DeployedNWrite__value", None)
        return value


class MappedParentCode:
    """One-use false-paired L0 code licensed only for an A-null fit identity."""

    __slots__ = (
        "__consumed", "__content_sha256", "__identity_sha256", "__issuer_id",
        "__program_sha256", "__release", "__sealed", "__value", "__version",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("mapped parent code type cannot be subclassed")

    def __init__(
        self, *, value: torch.Tensor, identity_sha256: str, issuer_id: str,
        program_sha256: str, release: Any,
    ) -> None:
        object.__setattr__(self, "_MappedParentCode__sealed", False)
        if not torch.is_tensor(value) or tuple(value.shape) != (
            BATCH_SIZE, SEQUENCE_LENGTH, CODE_DIM,
        ) or value.requires_grad or value.grad_fn is not None or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("mapped parent code is malformed")
        if not all(_sha256_text(item) for item in (
            identity_sha256, issuer_id, program_sha256,
        )) or not callable(release):
            raise ValueError("mapped parent provenance is malformed")
        owned = value.detach().clone().contiguous()
        self.__value = owned
        self.__version = owned._version
        self.__content_sha256 = tensor_identity_sha256(owned)
        self.__identity_sha256 = identity_sha256
        self.__issuer_id = issuer_id
        self.__program_sha256 = program_sha256
        self.__release = release
        self.__consumed = False
        object.__setattr__(self, "_MappedParentCode__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_MappedParentCode__sealed", False):
            raise AttributeError("mapped parent codes are sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("mapped parent codes cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("mapped parent codes cannot be copied")

    def __reduce__(self):
        raise RuntimeError("mapped parent codes cannot be serialized")

    @property
    def identity_sha256(self) -> str:
        return self.__identity_sha256

    @property
    def sha256(self) -> str:
        return logical_identity_sha256({
            "identity_sha256": self.__identity_sha256,
            "issuer_id": self.__issuer_id,
            "program_sha256": self.__program_sha256,
            "content_sha256": self.__content_sha256,
        })

    def _consume(self, *, issuer_id: str, program_sha256: str) -> torch.Tensor:
        if self.__consumed:
            raise RuntimeError("mapped parent code was already consumed")
        if issuer_id != self.__issuer_id or program_sha256 != self.__program_sha256:
            raise RuntimeError("mapped parent issuer or program identity changed")
        if self.__value._version != self.__version or tensor_identity_sha256(
            self.__value
        ) != self.__content_sha256 or not bool(torch.isfinite(self.__value).all()):
            raise RuntimeError("mapped parent code mutated before consumption")
        object.__setattr__(self, "_MappedParentCode__consumed", True)
        value = self.__value
        object.__setattr__(self, "_MappedParentCode__value", None)
        release = self.__release
        object.__setattr__(self, "_MappedParentCode__release", None)
        release(self.__identity_sha256)
        return value


class NativeOWrite:
    """Marker type for native-original writes; never legal on the student P path."""

    __slots__ = ("value",)

    def __init__(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("native O write requires a tensor")
        self.value = value


def mint_deployed_n_write(
    *, site: int, state: torch.Tensor, value: torch.Tensor,
    forward_nonce: str, issuer_id: str,
) -> DeployedNWrite:
    """Mint a typed N write; only the future observed adapter may authorize provenance."""

    return DeployedNWrite(
        site=site, state=state, value=value, forward_nonce=forward_nonce,
        issuer_id=issuer_id,
    )


def program_snapshot_sha256(program: JointAffineProgram | None) -> str:
    if program is None:
        return logical_identity_sha256({"program": None})
    if not isinstance(program, JointAffineProgram):
        raise TypeError("program snapshot requires the shared runtime program")
    program._validate_topology()
    program.require_exact_trainability()
    state = {
        name: tensor_identity_sha256(value)
        for name, value in sorted(program.state_dict().items())
    }
    return logical_identity_sha256({
        "route": program.route, "state": state,
        "trainable_parameter_names": list(program.trainable_parameter_names),
    })


class StudentCorrectionHook:
    """Original-free N/P student execution with one-use same-forward transport."""

    def __init__(
        self, bases: Mapping[int, torch.Tensor], *, issuer_id: str,
        coordinator: ScopeCoordinator,
    ) -> None:
        if set(bases) != {0, 1}:
            raise ValueError("runtime requires exactly the MLP0/1 bases")
        if not _sha256_text(issuer_id) or not isinstance(coordinator, ScopeCoordinator):
            raise ValueError("student trace issuer/coordinator is malformed")
        self._issuer_id = issuer_id
        self._coordinator = coordinator
        self._bases = {
            site: _finite_tensor(f"basis{site}", value, (D_MODEL, CODE_DIM))
            for site, value in bases.items()
        }
        for site, value in self._bases.items():
            contract.validate_orthonormal_basis(f"basis{site}", value)
        self._basis_sha256 = {
            site: tensor_identity_sha256(value) for site, value in self._bases.items()
        }
        self.program: JointAffineProgram | None = None
        self.states: dict[int, str] = {}
        self.capture_sites: frozenset[int] = frozenset()
        self.site0_edit: torch.Tensor | None = None
        self.parent_code: torch.Tensor | None = None
        self.mapped_parent_code: torch.Tensor | None = None
        self.mapped_parent_identity_sha256: str | None = None
        self.captured_z: dict[int, torch.Tensor] = {}
        self.captured_codes: dict[int, torch.Tensor] = {}
        self.calls = {0: 0, 1: 0}
        self.scope_calls = {0: 0, 1: 0}
        self.forward_nonce: str | None = None
        self.forward_identity: TraceIdentity | None = None
        self.parent_nonce: str | None = None
        self.parent_consumed = False
        self._pending_trace: StudentTrace | None = None
        self._pending_codes: dict[int, torch.Tensor] | None = None
        self._outstanding_trace_sha256: str | None = None
        self._spent_identity_sha256: set[str] = set()
        self.active = False

    @property
    def issuer_id(self) -> str:
        return self._issuer_id

    @property
    def coordinator(self) -> ScopeCoordinator:
        return self._coordinator

    @property
    def basis_sha256(self) -> Mapping[int, str]:
        return dict(self._basis_sha256)

    def _require_basis_integrity(self) -> None:
        for site in (0, 1):
            if tensor_identity_sha256(self._bases[site]) != self._basis_sha256[site]:
                raise RuntimeError("student basis mutated after hook construction")

    def configure(
        self,
        *,
        program: JointAffineProgram | None,
        states: Mapping[int, str],
        site0_edit: torch.Tensor | None = None,
        mapped_parent: MappedParentCode | None = None,
    ) -> None:
        if self.active or self._pending_trace is not None or self._pending_codes is not None or (
            self._outstanding_trace_sha256 is not None
        ):
            raise RuntimeError("student runtime cannot configure with an active trace")
        allowed = {0: {"N", "P"}, 1: {"N", "P"}}
        if any(site not in allowed or state not in allowed[site] for site, state in states.items()):
            raise ValueError("student runtime permits only MLP0/1 N/P states")
        if any(state == "P" for state in states.values()) and program is None:
            raise ValueError("predicted states require an executable program")
        if program is not None and not isinstance(program, JointAffineProgram):
            raise ValueError("program has the wrong runtime type")
        if site0_edit is not None and (
            not torch.is_tensor(site0_edit) or site0_edit.shape[-1] != CODE_DIM
            or not torch.isfinite(site0_edit).all()
        ):
            raise ValueError("site0 edit must be finite and end in code_dim")
        if site0_edit is not None and (states.get(0, "N") != "P" or program is None):
            raise ValueError("site0 edit requires an executable predicted MLP0 state")
        if mapped_parent is not None and (
            type(mapped_parent) is not MappedParentCode or program is None
            or program.route != "T" or states.get(0) != "P" or states.get(1) != "P"
        ):
            raise ValueError("mapped parent requires an executable P/P T program")
        mapped_code = None
        mapped_identity = None
        if mapped_parent is not None:
            mapped_identity = mapped_parent.identity_sha256
            mapped_code = mapped_parent._consume(
                issuer_id=self._issuer_id,
                program_sha256=program_snapshot_sha256(program),
            )
        self.program = program
        self.states = dict(states)
        self.capture_sites = frozenset()
        self.site0_edit = site0_edit
        self.parent_code = None
        self.mapped_parent_code = mapped_code
        self.mapped_parent_identity_sha256 = mapped_identity
        self.captured_z = {}
        self.captured_codes = {}
        self.calls = {0: 0, 1: 0}
        self.scope_calls = {0: 0, 1: 0}
        self.forward_nonce = None
        self.forward_identity = None
        self.parent_nonce = None
        self.parent_consumed = False

    def clear_configuration(self) -> None:
        """Drop all program/state/edit references after one batch transaction."""

        if self.active or self._pending_trace is not None or self._pending_codes is not None:
            raise RuntimeError("student runtime cannot clear during an active/pending forward")
        self.program = None
        self.states = {}
        self.site0_edit = None
        self.capture_sites = frozenset()
        self.captured_z = {}
        self.captured_codes = {}
        self.parent_code = None
        self.mapped_parent_code = None
        self.mapped_parent_identity_sha256 = None
        self.calls = {0: 0, 1: 0}
        self.scope_calls = {0: 0, 1: 0}
        self.forward_nonce = None
        self.forward_identity = None
        self.parent_nonce = None
        self.parent_consumed = False

    @property
    def has_mapped_parent(self) -> bool:
        return self.mapped_parent_code is not None and (
            self.mapped_parent_identity_sha256 is not None
        )

    @contextmanager
    def forward_scope(
        self, identity: TraceIdentity, *, capture_sites: Iterable[int] = (),
    ):
        if self.active:
            raise RuntimeError("runtime forward scope is already active")
        if not isinstance(identity, TraceIdentity):
            raise ValueError("student forward requires an exact trace identity")
        captures = frozenset(capture_sites)
        if captures not in {frozenset(), frozenset({0, 1})}:
            raise ValueError("student trace capture must be empty or exactly MLP0/1")
        if self._pending_trace is not None or self._pending_codes is not None or (
            self._outstanding_trace_sha256 is not None
        ):
            raise RuntimeError("previous student trace remains outstanding")
        if identity.sha256 in self._spent_identity_sha256:
            raise RuntimeError("student trace identity was already spent")
        if self.mapped_parent_identity_sha256 is not None and (
            self.mapped_parent_identity_sha256 != identity.sha256
        ):
            raise RuntimeError("mapped parent differs from the student identity")
        # The correction hook owns only MLP0/1. MLP2 background is carried by the
        # already-validated final identity and enforced by the observed adapter.
        background2 = dict(identity.student_states).get(2)
        configured_states = tuple(
            (site, self.states.get(site, "N")) for site in (0, 1)
        ) + ((2, background2 if identity.phase == "final" else "N"),)
        if identity.student_states != configured_states:
            raise RuntimeError("student configured states differ from trace identity")
        current_program_sha256 = program_snapshot_sha256(self.program)
        if identity.program_snapshot_sha256 != current_program_sha256:
            raise RuntimeError("student program differs from trace identity")
        expected_program_route = "L" if identity.route == "Q" else identity.route
        if self.program is None or self.program.route != expected_program_route:
            raise RuntimeError("student program route differs from trace route")
        self._spent_identity_sha256.add(identity.sha256)
        self._require_basis_integrity()
        with self._coordinator.enter("student"):
            self.active = True
            self.capture_sites = captures
            self.captured_z = {}
            self.captured_codes = {}
            self.scope_calls = {0: 0, 1: 0}
            self.forward_nonce = identity.nonce
            self.forward_identity = identity
            self.parent_code = None
            self.parent_nonce = None
            self.parent_consumed = False
            clean = False
            try:
                yield self
                clean = True
                if captures:
                    if program_snapshot_sha256(self.program) != current_program_sha256:
                        raise RuntimeError("student program mutated during its forward")
                    self._require_basis_integrity()
                    if self.scope_calls != {0: 1, 1: 1} or set(self.captured_z) != {0, 1}:
                        raise RuntimeError("student trace forward did not call/capture MLP0/1 once")
                    if set(self.captured_codes) != {0, 1}:
                        raise RuntimeError("student trace forward did not produce both fitted codes")
                    candidate = StudentTrace(
                        issuer_id=self._issuer_id, identity=identity,
                        program_sha256=current_program_sha256,
                        basis_sha256=self._basis_sha256, values=self.captured_z,
                        calls=self.scope_calls, release=self._release_trace,
                    )
                    self._pending_trace = candidate
                    # Preserve the exact autograd-bearing values.  The capability
                    # layer seals them into the same one-use student step as the
                    # detached state trace before configuration may be cleared.
                    self._pending_codes = dict(self.captured_codes)
                    self._outstanding_trace_sha256 = candidate.trace_sha256
            finally:
                if not clean:
                    self._pending_trace = None
                    self._pending_codes = None
                    self._outstanding_trace_sha256 = None
                self.capture_sites = frozenset()
                self.captured_z = {}
                self.captured_codes = {}
                self.parent_code = None
                self.mapped_parent_code = None
                self.mapped_parent_identity_sha256 = None
                self.parent_nonce = None
                self.forward_nonce = None
                self.forward_identity = None
                self.active = False

    def pop_trace(self, identity: TraceIdentity) -> StudentTrace:
        if self.active:
            raise RuntimeError("student trace cannot pop during its forward")
        trace = self._pending_trace
        if trace is None:
            raise RuntimeError("no completed student trace is pending")
        if identity.sha256 != trace.identity_sha256 or identity.nonce != trace.nonce:
            raise RuntimeError("pending student trace identity mismatch")
        self._pending_trace = None
        return trace

    def pop_student_codes(self, identity: TraceIdentity) -> tuple[torch.Tensor, torch.Tensor]:
        if self.active:
            raise RuntimeError("student codes cannot pop during their forward")
        codes = self._pending_codes
        if codes is None or set(codes) != {0, 1}:
            raise RuntimeError("no completed student codes are pending")
        if self._outstanding_trace_sha256 is None or identity.sha256 not in (
            self._spent_identity_sha256
        ):
            raise RuntimeError("pending student code identity mismatch")
        self._pending_codes = None
        return codes[0], codes[1]

    def discard_student_codes(self) -> None:
        if self.active:
            raise RuntimeError("student codes cannot discard during their forward")
        self._pending_codes = None

    def _release_trace(self, trace_sha256: str) -> None:
        if trace_sha256 != self._outstanding_trace_sha256:
            raise RuntimeError("student trace release identity changed")
        self._outstanding_trace_sha256 = None
        # Direct runtime callers do not receive the capability layer's
        # autograd-bearing student-output handle.  Once their detached trace is
        # spent, do not retain an inaccessible graph or block the next scope.
        self._pending_codes = None

    def __call__(
        self, site: int, z: torch.Tensor, deployed_n: DeployedNWrite, *,
        forward_nonce: str,
    ) -> torch.Tensor:
        if not self.active or forward_nonce != self.forward_nonce:
            raise RuntimeError("runtime call occurred outside a forward scope")
        if site not in (0, 1):
            raise ValueError("student runtime is restricted to MLP0/1")
        if type(deployed_n) is not DeployedNWrite:
            raise TypeError("student P_B[N] requires a typed deployed N write")
        mo = deployed_n._consume(
            site=site, state=z, forward_nonce=forward_nonce, issuer_id=self._issuer_id,
        )
        if self.scope_calls[site] != 0:
            raise RuntimeError(f"student MLP{site} was called more than once in one forward")
        self.scope_calls[site] += 1
        self.calls[site] += 1
        if site in self.capture_sites:
            captured = z.detach().cpu().contiguous().clone()
            if tuple(captured.shape) != (
                self.forward_identity.batch_rows, SEQUENCE_LENGTH, D_MODEL,
            ) or not bool(torch.isfinite(captured).all()):
                raise RuntimeError("captured student state shape or values changed")
            self.captured_z[site] = captured
        state = self.states.get(site, "N")
        if state == "N":
            if site == 0 and self.program is not None and self.program.route == "T":
                raise RuntimeError("T cannot source its parent from native/deployed MLP0")
            return mo
        if self.program is None:
            raise RuntimeError("predicted student runtime lacks a program")
        if site == 0:
            predicted = self.program.site0_code(z)
        else:
            if self.program.route == "T":
                if self.parent_consumed:
                    raise RuntimeError("T parent code was already consumed")
                if self.mapped_parent_code is not None:
                    if self.mapped_parent_identity_sha256 != self.forward_identity.sha256:
                        raise RuntimeError("mapped T parent identity changed")
                    transport_parent = self.mapped_parent_code.to(z.device)
                    self.mapped_parent_code = None
                    self.mapped_parent_identity_sha256 = None
                else:
                    if self.parent_code is None or self.parent_nonce != forward_nonce:
                        raise RuntimeError("T lacks one unused same-forward executable parent")
                    transport_parent = self.parent_code
                predicted = self.program.site1_code(z, transport_parent)
                self.parent_consumed = True
                self.parent_code = None
            else:
                predicted = self.program.site1_code(z)

        if site == 0 and self.site0_edit is not None:
            if self.site0_edit.shape != predicted.shape:
                raise RuntimeError("site0 edit shape differs from executable code")
            predicted = predicted + self.site0_edit.to(predicted.device, torch.float32)
        self.captured_codes[site] = predicted
        if site == 0:
            if self.parent_code is not None:
                raise RuntimeError("student parent code would be overwritten")
            self.parent_code = predicted
            self.parent_nonce = forward_nonce
        return JointAffineProgram.projected_replacement(mo, predicted, self._bases[site])


def scored_positions(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim < 2 or value.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("scored tensor must contain all 256 model positions")
    return value[:, SCORE_START:SCORE_STOP]


def _canonical_code_support(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != CODE_DIM:
        raise ValueError("code support must be [batch, token, code_dim]")
    if value.shape[1] == SEQUENCE_LENGTH:
        return scored_positions(value)
    if value.shape[1] != SCORE_STOP - SCORE_START:
        raise ValueError("code support must be full 256 or exact scored 192 positions")
    return value


@dataclass(frozen=True)
class MomentSufficientStatistics:
    """Mergeable float64 raw statistics for one site's frozen denominator."""

    count: int
    coordinate_sum: torch.Tensor
    coordinate_square_sum: torch.Tensor
    mean: torch.Tensor
    centered_m2: torch.Tensor

    @classmethod
    def from_labels(cls, labels: torch.Tensor) -> "MomentSufficientStatistics":
        labels = _canonical_code_support(labels)
        if labels.shape[0] <= 0 or not torch.isfinite(labels).all():
            raise ValueError("labels must contain finite scored code vectors")
        values = labels.detach().cpu().double().reshape(-1, CODE_DIM)
        mean = values.mean(dim=0)
        return cls(
            count=int(values.shape[0]),
            coordinate_sum=values.sum(dim=0),
            coordinate_square_sum=values.square().sum(dim=0),
            mean=mean,
            centered_m2=torch.sum((values - mean).square(), dim=0),
        )

    def merge(self, other: "MomentSufficientStatistics") -> "MomentSufficientStatistics":
        if not isinstance(other, MomentSufficientStatistics):
            raise TypeError("moment statistics can merge only with their own type")
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.count / count)
        centered_m2 = self.centered_m2 + other.centered_m2 + delta.square() * (
            self.count * other.count / count
        )
        return MomentSufficientStatistics(
            count=count,
            coordinate_sum=self.coordinate_sum + other.coordinate_sum,
            coordinate_square_sum=self.coordinate_square_sum + other.coordinate_square_sum,
            mean=mean,
            centered_m2=centered_m2,
        )

    def finalize(
        self, *, expected_count: int, ordered_support_sha256: str,
    ) -> Mapping[str, Any]:
        if self.count != expected_count:
            raise ValueError(f"moment support count changed: {self.count}!={expected_count}")
        if not isinstance(ordered_support_sha256, str) or len(ordered_support_sha256) != 64 \
                or any(character not in "0123456789abcdef" for character in ordered_support_sha256):
            raise ValueError("ordered support hash is malformed")
        if self.count < 2 or tuple(self.coordinate_sum.shape) != (CODE_DIM,) or (
            tuple(self.coordinate_square_sum.shape) != (CODE_DIM,)
        ) or not torch.isfinite(self.coordinate_sum).all() or not torch.isfinite(
            self.coordinate_square_sum
        ).all() or tuple(self.mean.shape) != (CODE_DIM,) or tuple(
            self.centered_m2.shape
        ) != (CODE_DIM,) or not torch.isfinite(self.mean).all() or not torch.isfinite(
            self.centered_m2
        ).all():
            raise ValueError("moment sufficient statistics are malformed")
        raw_centered_sum = torch.sum(
            self.coordinate_square_sum - self.coordinate_sum.square() / self.count
        )
        centered_sum = self.centered_m2.sum()
        denominator = centered_sum / (self.count * CODE_DIM)
        if not torch.isfinite(denominator) or float(denominator) <= 0:
            raise ValueError("label second moment must be positive and finite")
        return {
            "count": self.count,
            "coordinate_sum": self.coordinate_sum.clone(),
            "coordinate_square_sum": self.coordinate_square_sum.clone(),
            "mean": self.mean.clone(),
            "centered_sum_of_squares": centered_sum,
            "raw_sum_square_replay": raw_centered_sum,
            "denominator": denominator,
            "ordered_support_sha256": ordered_support_sha256,
        }


def centered_second_moment(
    labels: torch.Tensor, *, ordered_support_sha256: str,
) -> torch.Tensor:
    """Return the frozen scalar float64 centered second moment for one site."""

    return MomentSufficientStatistics.from_labels(labels).finalize(
        expected_count=384 * (SCORE_STOP - SCORE_START),
        ordered_support_sha256=ordered_support_sha256,
    )["denominator"]


def normalized_local_loss(
    predictions: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    denominators: Sequence[torch.Tensor | float],
) -> torch.Tensor:
    """Registered L loss; labels and frozen denominators never receive gradients."""

    if len(predictions) != 2 or len(labels) != 2 or len(denominators) != 2:
        raise ValueError("local loss requires exactly the two MLP0/1 sites")
    terms = []
    for prediction, label, denominator in zip(predictions, labels, denominators, strict=True):
        prediction = _canonical_code_support(prediction)
        label = _canonical_code_support(label)
        if prediction.shape != label.shape:
            raise ValueError("local prediction/label shape changed")
        target = label.detach().to(device=prediction.device, dtype=torch.float32)
        scale = torch.as_tensor(denominator, dtype=torch.float64).detach()
        if scale.numel() != 1 or not torch.isfinite(scale) or float(scale) <= 0:
            raise ValueError("local loss denominator must be positive and finite")
        terms.append(F.mse_loss(prediction.float(), target) / scale.float().to(prediction.device))
    return terms[0] + terms[1]


def teacher_student_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Token-weighted ``KL(teacher || student)`` with the teacher detached."""

    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3 or (
        teacher_logits.shape[-1] <= 1
    ):
        raise ValueError("teacher/student logits must be same-shaped [batch, token, vocab]")
    if not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
        raise ValueError("teacher/student logits must be finite")
    if teacher_logits.shape[1] == SEQUENCE_LENGTH:
        teacher_logits = scored_positions(teacher_logits)
        student_logits = scored_positions(student_logits)
    elif teacher_logits.shape[1] != SCORE_STOP - SCORE_START:
        raise ValueError("suffix KL support must be positions 64 through 255")
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    return torch.mean(torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), dim=-1))


def fit_permutations(row_count: int, trial: int) -> tuple[torch.Tensor, ...]:
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("row_count must be a positive integer")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial not in range(3):
        raise ValueError("trial must index the three registered learning rates")
    permutations = []
    for epoch in range(EPOCHS):
        generator = torch.Generator(device="cpu").manual_seed(2026083000 + 100 * trial + epoch)
        permutations.append(torch.randperm(row_count, generator=generator))
    return tuple(permutations)


def batch_indices(row_count: int, trial: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        permutation[start:start + BATCH_SIZE]
        for permutation in fit_permutations(row_count, trial)
        for start in range(0, row_count, BATCH_SIZE)
    )


def make_optimizer(parameters: Iterable[nn.Parameter], learning_rate: float) -> torch.optim.AdamW:
    if learning_rate not in LEARNING_RATES:
        raise ValueError("learning rate is outside the frozen grid")
    values = list(parameters)
    if not values or any(not isinstance(value, nn.Parameter) for value in values):
        raise ValueError("optimizer requires explicit trainable parameters")
    return torch.optim.AdamW(
        values, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )


def optimizer_step(loss: torch.Tensor, optimizer: torch.optim.AdamW) -> float:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("optimizer loss must be a finite scalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [value for group in optimizer.param_groups for value in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
    if not torch.isfinite(norm):
        raise RuntimeError("gradient norm is not finite")
    optimizer.step()
    return float(norm.detach().cpu())
