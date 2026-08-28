"""Pure, nonauthorizing capabilities for suffix-transport fit transactions.

There is deliberately no model loader, row loader, artifact I/O, or scoring authority
here. A transaction binds one exact row batch, one configured student program, its
actual autograd-bearing outputs, and one freshly produced teacher. Detached current
states may reach only the coordinate teacher; an autonomous O/O/N teacher receives
only tokens and a scope-revoked original-call gateway. A sealed model adapter must
later replace synthetic call/restoration assertions before any real run is legal.

An ordinary broker intentionally licenses only ``control=true``.  A broker may
instead be constructed with a separately implemented :class:`MappedRunAuthority`;
that mode is source-closed, uses a mapping-bound ledger identity, and exposes only
the mapped OON entry point implemented in this slice.  Mapping hashes therefore
cannot be decorative metadata on the ordinary interface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

import early_mlp_suffix_transport_v1 as contract
import early_mlp_suffix_transport_v1_runtime as runtime


EXACT_ZERO_CALLS = ((0, 0), (1, 0), (2, 0))
EXACT_EARLY_ORIGINAL_CALLS = ((0, 1), (1, 1), (2, 0))
FIT_ROW_COUNT = 384
FIT_BATCHES_PER_EPOCH = FIT_ROW_COUNT // runtime.BATCH_SIZE
VALIDATION_ROW_COUNT = 192
VALIDATION_BATCH_COUNT = VALIDATION_ROW_COUNT // runtime.BATCH_SIZE


def _call_tuple(value: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    if set(value) != {0, 1, 2} or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in value.values()
    ):
        raise RuntimeError("original-call ledger schema changed")
    return tuple((site, int(value[site])) for site in (0, 1, 2))


@dataclass(frozen=True)
class RunContext:
    """Exact immutable authorities expected by one process-local broker."""

    source_commit: str
    inherited_snapshot_sha256: str
    rows_receipt_sha256: str
    fit_role_tensor_sha256: str
    identity_teacher_mapping_sha256: str
    fit_row_count: int = FIT_ROW_COUNT

    def __post_init__(self) -> None:
        if not isinstance(self.source_commit, str) or len(self.source_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.source_commit
        ):
            raise ValueError("run-context source commit is malformed")
        if any(not runtime._sha256_text(value) for value in (
            self.inherited_snapshot_sha256, self.rows_receipt_sha256,
            self.fit_role_tensor_sha256, self.identity_teacher_mapping_sha256,
        )) or type(self.fit_row_count) is not int or self.fit_row_count != FIT_ROW_COUNT:
            raise ValueError("run context is malformed or changes the frozen fit rows")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            field: getattr(self, field) for field in self.__dataclass_fields__
        })

    def require_identity(
        self, identity: runtime.TraceIdentity, inputs: torch.Tensor,
        ordered_batch_indices: Sequence[int],
    ) -> None:
        self.require_common_identity(identity, inputs, ordered_batch_indices)
        if identity.control != "true":
            raise RuntimeError("mapped controls require the separate mapped-row capability")
        if identity.teacher_mapping_sha256 != self.identity_teacher_mapping_sha256:
            raise RuntimeError("true-route teacher mapping differs from the sealed identity map")

    def require_common_identity(
        self, identity: runtime.TraceIdentity, inputs: torch.Tensor,
        ordered_batch_indices: Sequence[int],
    ) -> None:
        """Validate immutable rows and schedule without granting a teacher mapping.

        The ordinary broker calls :meth:`require_identity`, which additionally
        insists on ``control=true``.  The separate mapped-row boundary may call this
        common portion only after independently proving its target-row permutation.
        """

        if not isinstance(identity, runtime.TraceIdentity):
            raise RuntimeError("student identity has the wrong runtime type")
        if identity.role != "early_mlp_suffix_transport_v1_fit" or identity.phase not in {
            "initial_denominator", "fit",
        }:
            raise RuntimeError("fit run context cannot authorize another role or phase")
        if (
            identity.source_commit != self.source_commit
            or identity.inherited_snapshot_sha256 != self.inherited_snapshot_sha256
            or identity.rows_receipt_sha256 != self.rows_receipt_sha256
            or identity.fit_role_tensor_sha256 != self.fit_role_tensor_sha256
        ):
            raise RuntimeError("trace differs from the sealed run context")
        identity.require_inputs(inputs)
        identity.require_batch_indices(ordered_batch_indices)
        indices = tuple(ordered_batch_indices)
        if any(index >= self.fit_row_count for index in indices):
            raise RuntimeError("ordered fit-row index is outside the frozen role")
        if identity.phase == "initial_denominator":
            if (identity.trial, identity.epoch, identity.optimizer_step) != (0, 0, 0) or not (
                0 <= identity.batch_ordinal < FIT_BATCHES_PER_EPOCH
            ):
                raise RuntimeError("initial-denominator schedule identity changed")
            expected = tuple(range(
                identity.batch_ordinal * runtime.BATCH_SIZE,
                (identity.batch_ordinal + 1) * runtime.BATCH_SIZE,
            ))
        else:
            if not 0 <= identity.batch_ordinal < FIT_BATCHES_PER_EPOCH or (
                identity.optimizer_step
                != identity.epoch * FIT_BATCHES_PER_EPOCH + identity.batch_ordinal
            ):
                raise RuntimeError("fit optimizer-step/batch schedule identity changed")
            permutation = runtime.fit_permutations(self.fit_row_count, identity.trial)[identity.epoch]
            start = identity.batch_ordinal * runtime.BATCH_SIZE
            expected = tuple(int(value) for value in permutation[start:start + runtime.BATCH_SIZE])
        if indices != expected:
            raise RuntimeError("ordered fit batch differs from the preregistered schedule")


@dataclass(frozen=True)
class ValidationRunContext:
    """Immutable authority for selection-only rows and the true evaluation teacher."""

    source_commit: str
    inherited_snapshot_sha256: str
    rows_receipt_sha256: str
    validation_role_tensor_sha256: str
    identity_teacher_mapping_sha256: str
    validation_row_count: int = VALIDATION_ROW_COUNT

    def __post_init__(self) -> None:
        if not isinstance(self.source_commit, str) or len(self.source_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.source_commit
        ):
            raise ValueError("validation source commit is malformed")
        if any(not runtime._sha256_text(value) for value in (
            self.inherited_snapshot_sha256, self.rows_receipt_sha256,
            self.validation_role_tensor_sha256, self.identity_teacher_mapping_sha256,
        )) or type(self.validation_row_count) is not int or self.validation_row_count != (
            VALIDATION_ROW_COUNT
        ):
            raise ValueError("validation run context changed the frozen role")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            field: getattr(self, field) for field in self.__dataclass_fields__
        })

    def require_identity(
        self, identity: runtime.TraceIdentity, inputs: torch.Tensor,
        ordered_batch_indices: Sequence[int],
    ) -> None:
        if not isinstance(identity, runtime.TraceIdentity) or identity.role != (
            "early_mlp_suffix_transport_v1_validation"
        ) or identity.phase != "validation":
            raise RuntimeError("selection execution lacks a validation identity")
        if identity.source_commit != self.source_commit or identity.inherited_snapshot_sha256 != (
            self.inherited_snapshot_sha256
        ) or identity.rows_receipt_sha256 != self.rows_receipt_sha256 or (
            # Trace schema v1 retains this historical field name; role disambiguates
            # whether it binds the fit tensor or the validation tensor.
            identity.fit_role_tensor_sha256 != self.validation_role_tensor_sha256
        ) or identity.teacher_mapping_sha256 != self.identity_teacher_mapping_sha256:
            raise RuntimeError("validation trace differs from the sealed run context")
        identity.require_inputs(inputs)
        identity.require_batch_indices(ordered_batch_indices)
        if identity.epoch != 0 or identity.optimizer_step != identity.batch_ordinal or not (
            0 <= identity.batch_ordinal < VALIDATION_BATCH_COUNT
        ):
            raise RuntimeError("validation batch schedule identity changed")
        start = identity.batch_ordinal * runtime.BATCH_SIZE
        expected = tuple(range(start, start + runtime.BATCH_SIZE))
        if tuple(ordered_batch_indices) != expected:
            raise RuntimeError("validation rows are not in canonical order")


class MappedRunAuthority:
    """Narrow interface a separate mapped-row module must implement.

    The capability module deliberately knows nothing about how document blocks are
    constructed.  It does insist that a mapped authority bind the same base role,
    authorize the source before the student forward, and authorize the exact target
    tokens before a teacher trace can be spent.
    """

    @property
    def base_context(self) -> RunContext:
        raise NotImplementedError

    @property
    def sha256(self) -> str:
        raise NotImplementedError

    def require_source_identity(
        self, identity: runtime.TraceIdentity, student_inputs: torch.Tensor,
        student_indices: Sequence[int],
    ) -> None:
        raise NotImplementedError

    def require_identity(
        self, identity: runtime.TraceIdentity, *, fit_rows: torch.Tensor,
        student_inputs: torch.Tensor, student_indices: Sequence[int],
        teacher_inputs: torch.Tensor, teacher_indices: Sequence[int],
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class StepClosure:
    identity_sha256: str
    forward_nonce: str
    scope: str
    producer_invocations: int
    outer_forward_count: int
    hook_calls: tuple[tuple[int, int], ...]
    original_calls: tuple[tuple[int, int], ...]
    outer_returned: bool
    hook_restored: bool
    hook_inert: bool
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    support: str
    requires_grad: bool
    grad_fn_absent: bool
    consumed: bool
    output_sha256: str
    ledger_sha256: str


@dataclass(frozen=True)
class ValidationBatchReductions:
    """Small raw reductions emitted by one sealed teacher/student transaction."""

    identity_sha256: str
    route: str
    program_sha256: str
    row_primary_sum: torch.Tensor
    row_primary_count: torch.Tensor
    row_ce_sum: torch.Tensor
    row_ce_count: torch.Tensor
    row_copy_ce_sum: torch.Tensor
    row_copy_count: torch.Tensor

    def __post_init__(self) -> None:
        if not runtime._sha256_text(self.identity_sha256) or self.route not in {
            "L", "R", "S0", "S1", "T",
        } or not runtime._sha256_text(self.program_sha256):
            raise ValueError("validation reduction identity is malformed")
        float_fields = ("row_primary_sum", "row_ce_sum", "row_copy_ce_sum")
        count_fields = ("row_primary_count", "row_ce_count", "row_copy_count")
        for name in (*float_fields, *count_fields):
            value = getattr(self, name)
            expected_dtype = torch.float64 if name in float_fields else torch.long
            if not torch.is_tensor(value) or tuple(value.shape) != (runtime.BATCH_SIZE,) or (
                value.dtype != expected_dtype
            ) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f"validation reduction {name} is malformed")
            object.__setattr__(self, name, value.detach().cpu().contiguous().clone())


@dataclass(frozen=True)
class LedgerSnapshot:
    run_context_sha256: str
    student_identity_count: int
    teacher_identity_count: int
    completed_identity_count: int
    student_identities_sha256: str
    teacher_identities_sha256: str
    completed_identities_sha256: str
    prepared_parent_identity_count: int
    prepared_parent_identities_sha256: str
    consumed_parent_identity_count: int
    consumed_parent_identities_sha256: str
    outstanding_parent_identity_sha256: str | None
    outstanding_identity_sha256: str | None
    rolling_ledger_sha256: str


class _CallCounter:
    def __init__(self, expected: tuple[tuple[int, int], ...]) -> None:
        self.expected = dict(expected)
        self.counts = {0: 0, 1: 0, 2: 0}
        self.active = True

    def record(self, site: int) -> None:
        if not self.active:
            raise RuntimeError("original-call ledger is inactive")
        if site not in self.counts:
            raise ValueError("original-call ledger permits only MLP0/1/2")
        self.counts[site] += 1
        if self.counts[site] > self.expected[site]:
            raise RuntimeError(f"original MLP{site} call exceeded its exact allowance")

    def close(self) -> tuple[tuple[int, int], ...]:
        self.active = False
        observed = _call_tuple(self.counts)
        if observed != tuple(self.expected.items()):
            raise RuntimeError(
                f"original-call ledger did not close exactly: {observed} != "
                f"{tuple(self.expected.items())}"
            )
        return observed

    def revoke(self) -> None:
        self.active = False


class StudentForwardCapability:
    """Narrow surface given to the future observed student adapter."""

    __slots__ = ("__bind_outer", "__counter")

    def __init__(
        self, counter: _CallCounter, bind_outer: Callable[[torch.Tensor], None],
    ) -> None:
        self.__counter = counter
        self.__bind_outer = bind_outer

    def record_original_call(self, site: int) -> None:
        self.__counter.record(site)

    def bind_outer_logits(self, logits: torch.Tensor) -> None:
        self.__bind_outer(logits)


class _EphemeralOriginalGateway:
    __slots__ = ("__calls", "__coordinator", "__lease", "__native", "__revoked")

    def __init__(
        self, *, native: Mapping[int, Callable[[torch.Tensor], torch.Tensor]],
        calls: _CallCounter, coordinator: runtime.ScopeCoordinator,
        lease: runtime.ScopeLease,
    ) -> None:
        self.__native = dict(native)
        self.__calls = calls
        self.__coordinator = coordinator
        self.__lease = lease
        self.__revoked = False

    def call(self, site: int, z: torch.Tensor) -> torch.Tensor:
        if self.__revoked:
            raise RuntimeError("original gateway was revoked")
        self.__coordinator.require_active(self.__lease)
        if site not in (0, 1):
            self.__calls.record(site)
            raise RuntimeError("suffix-transport teacher forbids original MLP2")
        if not torch.is_tensor(z) or tuple(z.shape) != (
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL,
        ) or not bool(torch.isfinite(z).all()):
            raise RuntimeError("original gateway input state is malformed")
        self.__calls.record(site)
        value = self.__native[site](z)
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(z.shape) or not bool(
            torch.isfinite(value).all()
        ):
            raise RuntimeError(f"original MLP{site} output is malformed")
        return value

    def revoke(self) -> None:
        self.__revoked = True
        self.__calls.revoke()
        self.__native.clear()


class _MappedCoordinateGateway:
    """One-use target-trajectory P/P/N correction and native-label capability."""

    __slots__ = (
        "__bases", "__calls", "__coordinator", "__labels", "__lease",
        "__native", "__program", "__program_sha256", "__replace_calls",
        "__revoked",
    )

    def __init__(
        self, *, native: Mapping[int, Callable[[torch.Tensor], torch.Tensor]],
        bases: Mapping[int, torch.Tensor], program: runtime.JointAffineProgram,
        program_sha256: str, calls: _CallCounter,
        coordinator: runtime.ScopeCoordinator, lease: runtime.ScopeLease,
    ) -> None:
        self.__native = dict(native)
        self.__bases = dict(bases)
        self.__program = program
        self.__program_sha256 = program_sha256
        self.__calls = calls
        self.__coordinator = coordinator
        self.__lease = lease
        self.__labels: dict[int, torch.Tensor] = {}
        self.__replace_calls = {0: 0, 1: 0}
        self.__revoked = False

    def correct_and_label(
        self, site: int, state: torch.Tensor, deployed_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.__revoked:
            raise RuntimeError("mapped coordinate gateway was revoked")
        self.__coordinator.require_active(self.__lease)
        if site not in (0, 1):
            raise ValueError("mapped coordinate gateway permits only MLP0/1")
        if self.__replace_calls[site] != 0:
            raise RuntimeError(f"mapped target MLP{site} was called more than once")
        if site == 1 and self.__replace_calls[0] != 1:
            raise RuntimeError("mapped target MLP1 executed before MLP0")
        if not torch.is_tensor(state) or not torch.is_tensor(deployed_output) or (
            tuple(state.shape) != (
                runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL,
            )
        ) or deployed_output.shape != state.shape or state.requires_grad or (
            state.grad_fn is not None
        ) or deployed_output.requires_grad or deployed_output.grad_fn is not None or not bool(
            torch.isfinite(state).all()
        ) or not bool(torch.isfinite(deployed_output).all()):
            raise RuntimeError("mapped target state or deployed write is malformed")
        self.__replace_calls[site] += 1
        self.__calls.record(site)
        native = self.__native[site](state)
        if not torch.is_tensor(native) or native.shape != state.shape or native.requires_grad or (
            native.grad_fn is not None
        ) or not bool(torch.isfinite(native).all()):
            raise RuntimeError(f"mapped native MLP{site} label output is malformed")
        basis = self.__bases[site].to(device=native.device, dtype=torch.float32)
        label = runtime.scored_positions(native.float() @ basis)
        predicted = self.__program.site0_code(state) if site == 0 else (
            self.__program.site1_code(state)
        )
        self.__labels[site] = label.detach().contiguous()
        return runtime.JointAffineProgram.projected_replacement(
            deployed_output, predicted, basis,
        )

    def take_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.__revoked:
            raise RuntimeError("mapped coordinate gateway was revoked")
        self.__coordinator.require_active(self.__lease)
        if self.__replace_calls != {0: 1, 1: 1} or set(self.__labels) != {0, 1}:
            raise RuntimeError("mapped target trajectory did not label MLP0/1 exactly once")
        if runtime.program_snapshot_sha256(self.__program) != self.__program_sha256:
            raise RuntimeError("mapped target program changed during label construction")
        labels = (self.__labels.pop(0), self.__labels.pop(1))
        if any(tuple(value.shape) != (
            runtime.BATCH_SIZE, runtime.SCORE_STOP - runtime.SCORE_START,
            runtime.CODE_DIM,
        ) or value.requires_grad or value.grad_fn is not None or not bool(
            torch.isfinite(value).all()
        ) for value in labels):
            raise RuntimeError("mapped coordinate labels are malformed")
        return labels

    def revoke(self) -> None:
        self.__revoked = True
        self.__calls.revoke()
        self.__native.clear()
        self.__bases.clear()
        self.__labels.clear()
        self.__program = None


class _MappedParentGateway:
    """Build one false-paired L0 code through a native-free target P/P/N path."""

    __slots__ = (
        "__bases", "__coordinator", "__lease", "__parent", "__program",
        "__program_sha256", "__replace_calls", "__revoked",
    )

    def __init__(
        self, *, bases: Mapping[int, torch.Tensor],
        program: runtime.JointAffineProgram, program_sha256: str,
        coordinator: runtime.ScopeCoordinator, lease: runtime.ScopeLease,
    ) -> None:
        self.__bases = dict(bases)
        self.__program = program
        self.__program_sha256 = program_sha256
        self.__coordinator = coordinator
        self.__lease = lease
        self.__parent: torch.Tensor | None = None
        self.__replace_calls = {0: 0, 1: 0}
        self.__revoked = False

    def correct(
        self, site: int, state: torch.Tensor, deployed_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.__revoked:
            raise RuntimeError("mapped parent gateway was revoked")
        self.__coordinator.require_active(self.__lease)
        if site not in (0, 1) or self.__replace_calls[site] != 0:
            raise RuntimeError("mapped parent target must call MLP0/1 exactly once")
        if site == 1 and self.__replace_calls[0] != 1:
            raise RuntimeError("mapped parent target MLP1 executed before MLP0")
        if not torch.is_tensor(state) or not torch.is_tensor(deployed_output) or tuple(
            state.shape
        ) != (runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL) or (
            deployed_output.shape != state.shape
        ) or state.requires_grad or state.grad_fn is not None or (
            deployed_output.requires_grad or deployed_output.grad_fn is not None
        ) or not bool(torch.isfinite(state).all()) or not bool(
            torch.isfinite(deployed_output).all()
        ):
            raise RuntimeError("mapped parent target state or deployed write is malformed")
        self.__replace_calls[site] += 1
        if site == 0:
            predicted = self.__program.site0_code(state)
            self.__parent = predicted.detach().clone().contiguous()
        else:
            # The target trajectory represents selected L, not the null's current A.
            predicted = self.__program.site1(state)
        basis = self.__bases[site].to(device=state.device, dtype=torch.float32)
        return runtime.JointAffineProgram.projected_replacement(
            deployed_output, predicted, basis,
        )

    def take_parent(self) -> torch.Tensor:
        if self.__revoked:
            raise RuntimeError("mapped parent gateway was revoked")
        self.__coordinator.require_active(self.__lease)
        if self.__replace_calls != {0: 1, 1: 1} or self.__parent is None:
            raise RuntimeError("mapped parent target trajectory did not close")
        if runtime.program_snapshot_sha256(self.__program) != self.__program_sha256:
            raise RuntimeError("mapped parent program changed during target construction")
        parent = self.__parent
        self.__parent = None
        if tuple(parent.shape) != (
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.CODE_DIM,
        ) or parent.requires_grad or parent.grad_fn is not None or not bool(
            torch.isfinite(parent).all()
        ):
            raise RuntimeError("mapped parent code is malformed")
        return parent

    def revoke(self) -> None:
        self.__revoked = True
        self.__bases.clear()
        self.__parent = None
        self.__program = None


class _TensorWitness:
    """Cheap one-process ownership witness for a tensor and its graph/storage.

    Full suffix logits are intentionally not copied to CPU for byte hashing.  This
    witness is paired with sealed one-use ownership and a source-closed adapter that
    closes the student/teacher transaction before releasing caller aliases.
    """

    __slots__ = (
        "content_sha256", "data_ptr", "descriptor", "object_id", "version",
    )

    def __init__(self, value: torch.Tensor, *, hash_content: bool) -> None:
        if not torch.is_tensor(value):
            raise TypeError("tensor witness requires a tensor")
        self.object_id = id(value)
        self.data_ptr = value.data_ptr()
        self.version = value._version
        self.descriptor = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "device_type": value.device.type, "device_index": value.device.index,
            "stride": list(value.stride()), "storage_offset": value.storage_offset(),
            "requires_grad": bool(value.requires_grad),
            "grad_fn_type": None if value.grad_fn is None else type(value.grad_fn).__name__,
        }
        self.content_sha256 = runtime.tensor_identity_sha256(value) if hash_content else None

    @property
    def logical_descriptor(self) -> Mapping[str, Any]:
        return {**self.descriptor, "content_sha256": self.content_sha256}

    def require(self, value: torch.Tensor) -> None:
        descriptor = {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "device_type": value.device.type, "device_index": value.device.index,
            "stride": list(value.stride()), "storage_offset": value.storage_offset(),
            "requires_grad": bool(value.requires_grad),
            "grad_fn_type": None if value.grad_fn is None else type(value.grad_fn).__name__,
        }
        if id(value) != self.object_id or value.data_ptr() != self.data_ptr or (
            value._version != self.version
        ) or descriptor != self.descriptor:
            raise RuntimeError("owned tensor graph/storage identity changed")
        if not bool(torch.isfinite(value.detach()).all()):
            raise RuntimeError("owned tensor became nonfinite")
        if self.content_sha256 is not None and runtime.tensor_identity_sha256(value) != (
            self.content_sha256
        ):
            raise RuntimeError("owned tensor content changed")


class _StudentOutputs:
    """One-use owner of exact student tensors and their autograd graphs."""

    __slots__ = (
        "__consumed", "__identity_sha256", "__metadata", "__sealed", "__values",
        "__witnesses",
    )

    def __init__(
        self, identity: runtime.TraceIdentity, codes: Sequence[torch.Tensor], logits: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "_StudentOutputs__sealed", False)
        if len(codes) != 2 or any(
            not torch.is_tensor(value) or tuple(value.shape) != (
                runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.CODE_DIM,
            ) or not bool(torch.isfinite(value.detach()).all()) for value in codes
        ):
            raise RuntimeError("student code outputs are malformed")
        if not torch.is_tensor(logits) or logits.ndim != 3 or tuple(logits.shape[:2]) != (
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH,
        ) or logits.shape[2] <= 1 or not bool(torch.isfinite(logits.detach()).all()):
            raise RuntimeError("student suffix logits are malformed")
        self.__identity_sha256 = identity.sha256
        self.__values = {"codes": tuple(codes), "logits": logits}
        self.__witnesses = {
            "codes": tuple(_TensorWitness(value, hash_content=True) for value in codes),
            "logits": _TensorWitness(logits, hash_content=False),
        }
        self.__metadata = {
            "codes": tuple(
                witness.logical_descriptor for witness in self.__witnesses["codes"]
            ),
            "logits_graph": self.__witnesses["logits"].logical_descriptor,
            "shapes": tuple(tuple(value.shape) for value in (*codes, logits)),
            "dtypes": tuple(str(value.dtype) for value in (*codes, logits)),
        }
        self.__consumed = False
        object.__setattr__(self, "_StudentOutputs__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_StudentOutputs__sealed", False):
            raise AttributeError("student outputs are sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("student outputs cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("student outputs cannot be copied")

    def __reduce__(self):
        raise RuntimeError("student outputs cannot be serialized")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "identity_sha256": self.__identity_sha256, **self.__metadata,
        })

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        return self.__metadata["shapes"]

    @property
    def dtypes(self) -> tuple[str, ...]:
        return self.__metadata["dtypes"]

    @property
    def any_requires_grad(self) -> bool:
        values = (*self.__values["codes"], self.__values["logits"])
        return any(value.requires_grad for value in values)

    @property
    def all_grad_fn_absent(self) -> bool:
        values = (*self.__values["codes"], self.__values["logits"])
        return all(value.grad_fn is None for value in values)

    def require_loss_graph(
        self, route: str, program: runtime.JointAffineProgram,
    ) -> None:
        required = self.__values["codes"] if route == "L" else (
            (self.__values["logits"],)
        )
        if any(not value.requires_grad or value.grad_fn is None for value in required):
            raise RuntimeError("fit student loss outputs are detached from their graph")
        program.require_exact_trainability()
        named = dict(program.named_parameters())
        names = program.expected_trainable_parameter_names
        parameters = tuple(named[name] for name in names)
        probe = sum(value.float().sum() for value in required)
        gradients = torch.autograd.grad(
            probe, parameters, retain_graph=True, create_graph=False, allow_unused=True,
        )
        if any(value is None for value in gradients):
            missing = tuple(name for name, value in zip(names, gradients, strict=True) if value is None)
            raise RuntimeError(
                f"student loss graph is disconnected from route parameters: {missing}"
            )

    def _require_integrity(self) -> None:
        codes = self.__values["codes"]
        logits = self.__values["logits"]
        try:
            for witness, value in zip(self.__witnesses["codes"], codes, strict=True):
                witness.require(value)
            self.__witnesses["logits"].require(logits)
        except RuntimeError as error:
            raise RuntimeError("student output mutated after its outer return") from error

    def consume(self, kind: str, identity: runtime.TraceIdentity):
        if self.__consumed:
            raise RuntimeError("student outputs were already consumed")
        if identity.sha256 != self.__identity_sha256 or kind not in {
            "coordinate", "oon", "discard", "validation",
        }:
            raise RuntimeError("student output identity or consumer kind changed")
        self._require_integrity()
        object.__setattr__(self, "_StudentOutputs__consumed", True)
        if kind == "coordinate":
            output = self.__values["codes"]
        elif kind == "oon":
            output = self.__values["logits"]
        elif kind == "validation":
            output = (self.__values["codes"], self.__values["logits"])
        else:
            output = None
        self.__values.clear()
        self.__witnesses.clear()
        return output

    def force_discard(self, identity: runtime.TraceIdentity) -> None:
        """Identity-bound cleanup which cannot be blocked by corrupted tensors."""

        if identity.sha256 != self.__identity_sha256:
            raise RuntimeError("student output cleanup identity changed")
        object.__setattr__(self, "_StudentOutputs__consumed", True)
        self.__values.clear()
        self.__witnesses.clear()


class StudentStep:
    """Sealed pairing of detached states and exact graph-bearing outputs."""

    __slots__ = (
        "__identity_sha256", "__issuer_id", "__outputs", "__sealed", "__spent", "__trace",
    )

    def __init__(
        self, *, issuer_id: str, identity: runtime.TraceIdentity,
        trace: runtime.StudentTrace, outputs: _StudentOutputs,
    ) -> None:
        object.__setattr__(self, "_StudentStep__sealed", False)
        self.__issuer_id = issuer_id
        self.__identity_sha256 = identity.sha256
        self.__trace = trace
        self.__outputs = outputs
        self.__spent = False
        object.__setattr__(self, "_StudentStep__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_StudentStep__sealed", False):
            raise AttributeError("student step is sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("student steps cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("student steps cannot be copied")

    def __reduce__(self):
        raise RuntimeError("student steps cannot be serialized")

    @property
    def output_sha256(self) -> str:
        return self.__outputs.sha256

    def _take(
        self, *, issuer_id: str, identity: runtime.TraceIdentity,
    ) -> tuple[runtime.StudentTrace, _StudentOutputs]:
        if self.__spent:
            raise RuntimeError("student step was already consumed")
        if issuer_id != self.__issuer_id or identity.sha256 != self.__identity_sha256:
            raise RuntimeError("student step identity or issuer mismatch")
        object.__setattr__(self, "_StudentStep__spent", True)
        return self.__trace, self.__outputs

    def _require_available(self, *, issuer_id: str, identity: runtime.TraceIdentity) -> None:
        if self.__spent:
            raise RuntimeError("student step was already consumed")
        if issuer_id != self.__issuer_id or identity.sha256 != self.__identity_sha256:
            raise RuntimeError("student step identity or issuer mismatch")


class _TeacherResult:
    __slots__ = (
        "__broker", "__consumed", "__identity", "__metadata", "__metadata_sha256",
        "__output_sha256", "__sealed", "__student_outputs", "__tensors",
        "__witnesses",
    )

    def __init__(
        self, *, broker: "CapabilityBroker", identity: runtime.TraceIdentity,
        tensors: Sequence[torch.Tensor], student_outputs: _StudentOutputs,
        metadata: Mapping[str, Any],
    ) -> None:
        object.__setattr__(self, "_TeacherResult__sealed", False)
        detached = tuple(value.detach().clone().contiguous() for value in tensors)
        if any(value.requires_grad or value.grad_fn is not None for value in detached):
            raise RuntimeError("teacher tensors did not detach")
        self.__broker = broker
        self.__identity = identity
        self.__student_outputs = student_outputs
        self.__tensors = list(detached)
        hash_content = identity.teacher_kind == "coordinate_labels"
        self.__witnesses = tuple(
            _TensorWitness(value, hash_content=hash_content) for value in detached
        )
        self.__consumed = False
        self.__metadata = dict(metadata)
        self.__metadata_sha256 = runtime.logical_identity_sha256(self.__metadata)
        self.__output_sha256 = runtime.logical_identity_sha256([
            witness.logical_descriptor for witness in self.__witnesses
        ])
        object.__setattr__(self, "_TeacherResult__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_TeacherResult__sealed", False):
            raise AttributeError("teacher result is sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("teacher results cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("teacher results cannot be copied")

    def __reduce__(self):
        raise RuntimeError("teacher results cannot be serialized")

    def _begin_consume(self, kind: str):
        if self.__consumed:
            raise RuntimeError("teacher result was already consumed")
        try:
            for witness, value in zip(self.__witnesses, self.__tensors, strict=True):
                witness.require(value)
        except RuntimeError as error:
            raise RuntimeError("teacher result tensor mutated before consumption") from error
        if runtime.logical_identity_sha256(self.__metadata) != self.__metadata_sha256:
            raise RuntimeError("teacher result tensor mutated before consumption")
        object.__setattr__(self, "_TeacherResult__consumed", True)
        student = self.__student_outputs.consume(kind, self.__identity)
        return tuple(self.__tensors), student

    def _finish_consume(self) -> StepClosure:
        closure = self.__broker._mint_closure(
            identity=self.__identity, output_sha256=self.__output_sha256,
            consumed=True, **self.__metadata,
        )
        self.__broker._complete_identity(self.__identity)
        return closure

    def _clear(self) -> None:
        self.__tensors.clear()
        object.__setattr__(self, "_TeacherResult__witnesses", ())

    def _abort_consume(self) -> None:
        self.__student_outputs.force_discard(self.__identity)
        self.__broker._abort_identity(self.__identity)


def _validation_role_batch(
    identity: runtime.TraceIdentity, role_rows: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or role_rows.ndim != 2 or (
        tuple(role_rows.shape) != (runtime.BATCH_SIZE, 513)
    ) or role_rows.device.type != "cpu":
        raise RuntimeError("validation reduction requires one CPU role-row batch")
    rows = role_rows.contiguous()
    identity.require_inputs(rows[:, :runtime.SEQUENCE_LENGTH])
    return rows


class CoordinateTeacherResult(_TeacherResult):
    def consume_loss(
        self, denominators: Sequence[torch.Tensor | float],
    ) -> tuple[torch.Tensor, StepClosure]:
        if self._TeacherResult__identity.phase != "fit" or (
            self._TeacherResult__identity.route != "L"
        ):
            raise RuntimeError("coordinate loss is licensed only for the fit/L route")
        complete = False
        try:
            labels, predictions = self._begin_consume("coordinate")
            loss = runtime.normalized_local_loss(predictions, labels, denominators)
            closure = self._finish_consume()
            complete = True
            return loss, closure
        finally:
            self._clear()
            if not complete:
                self._abort_consume()

    def consume_moments(
        self,
    ) -> tuple[tuple[runtime.MomentSufficientStatistics, ...], StepClosure]:
        if self._TeacherResult__identity.phase != "initial_denominator" or (
            self._TeacherResult__identity.route != "Q"
        ):
            raise RuntimeError("coordinate moments are licensed only for initial Q")
        complete = False
        try:
            labels, _ = self._begin_consume("discard")
            moments = tuple(runtime.MomentSufficientStatistics.from_labels(value) for value in labels)
            closure = self._finish_consume()
            complete = True
            return moments, closure
        finally:
            self._clear()
            if not complete:
                self._abort_consume()

    def consume_validation(
        self, role_rows: torch.Tensor, denominators: Sequence[torch.Tensor | float],
    ) -> tuple[ValidationBatchReductions, StepClosure]:
        if self._TeacherResult__identity.phase != "validation" or (
            self._TeacherResult__identity.route != "L"
        ):
            raise RuntimeError("coordinate validation is licensed only for validation/L")
        complete = False
        try:
            labels, student = self._begin_consume("validation")
            role_rows = _validation_role_batch(self._TeacherResult__identity, role_rows)
            predictions, logits = student
            import early_mlp_suffix_transport_v1_programs as programs

            primary_sum, primary_count = programs.local_primary_rows(
                predictions, labels, denominators,
            )
            ce_sum, ce_count, copy_sum, copy_count = programs.ce_and_copy_rows(
                logits, role_rows,
            )
            reductions = ValidationBatchReductions(
                identity_sha256=self._TeacherResult__identity.sha256,
                route="L",
                program_sha256=self._TeacherResult__identity.program_snapshot_sha256,
                row_primary_sum=primary_sum, row_primary_count=primary_count,
                row_ce_sum=ce_sum, row_ce_count=ce_count,
                row_copy_ce_sum=copy_sum, row_copy_count=copy_count,
            )
            closure = self._finish_consume()
            complete = True
            return reductions, closure
        finally:
            self._clear()
            if not complete:
                self._abort_consume()


class OONTeacherResult(_TeacherResult):
    def consume_loss(self) -> tuple[torch.Tensor, StepClosure]:
        if self._TeacherResult__identity.phase != "fit" or (
            self._TeacherResult__identity.route not in {"R", "S0", "S1", "T"}
        ):
            raise RuntimeError("OON loss is licensed only for fit R/S/T routes")
        complete = False
        try:
            (teacher_logits,), student_logits = self._begin_consume("oon")
            if student_logits.shape[1] == runtime.SEQUENCE_LENGTH:
                student_logits = runtime.scored_positions(student_logits)
            loss = runtime.teacher_student_kl(teacher_logits, student_logits)
            closure = self._finish_consume()
            complete = True
            return loss, closure
        finally:
            self._clear()
            if not complete:
                self._abort_consume()

    def consume_validation(
        self, role_rows: torch.Tensor,
    ) -> tuple[ValidationBatchReductions, StepClosure]:
        identity = self._TeacherResult__identity
        if identity.phase != "validation" or identity.route not in {"R", "S0", "S1", "T"}:
            raise RuntimeError("OON validation is licensed only for validation R/S/T")
        complete = False
        try:
            (teacher_logits,), student = self._begin_consume("validation")
            role_rows = _validation_role_batch(identity, role_rows)
            _, student_logits = student
            import early_mlp_suffix_transport_v1_programs as programs

            primary_sum, primary_count = programs.suffix_kl_rows(
                teacher_logits, student_logits,
            )
            ce_sum, ce_count, copy_sum, copy_count = programs.ce_and_copy_rows(
                student_logits, role_rows,
            )
            reductions = ValidationBatchReductions(
                identity_sha256=identity.sha256, route=identity.route,
                program_sha256=identity.program_snapshot_sha256,
                row_primary_sum=primary_sum, row_primary_count=primary_count,
                row_ce_sum=ce_sum, row_ce_count=ce_count,
                row_copy_ce_sum=copy_sum, row_copy_count=copy_count,
            )
            closure = self._finish_consume()
            complete = True
            return reductions, closure
        finally:
            self._clear()
            if not complete:
                self._abort_consume()


class StudentSession:
    __slots__ = (
        "__broker", "__closed", "__counter", "__forward_complete", "__hook",
        "__identity", "__monitor", "__outer_logits", "__spent",
    )

    def __init__(
        self, broker: "CapabilityBroker", identity: runtime.TraceIdentity,
        hook: runtime.StudentCorrectionHook,
    ) -> None:
        self.__broker = broker
        self.__identity = identity
        self.__hook = hook
        self.__counter = _CallCounter(EXACT_ZERO_CALLS)
        self.__outer_logits: torch.Tensor | None = None
        self.__monitor = StudentForwardCapability(self.__counter, self._bind_outer_logits)
        self.__forward_complete = False
        self.__closed = False
        self.__spent = False

    def _bind_outer_logits(self, logits: torch.Tensor) -> None:
        if self.__outer_logits is not None:
            raise RuntimeError("student outer logits were already bound")
        if not torch.is_tensor(logits) or logits.ndim != 3 or tuple(logits.shape[:2]) != (
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH,
        ) or logits.shape[2] <= 1 or not bool(torch.isfinite(logits.detach()).all()):
            raise RuntimeError("student outer logits are malformed")
        self.__outer_logits = logits

    @contextmanager
    def forward_scope(self):
        if self.__spent:
            raise RuntimeError("student session was already spent")
        self.__spent = True
        try:
            with self.__hook.forward_scope(self.__identity, capture_sites={0, 1}):
                yield self.__monitor
                if self.__outer_logits is None:
                    raise RuntimeError("student outer return did not bind its logits")
                self.__counter.close()
            self.__forward_complete = True
        except BaseException:
            self.__counter.revoke()
            self.__hook.discard_student_codes()
            self.__hook.clear_configuration()
            self.__broker._abort_identity(self.__identity)
            raise

    def close(
        self, *, outer_forward_count: int, outer_returned: bool,
        hook_restored: bool, hook_inert: bool,
    ) -> tuple[StudentStep, StepClosure]:
        if self.__closed:
            raise RuntimeError("student session was already closed")
        self.__closed = True
        if not self.__forward_complete or self.__outer_logits is None:
            try:
                self.__hook.discard_student_codes()
                self.__hook.clear_configuration()
            finally:
                self.__broker._abort_identity(self.__identity)
            raise RuntimeError("student session lacks a clean completed forward")
        trace: runtime.StudentTrace | None = None
        outputs: _StudentOutputs | None = None
        try:
            trace = self.__hook.pop_trace(self.__identity)
            codes = self.__hook.pop_student_codes(self.__identity)
            valid = (
                type(outer_forward_count) is int and outer_forward_count == 1
                and outer_returned is True and hook_restored is True and hook_inert is True
                and trace.student_calls == {0: 1, 1: 1}
            )
            if not valid:
                raise RuntimeError("student session execution closure failed")
            outputs = _StudentOutputs(self.__identity, codes, self.__outer_logits)
            if self.__identity.phase == "fit":
                if self.__hook.program is None:
                    raise RuntimeError("student program vanished before graph validation")
                outputs.require_loss_graph(self.__identity.route, self.__hook.program)
            step = StudentStep(
                issuer_id=self.__broker.issuer_id, identity=self.__identity,
                trace=trace, outputs=outputs,
            )
            self.__hook.clear_configuration()
        except BaseException:
            try:
                if outputs is not None:
                    outputs.force_discard(self.__identity)
                if trace is not None and not trace.consumed:
                    trace._discard(issuer_id=self.__broker.issuer_id, identity=self.__identity)
            finally:
                self.__hook.discard_student_codes()
                # Trace discard releases the hook's outstanding nonce.
                try:
                    self.__hook.clear_configuration()
                finally:
                    self.__broker._abort_identity(self.__identity)
            raise
        assert outputs is not None
        closure = self.__broker._mint_closure(
            identity=self.__identity, scope="student", producer_invocations=1,
            outer_forward_count=1, hook_calls=((0, 1), (1, 1), (2, 0)),
            original_calls=EXACT_ZERO_CALLS, outer_returned=True,
            hook_restored=True, hook_inert=True, output_shapes=outputs.shapes,
            output_dtypes=outputs.dtypes, support="0:256-current-state+student-output",
            requires_grad=outputs.any_requires_grad,
            grad_fn_absent=outputs.all_grad_fn_absent,
            consumed=False, output_sha256=outputs.sha256,
        )
        return step, closure


class CapabilityBroker:
    """Sealed process-local issuer for one-at-a-time student/teacher transactions."""

    __slots__ = (
        "__bases", "__basis_sha256", "__completed_identities", "__coordinator",
        "__consumed_parent_identities",
        "__issuer_id", "__mapped_authority", "__native_calls",
        "__outstanding_identity_sha256", "__outstanding_parent_identity_sha256",
        "__prepared_parent_identities", "__rolling_ledger_sha256",
        "__run_context", "__run_context_sha256", "__sealed",
        "__student_identities", "__teacher_identities",
    )

    def __init__(
        self, *, issuer_id: str, coordinator: runtime.ScopeCoordinator,
        run_context: RunContext | ValidationRunContext, bases: Mapping[int, torch.Tensor],
        native_calls: Mapping[int, Callable[[torch.Tensor], torch.Tensor]],
        mapped_authority: MappedRunAuthority | None = None,
    ) -> None:
        object.__setattr__(self, "_CapabilityBroker__sealed", False)
        if not runtime._sha256_text(issuer_id) or not isinstance(
            coordinator, runtime.ScopeCoordinator,
        ) or not isinstance(run_context, (RunContext, ValidationRunContext)) or set(
            bases
        ) != {0, 1} or set(
            native_calls
        ) != {0, 1}:
            raise ValueError("capability broker construction is malformed")
        if mapped_authority is not None and (
            not isinstance(run_context, RunContext)
            or
            not isinstance(mapped_authority, MappedRunAuthority)
            or mapped_authority.base_context != run_context
            or not runtime._sha256_text(mapped_authority.sha256)
        ):
            raise ValueError("mapped authority differs from the sealed run context")
        self.__issuer_id = issuer_id
        self.__coordinator = coordinator
        self.__run_context = run_context
        self.__mapped_authority = mapped_authority
        self.__run_context_sha256 = (
            run_context.sha256 if mapped_authority is None else mapped_authority.sha256
        )
        self.__bases = {}
        for site in (0, 1):
            basis = bases[site].detach().cpu().float().contiguous().clone()
            contract.validate_orthonormal_basis(f"basis{site}", basis)
            self.__bases[site] = basis
        self.__basis_sha256 = {
            site: runtime.tensor_identity_sha256(value) for site, value in self.__bases.items()
        }
        self.__native_calls = dict(native_calls)
        self.__student_identities: set[str] = set()
        self.__teacher_identities: set[str] = set()
        self.__completed_identities: set[str] = set()
        self.__prepared_parent_identities: set[str] = set()
        self.__consumed_parent_identities: set[str] = set()
        self.__outstanding_identity_sha256: str | None = None
        self.__outstanding_parent_identity_sha256: str | None = None
        self.__rolling_ledger_sha256 = "0" * 64
        object.__setattr__(self, "_CapabilityBroker__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_CapabilityBroker__sealed", False):
            raise AttributeError("capability broker is sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("capability brokers cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("capability brokers cannot be copied")

    def __reduce__(self):
        raise RuntimeError("capability brokers cannot be serialized")

    @property
    def issuer_id(self) -> str:
        return self.__issuer_id

    @property
    def ledger_snapshot(self) -> LedgerSnapshot:
        def digest(values: set[str]) -> str:
            return runtime.logical_identity_sha256(sorted(values))
        return LedgerSnapshot(
            run_context_sha256=self.__run_context_sha256,
            student_identity_count=len(self.__student_identities),
            teacher_identity_count=len(self.__teacher_identities),
            completed_identity_count=len(self.__completed_identities),
            student_identities_sha256=digest(self.__student_identities),
            teacher_identities_sha256=digest(self.__teacher_identities),
            completed_identities_sha256=digest(self.__completed_identities),
            prepared_parent_identity_count=len(self.__prepared_parent_identities),
            prepared_parent_identities_sha256=digest(self.__prepared_parent_identities),
            consumed_parent_identity_count=len(self.__consumed_parent_identities),
            consumed_parent_identities_sha256=digest(self.__consumed_parent_identities),
            outstanding_parent_identity_sha256=self.__outstanding_parent_identity_sha256,
            outstanding_identity_sha256=self.__outstanding_identity_sha256,
            rolling_ledger_sha256=self.__rolling_ledger_sha256,
        )

    def begin_student(
        self, identity: runtime.TraceIdentity, hook: runtime.StudentCorrectionHook,
        inputs: torch.Tensor, ordered_batch_indices: Sequence[int],
    ) -> StudentSession:
        if self.__mapped_authority is None:
            self.__run_context.require_identity(identity, inputs, ordered_batch_indices)
        else:
            legal_mapped = identity.control == "document_shuffle" and (
                (identity.route == "L" and identity.teacher_kind == "coordinate_labels")
                or (
                    identity.route in {"R", "S0", "S1"}
                    and identity.teacher_kind == "oon_logits"
                )
            )
            legal_mapped = legal_mapped or (
                identity.control.startswith("A_null_") and identity.route == "T"
                and identity.teacher_kind == "oon_logits"
            )
            if not legal_mapped:
                raise RuntimeError(
                    "mapped broker currently licenses only document-shuffled L/R/S"
                )
            self.__mapped_authority.require_source_identity(
                identity, inputs, ordered_batch_indices,
            )
        if identity.sha256 in self.__student_identities:
            raise RuntimeError("student logical identity is duplicated")
        if self.__outstanding_identity_sha256 is not None:
            raise RuntimeError("a previous student/teacher transaction remains outstanding")
        if hook.issuer_id != self.issuer_id or hook.coordinator is not self.__coordinator:
            raise RuntimeError("student hook belongs to a different capability broker")
        if dict(hook.basis_sha256) != self.__basis_sha256:
            raise RuntimeError("student hook and teacher broker use different bases")
        if identity.phase == "fit" and identity.control.startswith("A_null_"):
            if identity.sha256 not in self.__consumed_parent_identities or not (
                hook.has_mapped_parent
            ) or hook.mapped_parent_identity_sha256 != identity.sha256:
                raise RuntimeError("A-null student lacks its prepared mapped parent")
        elif hook.has_mapped_parent:
            raise RuntimeError("non-A-null student cannot consume a mapped parent")
        expected_program_route = "L" if identity.route == "Q" else identity.route
        if hook.program is None or hook.program.route != expected_program_route or (
            identity.program_snapshot_sha256 != runtime.program_snapshot_sha256(hook.program)
        ):
            raise RuntimeError("trace route/program differs from configured execution")
        hook.program.require_exact_trainability()
        if identity.student_states != ((0, "P"), (1, "P"), (2, "N")) or hook.states != {
            0: "P", 1: "P",
        }:
            raise RuntimeError("fit execution must be exact P/P/N")
        self.__student_identities.add(identity.sha256)
        object.__setattr__(self, "_CapabilityBroker__outstanding_identity_sha256", identity.sha256)
        return StudentSession(self, identity, hook)

    def _release_mapped_parent(self, identity_sha256: str) -> None:
        if self.__outstanding_parent_identity_sha256 != identity_sha256:
            raise RuntimeError("mapped parent release identity changed")
        self.__consumed_parent_identities.add(identity_sha256)
        object.__setattr__(
            self, "_CapabilityBroker__outstanding_parent_identity_sha256", None,
        )

    def prepare_mapped_parent(
        self, identity: runtime.TraceIdentity, *, fit_rows: torch.Tensor,
        student_inputs: torch.Tensor, student_indices: Sequence[int],
        teacher_inputs: torch.Tensor, teacher_indices: Sequence[int],
        program: runtime.JointAffineProgram,
        autonomous_forward: Callable[[Any, torch.Tensor], Mapping[str, Any]],
    ) -> tuple[runtime.MappedParentCode, StepClosure]:
        """Construct one sealed false-paired parent before its source forward."""

        if self.__mapped_authority is None or not identity.control.startswith(
            "A_null_"
        ) or identity.route != "T" or identity.teacher_kind != "oon_logits":
            raise RuntimeError("mapped parent preparation requires an A-null T identity")
        self.__mapped_authority.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_inputs,
            student_indices=student_indices, teacher_inputs=teacher_inputs,
            teacher_indices=teacher_indices,
        )
        if not isinstance(program, runtime.JointAffineProgram) or program.route != "T":
            raise RuntimeError("mapped parent preparation requires the frozen-L T topology")
        program.require_exact_trainability()
        program_sha256 = runtime.program_snapshot_sha256(program)
        if identity.program_snapshot_sha256 != program_sha256:
            raise RuntimeError("mapped parent program differs from the trace identity")
        if identity.sha256 in self.__prepared_parent_identities or (
            self.__outstanding_parent_identity_sha256 is not None
        ) or self.__outstanding_identity_sha256 is not None:
            raise RuntimeError("mapped parent identity is duplicated or another transaction exists")
        with self.__coordinator.enter("coordinate") as lease:
            gateway = _MappedParentGateway(
                bases=self.__bases, program=program, program_sha256=program_sha256,
                coordinator=self.__coordinator, lease=lease,
            )
            try:
                with torch.no_grad():
                    closure_data = autonomous_forward(
                        gateway, teacher_inputs.detach().clone(),
                    )
                required_closure = {
                    "outer_forward_count": 1,
                    "hook_calls": {0: 1, 1: 1, 2: 0},
                    "outer_returned": True, "hook_restored": True,
                    "hook_inert": True,
                }
                if not isinstance(closure_data, Mapping) or dict(
                    closure_data
                ) != required_closure:
                    raise RuntimeError("mapped parent execution closure changed")
                parent = gateway.take_parent()
            finally:
                gateway.revoke()
        self.__prepared_parent_identities.add(identity.sha256)
        object.__setattr__(
            self, "_CapabilityBroker__outstanding_parent_identity_sha256", identity.sha256,
        )
        try:
            handle = runtime.MappedParentCode(
                value=parent, identity_sha256=identity.sha256,
                issuer_id=self.issuer_id, program_sha256=program_sha256,
                release=self._release_mapped_parent,
            )
            closure = self._mint_closure(
                identity=identity, scope="mapped_parent", producer_invocations=1,
                outer_forward_count=1, hook_calls=((0, 1), (1, 1), (2, 0)),
                original_calls=EXACT_ZERO_CALLS, outer_returned=True,
                hook_restored=True, hook_inert=True,
                output_shapes=(tuple(parent.shape),), output_dtypes=(str(parent.dtype),),
                support="0:256-mapped-parent-code", requires_grad=False,
                grad_fn_absent=True, consumed=False, output_sha256=handle.sha256,
            )
            return handle, closure
        except BaseException:
            self.__prepared_parent_identities.discard(identity.sha256)
            object.__setattr__(
                self, "_CapabilityBroker__outstanding_parent_identity_sha256", None,
            )
            raise

    def _abort_identity(self, identity: runtime.TraceIdentity) -> None:
        if self.__outstanding_identity_sha256 == identity.sha256:
            object.__setattr__(self, "_CapabilityBroker__outstanding_identity_sha256", None)

    def _complete_identity(self, identity: runtime.TraceIdentity) -> None:
        if self.__outstanding_identity_sha256 != identity.sha256:
            raise RuntimeError("completed identity is not the outstanding transaction")
        self.__completed_identities.add(identity.sha256)
        object.__setattr__(self, "_CapabilityBroker__outstanding_identity_sha256", None)

    def _spend_teacher_identity(self, identity: runtime.TraceIdentity) -> None:
        if identity.sha256 not in self.__student_identities or identity.sha256 in (
            self.__teacher_identities
        ) or self.__outstanding_identity_sha256 != identity.sha256:
            raise RuntimeError("teacher identity lacks one unused paired student step")
        self.__teacher_identities.add(identity.sha256)

    def run_coordinate_teacher(
        self, identity: runtime.TraceIdentity, step: StudentStep,
    ) -> CoordinateTeacherResult:
        if self.__mapped_authority is not None:
            raise RuntimeError("mapped coordinate execution is not implemented")
        if identity.teacher_kind != "coordinate_labels":
            raise RuntimeError("coordinate capability received a non-coordinate identity")
        step._require_available(issuer_id=self.issuer_id, identity=identity)
        self._spend_teacher_identity(identity)
        trace, outputs = step._take(issuer_id=self.issuer_id, identity=identity)
        try:
            with self.__coordinator.enter("coordinate") as lease:
                states = trace._consume(issuer_id=self.issuer_id, identity=identity)
                counter = _CallCounter(EXACT_EARLY_ORIGINAL_CALLS)
                gateway = _EphemeralOriginalGateway(
                    native=self.__native_calls, calls=counter,
                    coordinator=self.__coordinator, lease=lease,
                )
                labels: list[torch.Tensor] = []
                try:
                    with torch.no_grad():
                        for site in (0, 1):
                            native = gateway.call(site, states.pop(site))
                            basis = self.__bases[site].to(native.device)
                            labels.append((native.float() @ basis)[:, runtime.SCORE_START:])
                    calls = counter.close()
                    for site, label in enumerate(labels):
                        if tuple(label.shape) != (
                            runtime.BATCH_SIZE, runtime.SCORE_STOP - runtime.SCORE_START,
                            runtime.CODE_DIM,
                        ) or not bool(torch.isfinite(label).all()):
                            raise RuntimeError(f"coordinate MLP{site} label is malformed")
                finally:
                    states.clear()
                    gateway.revoke()
        except BaseException:
            try:
                if not trace.consumed:
                    trace._discard(issuer_id=self.issuer_id, identity=identity)
            finally:
                outputs.force_discard(identity)
                self._abort_identity(identity)
            raise
        metadata = {
            "scope": "coordinate", "producer_invocations": 1,
            "outer_forward_count": 0, "hook_calls": EXACT_ZERO_CALLS,
            "original_calls": calls, "outer_returned": True,
            "hook_restored": True, "hook_inert": True,
            "output_shapes": tuple(tuple(value.shape) for value in labels),
            "output_dtypes": tuple(str(value.dtype) for value in labels),
            "support": "64:256", "requires_grad": False, "grad_fn_absent": True,
        }
        try:
            return CoordinateTeacherResult(
                broker=self, identity=identity, tensors=labels,
                student_outputs=outputs, metadata=metadata,
            )
        except BaseException:
            outputs.force_discard(identity)
            self._abort_identity(identity)
            raise

    def run_mapped_coordinate_teacher(
        self, identity: runtime.TraceIdentity, step: StudentStep, *,
        fit_rows: torch.Tensor, student_inputs: torch.Tensor,
        student_indices: Sequence[int], teacher_inputs: torch.Tensor,
        teacher_indices: Sequence[int], program: runtime.JointAffineProgram,
        autonomous_forward: Callable[[Any, torch.Tensor], Mapping[str, Any]],
    ) -> CoordinateTeacherResult:
        """Label a mapped P/P/N target trajectory against source predictions."""

        if self.__mapped_authority is None:
            raise RuntimeError("ordinary broker cannot execute a mapped coordinate teacher")
        if identity.control != "document_shuffle" or identity.route != "L" or (
            identity.teacher_kind != "coordinate_labels"
        ):
            raise RuntimeError("mapped coordinate capability requires document-shuffled L")
        self.__mapped_authority.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_inputs,
            student_indices=student_indices, teacher_inputs=teacher_inputs,
            teacher_indices=teacher_indices,
        )
        if not isinstance(program, runtime.JointAffineProgram) or program.route != "L":
            raise RuntimeError("mapped coordinate target requires the fitted L program")
        program.require_exact_trainability()
        program_sha256 = runtime.program_snapshot_sha256(program)
        if identity.program_snapshot_sha256 != program_sha256:
            raise RuntimeError("mapped target program differs from the student identity")
        step._require_available(issuer_id=self.issuer_id, identity=identity)
        self._spend_teacher_identity(identity)
        trace, outputs = step._take(issuer_id=self.issuer_id, identity=identity)
        gateway: _MappedCoordinateGateway | None = None
        try:
            with self.__coordinator.enter("coordinate") as lease:
                states = trace._consume(issuer_id=self.issuer_id, identity=identity)
                states.clear()
                counter = _CallCounter(EXACT_EARLY_ORIGINAL_CALLS)
                gateway = _MappedCoordinateGateway(
                    native=self.__native_calls, bases=self.__bases, program=program,
                    program_sha256=program_sha256, calls=counter,
                    coordinator=self.__coordinator, lease=lease,
                )
                try:
                    with torch.no_grad():
                        closure = autonomous_forward(
                            gateway, teacher_inputs.detach().clone(),
                        )
                    required_closure = {
                        "outer_forward_count": 1,
                        "hook_calls": {0: 1, 1: 1, 2: 0},
                        "outer_returned": True, "hook_restored": True,
                        "hook_inert": True,
                    }
                    if not isinstance(closure, Mapping) or dict(closure) != required_closure:
                        raise RuntimeError("mapped coordinate execution closure changed")
                    calls = counter.close()
                    labels = gateway.take_labels()
                finally:
                    gateway.revoke()
        except BaseException:
            try:
                if not trace.consumed:
                    trace._discard(issuer_id=self.issuer_id, identity=identity)
            finally:
                outputs.force_discard(identity)
                self._abort_identity(identity)
            raise
        metadata = {
            "scope": "mapped_coordinate", "producer_invocations": 1,
            "outer_forward_count": 1, "hook_calls": ((0, 1), (1, 1), (2, 0)),
            "original_calls": calls, "outer_returned": True,
            "hook_restored": True, "hook_inert": True,
            "output_shapes": tuple(tuple(value.shape) for value in labels),
            "output_dtypes": tuple(str(value.dtype) for value in labels),
            "support": "64:256", "requires_grad": False,
            "grad_fn_absent": True,
        }
        try:
            return CoordinateTeacherResult(
                broker=self, identity=identity, tensors=labels,
                student_outputs=outputs, metadata=metadata,
            )
        except BaseException:
            outputs.force_discard(identity)
            self._abort_identity(identity)
            raise

    def run_oon_teacher(
        self, identity: runtime.TraceIdentity, step: StudentStep, inputs: torch.Tensor,
        autonomous_forward: Callable[[Any, torch.Tensor], tuple[torch.Tensor, Mapping[str, Any]]],
    ) -> OONTeacherResult:
        if self.__mapped_authority is not None:
            raise RuntimeError("mapped broker requires the mapped OON entry point")
        if identity.teacher_kind != "oon_logits":
            raise RuntimeError("OON capability received a non-OON identity")
        identity.require_inputs(inputs)
        return self._run_oon_teacher_after_authority(
            identity, step, inputs, autonomous_forward,
        )

    def run_mapped_oon_teacher(
        self, identity: runtime.TraceIdentity, step: StudentStep, *,
        fit_rows: torch.Tensor, student_inputs: torch.Tensor,
        student_indices: Sequence[int], teacher_inputs: torch.Tensor,
        teacher_indices: Sequence[int],
        autonomous_forward: Callable[[Any, torch.Tensor], tuple[torch.Tensor, Mapping[str, Any]]],
    ) -> OONTeacherResult:
        """Execute an O/O/N teacher on the one plan-authorized target batch."""

        if self.__mapped_authority is None:
            raise RuntimeError("ordinary broker cannot execute a mapped OON teacher")
        if identity.teacher_kind != "oon_logits" or identity.control != (
            "document_shuffle"
        ) or identity.route not in {"R", "S0", "S1"}:
            raise RuntimeError(
                "mapped broker currently licenses only document-shuffled R/S OON"
            )
        self.__mapped_authority.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_inputs,
            student_indices=student_indices, teacher_inputs=teacher_inputs,
            teacher_indices=teacher_indices,
        )
        return self._run_oon_teacher_after_authority(
            identity, step, teacher_inputs, autonomous_forward,
        )

    def run_a_null_oon_teacher(
        self, identity: runtime.TraceIdentity, step: StudentStep, *,
        fit_rows: torch.Tensor, student_inputs: torch.Tensor,
        student_indices: Sequence[int], teacher_inputs: torch.Tensor,
        teacher_indices: Sequence[int],
        autonomous_forward: Callable[[Any, torch.Tensor], tuple[torch.Tensor, Mapping[str, Any]]],
    ) -> OONTeacherResult:
        """Score an A-null source trajectory against its true source O/O/N teacher."""

        if self.__mapped_authority is None or not identity.control.startswith(
            "A_null_"
        ) or identity.route != "T" or identity.teacher_kind != "oon_logits":
            raise RuntimeError("A-null OON capability requires a mapped T identity")
        self.__mapped_authority.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_inputs,
            student_indices=student_indices, teacher_inputs=teacher_inputs,
            teacher_indices=teacher_indices,
        )
        if identity.sha256 not in self.__consumed_parent_identities:
            raise RuntimeError("A-null OON teacher lacks a consumed mapped parent")
        return self._run_oon_teacher_after_authority(
            identity, step, student_inputs, autonomous_forward,
        )

    def _run_oon_teacher_after_authority(
        self, identity: runtime.TraceIdentity, step: StudentStep, inputs: torch.Tensor,
        autonomous_forward: Callable[[Any, torch.Tensor], tuple[torch.Tensor, Mapping[str, Any]]],
    ) -> OONTeacherResult:
        step._require_available(issuer_id=self.issuer_id, identity=identity)
        self._spend_teacher_identity(identity)
        trace, outputs = step._take(issuer_id=self.issuer_id, identity=identity)
        try:
            with self.__coordinator.enter("oon") as lease:
                states = trace._consume(issuer_id=self.issuer_id, identity=identity)
                states.clear()
                counter = _CallCounter(EXACT_EARLY_ORIGINAL_CALLS)
                gateway = _EphemeralOriginalGateway(
                    native=self.__native_calls, calls=counter,
                    coordinator=self.__coordinator, lease=lease,
                )
                try:
                    with torch.no_grad():
                        produced = autonomous_forward(gateway, inputs.detach().clone())
                    if not isinstance(produced, tuple) or len(produced) != 2 or not isinstance(
                        produced[1], Mapping
                    ):
                        raise RuntimeError("autonomous OON producer closure is malformed")
                    logits, closure = produced
                    calls = counter.close()
                    required_closure = {
                        "outer_forward_count": 1,
                        "hook_calls": {0: 1, 1: 1, 2: 0},
                        "outer_returned": True, "hook_restored": True, "hook_inert": True,
                    }
                    if dict(closure) != required_closure:
                        raise RuntimeError("autonomous OON execution closure changed")
                    if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[0] != (
                        runtime.BATCH_SIZE
                    ) or logits.shape[1] not in {
                        runtime.SCORE_STOP - runtime.SCORE_START, runtime.SEQUENCE_LENGTH,
                    } or logits.shape[2] <= 1 or not bool(torch.isfinite(logits).all()):
                        raise RuntimeError("autonomous OON logits are malformed")
                    if logits.shape[1] == runtime.SEQUENCE_LENGTH:
                        logits = runtime.scored_positions(logits)
                finally:
                    gateway.revoke()
        except BaseException:
            try:
                if not trace.consumed:
                    trace._discard(issuer_id=self.issuer_id, identity=identity)
            finally:
                outputs.force_discard(identity)
                self._abort_identity(identity)
            raise
        metadata = {
            "scope": "oon", "producer_invocations": 1, "outer_forward_count": 1,
            "hook_calls": ((0, 1), (1, 1), (2, 0)), "original_calls": calls,
            "outer_returned": True, "hook_restored": True, "hook_inert": True,
            "output_shapes": (tuple(logits.shape),),
            "output_dtypes": (str(logits.dtype),), "support": "64:256",
            "requires_grad": False, "grad_fn_absent": True,
        }
        try:
            return OONTeacherResult(
                broker=self, identity=identity, tensors=(logits,),
                student_outputs=outputs, metadata=metadata,
            )
        except BaseException:
            outputs.force_discard(identity)
            self._abort_identity(identity)
            raise

    def _mint_closure(
        self, *, identity: runtime.TraceIdentity, output_sha256: str,
        consumed: bool, **metadata: Any,
    ) -> StepClosure:
        payload = {
            "identity_sha256": identity.sha256, "forward_nonce": identity.nonce,
            **metadata, "consumed": consumed, "output_sha256": output_sha256,
            "previous_ledger_sha256": self.__rolling_ledger_sha256,
        }
        ledger_sha256 = runtime.logical_identity_sha256(payload)
        object.__setattr__(self, "_CapabilityBroker__rolling_ledger_sha256", ledger_sha256)
        return StepClosure(
            identity_sha256=identity.sha256, forward_nonce=identity.nonce,
            consumed=consumed, output_sha256=output_sha256,
            ledger_sha256=ledger_sha256, **metadata,
        )
