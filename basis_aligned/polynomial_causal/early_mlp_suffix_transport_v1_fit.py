"""Deterministic numerical fit orchestration for suffix-transport v1.

This module is the sole owner of the initial Q denominator pass, the true-row
L/R/S0/S1 optimization loops, and document-shuffled L/R/S negative controls.  It
also owns the twenty A-null/T false-parent control families.  It
deliberately performs no row loading, model
loading, validation selection, artifact publication, or final scoring.  Callers
must supply the already validated fit role, inherited program, observed adapter,
sealed capability broker, and student hook.

The separation is scientific, not cosmetic: every trace identity is constructed
before its forward, and every batch is consumed through the same observed-model
and teacher-capability transaction.  A future launcher may serialize the returned
CPU records only after checking the complete broker ledger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_mapped as mapped
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_rows as row_contract


TRUE_FIT_ROUTES = ("L", "R", "S0", "S1")
# T is fitted only after validation has selected and frozen L.  It uses the same
# true-row schedule and OON loss surface, but is deliberately excluded from the
# pre-selection objective-route bank above.
ALL_TRUE_FIT_ROUTES = (*TRUE_FIT_ROUTES, "T")
DOCUMENT_SHUFFLE_ROUTES = ("L", "R", "S0", "S1")
STUDENT_STATES = ((0, "P"), (1, "P"), (2, "N"))


def validate_fit_rows(rows: torch.Tensor, context: capabilities.RunContext) -> torch.Tensor:
    """Return a contiguous CPU view after binding the complete frozen fit role."""

    if not isinstance(context, capabilities.RunContext):
        raise TypeError("fit rows require the sealed suffix run context")
    if not torch.is_tensor(rows) or rows.dtype != torch.long or tuple(rows.shape) != (
        capabilities.FIT_ROW_COUNT, row_contract.TOKEN_LENGTH,
    ) or rows.device.type != "cpu":
        raise ValueError("fit role must be contiguous CPU int64 [384,513]")
    if runtime.tensor_identity_sha256(rows) != context.fit_role_tensor_sha256:
        raise RuntimeError("fit role tensor differs from the sealed run context")
    return rows.contiguous()


def scheduled_indices(
    *, phase: str, trial: int, epoch: int, batch_ordinal: int,
) -> tuple[int, ...]:
    """Materialize the exact preregistered ordered indices for one batch."""

    if phase == "initial_denominator":
        if (trial, epoch) != (0, 0) or not 0 <= batch_ordinal < (
            capabilities.FIT_BATCHES_PER_EPOCH
        ):
            raise ValueError("initial denominator batch schedule changed")
        start = batch_ordinal * runtime.BATCH_SIZE
        return tuple(range(start, start + runtime.BATCH_SIZE))
    if phase != "fit" or trial not in range(len(runtime.LEARNING_RATES)) or epoch not in range(
        runtime.EPOCHS
    ) or not 0 <= batch_ordinal < capabilities.FIT_BATCHES_PER_EPOCH:
        raise ValueError("fit batch schedule changed")
    permutation = runtime.fit_permutations(capabilities.FIT_ROW_COUNT, trial)[epoch]
    start = batch_ordinal * runtime.BATCH_SIZE
    return tuple(int(value) for value in permutation[start:start + runtime.BATCH_SIZE])


def make_identity(
    *, context: capabilities.RunContext, program: runtime.JointAffineProgram,
    inputs: torch.Tensor, indices: Sequence[int], phase: str, route: str,
    trial: int, epoch: int, batch_ordinal: int,
) -> runtime.TraceIdentity:
    """Bind one program snapshot and one ordered batch before any forward executes."""

    if phase == "initial_denominator":
        if route != "Q" or program.route != "L":
            raise ValueError("initial denominator requires Q over the initialized L topology")
        teacher_kind = "coordinate_labels"
        optimizer_step = 0
    else:
        if phase != "fit" or route not in ALL_TRUE_FIT_ROUTES or program.route != route:
            raise ValueError("fit identity route differs from its program")
        teacher_kind = "coordinate_labels" if route == "L" else "oon_logits"
        optimizer_step = epoch * capabilities.FIT_BATCHES_PER_EPOCH + batch_ordinal
    expected = scheduled_indices(
        phase=phase, trial=trial, epoch=epoch, batch_ordinal=batch_ordinal,
    )
    if tuple(indices) != expected:
        raise RuntimeError("caller batch differs from the preregistered schedule")
    return runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=indices,
        source_commit=context.source_commit,
        inherited_snapshot_sha256=context.inherited_snapshot_sha256,
        rows_receipt_sha256=context.rows_receipt_sha256,
        fit_role_tensor_sha256=context.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=context.identity_teacher_mapping_sha256,
        phase=phase, route=route, control="true", teacher_kind=teacher_kind,
        trial=trial, epoch=epoch, optimizer_step=optimizer_step,
        batch_ordinal=batch_ordinal, student_states=STUDENT_STATES,
    )


def make_document_shuffle_identity(
    *, context: mapped.MappedRunContext, program: runtime.JointAffineProgram,
    inputs: torch.Tensor, indices: Sequence[int], route: str, trial: int,
    epoch: int, batch_ordinal: int,
) -> runtime.TraceIdentity:
    """Bind one source batch to the frozen document-shuffle plan before execution."""

    if not isinstance(context, mapped.MappedRunContext) or context.plan.control != (
        "document_shuffle"
    ) or route not in DOCUMENT_SHUFFLE_ROUTES or program.route != route:
        raise ValueError("document-shuffle fit identity is malformed")
    expected = scheduled_indices(
        phase="fit", trial=trial, epoch=epoch, batch_ordinal=batch_ordinal,
    )
    if tuple(indices) != expected:
        raise RuntimeError("caller batch differs from the preregistered schedule")
    return runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=indices,
        source_commit=context.base.source_commit,
        inherited_snapshot_sha256=context.base.inherited_snapshot_sha256,
        rows_receipt_sha256=context.base.rows_receipt_sha256,
        fit_role_tensor_sha256=context.base.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=context.plan.sha256, phase="fit", route=route,
        control="document_shuffle",
        teacher_kind="coordinate_labels" if route == "L" else "oon_logits",
        trial=trial,
        epoch=epoch,
        optimizer_step=epoch * capabilities.FIT_BATCHES_PER_EPOCH + batch_ordinal,
        batch_ordinal=batch_ordinal, student_states=STUDENT_STATES,
    )


def make_a_null_identity(
    *, context: mapped.MappedRunContext, program: runtime.JointAffineProgram,
    inputs: torch.Tensor, indices: Sequence[int], trial: int, epoch: int,
    batch_ordinal: int,
) -> runtime.TraceIdentity:
    """Bind one T step to an exact registered A-null document map."""

    if not isinstance(context, mapped.MappedRunContext) or not context.plan.control.startswith(
        "A_null_"
    ) or program.route != "T":
        raise ValueError("A-null fit identity is malformed")
    expected = scheduled_indices(
        phase="fit", trial=trial, epoch=epoch, batch_ordinal=batch_ordinal,
    )
    if tuple(indices) != expected:
        raise RuntimeError("caller batch differs from the preregistered schedule")
    return runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=indices,
        source_commit=context.base.source_commit,
        inherited_snapshot_sha256=context.base.inherited_snapshot_sha256,
        rows_receipt_sha256=context.base.rows_receipt_sha256,
        fit_role_tensor_sha256=context.base.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=context.plan.sha256, phase="fit", route="T",
        control=context.plan.control, teacher_kind="oon_logits", trial=trial,
        epoch=epoch,
        optimizer_step=epoch * capabilities.FIT_BATCHES_PER_EPOCH + batch_ordinal,
        batch_ordinal=batch_ordinal, student_states=STUDENT_STATES,
    )


def _batch_tokens(
    rows: torch.Tensor, indices: Sequence[int], device: torch.device | str | None,
) -> torch.Tensor:
    selected = rows[
        torch.tensor(tuple(indices), dtype=torch.long), :runtime.SEQUENCE_LENGTH
    ].contiguous()
    if device is not None:
        selected = selected.to(device=device)
    return selected


def _student_step(
    *, adapter: Any, broker: capabilities.CapabilityBroker,
    hook: runtime.StudentCorrectionHook, program: runtime.JointAffineProgram,
    identity: runtime.TraceIdentity, inputs: torch.Tensor, indices: Sequence[int],
    mapped_parent: runtime.MappedParentCode | None = None,
) -> tuple[Any, Any, Any]:
    program.require_exact_trainability()
    hook.configure(
        program=program, states={0: "P", 1: "P"}, mapped_parent=mapped_parent,
    )
    session = broker.begin_student(identity, hook, inputs, indices)
    return adapter.run_student(
        session=session, hook=hook, identity=identity, tokens=inputs,
    )


def _merge_moment_pair(
    aggregate: tuple[runtime.MomentSufficientStatistics, ...] | None,
    batch: tuple[runtime.MomentSufficientStatistics, ...],
) -> tuple[runtime.MomentSufficientStatistics, ...]:
    if len(batch) != 2:
        raise RuntimeError("denominator teacher did not return exactly MLP0/1 moments")
    if aggregate is None:
        return batch
    return tuple(left.merge(right) for left, right in zip(aggregate, batch, strict=True))


def _closure_payload(value: Any) -> Mapping[str, Any]:
    """Convert a reviewed immutable closure into deterministic hash material."""

    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("observed execution closure is not deterministic data")
    # Fail before receipt construction if an adapter ever grows a raw tensor field.
    if any(torch.is_tensor(item) for item in payload.values()):
        raise RuntimeError("observed execution closure leaked a raw tensor")
    return payload


@dataclass(frozen=True)
class DenominatorPass:
    """Frozen fit-label moments and an ordered transaction-history commitment."""

    site_records: tuple[Mapping[str, Any], Mapping[str, Any]]
    transaction_history_sha256: str
    completed_steps: int

    @property
    def denominators(self) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(record["denominator"] for record in self.site_records)  # type: ignore[return-value]

    @property
    def sha256(self) -> str:
        if len(self.site_records) != 2 or type(self.completed_steps) is not int or (
            self.completed_steps != capabilities.FIT_BATCHES_PER_EPOCH
        ) or not runtime._sha256_text(self.transaction_history_sha256):
            raise RuntimeError("denominator pass identity is malformed")
        records = []
        required = {
            "count", "coordinate_sum", "coordinate_square_sum", "mean",
            "centered_sum_of_squares", "raw_sum_square_replay", "denominator",
            "ordered_support_sha256",
        }
        for value in self.site_records:
            if not isinstance(value, Mapping) or set(value) != required or not (
                runtime._sha256_text(value["ordered_support_sha256"])
            ):
                raise RuntimeError("denominator pass site record is malformed")
            tensor_fields = {}
            for name in required - {"count", "ordered_support_sha256"}:
                tensor = value[name]
                if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
                    raise RuntimeError("denominator pass site tensor is malformed")
                tensor_fields[name] = runtime.tensor_identity_sha256(tensor)
            records.append({
                "count": value["count"],
                "ordered_support_sha256": value["ordered_support_sha256"],
                "tensors": tensor_fields,
            })
        if records[0]["count"] != capabilities.FIT_ROW_COUNT * (
            runtime.SCORE_STOP - runtime.SCORE_START
        ) or records[1]["count"] != records[0]["count"] or (
            records[0]["ordered_support_sha256"] != records[1]["ordered_support_sha256"]
        ):
            raise RuntimeError("denominator pass support changed")
        return runtime.logical_identity_sha256({
            "site_records": records,
            "transaction_history_sha256": self.transaction_history_sha256,
            "completed_steps": self.completed_steps,
        })


def run_initial_denominator_pass(
    *, rows: torch.Tensor, context: capabilities.RunContext,
    program: runtime.JointAffineProgram, adapter: Any,
    broker: capabilities.CapabilityBroker, hook: runtime.StudentCorrectionHook,
    device: torch.device | str | None = None,
) -> DenominatorPass:
    """Run the sequential initialized-Q label pass over all 384 fit rows once."""

    rows = validate_fit_rows(rows, context)
    if program.route != "L":
        raise ValueError("Q denominator program must use the initialized L topology")
    aggregate = None
    history: list[Mapping[str, Any]] = []
    for ordinal in range(capabilities.FIT_BATCHES_PER_EPOCH):
        indices = scheduled_indices(
            phase="initial_denominator", trial=0, epoch=0, batch_ordinal=ordinal,
        )
        inputs = _batch_tokens(rows, indices, device)
        identity = make_identity(
            context=context, program=program, inputs=inputs, indices=indices,
            phase="initial_denominator", route="Q", trial=0, epoch=0,
            batch_ordinal=ordinal,
        )
        step, student_closure, observed = _student_step(
            adapter=adapter, broker=broker, hook=hook, program=program,
            identity=identity, inputs=inputs, indices=indices,
        )
        result = broker.run_coordinate_teacher(identity, step)
        batch_moments, teacher_closure = result.consume_moments()
        aggregate = _merge_moment_pair(aggregate, batch_moments)
        history.append({
            "identity_sha256": identity.sha256,
            "student_ledger_sha256": student_closure.ledger_sha256,
            "teacher_ledger_sha256": teacher_closure.ledger_sha256,
            "observed": _closure_payload(observed),
        })
    assert aggregate is not None
    support_sha256 = runtime.logical_identity_sha256({
        "fit_role_tensor_sha256": context.fit_role_tensor_sha256,
        "ordered_rows": list(range(capabilities.FIT_ROW_COUNT)),
        "score_support": [runtime.SCORE_START, runtime.SCORE_STOP],
    })
    records = tuple(MappingProxyType(dict(value.finalize(
        expected_count=capabilities.FIT_ROW_COUNT * (
            runtime.SCORE_STOP - runtime.SCORE_START
        ),
        ordered_support_sha256=support_sha256,
    ))) for value in aggregate)
    return DenominatorPass(
        site_records=records,  # type: ignore[arg-type]
        transaction_history_sha256=runtime.logical_identity_sha256(history),
        completed_steps=len(history),
    )


@dataclass(frozen=True)
class FitCandidate:
    """One complete true-row optimizer trajectory, still unselected and unscored."""

    route: str
    trial: int
    learning_rate: float
    completed_steps: int
    loss_sum: float
    loss_min: float
    loss_max: float
    final_program_sha256: str
    transaction_history_sha256: str
    state_dict: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class MappedFitCandidate:
    """One negative-control trajectory, ineligible for true-row selection."""

    control: str
    mapping_sha256: str
    route: str
    trial: int
    learning_rate: float
    completed_steps: int
    loss_sum: float
    loss_min: float
    loss_max: float
    final_program_sha256: str
    transaction_history_sha256: str
    state_dict: Mapping[str, torch.Tensor]


def run_true_fit_trial(
    *, rows: torch.Tensor, context: capabilities.RunContext,
    program: runtime.JointAffineProgram, route: str, trial: int,
    denominators: Sequence[torch.Tensor | float] | None, adapter: Any,
    broker: capabilities.CapabilityBroker, hook: runtime.StudentCorrectionHook,
    device: torch.device | str | None = None,
) -> FitCandidate:
    """Fit one preregistered L/R/S trial without selection or artifact writes."""

    rows = validate_fit_rows(rows, context)
    if route not in ALL_TRUE_FIT_ROUTES or program.route != route or trial not in range(
        len(runtime.LEARNING_RATES)
    ):
        raise ValueError("true fit route/trial/program is malformed")
    if route == "L":
        if denominators is None or len(tuple(denominators)) != 2:
            raise ValueError("L fit requires both frozen Q denominators")
    elif denominators is not None:
        raise ValueError("suffix fit routes must not receive local denominators")
    parameters = program.set_route_trainability()
    program.require_exact_trainability()
    learning_rate = runtime.LEARNING_RATES[trial]
    optimizer = runtime.make_optimizer(parameters, learning_rate)
    history: list[Mapping[str, Any]] = []
    losses: list[float] = []
    for epoch in range(runtime.EPOCHS):
        for ordinal in range(capabilities.FIT_BATCHES_PER_EPOCH):
            indices = scheduled_indices(
                phase="fit", trial=trial, epoch=epoch, batch_ordinal=ordinal,
            )
            inputs = _batch_tokens(rows, indices, device)
            identity = make_identity(
                context=context, program=program, inputs=inputs, indices=indices,
                phase="fit", route=route, trial=trial, epoch=epoch,
                batch_ordinal=ordinal,
            )
            step, student_closure, observed = _student_step(
                adapter=adapter, broker=broker, hook=hook, program=program,
                identity=identity, inputs=inputs, indices=indices,
            )
            if route == "L":
                result = broker.run_coordinate_teacher(identity, step)
                loss, teacher_closure = result.consume_loss(denominators)
            else:
                result = adapter.run_oon_teacher(
                    broker=broker, identity=identity, step=step, tokens=inputs,
                )
                loss, teacher_closure = result.consume_loss()
            loss_value = float(loss.detach().double().cpu())
            gradient_norm = runtime.optimizer_step(loss, optimizer)
            losses.append(loss_value)
            history.append({
                "identity_sha256": identity.sha256,
                "loss": loss_value,
                "gradient_norm": gradient_norm,
                "student_ledger_sha256": student_closure.ledger_sha256,
                "teacher_ledger_sha256": teacher_closure.ledger_sha256,
                "observed": _closure_payload(observed),
            })
    state = MappingProxyType({
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(program.state_dict().items())
    })
    return FitCandidate(
        route=route, trial=trial, learning_rate=learning_rate,
        completed_steps=len(losses), loss_sum=float(sum(losses)),
        loss_min=float(min(losses)), loss_max=float(max(losses)),
        final_program_sha256=runtime.program_snapshot_sha256(program),
        transaction_history_sha256=runtime.logical_identity_sha256(history),
        state_dict=state,
    )


def run_document_shuffle_fit_trial(
    *, rows: torch.Tensor, context: mapped.MappedRunContext,
    program: runtime.JointAffineProgram, route: str, trial: int,
    adapter: Any, broker: capabilities.CapabilityBroker,
    hook: runtime.StudentCorrectionHook,
    device: torch.device | str | None = None,
    denominators: Sequence[torch.Tensor | float] | None = None,
) -> MappedFitCandidate:
    """Fit one document-shuffled L/R/S control through the sealed mapped teacher."""

    if not isinstance(context, mapped.MappedRunContext) or context.plan.control != (
        "document_shuffle"
    ):
        raise ValueError("mapped fit requires the document-shuffle run context")
    rows = validate_fit_rows(rows, context.base)
    if route not in DOCUMENT_SHUFFLE_ROUTES or program.route != route or trial not in range(
        len(runtime.LEARNING_RATES)
    ):
        raise ValueError("document-shuffle route/trial/program is malformed")
    if route == "L":
        if denominators is None or len(tuple(denominators)) != 2:
            raise ValueError("document-shuffled L requires both frozen Q denominators")
    elif denominators is not None:
        raise ValueError("document-shuffled suffix routes must not receive denominators")
    parameters = program.set_route_trainability()
    program.require_exact_trainability()
    learning_rate = runtime.LEARNING_RATES[trial]
    optimizer = runtime.make_optimizer(parameters, learning_rate)
    history: list[Mapping[str, Any]] = []
    losses: list[float] = []
    for epoch in range(runtime.EPOCHS):
        for ordinal in range(capabilities.FIT_BATCHES_PER_EPOCH):
            source_indices = scheduled_indices(
                phase="fit", trial=trial, epoch=epoch, batch_ordinal=ordinal,
            )
            target_indices = context.plan.target_indices(source_indices)
            source_inputs = _batch_tokens(rows, source_indices, device)
            target_inputs = _batch_tokens(rows, target_indices, device)
            identity = make_document_shuffle_identity(
                context=context, program=program, inputs=source_inputs,
                indices=source_indices, route=route, trial=trial, epoch=epoch,
                batch_ordinal=ordinal,
            )
            step, student_closure, observed = _student_step(
                adapter=adapter, broker=broker, hook=hook, program=program,
                identity=identity, inputs=source_inputs, indices=source_indices,
            )
            if route == "L":
                result = adapter.run_mapped_coordinate_teacher(
                    broker=broker, identity=identity, step=step, fit_rows=rows,
                    student_tokens=source_inputs, student_indices=source_indices,
                    teacher_tokens=target_inputs, teacher_indices=target_indices,
                    program=program,
                )
                loss, teacher_closure = result.consume_loss(denominators)
            else:
                result = adapter.run_mapped_oon_teacher(
                    broker=broker, identity=identity, step=step, fit_rows=rows,
                    student_tokens=source_inputs, student_indices=source_indices,
                    teacher_tokens=target_inputs, teacher_indices=target_indices,
                )
                loss, teacher_closure = result.consume_loss()
            loss_value = float(loss.detach().double().cpu())
            gradient_norm = runtime.optimizer_step(loss, optimizer)
            losses.append(loss_value)
            history.append({
                "identity_sha256": identity.sha256,
                "mapping_sha256": context.plan.sha256,
                "source_indices": list(source_indices),
                "target_indices": list(target_indices),
                "loss": loss_value, "gradient_norm": gradient_norm,
                "student_ledger_sha256": student_closure.ledger_sha256,
                "teacher_ledger_sha256": teacher_closure.ledger_sha256,
                "observed": _closure_payload(observed),
            })
    state = MappingProxyType({
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(program.state_dict().items())
    })
    return MappedFitCandidate(
        control="document_shuffle", mapping_sha256=context.plan.sha256,
        route=route, trial=trial, learning_rate=learning_rate,
        completed_steps=len(losses), loss_sum=float(sum(losses)),
        loss_min=float(min(losses)), loss_max=float(max(losses)),
        final_program_sha256=runtime.program_snapshot_sha256(program),
        transaction_history_sha256=runtime.logical_identity_sha256(history),
        state_dict=state,
    )


def run_a_null_fit_trial(
    *, rows: torch.Tensor, context: mapped.MappedRunContext,
    program: runtime.JointAffineProgram, trial: int, adapter: Any,
    broker: capabilities.CapabilityBroker, hook: runtime.StudentCorrectionHook,
    device: torch.device | str | None = None,
) -> MappedFitCandidate:
    """Fit one registered A-null with false parent codes and true source teachers."""

    if not isinstance(context, mapped.MappedRunContext) or not context.plan.control.startswith(
        "A_null_"
    ):
        raise ValueError("A-null fit requires one registered mapped run context")
    rows = validate_fit_rows(rows, context.base)
    if program.route != "T" or trial not in range(len(runtime.LEARNING_RATES)):
        raise ValueError("A-null trial/program is malformed")
    parameters = program.set_route_trainability()
    program.require_exact_trainability()
    learning_rate = runtime.LEARNING_RATES[trial]
    optimizer = runtime.make_optimizer(parameters, learning_rate)
    history: list[Mapping[str, Any]] = []
    losses: list[float] = []
    for epoch in range(runtime.EPOCHS):
        for ordinal in range(capabilities.FIT_BATCHES_PER_EPOCH):
            source_indices = scheduled_indices(
                phase="fit", trial=trial, epoch=epoch, batch_ordinal=ordinal,
            )
            target_indices = context.plan.target_indices(source_indices)
            source_inputs = _batch_tokens(rows, source_indices, device)
            target_inputs = _batch_tokens(rows, target_indices, device)
            identity = make_a_null_identity(
                context=context, program=program, inputs=source_inputs,
                indices=source_indices, trial=trial, epoch=epoch,
                batch_ordinal=ordinal,
            )
            parent, parent_closure = adapter.prepare_mapped_parent(
                broker=broker, identity=identity, fit_rows=rows,
                student_tokens=source_inputs, student_indices=source_indices,
                teacher_tokens=target_inputs, teacher_indices=target_indices,
                program=program,
            )
            step, student_closure, observed = _student_step(
                adapter=adapter, broker=broker, hook=hook, program=program,
                identity=identity, inputs=source_inputs, indices=source_indices,
                mapped_parent=parent,
            )
            result = adapter.run_a_null_oon_teacher(
                broker=broker, identity=identity, step=step, fit_rows=rows,
                student_tokens=source_inputs, student_indices=source_indices,
                teacher_tokens=target_inputs, teacher_indices=target_indices,
            )
            loss, teacher_closure = result.consume_loss()
            loss_value = float(loss.detach().double().cpu())
            gradient_norm = runtime.optimizer_step(loss, optimizer)
            losses.append(loss_value)
            history.append({
                "identity_sha256": identity.sha256,
                "mapping_sha256": context.plan.sha256,
                "source_indices": list(source_indices),
                "target_indices": list(target_indices),
                "loss": loss_value, "gradient_norm": gradient_norm,
                "parent_ledger_sha256": parent_closure.ledger_sha256,
                "student_ledger_sha256": student_closure.ledger_sha256,
                "teacher_ledger_sha256": teacher_closure.ledger_sha256,
                "observed": _closure_payload(observed),
            })
    state = MappingProxyType({
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(program.state_dict().items())
    })
    return MappedFitCandidate(
        control=context.plan.control, mapping_sha256=context.plan.sha256,
        route="T", trial=trial, learning_rate=learning_rate,
        completed_steps=len(losses), loss_sum=float(sum(losses)),
        loss_min=float(min(losses)), loss_max=float(max(losses)),
        final_program_sha256=runtime.program_snapshot_sha256(program),
        transaction_history_sha256=runtime.logical_identity_sha256(history),
        state_dict=state,
    )
