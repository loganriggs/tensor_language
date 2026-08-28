"""Deterministic numerical fit orchestration for suffix-transport v1.

This module is the sole owner of the initial Q denominator pass and the true-row
L/R/S0/S1 optimization loops.  It deliberately performs no row loading, model
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
import early_mlp_suffix_transport_v1_runtime as runtime


TRUE_FIT_ROUTES = ("L", "R", "S0", "S1")
STUDENT_STATES = ((0, "P"), (1, "P"), (2, "N"))


def validate_fit_rows(rows: torch.Tensor, context: capabilities.RunContext) -> torch.Tensor:
    """Return a contiguous CPU view after binding the complete frozen fit role."""

    if not isinstance(context, capabilities.RunContext):
        raise TypeError("fit rows require the sealed suffix run context")
    if not torch.is_tensor(rows) or rows.dtype != torch.long or tuple(rows.shape) != (
        capabilities.FIT_ROW_COUNT, runtime.SEQUENCE_LENGTH,
    ) or rows.device.type != "cpu":
        raise ValueError("fit role must be contiguous CPU int64 [384,256]")
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
        if phase != "fit" or route not in TRUE_FIT_ROUTES or program.route != route:
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


def _batch_tokens(
    rows: torch.Tensor, indices: Sequence[int], device: torch.device | str | None,
) -> torch.Tensor:
    selected = rows[torch.tensor(tuple(indices), dtype=torch.long)].contiguous()
    if device is not None:
        selected = selected.to(device=device)
    return selected


def _student_step(
    *, adapter: Any, broker: capabilities.CapabilityBroker,
    hook: runtime.StudentCorrectionHook, program: runtime.JointAffineProgram,
    identity: runtime.TraceIdentity, inputs: torch.Tensor, indices: Sequence[int],
) -> tuple[Any, Any, Any]:
    program.require_exact_trainability()
    hook.configure(program=program, states={0: "P", 1: "P"})
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


def run_true_fit_trial(
    *, rows: torch.Tensor, context: capabilities.RunContext,
    program: runtime.JointAffineProgram, route: str, trial: int,
    denominators: Sequence[torch.Tensor | float] | None, adapter: Any,
    broker: capabilities.CapabilityBroker, hook: runtime.StudentCorrectionHook,
    device: torch.device | str | None = None,
) -> FitCandidate:
    """Fit one preregistered L/R/S trial without selection or artifact writes."""

    rows = validate_fit_rows(rows, context)
    if route not in TRUE_FIT_ROUTES or program.route != route or trial not in range(
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
