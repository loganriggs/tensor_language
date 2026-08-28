"""Source-closed execution bridge from the frozen ship to suffix capabilities.

The adapter owns a complete outer forward and returns only sealed capability objects
and an immutable execution receipt.  Student callers never receive logits, live
states, dispatcher callbacks, or deployed-N handles.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any, Iterator

import torch

import bilin18_observed_model_facade as facade
import early_mlp_suffix_transport_v1 as contract
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_response_execution as response_execution
import early_mlp_suffix_transport_v1_response_plan as response_plan
import early_mlp_suffix_transport_v1_response_reductions as response_reductions
import early_mlp_suffix_transport_v1_runtime as runtime


EARLY_SITES = (0, 1, 2)
CORRECTION_SITES = (0, 1)


def _counts(values: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(site), int(count)) for site, count in values.items()))


def _final_batch_vector(name: str, value: Any, *, count: bool) -> torch.Tensor:
    expected_dtype = torch.long if count else torch.float64
    if not torch.is_tensor(value) or tuple(value.shape) != (
        runtime.BATCH_SIZE,
    ) or value.dtype != expected_dtype or value.device.type != "cpu" or (
        value.requires_grad
    ) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(f"final baseline reduction {name} is malformed")
    return value.detach().clone().contiguous()


@dataclass(frozen=True)
class ObservedClosure:
    """Adapter-side proof of the literal execution path for one outer forward."""

    scope: str
    outer_forward_count: int
    outer_returned: bool
    attention_dispatch_calls: tuple[tuple[int, int], ...]
    mlp_dispatch_calls: tuple[tuple[int, int], ...]
    deployed_n_calls: tuple[tuple[int, int], ...]
    correction_calls: tuple[tuple[int, int], ...]
    literal_early_mlp_calls: tuple[tuple[int, int], ...]
    native_guard_restored: bool
    native_guard_inert: bool
    logit_shape: tuple[int, ...]
    logit_dtype: str


@dataclass(frozen=True)
class ObservedValidationReceipt:
    """Tensor-free receipt for one complete selection-only model transaction."""

    identity_sha256: str
    route: str
    control: str
    batch_ordinal: int
    ordered_row_indices_sha256: str
    reduction_sha256: str
    student_ledger_sha256: str
    teacher_ledger_sha256: str
    observed_closure_sha256: str


@dataclass(frozen=True)
class ObservedValidationBaselineReceipt:
    """Tensor-free receipt for one deployed-N/N validation baseline batch."""

    identity_sha256: str
    batch_ordinal: int
    reduction_sha256: str
    observed_closure_sha256: str


@dataclass(frozen=True)
class ObservedFinalProgramBatchReceipt:
    """Tensor-free closure receipt for one final P/P/N or P/P/E program batch."""

    identity_sha256: str
    route: str
    control: str
    batch_ordinal: int
    ordered_row_indices_sha256: str
    reduction_sha256: str
    frequency_assignment_sha256: str
    student_ledger_sha256: str
    teacher_ledger_sha256: str
    observed_closure_sha256: str


@dataclass(frozen=True)
class ObservedMaterializedFinalProgramBatchReceipt:
    """Tensor-free proof joining one named action to its broker transaction."""

    action_key: str
    final_action_identity_sha256: str
    materialization_sha256: str
    binding_sha256: str
    runtime_identity_sha256: str
    runtime_receipt_sha256: str
    reduction_sha256: str
    frequency_assignment_sha256: str
    batch_ordinal: int

    def __post_init__(self) -> None:
        if self.action_key not in final_actions.CANONICAL_ACTION_KEYS or any(
            not runtime._sha256_text(value) for value in (
                self.final_action_identity_sha256, self.materialization_sha256,
                self.binding_sha256, self.runtime_identity_sha256,
                self.runtime_receipt_sha256, self.reduction_sha256,
                self.frequency_assignment_sha256,
            )
        ) or type(self.batch_ordinal) is not int or not 0 <= self.batch_ordinal < (
            final_actions.OBSERVATIONAL_BATCH_COUNT
        ):
            raise ValueError("observed materialized final receipt is malformed")


@dataclass(frozen=True)
class ObservedFinalBaselineBatchReductions:
    """Only per-row scalars released by one native/deployed final baseline."""

    identity_sha256: str
    action_key: str
    row_primary_sum: torch.Tensor | None
    row_primary_count: torch.Tensor | None
    row_ce_sum: torch.Tensor
    row_ce_count: torch.Tensor
    row_copy_ce_sum: torch.Tensor
    row_copy_count: torch.Tensor
    row_frequency_ce_sum: torch.Tensor
    row_frequency_count: torch.Tensor

    def __post_init__(self) -> None:
        if not runtime._sha256_text(self.identity_sha256) or self.action_key not in {
            f"{arm}/{background}"
            for arm in ("n_n", "o_o")
            for background in final_actions.BACKGROUNDS
        }:
            raise ValueError("final baseline reduction identity is malformed")
        background = self.action_key.split("/")[1]
        if background == "N":
            if self.row_primary_sum is None or self.row_primary_count is None:
                raise ValueError("deployed-MLP2 baseline requires exact OON teacher KL")
            object.__setattr__(
                self, "row_primary_sum",
                _final_batch_vector("row_primary_sum", self.row_primary_sum, count=False),
            )
            object.__setattr__(
                self, "row_primary_count",
                _final_batch_vector("row_primary_count", self.row_primary_count, count=True),
            )
        elif self.row_primary_sum is not None or self.row_primary_count is not None:
            raise ValueError("exact-MLP2 baseline must remain CE-only")
        for name in ("row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count"):
            object.__setattr__(
                self, name,
                _final_batch_vector(name, getattr(self, name), count=name.endswith("count")),
            )
        for name in ("row_frequency_ce_sum", "row_frequency_count"):
            value = getattr(self, name)
            expected = torch.float64 if name.endswith("ce_sum") else torch.long
            if not torch.is_tensor(value) or tuple(value.shape) != (
                runtime.BATCH_SIZE, 9
            ) or value.dtype != expected or value.device.type != "cpu" or (
                value.requires_grad
            ) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f"final baseline reduction {name} is malformed")
            object.__setattr__(self, name, value.detach().clone().contiguous())
        if not torch.equal(self.row_frequency_count.sum(dim=1), self.row_ce_count) or not (
            torch.allclose(self.row_frequency_ce_sum.sum(dim=1), self.row_ce_sum, atol=1e-10)
        ):
            raise ValueError("final baseline frequency reduction does not partition CE")


@dataclass(frozen=True)
class ObservedFinalBaselineBatchReceipt:
    """Tensor-free closure receipt for one N/N or O/O final baseline batch."""

    identity_sha256: str
    action_key: str
    batch_ordinal: int
    ordered_row_indices_sha256: str
    reduction_sha256: str
    frequency_assignment_sha256: str
    observed_student_closure_sha256: str
    observed_teacher_closure_sha256: str | None
    teacher_reused_student: bool


@dataclass(frozen=True)
class _ObservedResponseTeacherForward:
    """Private exact-teacher tensors for immediate in-adapter paired reduction."""

    code1: torch.Tensor
    logits: torch.Tensor
    edit_sha256: str
    unit_identity_sha256: str
    observed_closure: ObservedClosure

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.code1) or tuple(self.code1.shape) != (
            runtime.BATCH_SIZE, runtime.SCORE_STOP - runtime.SCORE_START,
            runtime.CODE_DIM,
        ) or self.code1.device.type != "cpu" or self.code1.requires_grad or not (
            bool(torch.isfinite(self.code1).all())
        ):
            raise RuntimeError("response teacher MLP1 code is malformed")
        if not torch.is_tensor(self.logits) or self.logits.ndim != 3 or tuple(
            self.logits.shape[:2]
        ) != (
            runtime.BATCH_SIZE, runtime.SCORE_STOP - runtime.SCORE_START,
        ) or self.logits.shape[-1] <= 1 or self.logits.device.type != "cpu" or (
            self.logits.requires_grad
        ) or not bool(torch.isfinite(self.logits).all()):
            raise RuntimeError("response teacher logits are malformed")
        if not runtime._sha256_text(self.edit_sha256) or not runtime._sha256_text(
            self.unit_identity_sha256
        ) or not isinstance(self.observed_closure, ObservedClosure):
            raise RuntimeError("response teacher identity or closure is malformed")
        object.__setattr__(self, "code1", self.code1.detach().clone().contiguous())
        object.__setattr__(self, "logits", self.logits.detach().clone().contiguous())


@dataclass(frozen=True)
class _ObservedResponseStudentForward:
    """Private student tensors and receipts for immediate triplet reduction."""

    code1: torch.Tensor
    logits: torch.Tensor
    response_execution_identity_sha256: str
    edit_sha256: str
    unit_identity_sha256: str
    student_step_ledger_sha256: str
    consumer_ledger_sha256: str
    broker_ledger_sha256: str
    observed_closure: ObservedClosure

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.code1) or tuple(self.code1.shape) != (
            runtime.BATCH_SIZE, runtime.SCORE_STOP - runtime.SCORE_START,
            runtime.CODE_DIM,
        ) or self.code1.device.type != "cpu" or self.code1.requires_grad or not (
            bool(torch.isfinite(self.code1).all())
        ):
            raise RuntimeError("response student MLP1 code is malformed")
        if not torch.is_tensor(self.logits) or self.logits.ndim != 3 or tuple(
            self.logits.shape[:2]
        ) != (
            runtime.BATCH_SIZE, runtime.SCORE_STOP - runtime.SCORE_START,
        ) or self.logits.shape[-1] <= 1 or self.logits.device.type != "cpu" or (
            self.logits.requires_grad
        ) or not bool(torch.isfinite(self.logits).all()):
            raise RuntimeError("response student logits are malformed")
        for name in (
            "response_execution_identity_sha256", "edit_sha256",
            "unit_identity_sha256", "student_step_ledger_sha256",
            "consumer_ledger_sha256", "broker_ledger_sha256",
        ):
            if not runtime._sha256_text(getattr(self, name)):
                raise RuntimeError(f"response student {name} is malformed")
        if not isinstance(self.observed_closure, ObservedClosure):
            raise RuntimeError("response student observed closure is malformed")
        object.__setattr__(self, "code1", self.code1.detach().clone().contiguous())
        object.__setattr__(self, "logits", self.logits.detach().clone().contiguous())


class _EarlyNativePoison:
    """Fail before any forbidden literal early-MLP call, then restore exactly."""

    def __init__(
        self, model: torch.nn.Module, *, poison_sites: tuple[int, ...] = EARLY_SITES,
    ) -> None:
        if any(site not in EARLY_SITES for site in poison_sites) or len(
            set(poison_sites)
        ) != len(poison_sites):
            raise ValueError("native poison sites are malformed")
        self._modules = {
            site: model.transformer.h[site].mlp for site in poison_sites
        }
        self._snapshots: dict[int, tuple[bool, Any]] = {}
        self._installed: dict[int, Any] = {}
        self.calls = {site: 0 for site in EARLY_SITES}
        self.restored = False
        self.inert = False

    @contextmanager
    def scope(self) -> Iterator[None]:
        if self._snapshots:
            raise RuntimeError("native-call poison is not reusable")
        for site, module in self._modules.items():
            had_instance_forward = "forward" in module.__dict__
            previous = module.__dict__.get("forward")
            self._snapshots[site] = (had_instance_forward, previous)

            def poison(_module, *args, _site=site, **kwargs):
                self.calls[_site] += 1
                raise RuntimeError(f"literal native MLP{_site} call is forbidden")

            installed = MethodType(poison, module)
            self._installed[site] = installed
            module.forward = installed
        try:
            yield
        finally:
            for site, module in self._modules.items():
                had_instance_forward, previous = self._snapshots[site]
                if had_instance_forward:
                    module.__dict__["forward"] = previous
                else:
                    module.__dict__.pop("forward", None)
            self.restored = all(
                (("forward" in module.__dict__) == had)
                and (not had or module.__dict__.get("forward") is previous)
                for site, module in self._modules.items()
                for had, previous in (self._snapshots[site],)
            )
            self.inert = self.restored and all(
                module.__dict__.get("forward") is not self._installed[site]
                for site, module in self._modules.items()
            )


class ObservedBilin18Adapter:
    """Execute student P/P/N and autonomous O/O/N through one reviewed surface."""

    def __init__(
        self, model: torch.nn.Module, frozen_ship: Any, *, production: bool = True,
    ) -> None:
        if production:
            facade.validate_production_model(model)
            if not bool(getattr(frozen_ship, "production", False)):
                raise RuntimeError("production adapter requires the validated frozen ship")
        if not callable(getattr(frozen_ship, "attention", None)) or not callable(
            getattr(frozen_ship, "mlp", None)
        ):
            raise TypeError("frozen ship must provide attention and MLP dispatchers")
        self._model = model
        self._ship = frozen_ship
        self._production = bool(production)

    def make_capability_broker(
        self, *, issuer_id: str, coordinator: runtime.ScopeCoordinator,
        run_context: Any, bases: dict[int, torch.Tensor],
    ) -> Any:
        """Construct the only production broker binding for native MLP0/1.

        Student traces deliberately own detached CPU copies of the current states.
        Coordinate teachers, however, execute the pinned native modules on the model
        device.  Keeping this bridge here prevents runners from passing raw CUDA
        modules to a CPU-state capability or inventing a second native-call path.
        """

        return self._make_capability_broker(
            issuer_id=issuer_id, coordinator=coordinator,
            run_context=run_context, bases=bases, mapped_authority=None,
        )

    def make_mapped_capability_broker(
        self, *, issuer_id: str, coordinator: runtime.ScopeCoordinator,
        mapped_context: Any, bases: dict[int, torch.Tensor],
    ) -> Any:
        """Construct a mapping-bound broker without exposing native model calls."""

        return self._make_capability_broker(
            issuer_id=issuer_id, coordinator=coordinator,
            run_context=mapped_context.base_context, bases=bases,
            mapped_authority=mapped_context,
        )

    def _make_capability_broker(
        self, *, issuer_id: str, coordinator: runtime.ScopeCoordinator,
        run_context: Any, bases: dict[int, torch.Tensor], mapped_authority: Any,
    ) -> Any:
        import early_mlp_suffix_transport_v1_capabilities as capabilities

        try:
            model_parameter = next(self._model.parameters())
        except StopIteration as error:
            raise RuntimeError("observed model has no device-bearing parameters") from error
        model_device = model_parameter.device

        def native(site: int):
            module = self._model.transformer.h[site].mlp
            # Capture the reviewed native implementation before any per-forward
            # instance poison is installed. Calling this bound method directly is
            # the broker's sole authorized O path; accidental ``module(...)`` calls
            # made by the frozen ship still encounter the poison.
            native_forward = module.forward

            def call(state: torch.Tensor) -> torch.Tensor:
                if not torch.is_tensor(state) or tuple(state.shape) != (
                    runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL,
                ) or state.requires_grad or state.grad_fn is not None or not bool(
                    torch.isfinite(state).all()
                ):
                    raise RuntimeError(f"coordinate native MLP{site} state is malformed")
                moved = state.to(device=model_device, dtype=model_parameter.dtype)
                return native_forward(moved)

            return call

        return capabilities.CapabilityBroker(
            issuer_id=issuer_id,
            coordinator=coordinator,
            run_context=run_context,
            bases=bases,
            native_calls={site: native(site) for site in CORRECTION_SITES},
            mapped_authority=mapped_authority,
        )

    def run_student(
        self, *, session: Any, hook: runtime.StudentCorrectionHook,
        identity: runtime.TraceIdentity, tokens: torch.Tensor,
    ) -> tuple[Any, Any, ObservedClosure]:
        """Run one registered P/P/N or final P/P/E student transaction."""

        states = dict(getattr(identity, "student_states", ()))
        mlp2_background = states.get(2, "N")
        if mlp2_background not in {"N", "E"}:
            raise RuntimeError("observed student has an unknown MLP2 background")
        if mlp2_background == "E" and getattr(identity, "phase", None) != "final":
            raise RuntimeError("exact MLP2 background is licensed only during final scoring")
        poison = _EarlyNativePoison(
            self._model,
            poison_sites=EARLY_SITES if mlp2_background == "N" else (0, 1),
        )
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        deployed_n_calls = {site: 0 for site in EARLY_SITES}
        exact_background_calls = {site: 0 for site in EARLY_SITES}
        correction_calls = {site: 0 for site in EARLY_SITES}
        outer_forward_count = 0
        outer_returned = False
        logits: torch.Tensor | None = None

        def attention(event: facade.AttentionEvent):
            attention_calls[event.site] += 1
            return self._ship.attention(event)

        def mlp(event: facade.EarlyMLPEvent):
            mlp_calls[event.site] += 1
            if event.site not in EARLY_SITES:
                return self._ship.mlp(event)
            if event.site == 2 and mlp2_background == "E":
                exact_background_calls[2] += 1
                return event.block.mlp(event.state)
            deployed = self._ship.mlp(event)
            deployed_n_calls[event.site] += 1
            if event.site == 2:
                return deployed
            handle = runtime.mint_deployed_n_write(
                site=event.site,
                state=event.state,
                value=deployed,
                forward_nonce=identity.nonce,
                issuer_id=hook.issuer_id,
            )
            correction_calls[event.site] += 1
            return hook(
                event.site, event.state, handle, forward_nonce=identity.nonce,
            )

        try:
            with poison.scope():
                with session.forward_scope() as capability:
                    outer_forward_count += 1
                    logits = facade.forward_with_dispatch(
                        self._model,
                        tokens,
                        attention,
                        mlp,
                        require_production=self._production,
                    )
                    outer_returned = True
                    capability.bind_outer_logits(logits)
        except BaseException:
            raise

        expected_sites = tuple(range(len(self._model.transformer.h)))
        expected_all = tuple((site, 1) for site in expected_sites)
        try:
            if _counts(attention_calls) != expected_all or _counts(mlp_calls) != expected_all:
                raise RuntimeError("observed outer forward did not dispatch every site exactly once")
            expected_deployed = (
                ((0, 1), (1, 1), (2, 1)) if mlp2_background == "N"
                else ((0, 1), (1, 1), (2, 0))
            )
            expected_exact = (
                ((0, 0), (1, 0), (2, 0)) if mlp2_background == "N"
                else ((0, 0), (1, 0), (2, 1))
            )
            literal_calls = {
                site: poison.calls[site] + exact_background_calls[site]
                for site in EARLY_SITES
            }
            if _counts(deployed_n_calls) != expected_deployed or _counts(
                correction_calls
            ) != ((0, 1), (1, 1), (2, 0)):
                raise RuntimeError("observed student call ledger did not close exactly")
            if not poison.restored or not poison.inert or _counts(literal_calls) != expected_exact:
                raise RuntimeError("literal early-MLP poison did not close cleanly")
            if logits is None:
                raise RuntimeError("observed student outer forward returned no logits")
        except BaseException:
            # A completed forward leaves a pending trace in StudentSession.  Force
            # its ordinary failure closure so no broker identity or graph survives
            # an adapter-ledger failure.
            try:
                session.close(
                    outer_forward_count=0,
                    outer_returned=False,
                    hook_restored=poison.restored,
                    hook_inert=poison.inert,
                )
            except BaseException:
                pass
            raise

        step, step_closure = session.close(
            outer_forward_count=outer_forward_count,
            outer_returned=outer_returned,
            hook_restored=poison.restored,
            hook_inert=poison.inert,
        )
        observed = ObservedClosure(
            scope="student",
            outer_forward_count=outer_forward_count,
            outer_returned=outer_returned,
            attention_dispatch_calls=_counts(attention_calls),
            mlp_dispatch_calls=_counts(mlp_calls),
            deployed_n_calls=_counts(deployed_n_calls),
            correction_calls=_counts(correction_calls),
            literal_early_mlp_calls=_counts(literal_calls),
            native_guard_restored=poison.restored,
            native_guard_inert=poison.inert,
            logit_shape=tuple(logits.shape),
            logit_dtype=str(logits.dtype),
        )
        # Deliberately drop the only adapter-local alias before returning.
        logits = None
        return step, step_closure, observed

    def run_oon_teacher(
        self, *, broker: Any, identity: runtime.TraceIdentity, step: Any,
        tokens: torch.Tensor,
    ) -> Any:
        """Return the sealed OON teacher result; never return teacher logits directly."""

        return broker.run_oon_teacher(
            identity, step, tokens, self._autonomous_oon_forward,
        )

    def run_validation_batch(
        self, *, broker: Any, hook: runtime.StudentCorrectionHook,
        program: runtime.JointAffineProgram, identity: runtime.TraceIdentity,
        role_rows: torch.Tensor, ordered_row_indices: Any, collector: Any,
        denominators: Any = None,
    ) -> ObservedValidationReceipt:
        """Reduce one true-row validation transaction without releasing held-out tensors."""

        import early_mlp_suffix_transport_v1_capabilities as capabilities
        import early_mlp_suffix_transport_v1_programs as programs

        if not isinstance(identity, runtime.TraceIdentity) or identity.phase != (
            "validation"
        ) or identity.role != "early_mlp_suffix_transport_v1_validation" or not isinstance(
            program, runtime.JointAffineProgram
        ) or program.route != identity.route or runtime.program_snapshot_sha256(program) != (
            identity.program_snapshot_sha256
        ) or not isinstance(collector, programs.ValidationStatisticsCollector):
            raise RuntimeError("observed validation identity/program/collector is malformed")
        if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or tuple(
            role_rows.shape
        ) != (runtime.BATCH_SIZE, 513) or role_rows.device.type != "cpu":
            raise RuntimeError("observed validation requires one CPU role-row batch")
        indices = tuple(ordered_row_indices)
        inputs = role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous()
        identity.require_inputs(inputs)
        identity.require_batch_indices(indices)
        collector.require_identity(
            route=identity.route, program_sha256=identity.program_snapshot_sha256,
        )
        if identity.route == "L":
            if denominators is None or len(denominators) != 2:
                raise RuntimeError("local validation requires two frozen denominators")
        elif denominators is not None:
            raise RuntimeError("suffix validation cannot receive local denominators")

        try:
            hook.configure(program=program, states={0: "P", 1: "P"})
            try:
                model_device = next(self._model.parameters()).device
            except StopIteration as error:
                raise RuntimeError("observed model has no device-bearing parameters") from error
            model_inputs = inputs.to(device=model_device)
            session = broker.begin_student(identity, hook, model_inputs, indices)
        except BaseException:
            try:
                hook.clear_configuration()
            except BaseException:
                pass
            raise

        with torch.no_grad():
            step, student_closure, observed = self.run_student(
                session=session, hook=hook, identity=identity, tokens=model_inputs,
            )
            if identity.route == "L":
                result = broker.run_coordinate_teacher(identity, step)
                reductions, teacher_closure = result.consume_validation(
                    role_rows, denominators,
                )
            else:
                result = self.run_oon_teacher(
                    broker=broker, identity=identity, step=step, tokens=model_inputs,
                )
                reductions, teacher_closure = result.consume_validation(role_rows)

        if student_closure.scope != "student" or student_closure.original_calls != (
            capabilities.EXACT_ZERO_CALLS
        ) or student_closure.hook_restored is not True or student_closure.hook_inert is not (
            True
        ) or observed.literal_early_mlp_calls != ((0, 0), (1, 0), (2, 0)) or (
            observed.native_guard_restored is not True
        ) or observed.native_guard_inert is not True or teacher_closure.original_calls != (
            capabilities.EXACT_EARLY_ORIGINAL_CALLS
        ) or teacher_closure.consumed is not True:
            raise RuntimeError("observed validation transaction did not close exactly")
        collector.add_batch(
            batch_ordinal=identity.batch_ordinal, ordered_row_indices=indices,
            row_primary_sum=reductions.row_primary_sum,
            row_primary_count=reductions.row_primary_count,
            row_ce_sum=reductions.row_ce_sum, row_ce_count=reductions.row_ce_count,
            row_copy_ce_sum=reductions.row_copy_ce_sum,
            row_copy_count=reductions.row_copy_count,
            student_original_calls=student_closure.original_calls,
            hook_restored=student_closure.hook_restored,
            hook_inert=student_closure.hook_inert,
        )
        reduction_sha256 = runtime.logical_identity_sha256({
            name: runtime.tensor_identity_sha256(getattr(reductions, name))
            for name in (
                "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
                "row_copy_ce_sum", "row_copy_count",
            )
        })
        return ObservedValidationReceipt(
            identity_sha256=identity.sha256, route=identity.route,
            control=identity.control, batch_ordinal=identity.batch_ordinal,
            ordered_row_indices_sha256=runtime.logical_identity_sha256(list(indices)),
            reduction_sha256=reduction_sha256,
            student_ledger_sha256=student_closure.ledger_sha256,
            teacher_ledger_sha256=teacher_closure.ledger_sha256,
            observed_closure_sha256=runtime.logical_identity_sha256(asdict(observed)),
        )

    def run_validation_baseline_batch(
        self, *, identity: Any, role_rows: torch.Tensor,
        ordered_row_indices: Any, collector: Any,
    ) -> ObservedValidationBaselineReceipt:
        """Run and reduce one deployed-N/N baseline batch without releasing logits."""

        import early_mlp_suffix_transport_v1_programs as programs

        if not isinstance(identity, programs.ValidationBaselineIdentity) or not isinstance(
            collector, programs.ValidationBaselineCollector
        ):
            raise RuntimeError("observed validation baseline identity/collector is malformed")
        indices = tuple(ordered_row_indices)
        identity.require_batch(role_rows, indices)
        collector.require_identity(common_support_sha256=identity.common_support_sha256)
        tokens = role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous()
        try:
            model_device = next(self._model.parameters()).device
        except StopIteration as error:
            raise RuntimeError("observed model has no device-bearing parameters") from error
        tokens = tokens.to(device=model_device)
        poison = _EarlyNativePoison(self._model)
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        logits: torch.Tensor | None = None
        outer_returned = False

        def attention(event: facade.AttentionEvent):
            attention_calls[event.site] += 1
            return self._ship.attention(event)

        def mlp(event: facade.EarlyMLPEvent):
            mlp_calls[event.site] += 1
            return self._ship.mlp(event)

        with torch.no_grad():
            with poison.scope():
                logits = facade.forward_with_dispatch(
                    self._model, tokens, attention, mlp,
                    require_production=self._production,
                )
                outer_returned = True
            expected_all = tuple(
                (site, 1) for site in range(len(self._model.transformer.h))
            )
            if _counts(attention_calls) != expected_all or _counts(mlp_calls) != expected_all or (
                not outer_returned
            ) or not poison.restored or not poison.inert or any(poison.calls.values()) or (
                logits is None
            ):
                raise RuntimeError("observed deployed-N/N baseline did not close exactly")
            ce_sum, ce_count, copy_sum, copy_count = programs.ce_and_copy_rows(
                logits, role_rows,
            )
        observed = ObservedClosure(
            scope="validation_baseline", outer_forward_count=1, outer_returned=True,
            attention_dispatch_calls=_counts(attention_calls),
            mlp_dispatch_calls=_counts(mlp_calls),
            deployed_n_calls=((0, 1), (1, 1), (2, 1)),
            correction_calls=((0, 0), (1, 0), (2, 0)),
            literal_early_mlp_calls=_counts(poison.calls),
            native_guard_restored=poison.restored, native_guard_inert=poison.inert,
            logit_shape=tuple(logits.shape), logit_dtype=str(logits.dtype),
        )
        logits = None
        collector.add_batch(
            batch_ordinal=identity.batch_ordinal, ordered_row_indices=indices,
            row_ce_sum=ce_sum, row_ce_count=ce_count,
            row_copy_ce_sum=copy_sum, row_copy_count=copy_count,
            literal_early_mlp_calls=observed.literal_early_mlp_calls,
            native_guard_restored=observed.native_guard_restored,
            native_guard_inert=observed.native_guard_inert,
        )
        reduction_sha256 = runtime.logical_identity_sha256({
            name: runtime.tensor_identity_sha256(value)
            for name, value in {
                "row_ce_sum": ce_sum, "row_ce_count": ce_count,
                "row_copy_ce_sum": copy_sum, "row_copy_count": copy_count,
            }.items()
        })
        return ObservedValidationBaselineReceipt(
            identity_sha256=identity.sha256, batch_ordinal=identity.batch_ordinal,
            reduction_sha256=reduction_sha256,
            observed_closure_sha256=runtime.logical_identity_sha256(asdict(observed)),
        )

    def run_final_program_batch(
        self, *, broker: Any, hook: runtime.StudentCorrectionHook,
        program: runtime.JointAffineProgram, identity: runtime.TraceIdentity,
        role_rows: torch.Tensor, ordered_row_indices: Any,
        denominators: Any = None, frequency_bins: torch.Tensor | None = None,
    ) -> tuple[Any, ObservedFinalProgramBatchReceipt]:
        """Run one true final P/P/N or P/P/E program and return typed reductions.

        This is the observed backend for the fitted L/R/S/T families under deployed
        MLP2 N and exact-restored MLP2 E.  The E background is CE-only.  Action-level
        aggregation, non-program baselines, interventions, and null mapping remain
        separate typed capabilities; this method cannot silently stand in for them.
        """

        import early_mlp_suffix_transport_v1_capabilities as capabilities

        if not isinstance(identity, runtime.TraceIdentity) or identity.phase != (
            "final"
        ) or identity.role != "early_mlp_suffix_transport_v1_final" or not isinstance(
            program, runtime.JointAffineProgram
        ) or program.route != identity.route or runtime.program_snapshot_sha256(program) != (
            identity.program_snapshot_sha256
        ):
            raise RuntimeError("observed final identity/program is malformed")
        if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or tuple(
            role_rows.shape
        ) != (runtime.BATCH_SIZE, 513) or role_rows.device.type != "cpu":
            raise RuntimeError("observed final requires one CPU role-row batch")
        indices = tuple(ordered_row_indices)
        inputs = role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous()
        identity.require_inputs(inputs)
        identity.require_batch_indices(indices)
        if not torch.is_tensor(frequency_bins) or frequency_bins.dtype != torch.long or tuple(
            frequency_bins.shape
        ) != (runtime.BATCH_SIZE, runtime.SCORE_STOP - runtime.SCORE_START) or (
            frequency_bins.device.type != "cpu"
        ):
            raise RuntimeError("observed final frequency assignment is malformed")
        mlp2_background = dict(identity.student_states)[2]
        if mlp2_background == "N" and identity.route == "L":
            if denominators is None or len(denominators) != 2:
                raise RuntimeError("local final reduction requires two frozen denominators")
        elif denominators is not None:
            raise RuntimeError("CE-only or suffix final reduction cannot receive denominators")

        try:
            hook.configure(program=program, states={0: "P", 1: "P"})
            try:
                model_device = next(self._model.parameters()).device
            except StopIteration as error:
                raise RuntimeError("observed model has no device-bearing parameters") from error
            model_inputs = inputs.to(device=model_device)
            session = broker.begin_student(identity, hook, model_inputs, indices)
        except BaseException:
            try:
                hook.clear_configuration()
            except BaseException:
                pass
            raise

        with torch.no_grad():
            step, student_closure, observed = self.run_student(
                session=session, hook=hook, identity=identity, tokens=model_inputs,
            )
            if mlp2_background == "E":
                reductions, teacher_closure = broker.consume_final_ce(
                    identity, step, role_rows, frequency_bins,
                )
            elif identity.route == "L":
                result = broker.run_coordinate_teacher(identity, step)
                reductions, teacher_closure = result.consume_final(
                    role_rows, denominators, frequency_bins,
                )
            else:
                result = self.run_oon_teacher(
                    broker=broker, identity=identity, step=step, tokens=model_inputs,
                )
                reductions, teacher_closure = result.consume_final(
                    role_rows, frequency_bins,
                )

        expected_reduction_type = (
            capabilities.FinalCEBatchReductions
            if mlp2_background == "E" else capabilities.FinalBatchReductions
        )
        expected_literal_calls = (
            ((0, 0), (1, 0), (2, 1)) if mlp2_background == "E"
            else ((0, 0), (1, 0), (2, 0))
        )
        expected_consumer_calls = (
            capabilities.EXACT_ZERO_CALLS
            if mlp2_background == "E" else capabilities.EXACT_EARLY_ORIGINAL_CALLS
        )
        if type(reductions) is not expected_reduction_type or (
            reductions.identity_sha256 != identity.sha256
        ) or student_closure.scope != "student" or student_closure.original_calls != (
            capabilities.EXACT_ZERO_CALLS
        ) or student_closure.hook_restored is not True or student_closure.hook_inert is not (
            True
        ) or observed.literal_early_mlp_calls != expected_literal_calls or (
            observed.native_guard_restored is not True
        ) or observed.native_guard_inert is not True or teacher_closure.original_calls != (
            expected_consumer_calls
        ) or teacher_closure.consumed is not True:
            raise RuntimeError("observed final program transaction did not close exactly")
        reduction_fields = (
            "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
            "row_frequency_ce_sum", "row_frequency_count",
        ) if mlp2_background == "E" else (
            "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
            "row_copy_ce_sum", "row_copy_count",
            "row_frequency_ce_sum", "row_frequency_count",
        )
        reduction_sha256 = runtime.logical_identity_sha256({
            name: runtime.tensor_identity_sha256(getattr(reductions, name))
            for name in reduction_fields
        })
        receipt = ObservedFinalProgramBatchReceipt(
            identity_sha256=identity.sha256, route=identity.route,
            control=identity.control, batch_ordinal=identity.batch_ordinal,
            ordered_row_indices_sha256=runtime.logical_identity_sha256(list(indices)),
            reduction_sha256=reduction_sha256,
            frequency_assignment_sha256=runtime.tensor_identity_sha256(frequency_bins),
            student_ledger_sha256=student_closure.ledger_sha256,
            teacher_ledger_sha256=teacher_closure.ledger_sha256,
            observed_closure_sha256=runtime.logical_identity_sha256(asdict(observed)),
        )
        return reductions, receipt

    def run_materialized_final_program_batch(
        self, *, broker: Any, hook: runtime.StudentCorrectionHook,
        materialized: final_actions.MaterializedFinalAction,
        identity: final_actions.FinalActionBatchIdentity,
        final_context: Any, role_rows: torch.Tensor,
        ordered_row_indices: Any, denominators: Any = None,
        frequency_bins: torch.Tensor | None = None,
    ) -> tuple[Any, ObservedMaterializedFinalProgramBatchReceipt]:
        """Execute one named program action without accepting a caller-made trace.

        This is the source-closed outer entry point for QQ/LL/RR, hybrids, transport,
        nulls, shuffles, and the frozen mean.  It derives the broker trace from the
        sealed materialization and full 513-token row identity, then binds the lower
        transaction receipt back to that semantic action.
        """

        import early_mlp_suffix_transport_v1_capabilities as capabilities

        if not isinstance(broker, capabilities.CapabilityBroker) or not isinstance(
            final_context, capabilities.FinalRunContext
        ) or not isinstance(materialized, final_actions.MaterializedFinalAction) or not (
            isinstance(identity, final_actions.FinalActionBatchIdentity)
        ):
            raise RuntimeError("materialized final execution lacks typed authorities")
        if broker.ledger_snapshot.run_context_sha256 != final_context.sha256:
            raise RuntimeError("final broker and action run context differ")
        indices = tuple(ordered_row_indices)
        binding = final_actions.bind_runtime_program_batch(
            materialized=materialized, identity=identity, role_rows=role_rows,
            ordered_batch_indices=indices,
            teacher_mapping_sha256=final_context.identity_teacher_mapping_sha256,
        )
        trace = binding.runtime_identity
        final_context.require_identity(
            trace, role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous(), indices,
        )
        program = materialized.make_program()
        reductions, runtime_receipt = self.run_final_program_batch(
            broker=broker, hook=hook, program=program, identity=trace,
            role_rows=role_rows, ordered_row_indices=indices,
            denominators=denominators, frequency_bins=frequency_bins,
        )
        if not isinstance(runtime_receipt, ObservedFinalProgramBatchReceipt) or (
            runtime_receipt.identity_sha256 != trace.sha256
        ) or runtime_receipt.route != trace.route or runtime_receipt.control != (
            trace.control
        ) or runtime_receipt.batch_ordinal != identity.batch_ordinal:
            raise RuntimeError("runtime receipt escaped its final action binding")
        outer_receipt = ObservedMaterializedFinalProgramBatchReceipt(
            action_key=identity.action_key,
            final_action_identity_sha256=identity.sha256,
            materialization_sha256=materialized.sha256,
            binding_sha256=binding.sha256,
            runtime_identity_sha256=trace.sha256,
            runtime_receipt_sha256=runtime.logical_identity_sha256(
                asdict(runtime_receipt)
            ),
            reduction_sha256=runtime_receipt.reduction_sha256,
            frequency_assignment_sha256=runtime_receipt.frequency_assignment_sha256,
            batch_ordinal=identity.batch_ordinal,
        )
        program = None
        return reductions, outer_receipt

    def _run_final_baseline_forward(
        self, *, tokens: torch.Tensor, execution_kind: str, background: str,
    ) -> tuple[torch.Tensor, ObservedClosure]:
        """Execute exactly one N/N or O/O baseline without exposing dispatch handles."""

        if execution_kind not in {"deployed_baseline", "native_baseline"} or background not in (
            final_actions.BACKGROUNDS
        ):
            raise ValueError("final baseline physical path is malformed")
        exact_sites = set()
        if execution_kind == "native_baseline":
            exact_sites.update((0, 1))
        if background == "E":
            exact_sites.add(2)
        poison = _EarlyNativePoison(
            self._model,
            poison_sites=tuple(site for site in EARLY_SITES if site not in exact_sites),
        )
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        deployed_calls = {site: 0 for site in EARLY_SITES}
        exact_calls = {site: 0 for site in EARLY_SITES}
        logits: torch.Tensor | None = None
        outer_returned = False

        def attention(event: facade.AttentionEvent):
            attention_calls[event.site] += 1
            return self._ship.attention(event)

        def mlp(event: facade.EarlyMLPEvent):
            mlp_calls[event.site] += 1
            if event.site not in EARLY_SITES:
                return self._ship.mlp(event)
            if event.site in exact_sites:
                exact_calls[event.site] += 1
                return event.block.mlp(event.state)
            deployed_calls[event.site] += 1
            return self._ship.mlp(event)

        with poison.scope():
            logits = facade.forward_with_dispatch(
                self._model, tokens, attention, mlp,
                require_production=self._production,
            )
            outer_returned = True
        expected_all = tuple((site, 1) for site in range(len(self._model.transformer.h)))
        semantic_arm = "n_n" if execution_kind == "deployed_baseline" else "o_o"
        expected_pattern = final_actions.expected_early_call_pattern(
            final_actions.plan_for(semantic_arm, background),
        )
        literal_calls = {
            site: exact_calls[site] + poison.calls[site] for site in EARLY_SITES
        }
        if _counts(attention_calls) != expected_all or _counts(mlp_calls) != (
            expected_all
        ) or _counts(deployed_calls) != expected_pattern.deployed_n_calls or _counts(
            literal_calls
        ) != expected_pattern.literal_early_mlp_calls or not outer_returned or not poison.restored or not (
            poison.inert
        ) or logits is None:
            raise RuntimeError("observed final baseline did not close exactly")
        observed = ObservedClosure(
            scope=f"final_{execution_kind}_{background}",
            outer_forward_count=1, outer_returned=True,
            attention_dispatch_calls=_counts(attention_calls),
            mlp_dispatch_calls=_counts(mlp_calls),
            deployed_n_calls=_counts(deployed_calls),
            correction_calls=((0, 0), (1, 0), (2, 0)),
            literal_early_mlp_calls=_counts(literal_calls),
            native_guard_restored=poison.restored,
            native_guard_inert=poison.inert,
            logit_shape=tuple(logits.shape), logit_dtype=str(logits.dtype),
        )
        return logits, observed

    def run_final_baseline_batch(
        self, *, materialized: final_actions.MaterializedFinalAction,
        identity: final_actions.FinalActionBatchIdentity,
        role_rows: torch.Tensor, ordered_row_indices: Any,
        frequency_bins: torch.Tensor,
    ) -> tuple[ObservedFinalBaselineBatchReductions, ObservedFinalBaselineBatchReceipt]:
        """Run one N/N or O/O action and release only bound per-row reductions."""

        import early_mlp_suffix_transport_v1_programs as programs

        if not isinstance(materialized, final_actions.MaterializedFinalAction) or not isinstance(
            identity, final_actions.FinalActionBatchIdentity
        ) or materialized.plan.arm_plan.execution_kind not in {
            "deployed_baseline", "native_baseline",
        }:
            raise RuntimeError("observed final baseline identity/action is malformed")
        indices = tuple(ordered_row_indices)
        identity.require_role_rows(
            materialized=materialized, role_rows=role_rows,
            ordered_batch_indices=indices,
        )
        if identity.action_key != materialized.plan.key:
            raise RuntimeError("observed final baseline action identity changed")
        try:
            model_device = next(self._model.parameters()).device
        except StopIteration as error:
            raise RuntimeError("observed model has no device-bearing parameters") from error
        tokens = role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous().to(model_device)
        execution_kind = materialized.plan.arm_plan.execution_kind
        background = materialized.plan.background
        teacher_logits: torch.Tensor | None = None
        teacher_observed: ObservedClosure | None = None
        teacher_reused_student = False
        with torch.no_grad():
            student_logits, student_observed = self._run_final_baseline_forward(
                tokens=tokens, execution_kind=execution_kind, background=background,
            )
            if background == "N":
                if execution_kind == "native_baseline":
                    teacher_logits = student_logits
                    teacher_observed = student_observed
                    teacher_reused_student = True
                else:
                    teacher_logits, teacher_observed = self._run_final_baseline_forward(
                        tokens=tokens, execution_kind="native_baseline", background="N",
                    )
                primary_sum, primary_count = programs.suffix_kl_rows(
                    teacher_logits, student_logits,
                )
            else:
                primary_sum = primary_count = None
            ce_sum, ce_count, copy_sum, copy_count, frequency_sum, frequency_count = (
                programs.final_ce_copy_frequency_rows(
                    student_logits, role_rows, frequency_bins,
                )
            )
        reductions = ObservedFinalBaselineBatchReductions(
            identity_sha256=identity.sha256, action_key=identity.action_key,
            row_primary_sum=primary_sum, row_primary_count=primary_count,
            row_ce_sum=ce_sum, row_ce_count=ce_count,
            row_copy_ce_sum=copy_sum, row_copy_count=copy_count,
            row_frequency_ce_sum=frequency_sum,
            row_frequency_count=frequency_count,
        )
        reduction_fields = (
            "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
            "row_frequency_ce_sum", "row_frequency_count",
        ) if background == "E" else (
            "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
            "row_copy_ce_sum", "row_copy_count",
            "row_frequency_ce_sum", "row_frequency_count",
        )
        reduction_sha256 = runtime.logical_identity_sha256({
            "action_key": identity.action_key,
            **{
                name: runtime.tensor_identity_sha256(getattr(reductions, name))
                for name in reduction_fields
            },
        })
        student_closure_sha256 = runtime.logical_identity_sha256(asdict(student_observed))
        teacher_closure_sha256 = None if teacher_observed is None else (
            runtime.logical_identity_sha256(asdict(teacher_observed))
        )
        receipt = ObservedFinalBaselineBatchReceipt(
            identity_sha256=identity.sha256, action_key=identity.action_key,
            batch_ordinal=identity.batch_ordinal,
            ordered_row_indices_sha256=runtime.logical_identity_sha256(list(indices)),
            reduction_sha256=reduction_sha256,
            frequency_assignment_sha256=runtime.tensor_identity_sha256(frequency_bins),
            observed_student_closure_sha256=student_closure_sha256,
            observed_teacher_closure_sha256=teacher_closure_sha256,
            teacher_reused_student=teacher_reused_student,
        )
        student_logits = None
        teacher_logits = None
        return reductions, receipt

    def run_mapped_oon_teacher(
        self, *, broker: Any, identity: runtime.TraceIdentity, step: Any,
        fit_rows: torch.Tensor, student_tokens: torch.Tensor,
        student_indices: Any, teacher_tokens: torch.Tensor,
        teacher_indices: Any,
    ) -> Any:
        """Run a plan-authorized target-token teacher without releasing its logits."""

        return broker.run_mapped_oon_teacher(
            identity, step, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices,
            autonomous_forward=self._autonomous_oon_forward,
        )

    def run_mapped_coordinate_teacher(
        self, *, broker: Any, identity: runtime.TraceIdentity, step: Any,
        fit_rows: torch.Tensor, student_tokens: torch.Tensor,
        student_indices: Any, teacher_tokens: torch.Tensor,
        teacher_indices: Any, program: runtime.JointAffineProgram,
    ) -> Any:
        """Run mapped P/P/N label construction without releasing target states."""

        return broker.run_mapped_coordinate_teacher(
            identity, step, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices, program=program,
            autonomous_forward=self._autonomous_mapped_coordinate_forward,
        )

    def prepare_mapped_parent(
        self, *, broker: Any, identity: runtime.TraceIdentity,
        fit_rows: torch.Tensor, student_tokens: torch.Tensor,
        student_indices: Any, teacher_tokens: torch.Tensor,
        teacher_indices: Any, program: runtime.JointAffineProgram,
    ) -> tuple[Any, Any]:
        """Return a sealed false-paired parent and its target-forward receipt."""

        return broker.prepare_mapped_parent(
            identity, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices, program=program,
            autonomous_forward=self._autonomous_mapped_parent_forward,
        )

    def run_a_null_oon_teacher(
        self, *, broker: Any, identity: runtime.TraceIdentity, step: Any,
        fit_rows: torch.Tensor, student_tokens: torch.Tensor,
        student_indices: Any, teacher_tokens: torch.Tensor,
        teacher_indices: Any,
    ) -> Any:
        """Score the false-parent source trajectory against true source O/O/N."""

        return broker.run_a_null_oon_teacher(
            identity, step, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices,
            autonomous_forward=self._autonomous_oon_forward,
        )

    def _autonomous_mapped_parent_forward(
        self, gateway: Any, tokens: torch.Tensor,
    ) -> dict[str, Any]:
        """Run native-free selected-L P/P/N and seal its executable L0 code."""

        poison = _EarlyNativePoison(self._model)
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        deployed_n_calls = {site: 0 for site in EARLY_SITES}
        correction_calls = {site: 0 for site in EARLY_SITES}
        outer_returned = False

        def attention(event: facade.AttentionEvent):
            attention_calls[event.site] += 1
            return self._ship.attention(event)

        def mlp(event: facade.EarlyMLPEvent):
            mlp_calls[event.site] += 1
            deployed = self._ship.mlp(event)
            if event.site not in EARLY_SITES:
                return deployed
            deployed_n_calls[event.site] += 1
            if event.site == 2:
                return deployed
            correction_calls[event.site] += 1
            return gateway.correct(event.site, event.state, deployed)

        with poison.scope():
            facade.forward_with_dispatch(
                self._model, tokens, attention, mlp,
                require_production=self._production,
            )
            outer_returned = True
        expected_all = tuple((site, 1) for site in range(len(self._model.transformer.h)))
        if _counts(attention_calls) != expected_all or _counts(mlp_calls) != expected_all:
            raise RuntimeError("mapped parent forward did not dispatch every site once")
        if _counts(deployed_n_calls) != ((0, 1), (1, 1), (2, 1)) or _counts(
            correction_calls
        ) != ((0, 1), (1, 1), (2, 0)):
            raise RuntimeError("mapped parent P/P/N call ledger did not close exactly")
        if not outer_returned or not poison.restored or not poison.inert or any(
            poison.calls.values()
        ):
            raise RuntimeError("mapped parent native guard did not close cleanly")
        return {
            "outer_forward_count": 1,
            "hook_calls": {0: 1, 1: 1, 2: 0},
            "outer_returned": True,
            "hook_restored": True,
            "hook_inert": True,
        }

    def _autonomous_mapped_coordinate_forward(
        self, gateway: Any, tokens: torch.Tensor,
    ) -> dict[str, Any]:
        """Run the mapped document through the same P/P/N target trajectory."""

        poison = _EarlyNativePoison(self._model)
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        deployed_n_calls = {site: 0 for site in EARLY_SITES}
        correction_calls = {site: 0 for site in EARLY_SITES}
        outer_returned = False

        def attention(event: facade.AttentionEvent):
            attention_calls[event.site] += 1
            return self._ship.attention(event)

        def mlp(event: facade.EarlyMLPEvent):
            mlp_calls[event.site] += 1
            deployed = self._ship.mlp(event)
            if event.site not in EARLY_SITES:
                return deployed
            deployed_n_calls[event.site] += 1
            if event.site == 2:
                return deployed
            correction_calls[event.site] += 1
            return gateway.correct_and_label(event.site, event.state, deployed)

        with poison.scope():
            facade.forward_with_dispatch(
                self._model, tokens, attention, mlp,
                require_production=self._production,
            )
            outer_returned = True
        expected_all = tuple((site, 1) for site in range(len(self._model.transformer.h)))
        if _counts(attention_calls) != expected_all or _counts(mlp_calls) != expected_all:
            raise RuntimeError("mapped coordinate forward did not dispatch every site once")
        if _counts(deployed_n_calls) != ((0, 1), (1, 1), (2, 1)) or _counts(
            correction_calls
        ) != ((0, 1), (1, 1), (2, 0)):
            raise RuntimeError("mapped coordinate P/P/N call ledger did not close exactly")
        if not outer_returned or not poison.restored or not poison.inert or any(
            poison.calls.values()
        ):
            raise RuntimeError("mapped coordinate native guard did not close cleanly")
        return {
            "outer_forward_count": 1,
            "hook_calls": {0: 1, 1: 1, 2: 0},
            "outer_returned": True,
            "hook_restored": True,
            "hook_inert": True,
        }

    def _autonomous_oon_forward(self, gateway: Any, tokens: torch.Tensor):
        calls = {site: 0 for site in EARLY_SITES}
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}

        def attention(event: facade.AttentionEvent):
            attention_calls[event.site] += 1
            return self._ship.attention(event)

        def mlp(event: facade.EarlyMLPEvent):
            mlp_calls[event.site] += 1
            if event.site in CORRECTION_SITES:
                calls[event.site] += 1
                return gateway.call(event.site, event.state)
            if event.site == 2:
                calls[2] += 0
            return self._ship.mlp(event)

        logits = facade.forward_with_dispatch(
            self._model,
            tokens,
            attention,
            mlp,
            require_production=self._production,
        )
        expected_all = tuple(
            (site, 1) for site in range(len(self._model.transformer.h))
        )
        if _counts(attention_calls) != expected_all or _counts(mlp_calls) != expected_all:
            raise RuntimeError("autonomous OON forward did not dispatch every site once")
        if _counts(calls) != ((0, 1), (1, 1), (2, 0)):
            raise RuntimeError("autonomous O/O/N call ledger did not close exactly")
        return logits, {
            "outer_forward_count": 1,
            "hook_calls": {0: 1, 1: 1, 2: 0},
            "outer_returned": True,
            "hook_restored": True,
            "hook_inert": True,
        }

    def _run_final_response_teacher_forward(
        self, *, edit: response_execution.FinalResponseEdit,
        role_rows: torch.Tensor, ordered_batch_indices: Any,
        basis0: torch.Tensor, basis1: torch.Tensor,
    ) -> _ObservedResponseTeacherForward:
        """Execute one exact O/O/N teacher edit without releasing live tensors.

        This is deliberately private.  The eventual public response-batch method must
        call it three times and reduce those tensors with all 22 student triplets
        before returning anything.
        """

        if not isinstance(edit, response_execution.FinalResponseEdit):
            raise TypeError("response teacher requires an authority-derived edit")
        indices = tuple(ordered_batch_indices)
        edit.require_pristine(
            role_rows=role_rows, ordered_batch_indices=indices, basis0=basis0,
        )
        checked_basis1 = basis1.detach().cpu().float().contiguous()
        contract.validate_orthonormal_basis("response basis1", checked_basis1)
        try:
            model_parameter = next(self._model.parameters())
        except StopIteration as error:
            raise RuntimeError("observed model has no device-bearing parameters") from error
        model_device = model_parameter.device
        tokens = role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous().to(model_device)
        native_forwards = {
            site: self._model.transformer.h[site].mlp.forward
            for site in CORRECTION_SITES
        }
        poison = _EarlyNativePoison(self._model)
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        deployed_n_calls = {site: 0 for site in EARLY_SITES}
        exact_calls = {site: 0 for site in EARLY_SITES}
        code1: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        outer_returned = False

        def attention(event: facade.AttentionEvent):
            attention_calls[event.site] += 1
            return self._ship.attention(event)

        def mlp(event: facade.EarlyMLPEvent):
            nonlocal code1
            mlp_calls[event.site] += 1
            if event.site == 0:
                exact_calls[0] += 1
                native = native_forwards[0](event.state)
                physical = edit.physical_edit.to(
                    device=native.device, dtype=native.dtype,
                )
                return native + physical
            if event.site == 1:
                exact_calls[1] += 1
                native = native_forwards[1](event.state)
                projected = native.float() @ checked_basis1.to(native.device)
                code1 = runtime.scored_positions(projected).detach().cpu().float().contiguous()
                return native
            if event.site == 2:
                deployed_n_calls[2] += 1
            return self._ship.mlp(event)

        with torch.no_grad():
            with poison.scope():
                logits = facade.forward_with_dispatch(
                    self._model, tokens, attention, mlp,
                    require_production=self._production,
                )
                outer_returned = True
        expected_all = tuple((site, 1) for site in range(len(self._model.transformer.h)))
        literal_calls = {
            site: exact_calls[site] + poison.calls[site] for site in EARLY_SITES
        }
        if _counts(attention_calls) != expected_all or _counts(mlp_calls) != (
            expected_all
        ) or _counts(deployed_n_calls) != ((0, 0), (1, 0), (2, 1)) or _counts(
            literal_calls
        ) != ((0, 1), (1, 1), (2, 0)) or not outer_returned or not poison.restored or not (
            poison.inert
        ) or any(poison.calls.values()) or code1 is None or logits is None:
            raise RuntimeError("exact response teacher forward did not close exactly")
        scored_logits = runtime.scored_positions(logits).detach().cpu().float().contiguous()
        observed = ObservedClosure(
            scope=f"final_response_exact_teacher_{edit.edit_sign:+d}",
            outer_forward_count=1, outer_returned=True,
            attention_dispatch_calls=_counts(attention_calls),
            mlp_dispatch_calls=_counts(mlp_calls),
            deployed_n_calls=_counts(deployed_n_calls),
            correction_calls=((0, 0), (1, 0), (2, 0)),
            literal_early_mlp_calls=_counts(literal_calls),
            native_guard_restored=poison.restored,
            native_guard_inert=poison.inert,
            logit_shape=tuple(scored_logits.shape),
            logit_dtype=str(scored_logits.dtype),
        )
        return _ObservedResponseTeacherForward(
            code1=code1, logits=scored_logits, edit_sha256=edit.sha256,
            unit_identity_sha256=edit.unit_identity_sha256,
            observed_closure=observed,
        )

    def _run_final_response_student_forward(
        self, *, broker: Any, hook: runtime.StudentCorrectionHook,
        materialized: final_actions.MaterializedFinalAction,
        final_action_identity: final_actions.FinalActionBatchIdentity,
        binding: response_execution.ResponseProgramBatchBinding,
        edit: response_execution.FinalResponseEdit, final_context: Any,
        role_rows: torch.Tensor, ordered_batch_indices: Any,
        basis0: torch.Tensor,
    ) -> _ObservedResponseStudentForward:
        """Execute and consume one perturbation-bound P/P/N student forward."""

        import early_mlp_suffix_transport_v1_capabilities as capabilities

        if not isinstance(broker, capabilities.CapabilityBroker) or not isinstance(
            hook, runtime.StudentCorrectionHook
        ) or not isinstance(materialized, final_actions.MaterializedFinalAction) or not isinstance(
            final_action_identity, final_actions.FinalActionBatchIdentity
        ) or not isinstance(binding, response_execution.ResponseProgramBatchBinding) or not isinstance(
            edit, response_execution.FinalResponseEdit
        ) or not isinstance(final_context, capabilities.FinalRunContext):
            raise TypeError("response student requires typed execution authorities")
        indices = tuple(ordered_batch_indices)
        trace = binding.runtime_identity
        execution_identity = binding.execution_identity
        final_action_identity.require_role_rows(
            materialized=materialized, role_rows=role_rows,
            ordered_batch_indices=indices,
        )
        edit.require_pristine(
            role_rows=role_rows, ordered_batch_indices=indices, basis0=basis0,
        )
        final_context.require_identity(
            trace, role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous(), indices,
        )
        if broker.ledger_snapshot.run_context_sha256 != final_context.sha256 or (
            execution_identity.final_action_identity_sha256 != final_action_identity.sha256
        ) or execution_identity.materialization_sha256 != materialized.sha256 or (
            execution_identity.edit_sha256 != edit.sha256
        ) or execution_identity.runtime_identity_sha256 != trace.sha256 or (
            execution_identity.unit_identity_sha256 != edit.unit_identity_sha256
        ):
            raise RuntimeError("response student authorities do not share one identity")
        program = materialized.make_program()
        try:
            hook.configure(
                program=program, states={0: "P", 1: "P"},
                site0_edit=edit.code_edit,
            )
            try:
                model_device = next(self._model.parameters()).device
            except StopIteration as error:
                raise RuntimeError("observed model has no device-bearing parameters") from error
            model_inputs = role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous().to(
                model_device
            )
            session = broker.begin_student(trace, hook, model_inputs, indices)
        except BaseException:
            try:
                hook.clear_configuration()
            except BaseException:
                pass
            raise
        with torch.no_grad():
            step, student_closure, observed = self.run_student(
                session=session, hook=hook, identity=trace, tokens=model_inputs,
            )
            private_tensors, consumer_closure = broker._consume_final_response_student(
                trace, step,
            )
            private_sha256 = private_tensors.sha256
            code1, logits = private_tensors._take_for_observed_adapter(trace)
        if student_closure.scope != "student" or student_closure.original_calls != (
            capabilities.EXACT_ZERO_CALLS
        ) or not student_closure.hook_restored or not student_closure.hook_inert or (
            consumer_closure.scope != "final_response_student"
        ) or consumer_closure.original_calls != capabilities.EXACT_ZERO_CALLS or not (
            consumer_closure.consumed
        ) or observed.deployed_n_calls != ((0, 1), (1, 1), (2, 1)) or (
            observed.correction_calls != ((0, 1), (1, 1), (2, 0))
        ) or observed.literal_early_mlp_calls != ((0, 0), (1, 0), (2, 0)) or not (
            observed.native_guard_restored and observed.native_guard_inert
        ) or consumer_closure.output_sha256 != private_sha256:
            raise RuntimeError("response student forward did not close exactly")
        broker_snapshot = broker.ledger_snapshot
        if broker_snapshot.outstanding_identity_sha256 is not None:
            raise RuntimeError("response student broker retained an outstanding identity")
        return _ObservedResponseStudentForward(
            code1=code1, logits=logits,
            response_execution_identity_sha256=execution_identity.sha256,
            edit_sha256=edit.sha256,
            unit_identity_sha256=edit.unit_identity_sha256,
            student_step_ledger_sha256=student_closure.ledger_sha256,
            consumer_ledger_sha256=consumer_closure.ledger_sha256,
            broker_ledger_sha256=runtime.logical_identity_sha256(asdict(broker_snapshot)),
            observed_closure=observed,
        )

    def run_final_response_batch(
        self, *, validated_program_bank: Any, inherited_initialization: Any,
        final_context: Any, role_rows: torch.Tensor,
        ordered_batch_indices: Any, batch_ordinal: int,
    ) -> response_execution.ObservedResponseBatchResult:
        """Execute the exact ordered 69-forward response transaction for one batch."""

        import early_mlp_suffix_transport_v1_capabilities as capabilities

        if not isinstance(final_context, capabilities.FinalRunContext) or not callable(
            getattr(inherited_initialization, "clone_bases", None)
        ) or not callable(getattr(inherited_initialization, "make_program", None)) or not (
            hasattr(inherited_initialization, "authority")
        ):
            raise TypeError("response batch requires final and inherited authorities")
        indices = tuple(ordered_batch_indices)
        expected_indices = tuple(range(
            batch_ordinal * runtime.BATCH_SIZE,
            (batch_ordinal + 1) * runtime.BATCH_SIZE,
        ))
        if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or tuple(
            role_rows.shape
        ) != (runtime.BATCH_SIZE, 513) or role_rows.device.type != "cpu" or not (
            role_rows.is_contiguous()
        ) or indices != expected_indices or final_context.inherited_snapshot_sha256 != (
            inherited_initialization.authority.snapshot_sha256
        ):
            raise RuntimeError("response batch rows, schedule, or inherited authority changed")
        bases = inherited_initialization.clone_bases()
        if set(bases) != {0, 1}:
            raise RuntimeError("response batch inherited bases are incomplete")
        inherited_q = inherited_initialization.make_program("L")
        sources = final_actions.source_bank_from_validated(
            validated_program_bank, inherited_q=inherited_q,
        )
        payload_sha256 = validated_program_bank.get("payload_sha256")
        if not runtime._sha256_text(payload_sha256):
            raise RuntimeError("response batch program payload identity is malformed")
        common_support_sha256 = runtime.logical_identity_sha256({
            "role": "early_mlp_suffix_transport_v1_final",
            "final_role_tensor_sha256": final_context.final_role_tensor_sha256,
            "rows_receipt_sha256": final_context.rows_receipt_sha256,
            "row_count": final_context.final_row_count,
            "score_start": runtime.SCORE_START,
            "score_stop": runtime.SCORE_STOP,
        })
        edits = {
            sign: response_execution.build_final_response_edit(
                validated_program_bank=validated_program_bank, role_rows=role_rows,
                ordered_batch_indices=indices, batch_ordinal=batch_ordinal,
                basis0=bases[0], edit_sign=sign,
            ) for sign in response_execution.EDIT_SIGNS
        }
        if len({value.unit_identity_sha256 for value in edits.values()}) != 1:
            raise RuntimeError("response batch edit triplet does not share one unit")
        unit_identity = edits[0].unit_identity_sha256
        plan = response_plan.build_response_batch_plan(
            batch_ordinal=batch_ordinal,
            ordered_role_rows_sha256=runtime.tensor_identity_sha256(role_rows),
            intervention_unit_sha256=unit_identity,
        )
        issuer_id = runtime.logical_identity_sha256({
            "kind": "final_response_batch", "batch_plan_sha256": plan.sha256,
            "final_context_sha256": final_context.sha256,
            "source_bank_sha256": sources.sha256,
            "program_payload_sha256": payload_sha256,
        })
        coordinator = runtime.ScopeCoordinator()
        hook = runtime.StudentCorrectionHook(
            bases, issuer_id=issuer_id, coordinator=coordinator,
        )
        broker = self.make_capability_broker(
            issuer_id=issuer_id, coordinator=coordinator,
            run_context=final_context, bases=bases,
        )
        sign_for = {"baseline": 0, "positive": 1, "negative": -1}
        forward_receipts: list[response_execution.ObservedResponseForwardReceipt] = []
        teacher_forwards: list[_ObservedResponseTeacherForward] = []

        for forward_plan in plan.forwards[:3]:
            edit = edits[sign_for[forward_plan.perturbation]]
            observed_teacher = self._run_final_response_teacher_forward(
                edit=edit, role_rows=role_rows, ordered_batch_indices=indices,
                basis0=bases[0], basis1=bases[1],
            )
            execution_identity = runtime.logical_identity_sha256({
                "kind": "exact_oon_response_teacher",
                "response_execution_amendment_sha256": (
                    response_execution.RESPONSE_EXECUTION_AMENDMENT_SHA256
                ),
                "forward_plan_sha256": forward_plan.sha256,
                "edit_sha256": edit.sha256,
                "final_context_sha256": final_context.sha256,
                "source_bank_sha256": sources.sha256,
                "basis1_sha256": runtime.tensor_identity_sha256(
                    bases[1].detach().cpu().float().contiguous()
                ),
            })
            receipt = response_execution.ObservedResponseForwardReceipt(
                forward_plan_sha256=forward_plan.sha256,
                subject_key=forward_plan.subject_key,
                perturbation=forward_plan.perturbation,
                batch_ordinal=batch_ordinal,
                execution_identity_sha256=execution_identity,
                final_action_identity_sha256=None, materialization_sha256=None,
                edit_sha256=edit.sha256,
                semantic_delta_sha256=edit.semantic_delta_sha256,
                code_edit_sha256=edit.code_edit_sha256,
                physical_edit_sha256=edit.physical_edit_sha256,
                code1_sha256=runtime.tensor_identity_sha256(observed_teacher.code1),
                logits_sha256=runtime.tensor_identity_sha256(observed_teacher.logits),
                observed_closure_sha256=runtime.logical_identity_sha256(
                    asdict(observed_teacher.observed_closure)
                ),
                student_step_ledger_sha256=None,
                consumer_ledger_sha256=None, broker_ledger_sha256=None,
            )
            teacher_forwards.append(observed_teacher)
            forward_receipts.append(receipt)
        teacher_code = response_reductions.ResponseTriplet(
            baseline=teacher_forwards[0].code1,
            positive=teacher_forwards[1].code1,
            negative=teacher_forwards[2].code1,
        )
        teacher_logits = response_reductions.ResponseTriplet(
            baseline=teacher_forwards[0].logits,
            positive=teacher_forwards[1].logits,
            negative=teacher_forwards[2].logits,
        )
        teacher_receipt_sha256s = tuple(
            value.sha256 for value in forward_receipts[:3]
        )
        arm_reductions: list[response_execution.ObservedResponseArmReduction] = []
        offset = 3
        for action_key in response_plan.RESPONSE_ACTION_KEYS:
            arm, background = action_key.split("/")
            materialized = final_actions.materialize(
                final_actions.plan_for(arm, background), sources,
            )
            final_identity = final_actions.FinalActionBatchIdentity.from_role_rows(
                materialized=materialized, role_rows=role_rows,
                ordered_batch_indices=indices, batch_ordinal=batch_ordinal,
                source_commit=final_context.source_commit,
                inherited_snapshot_sha256=final_context.inherited_snapshot_sha256,
                rows_receipt_sha256=final_context.rows_receipt_sha256,
                final_role_tensor_sha256=final_context.final_role_tensor_sha256,
                program_payload_sha256=payload_sha256,
                common_support_sha256=common_support_sha256,
            )
            student_forwards: list[_ObservedResponseStudentForward] = []
            student_receipts: list[
                response_execution.ObservedResponseForwardReceipt
            ] = []
            action_plans = plan.forwards[offset:offset + 3]
            offset += 3
            for forward_plan in action_plans:
                edit = edits[sign_for[forward_plan.perturbation]]
                binding = response_execution.bind_runtime_response_program_batch(
                    materialized=materialized,
                    final_action_identity=final_identity,
                    forward_plan=forward_plan, edit=edit,
                    role_rows=role_rows, ordered_batch_indices=indices,
                    teacher_mapping_sha256=(
                        final_context.identity_teacher_mapping_sha256
                    ),
                )
                observed_student = self._run_final_response_student_forward(
                    broker=broker, hook=hook, materialized=materialized,
                    final_action_identity=final_identity, binding=binding,
                    edit=edit, final_context=final_context, role_rows=role_rows,
                    ordered_batch_indices=indices, basis0=bases[0],
                )
                receipt = response_execution.ObservedResponseForwardReceipt(
                    forward_plan_sha256=forward_plan.sha256,
                    subject_key=forward_plan.subject_key,
                    perturbation=forward_plan.perturbation,
                    batch_ordinal=batch_ordinal,
                    execution_identity_sha256=(
                        observed_student.response_execution_identity_sha256
                    ),
                    final_action_identity_sha256=final_identity.sha256,
                    materialization_sha256=materialized.sha256,
                    edit_sha256=edit.sha256,
                    semantic_delta_sha256=edit.semantic_delta_sha256,
                    code_edit_sha256=edit.code_edit_sha256,
                    physical_edit_sha256=edit.physical_edit_sha256,
                    code1_sha256=runtime.tensor_identity_sha256(
                        observed_student.code1
                    ),
                    logits_sha256=runtime.tensor_identity_sha256(
                        observed_student.logits
                    ),
                    observed_closure_sha256=runtime.logical_identity_sha256(
                        asdict(observed_student.observed_closure)
                    ),
                    student_step_ledger_sha256=(
                        observed_student.student_step_ledger_sha256
                    ),
                    consumer_ledger_sha256=(
                        observed_student.consumer_ledger_sha256
                    ),
                    broker_ledger_sha256=observed_student.broker_ledger_sha256,
                )
                student_forwards.append(observed_student)
                student_receipts.append(receipt)
                forward_receipts.append(receipt)
            student_code = response_reductions.ResponseTriplet(
                baseline=student_forwards[0].code1,
                positive=student_forwards[1].code1,
                negative=student_forwards[2].code1,
            )
            student_logits = response_reductions.ResponseTriplet(
                baseline=student_forwards[0].logits,
                positive=student_forwards[1].logits,
                negative=student_forwards[2].logits,
            )
            code_reduction = (
                response_reductions.reduce_code_response(
                    teacher=teacher_code, student=student_code,
                    unit_identity=unit_identity,
                ) if action_key in {"ll/N", "lt/N"} else None
            )
            logit_reduction = response_reductions.reduce_centered_logit_response(
                teacher=teacher_logits, student=student_logits,
                unit_identity=unit_identity,
            )
            output_kl_reduction = response_reductions.reduce_output_kl_response(
                teacher=teacher_logits, student=student_logits,
                unit_identity=unit_identity,
            )
            arm_reductions.append(response_execution.ObservedResponseArmReduction(
                action_key=action_key, batch_plan_sha256=plan.sha256,
                teacher_forward_receipt_sha256s=teacher_receipt_sha256s,
                student_forward_receipt_sha256s=tuple(
                    value.sha256 for value in student_receipts
                ),
                code_response=code_reduction, logit_response=logit_reduction,
                output_kl_response=output_kl_reduction,
            ))
            student_forwards.clear()
            student_receipts.clear()
        if offset != len(plan.forwards) or len(forward_receipts) != 69 or not (
            coordinator.idle
        ):
            raise RuntimeError("response batch forward schedule did not close")
        broker_snapshot = broker.ledger_snapshot
        if broker_snapshot.student_identity_count != 66 or (
            broker_snapshot.teacher_identity_count != 66
        ) or broker_snapshot.completed_identity_count != 66 or (
            broker_snapshot.outstanding_identity_sha256 is not None
        ):
            raise RuntimeError("response batch broker ledger did not close")
        batch_receipt = response_execution.ObservedResponseBatchReceipt(
            batch_ordinal=batch_ordinal, batch_plan_sha256=plan.sha256,
            source_bank_sha256=sources.sha256,
            program_payload_sha256=payload_sha256,
            final_context_sha256=final_context.sha256,
            common_support_sha256=common_support_sha256,
            basis0_sha256=runtime.tensor_identity_sha256(
                bases[0].detach().cpu().float().contiguous()
            ),
            basis1_sha256=runtime.tensor_identity_sha256(
                bases[1].detach().cpu().float().contiguous()
            ),
            forward_receipt_sha256s=tuple(
                value.sha256 for value in forward_receipts
            ),
            arm_reduction_sha256s=tuple(
                (value.action_key, value.sha256) for value in arm_reductions
            ),
            broker_ledger_sha256=runtime.logical_identity_sha256(
                asdict(broker_snapshot)
            ),
            teacher_forward_count=3, student_forward_count=66,
            atomic_complete=True,
        )
        teacher_forwards.clear()
        forward_receipts.clear()
        return response_execution.ObservedResponseBatchResult(
            arm_reductions=tuple(arm_reductions), receipt=batch_receipt,
        )

    def run_final_response_role(
        self, *, validated_program_bank: Any, inherited_initialization: Any,
        final_context: Any, final_rows: torch.Tensor,
    ) -> response_execution.ObservedResponseRunResult:
        """Execute all 48 canonical response batches without exposing a partial run."""

        import early_mlp_suffix_transport_v1_capabilities as capabilities

        if not isinstance(final_context, capabilities.FinalRunContext) or not torch.is_tensor(
            final_rows
        ) or final_rows.dtype != torch.long or tuple(final_rows.shape) != (
            response_execution.FINAL_ROW_COUNT, response_execution.FINAL_ROW_WIDTH,
        ) or final_rows.device.type != "cpu" or not final_rows.is_contiguous() or (
            runtime.tensor_identity_sha256(final_rows)
            != final_context.final_role_tensor_sha256
        ):
            raise RuntimeError("final response role authority or tensor changed")
        before_sha256 = runtime.tensor_identity_sha256(final_rows)
        before_version = final_rows._version
        accumulator = response_execution.ObservedResponseRunAccumulator()
        for batch_ordinal in range(response_execution.FINAL_BATCH_COUNT):
            start = batch_ordinal * runtime.BATCH_SIZE
            indices = tuple(range(start, start + runtime.BATCH_SIZE))
            batch_rows = final_rows[start:start + runtime.BATCH_SIZE].contiguous()
            accumulator.add(self.run_final_response_batch(
                validated_program_bank=validated_program_bank,
                inherited_initialization=inherited_initialization,
                final_context=final_context, role_rows=batch_rows,
                ordered_batch_indices=indices, batch_ordinal=batch_ordinal,
            ))
        result = accumulator.finish()
        if result.receipt.final_context_sha256 != final_context.sha256 or (
            final_rows._version != before_version
        ) or runtime.tensor_identity_sha256(final_rows) != before_sha256:
            raise RuntimeError("final response role mutated or changed authority")
        return result
