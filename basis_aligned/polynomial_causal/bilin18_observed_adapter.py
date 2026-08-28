"""Source-closed execution bridge from the frozen ship to suffix capabilities.

The adapter owns a complete outer forward and returns only sealed capability objects
and an immutable execution receipt.  Student callers never receive logits, live
states, dispatcher callbacks, or deployed-N handles.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator

import torch

import bilin18_observed_model_facade as facade
import early_mlp_suffix_transport_v1_runtime as runtime


EARLY_SITES = (0, 1, 2)
CORRECTION_SITES = (0, 1)


def _counts(values: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(site), int(count)) for site, count in values.items()))


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


class _EarlyNativePoison:
    """Fail before any literal MLP0/1/2 forward can execute, then restore exactly."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._modules = {
            site: model.transformer.h[site].mlp for site in EARLY_SITES
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
        """Run one P/P/N student transaction without releasing graph-bearing tensors."""

        poison = _EarlyNativePoison(self._model)
        attention_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        mlp_calls = {site: 0 for site in range(len(self._model.transformer.h))}
        deployed_n_calls = {site: 0 for site in EARLY_SITES}
        correction_calls = {site: 0 for site in EARLY_SITES}
        outer_forward_count = 0
        outer_returned = False
        logits: torch.Tensor | None = None

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
            if _counts(deployed_n_calls) != ((0, 1), (1, 1), (2, 1)) or _counts(
                correction_calls
            ) != ((0, 1), (1, 1), (2, 0)):
                raise RuntimeError("observed P/P/N call ledger did not close exactly")
            if not poison.restored or not poison.inert or any(poison.calls.values()):
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
            literal_early_mlp_calls=_counts(poison.calls),
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
