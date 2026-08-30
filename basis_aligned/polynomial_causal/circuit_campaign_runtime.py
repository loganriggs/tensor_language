"""Hook-free, one-use bilin18 execution boundary for circuit campaigns.

The module owns no row selection, statistics, model loading, or publication.  It
routes one already-loaded model forward through :mod:`bilin18_observed_model_facade`
according to immutable arm plans and returns an exact component-call ledger.
Replacement callbacks see value-preserving tensor clones but never the native block,
so the runtime itself cannot provide a path to call a replaced native component.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import torch

import bilin18_observed_model_facade as facade


PRODUCTION_SITE_COUNT = 18


class ArmKind(str, Enum):
    NATIVE = "native"
    CANDIDATE = "candidate"


class ComponentAction(str, Enum):
    NATIVE = "native"
    REPLACE = "replace"


@dataclass(frozen=True)
class ComponentPlan:
    site: int
    action: ComponentAction
    replacement: str | None = None

    def __post_init__(self) -> None:
        if type(self.site) is not int or self.site < 0:
            raise ValueError("component site must be a nonnegative integer")
        if type(self.action) is not ComponentAction:
            raise ValueError("component action must be a ComponentAction")
        if self.action is ComponentAction.REPLACE:
            if not isinstance(self.replacement, str) or not self.replacement:
                raise ValueError("replacement action requires a nonempty replacement ID")
        elif self.replacement is not None:
            raise ValueError("native action cannot name a replacement")


@dataclass(frozen=True)
class ArmPlan:
    name: str
    kind: ArmKind
    attention: tuple[ComponentPlan, ...]
    mlp: tuple[ComponentPlan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("arm name must be a nonempty string")
        if type(self.kind) is not ArmKind:
            raise ValueError("arm kind must be an ArmKind")
        if type(self.attention) is not tuple or type(self.mlp) is not tuple or any(
            type(item) is not ComponentPlan for item in (*self.attention, *self.mlp)
        ):
            raise ValueError("arm component plans must have exact ComponentPlan type")
        if self.kind is ArmKind.NATIVE and any(
            item.action is not ComponentAction.NATIVE
            for item in (*self.attention, *self.mlp)
        ):
            raise ValueError("native arm cannot contain replacements")
        if self.kind is ArmKind.CANDIDATE and not any(
            item.action is ComponentAction.REPLACE
            for item in (*self.attention, *self.mlp)
        ):
            raise ValueError("candidate arm must replace at least one component")

    @classmethod
    def build(
        cls,
        name: str,
        kind: ArmKind,
        *,
        site_count: int = PRODUCTION_SITE_COUNT,
        attention_replacements: Mapping[int, str] | None = None,
        mlp_replacements: Mapping[int, str] | None = None,
    ) -> "ArmPlan":
        if type(site_count) is not int or site_count <= 0:
            raise ValueError("site_count must be a positive integer")
        attention_map = dict(attention_replacements or {})
        mlp_map = dict(mlp_replacements or {})
        for mapping in (attention_map, mlp_map):
            if any(type(site) is not int or not 0 <= site < site_count for site in mapping):
                raise ValueError("replacement site is outside the arm topology")
            if any(not isinstance(value, str) or not value for value in mapping.values()):
                raise ValueError("replacement IDs must be nonempty strings")

        def plans(replacements: Mapping[int, str]) -> tuple[ComponentPlan, ...]:
            return tuple(
                ComponentPlan(
                    site,
                    ComponentAction.REPLACE if site in replacements else ComponentAction.NATIVE,
                    replacements.get(site),
                )
                for site in range(site_count)
            )

        return cls(
            name=name,
            kind=kind,
            attention=plans(attention_map),
            mlp=plans(mlp_map),
        )


@dataclass(frozen=True)
class CircuitPlan:
    name: str
    site_count: int
    arms: tuple[ArmPlan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("circuit plan name must be a nonempty string")
        if type(self.site_count) is not int or self.site_count <= 0:
            raise ValueError("circuit site_count must be a positive integer")
        if type(self.arms) is not tuple or not self.arms or any(
            type(arm) is not ArmPlan for arm in self.arms
        ):
            raise ValueError("circuit arms must be a nonempty exact ArmPlan tuple")
        names = tuple(arm.name for arm in self.arms)
        if len(set(names)) != len(names):
            raise ValueError("circuit arm names must be unique")
        if sum(arm.kind is ArmKind.NATIVE for arm in self.arms) != 1:
            raise ValueError("circuit plan requires exactly one native arm")
        expected_sites = tuple(range(self.site_count))
        for arm in self.arms:
            if tuple(item.site for item in arm.attention) != expected_sites or tuple(
                item.site for item in arm.mlp
            ) != expected_sites:
                raise ValueError("every arm must cover every site exactly once in order")

    def arm(self, name: str) -> ArmPlan:
        matches = tuple(arm for arm in self.arms if arm.name == name)
        if len(matches) != 1:
            raise ValueError("requested arm is not uniquely present in the circuit plan")
        return matches[0]


@dataclass(frozen=True)
class AttentionReplacementEvent:
    site: int
    state: torch.Tensor
    tokens: torch.Tensor
    first_value: torch.Tensor | None


@dataclass(frozen=True)
class MLPReplacementEvent:
    site: int
    state: torch.Tensor
    attention_write: torch.Tensor
    tokens: torch.Tensor
    prior_writes: tuple[torch.Tensor, ...]


AttentionReplacement = Callable[
    [AttentionReplacementEvent], tuple[torch.Tensor, torch.Tensor]
]
MLPReplacement = Callable[[MLPReplacementEvent], torch.Tensor]


@dataclass(frozen=True)
class SiteCallLedger:
    site: int
    native_attention_calls: int
    replacement_attention_calls: int
    native_mlp_calls: int
    replacement_mlp_calls: int


@dataclass(frozen=True)
class ForwardClosure:
    circuit: str
    arm: str
    arm_kind: ArmKind
    attempted_outer_forwards: int
    completed_outer_forwards: int
    outer_returns: int
    document_count: int
    sites: tuple[SiteCallLedger, ...]
    candidate_native_call_prohibition_passed: bool
    closed: bool


def _owned_tensor(value: torch.Tensor) -> torch.Tensor:
    # Clone preserves dtype/device/value and graph behavior; owner execution itself is
    # no-grad, while the clone prevents callbacks from retaining live facade aliases.
    return value.clone()


class CircuitForwardOwner:
    """Consume exactly one arm forward and expose its immutable call closure."""

    __slots__ = (
        "_plan", "_arm", "_attention_callbacks", "_mlp_callbacks", "_state",
        "_native_attention", "_replacement_attention", "_native_mlp",
        "_replacement_mlp", "_closure", "_sealed",
    )

    def __init__(
        self,
        *,
        plan: CircuitPlan,
        arm: str,
        attention_replacements: Mapping[str, AttentionReplacement] | None = None,
        mlp_replacements: Mapping[str, MLPReplacement] | None = None,
    ) -> None:
        if type(plan) is not CircuitPlan:
            raise ValueError("owner requires an exact CircuitPlan")
        selected = plan.arm(arm)
        attention_callbacks = dict(attention_replacements or {})
        mlp_callbacks = dict(mlp_replacements or {})
        required_attention = {
            item.replacement for item in selected.attention
            if item.action is ComponentAction.REPLACE
        }
        required_mlp = {
            item.replacement for item in selected.mlp
            if item.action is ComponentAction.REPLACE
        }
        if set(attention_callbacks) != required_attention or any(
            not callable(value) for value in attention_callbacks.values()
        ):
            raise ValueError("attention callbacks do not exactly match the selected arm")
        if set(mlp_callbacks) != required_mlp or any(
            not callable(value) for value in mlp_callbacks.values()
        ):
            raise ValueError("MLP callbacks do not exactly match the selected arm")
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_arm", selected)
        object.__setattr__(self, "_attention_callbacks", MappingProxyType(attention_callbacks))
        object.__setattr__(self, "_mlp_callbacks", MappingProxyType(mlp_callbacks))
        object.__setattr__(self, "_state", "fresh")
        object.__setattr__(self, "_native_attention", (0,) * plan.site_count)
        object.__setattr__(self, "_replacement_attention", (0,) * plan.site_count)
        object.__setattr__(self, "_native_mlp", (0,) * plan.site_count)
        object.__setattr__(self, "_replacement_mlp", (0,) * plan.site_count)
        object.__setattr__(self, "_closure", None)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("circuit forward owner is sealed")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        raise TypeError("circuit forward owner cannot be copied")

    def __deepcopy__(self, _memo):
        raise TypeError("circuit forward owner cannot be deep-copied")

    def __reduce__(self):
        raise TypeError("circuit forward owner cannot be serialized")

    def _site_ledgers(self) -> tuple[SiteCallLedger, ...]:
        return tuple(
            SiteCallLedger(
                site=site,
                native_attention_calls=self._native_attention[site],
                replacement_attention_calls=self._replacement_attention[site],
                native_mlp_calls=self._native_mlp[site],
                replacement_mlp_calls=self._replacement_mlp[site],
            )
            for site in range(self._plan.site_count)
        )

    def _candidate_prohibition_passed(self) -> bool:
        if self._arm.kind is ArmKind.NATIVE:
            return True
        return all(
            self._native_attention[item.site] == 0
            for item in self._arm.attention if item.action is ComponentAction.REPLACE
        ) and all(
            self._native_mlp[item.site] == 0
            for item in self._arm.mlp if item.action is ComponentAction.REPLACE
        )

    def _successful_closure(self, document_count: int) -> ForwardClosure:
        sites = self._site_ledgers()
        for attention_plan, mlp_plan, ledger in zip(
            self._arm.attention, self._arm.mlp, sites, strict=True,
        ):
            expected_attention = (
                (0, 1) if attention_plan.action is ComponentAction.REPLACE else (1, 0)
            )
            expected_mlp = (
                (0, 1) if mlp_plan.action is ComponentAction.REPLACE else (1, 0)
            )
            if (
                (ledger.native_attention_calls, ledger.replacement_attention_calls)
                != expected_attention
                or (ledger.native_mlp_calls, ledger.replacement_mlp_calls) != expected_mlp
            ):
                raise RuntimeError("successful forward call ledger is not exact")
        prohibition = self._candidate_prohibition_passed()
        if not prohibition:
            raise RuntimeError("candidate arm called a replaced native component")
        return ForwardClosure(
            circuit=self._plan.name,
            arm=self._arm.name,
            arm_kind=self._arm.kind,
            attempted_outer_forwards=1,
            completed_outer_forwards=1,
            outer_returns=1,
            document_count=document_count,
            sites=sites,
            candidate_native_call_prohibition_passed=prohibition,
            closed=True,
        )

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        *,
        require_production: bool = True,
    ) -> torch.Tensor:
        if self._state != "fresh":
            raise RuntimeError("circuit forward owner is already spent or failed")
        try:
            blocks = model.transformer.h
        except AttributeError as error:
            object.__setattr__(self, "_state", "failed")
            raise ValueError("model does not expose transformer blocks") from error
        if len(blocks) != self._plan.site_count:
            object.__setattr__(self, "_state", "failed")
            raise ValueError("model depth differs from the immutable circuit plan")
        object.__setattr__(self, "_state", "active")

        def attention(event: facade.AttentionEvent) -> tuple[torch.Tensor, torch.Tensor]:
            component = self._arm.attention[event.site]
            if component.action is ComponentAction.NATIVE:
                values = list(self._native_attention)
                values[event.site] += 1
                object.__setattr__(self, "_native_attention", tuple(values))
                return event.block.attn(event.state, event.first_value)
            values = list(self._replacement_attention)
            values[event.site] += 1
            object.__setattr__(self, "_replacement_attention", tuple(values))
            callback = self._attention_callbacks[component.replacement]
            result = callback(AttentionReplacementEvent(
                site=event.site,
                state=_owned_tensor(event.state),
                tokens=_owned_tensor(event.tokens),
                first_value=None if event.first_value is None else _owned_tensor(event.first_value),
            ))
            if not isinstance(result, tuple) or len(result) != 2 or not all(
                torch.is_tensor(value) for value in result
            ):
                raise RuntimeError("attention replacement result is malformed")
            return _owned_tensor(result[0]), _owned_tensor(result[1])

        def mlp(event: facade.EarlyMLPEvent) -> torch.Tensor:
            component = self._arm.mlp[event.site]
            if component.action is ComponentAction.NATIVE:
                values = list(self._native_mlp)
                values[event.site] += 1
                object.__setattr__(self, "_native_mlp", tuple(values))
                return event.block.mlp(event.state)
            values = list(self._replacement_mlp)
            values[event.site] += 1
            object.__setattr__(self, "_replacement_mlp", tuple(values))
            callback = self._mlp_callbacks[component.replacement]
            result = callback(MLPReplacementEvent(
                site=event.site,
                state=_owned_tensor(event.state),
                attention_write=_owned_tensor(event.attention_write),
                tokens=_owned_tensor(event.tokens),
                prior_writes=tuple(_owned_tensor(value) for value in event.prior_writes),
            ))
            if not torch.is_tensor(result):
                raise RuntimeError("MLP replacement result is malformed")
            return _owned_tensor(result)

        try:
            logits = facade.forward_with_dispatch(
                model,
                tokens,
                attention,
                mlp,
                require_production=require_production,
            )
            object.__setattr__(
                self, "_closure", self._successful_closure(int(tokens.shape[0])),
            )
        except BaseException:
            object.__setattr__(self, "_state", "failed")
            raise
        object.__setattr__(self, "_state", "spent")
        return logits

    @property
    def closure(self) -> ForwardClosure:
        if self._state != "spent" or self._closure is None:
            raise RuntimeError("successful circuit forward closure is unavailable")
        return self._closure

    @property
    def failure_ledger(self) -> tuple[SiteCallLedger, ...]:
        if self._state != "failed":
            raise RuntimeError("failure ledger is available only after failure")
        return self._site_ledgers()


__all__ = (
    "ArmKind",
    "ArmPlan",
    "AttentionReplacement",
    "AttentionReplacementEvent",
    "CircuitForwardOwner",
    "CircuitPlan",
    "ComponentAction",
    "ComponentPlan",
    "ForwardClosure",
    "MLPReplacement",
    "MLPReplacementEvent",
    "PRODUCTION_SITE_COUNT",
    "SiteCallLedger",
)
