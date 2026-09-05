"""Reusable, circuit-level intervention and scoring primitives.

The objects here deliberately know nothing about Task14, attention-head numbers,
semantic labels, ranks, optimizers, or publication.  A circuit runner supplies a
declarative, depth-ordered product plan and registers ``runtime.hook(layer)`` on
the corresponding bilinear MLP ``Down`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

import numpy as np


Operation = Literal["observe", "reset", "rescue"]


class CircuitInterventionError(ValueError):
    """A declarative plan, tensor, or execution order was invalid."""


@dataclass(frozen=True)
class ProductAction:
    """One action at a bilinear MLP product (the input to ``Down``)."""

    layer: int
    operation: Operation
    base_key: str | None = None
    capture_key: str | None = None
    events_before: tuple[str, ...] = ()
    capture_event: str | None = None
    events_after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise CircuitInterventionError("product-action layer must be a nonnegative integer")
        if self.operation not in {"observe", "reset", "rescue"}:
            raise CircuitInterventionError("unknown product operation")
        if self.operation != "observe" and not self.base_key:
            raise CircuitInterventionError("reset/rescue action requires a base cache key")
        if self.capture_event is not None and self.capture_key is None:
            raise CircuitInterventionError("capture event requires a captured live product")
        events = self.events_before + ((self.capture_event,) if self.capture_event else ()) + self.events_after
        if any(not isinstance(event, str) or not event for event in events):
            raise CircuitInterventionError("provenance event names must be nonempty strings")
        if len(events) != len(set(events)):
            raise CircuitInterventionError("one action cannot emit a duplicate event")


@dataclass(frozen=True)
class ProductPlan:
    """A strictly depth-ordered collection of MLP product actions."""

    name: str
    actions: tuple[ProductAction, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.actions:
            raise CircuitInterventionError("product plan must have a name and actions")
        layers = tuple(action.layer for action in self.actions)
        if layers != tuple(sorted(layers)) or len(layers) != len(set(layers)):
            raise CircuitInterventionError("product actions must have unique increasing layers")
        events = tuple(event for action in self.actions for event in (
            action.events_before
            + ((action.capture_event,) if action.capture_event else ())
            + action.events_after
        ))
        if len(events) != len(set(events)):
            raise CircuitInterventionError("plan provenance events must be globally unique")

    def action_at(self, layer: int) -> ProductAction | None:
        return next((action for action in self.actions if action.layer == layer), None)


def _linear(operator, value):
    return operator(value) if callable(operator) else value @ operator.T


def exact_bilinear_response_terms(left, right, x_base, x_changed):
    """Return the exact three terms in z(x_changed)-z(x_base).

    Inputs are torch tensors.  ``left`` and ``right`` are the live linear
    modules, which prevents a caller from validating against stale weights.
    """

    dx = x_changed - x_base
    return (
        _linear(left, x_base) * _linear(right, dx),
        _linear(left, dx) * _linear(right, x_base),
        _linear(left, dx) * _linear(right, dx),
    )


def bilinear_closure_max_abs(left, right, x_base, x_changed, z_base, z_changed) -> float:
    terms = exact_bilinear_response_terms(left, right, x_base, x_changed)
    error = (z_changed - z_base) - sum(terms)
    return float(error.detach().abs().max())


class ProductInterventionRuntime:
    """Stateful hook runtime for one batch and one declarative product plan.

    Only ``product[row, final_position]`` may change.  Base vectors are supplied
    by row ID and cache key.  The runtime rejects repeated or out-of-depth-order
    hook calls, records detached live products, and exposes per-row provenance.
    """

    def __init__(
        self, plan: ProductPlan, *, row_ids: Sequence[str], positions: Sequence[int],
        base_vectors: Mapping[tuple[str, str], object],
    ) -> None:
        if len(row_ids) != len(positions) or not row_ids or len(set(row_ids)) != len(row_ids):
            raise CircuitInterventionError("runtime rows/positions are malformed")
        if any(type(position) is not int or position < 0 for position in positions):
            raise CircuitInterventionError("final positions must be nonnegative integers")
        self.plan = plan
        self.row_ids = tuple(row_ids)
        self.positions = tuple(positions)
        self.base_vectors = base_vectors
        self.events: dict[str, list[str]] = {row_id: [] for row_id in self.row_ids}
        self.captures: dict[tuple[str, str], object] = {}
        self.endpoint_error: dict[str, float] = {row_id: 0.0 for row_id in self.row_ids}
        self._next_action = 0

    def hook(self, layer: int):
        action = self.plan.action_at(layer)
        if action is None:
            raise CircuitInterventionError(f"plan {self.plan.name} has no layer {layer} action")

        def down_pre(_module, arguments):
            if self._next_action >= len(self.plan.actions) or self.plan.actions[self._next_action] != action:
                raise CircuitInterventionError("product hooks executed repeatedly or out of depth order")
            if not isinstance(arguments, tuple) or not arguments:
                raise CircuitInterventionError("Down pre-hook received malformed arguments")
            product = arguments[0]
            if getattr(product, "ndim", None) != 3 or product.shape[0] != len(self.row_ids):
                raise CircuitInterventionError("Down input must have shape [batch,tokens,products]")
            if any(position >= product.shape[1] for position in self.positions):
                raise CircuitInterventionError("final position is outside the Down input")
            changed = product.clone() if action.operation != "observe" else product
            for index, (row_id, position) in enumerate(zip(self.row_ids, self.positions)):
                live = product[index, position]
                self.events[row_id].extend(action.events_before)
                if action.capture_key is not None:
                    self.captures[(row_id, action.capture_key)] = live.detach().cpu().clone()
                    if action.capture_event is not None:
                        self.events[row_id].append(action.capture_event)
                if action.operation != "observe":
                    base = self.base_vectors.get((row_id, action.base_key or ""))
                    if base is None:
                        raise CircuitInterventionError(
                            f"base product missing for {row_id}/{action.base_key}"
                        )
                    if hasattr(base, "detach") and hasattr(base, "to"):
                        base = base.detach().to(device=live.device, dtype=live.dtype)
                    else:
                        base = live.new_tensor(base)
                    if tuple(base.shape) != tuple(live.shape):
                        raise CircuitInterventionError("base product has wrong shape")
                    delta = live - base
                    replacement = live - delta if action.operation == "reset" else base + delta
                    expected = base if action.operation == "reset" else live
                    self.endpoint_error[row_id] = max(
                        self.endpoint_error[row_id], float((replacement - expected).abs().max()),
                    )
                    changed[index, position] = replacement
                self.events[row_id].extend(action.events_after)
            self._next_action += 1
            return (changed,) + tuple(arguments[1:]) if action.operation != "observe" else None

        return down_pre

    def validate_complete(self) -> None:
        if self._next_action != len(self.plan.actions):
            raise CircuitInterventionError("not every declared product action executed")

    def provenance(self) -> tuple[tuple[str, ...], ...]:
        self.validate_complete()
        return tuple(tuple(self.events[row_id]) for row_id in self.row_ids)


def full_vocab_difference(a: object, b: object) -> dict[str, float]:
    """Summarize a full-logit difference without retaining the vocabulary vector."""

    delta = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    if delta.ndim != 1 or not np.isfinite(delta).all() or delta.size == 0:
        raise CircuitInterventionError("full-vocabulary logits must be finite vectors")
    return {
        "rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def signed_response_metrics(head_effects: Sequence[float], effects: Sequence[float]) -> dict[str, float]:
    """Compute signed causal-response metrics for one predeclared semantic cell."""

    h = np.asarray(head_effects, dtype=np.float64)
    e = np.asarray(effects, dtype=np.float64)
    if h.ndim != 1 or e.shape != h.shape or h.size == 0 or not (
        np.isfinite(h).all() and np.isfinite(e).all()
    ):
        raise CircuitInterventionError("cell effects are malformed")
    dot = float(e @ h)
    h2 = float(h @ h)
    hn, en = math.sqrt(h2), float(np.linalg.norm(e))
    return {
        "count": int(h.size),
        "beta": float(-dot / max(h2, 1e-24)),
        "q": float(np.sqrt(np.mean(e * e)) / max(float(np.sqrt(np.mean(h * h))), 1e-12)),
        "absolute_coherence": float(abs(dot) / max(hn * en, 1e-24)),
        "signed_dot": dot,
        "mean_effect": float(np.mean(e)),
    }
