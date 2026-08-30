"""Source-closed observed-model backend for causal-response tensor collection.

No historical census helper is imported.  Every model call goes through the typed
``bilin18_observed_model_facade`` dispatch surface; component capture and rank-one
projection are synchronous, hook-free, and accounted in an exact physical call
ledger.  Inputs and masks are supplied by an authority-bound outer transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
import json

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from causal_response_tensor_collection import (
    STATISTIC_NAMES,
    aggregate_document_responses,
    document_position_index,
    local_mask_from_global,
    validate_response_tensors,
)


PHASES = ("full", "residual")
PRODUCTION_COMPONENT_ORDER = ("a8", "a16", "m16", "a3", "m14", "m13")
PRODUCTION_SPEC_ORDER_SHA256 = (
    "86d0bd7250102fc8dcdee517562fcadda74f2f6bf6d026582bcab71a33f24ca0"
)


@dataclass(frozen=True)
class CircuitSpec:
    tag: str
    component: str
    member_mask: torch.Tensor
    slice_mask: torch.Tensor

    def validate(self, *, grid_size: int, layer_count: int) -> None:
        kind, site = parse_component(self.component, layer_count=layer_count)
        if kind not in ("a", "m") or not self.tag:
            raise ValueError("circuit tag/component is malformed")
        member = torch.as_tensor(self.member_mask, dtype=torch.bool).reshape(-1)
        slice_mask = torch.as_tensor(self.slice_mask, dtype=torch.bool).reshape(-1)
        if member.numel() != grid_size or slice_mask.numel() != grid_size:
            raise ValueError("circuit masks do not cover the exact row grid")
        if not member.any() or not torch.all(slice_mask[member]):
            raise ValueError("circuit members must be a nonempty subset of its slice")
        if bool(slice_mask.all()):
            raise ValueError("circuit slice must leave off-slice positions")


def parse_component(component: str, *, layer_count: int) -> tuple[str, int]:
    if not isinstance(component, str) or len(component) < 2 or component[0] not in "am":
        raise ValueError("component must be a<site> or m<site>")
    try:
        site = int(component[1:])
    except ValueError as exc:
        raise ValueError("component site is not an integer") from exc
    if not 0 <= site < layer_count:
        raise ValueError("component site is outside the model")
    return component[0], site


@dataclass
class PhysicalCallLedger:
    outer_forwards: int = 0
    attention_native_calls: list[int] = field(default_factory=list)
    mlp_native_calls: list[int] = field(default_factory=list)
    projection_calls: dict[str, int] = field(default_factory=dict)
    capture_calls: dict[str, int] = field(default_factory=dict)
    projection_events: dict[str, int] = field(default_factory=dict)
    capture_events: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        attention_sites: Sequence[int],
        mlp_sites: Sequence[int],
        *,
        layer_count: int,
    ) -> None:
        expected = list(range(layer_count))
        if list(attention_sites) != expected or list(mlp_sites) != expected:
            raise RuntimeError("typed dispatch did not call every native component exactly once")
        self.outer_forwards += 1
        self.attention_native_calls.extend(attention_sites)
        self.mlp_native_calls.extend(mlp_sites)

    def payload(self) -> dict[str, object]:
        return {
            "outer_forwards": self.outer_forwards,
            "attention_native_calls": len(self.attention_native_calls),
            "mlp_native_calls": len(self.mlp_native_calls),
            "attention_calls_by_site": {
                str(site): self.attention_native_calls.count(site)
                for site in sorted(set(self.attention_native_calls))
            },
            "mlp_calls_by_site": {
                str(site): self.mlp_native_calls.count(site)
                for site in sorted(set(self.mlp_native_calls))
            },
            "projection_calls": dict(sorted(self.projection_calls.items())),
            "capture_calls": dict(sorted(self.capture_calls.items())),
        }

    def structured_payload(
        self,
        specs: Sequence[CircuitSpec],
        *,
        batches: int,
        include_capture: bool,
    ) -> dict[str, object]:
        """Encode exact event identities as compact dense integer tensors."""
        tags = [spec.tag for spec in specs]
        components = [spec.component for spec in specs]
        capture_components = list(dict.fromkeys(components)) if include_capture else []
        projection_counts = torch.empty(
            (len(PHASES), len(specs), batches), dtype=torch.int64
        )
        for phase_index, phase in enumerate(PHASES):
            for source_index, spec in enumerate(specs):
                for batch_index in range(batches):
                    projection_counts[phase_index, source_index, batch_index] = (
                        self.projection_events.get(
                            projection_event_key(
                                phase, spec.tag, spec.component, batch_index
                            ),
                            0,
                        )
                    )
        capture_counts = torch.empty(
            (len(capture_components), batches), dtype=torch.int64
        )
        for component_index, component in enumerate(capture_components):
            for batch_index in range(batches):
                capture_counts[component_index, batch_index] = self.capture_events.get(
                    capture_event_key(component, batch_index), 0
                )
        return {
            **self.payload(),
            "projection_phases": list(PHASES),
            "projection_source_tags": tags,
            "projection_source_components": components,
            "projection_batch_indices": list(range(batches)),
            "projection_event_counts": projection_counts,
            "capture_components": capture_components,
            "capture_batch_indices": list(range(batches)),
            "capture_event_counts": capture_counts,
        }

    def record_projection(
        self, *, phase: str, source_tag: str, component: str, batch_index: int
    ) -> None:
        if phase not in PHASES or not source_tag or batch_index < 0:
            raise RuntimeError("projection event identity is malformed")
        key = projection_event_key(phase, source_tag, component, batch_index)
        self.projection_events[key] = self.projection_events.get(key, 0) + 1
        self.projection_calls[component] = self.projection_calls.get(component, 0) + 1

    def record_capture(self, *, component: str, batch_index: int) -> None:
        if not component or batch_index < 0:
            raise RuntimeError("capture event identity is malformed")
        key = capture_event_key(component, batch_index)
        self.capture_events[key] = self.capture_events.get(key, 0) + 1
        self.capture_calls[component] = self.capture_calls.get(component, 0) + 1


def projection_event_key(
    phase: str, source_tag: str, component: str, batch_index: int
) -> str:
    return f"{phase}\t{source_tag}\t{component}\t{batch_index}"


def capture_event_key(component: str, batch_index: int) -> str:
    return f"{component}\t{batch_index}"


def tensor_sha256(value: torch.Tensor) -> str:
    if type(value) is not torch.Tensor or value.device.type != "cpu":
        raise TypeError("only owned CPU tensors may be hashed")
    tensor = value.contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonicalize_sign(vector: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(vector)
    pivot = int(vector.abs().argmax())
    return -vector if vector[pivot] < 0 else vector


def leading_shared_direction(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a float64 sign-fixed direction, rejecting an unstable top subspace."""
    if type(matrix) is not torch.Tensor or matrix.dtype != torch.float64 or (
        matrix.device.type != "cpu" or matrix.ndim != 2 or matrix.shape[0] < 2
    ):
        raise TypeError("shared-direction matrix must be a 2D CPU float64 tensor")
    _left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
    if not torch.isfinite(singular_values).all() or singular_values[0] <= 0:
        raise RuntimeError("shared-direction singular spectrum is invalid")
    relative_gap = (singular_values[0] - singular_values[1]) / singular_values[0]
    if relative_gap <= 1e-6:
        raise RuntimeError("shared-direction top singular value is tied")
    return canonicalize_sign(right[0]), singular_values


class ObservedResponseCollector:
    """One-run owner around a frozen model and an exact row/mask grid."""

    def __init__(
        self,
        model: torch.nn.Module,
        rows: torch.Tensor,
        row_document_ids: torch.Tensor,
        specs: Sequence[CircuitSpec],
        *,
        batch_size: int = 4,
        require_production: bool = True,
    ) -> None:
        if batch_size != 4:
            raise ValueError("production collector uses the facade's four-row batch")
        if type(rows) is not torch.Tensor or rows.dtype != torch.int64 or (
            rows.device.type != "cpu" or not rows.is_contiguous()
        ):
            raise TypeError("rows must be an exact contiguous CPU int64 tensor")
        if type(row_document_ids) is not torch.Tensor or (
            row_document_ids.dtype != torch.int64
            or row_document_ids.device.type != "cpu"
            or not row_document_ids.is_contiguous()
        ):
            raise TypeError(
                "row document IDs must be an exact contiguous CPU int64 tensor"
            )
        self.model = model
        self.rows = rows.clone()
        self.row_document_ids = row_document_ids.clone()
        owned_specs: list[CircuitSpec] = []
        for spec in specs:
            if type(spec) is not CircuitSpec or type(spec.member_mask) is not torch.Tensor or (
                type(spec.slice_mask) is not torch.Tensor
                or spec.member_mask.dtype != torch.bool
                or spec.slice_mask.dtype != torch.bool
                or spec.member_mask.device.type != "cpu"
                or spec.slice_mask.device.type != "cpu"
                or not spec.member_mask.is_contiguous()
                or not spec.slice_mask.is_contiguous()
            ):
                raise TypeError("circuit masks must be exact contiguous CPU bool tensors")
            owned_specs.append(CircuitSpec(
                tag=spec.tag,
                component=spec.component,
                member_mask=spec.member_mask.clone(),
                slice_mask=spec.slice_mask.clone(),
            ))
        self.specs = tuple(owned_specs)
        self.batch_size = batch_size
        self.require_production = require_production
        self.layer_count = len(model.transformer.h)
        self.positions = self.rows.shape[1] - 1
        self.width = model.config.n_embd
        self.device = next(model.parameters()).device
        self.ledger = PhysicalCallLedger()
        self._spent = False
        if self.rows.ndim != 2 or self.positions <= 0 or (
            self.row_document_ids.shape != (self.rows.shape[0],)
        ):
            raise ValueError("rows/document IDs are malformed")
        if require_production and self.positions != 256:
            raise ValueError("production rows must contain 256 predictions")
        if require_production and self.rows.shape[0] != 1_000:
            raise ValueError("production census must contain exactly 1,000 rows")
        if not self.specs or len({spec.tag for spec in self.specs}) != len(self.specs):
            raise ValueError("circuit specs must be nonempty with unique tags")
        grid_size = self.rows.shape[0] * self.positions
        for spec in self.specs:
            spec.validate(grid_size=grid_size, layer_count=self.layer_count)
        if require_production:
            self._require_production_spec_order()
        self._require_hook_free()

    def _require_production_spec_order(self) -> None:
        if len(self.specs) != 49:
            raise ValueError("production circuit inventory must contain exactly 49 specs")
        component_rank = {
            component: index for index, component in enumerate(PRODUCTION_COMPONENT_ORDER)
        }
        if any(spec.component not in component_rank for spec in self.specs):
            raise ValueError("production circuit inventory contains an unknown component")
        ordered = sorted(
            self.specs, key=lambda spec: (component_rank[spec.component], spec.tag)
        )
        if list(self.specs) != ordered:
            raise ValueError("production circuit specs are not in the frozen order")
        serialized = "".join(
            f"{spec.component}\t{spec.tag}\n" for spec in self.specs
        ).encode()
        if hashlib.sha256(serialized).hexdigest() != PRODUCTION_SPEC_ORDER_SHA256:
            raise ValueError("production circuit ordering hash changed")

    def _require_hook_free(self) -> None:
        global_hook_names = (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
        module_state = torch.nn.modules.module
        if any(getattr(module_state, name, {}) for name in global_hook_names):
            raise RuntimeError("process contains a global module hook")
        for module in self.model.modules():
            if module._forward_hooks or module._forward_pre_hooks or (
                module._backward_hooks
            ):
                raise RuntimeError("model contains a preexisting module hook")

    def _validate_role(self, selected_rows: torch.Tensor, *, label: str) -> torch.Tensor:
        if type(selected_rows) is not torch.Tensor or selected_rows.dtype != torch.int64 or (
            selected_rows.device.type != "cpu" or not selected_rows.is_contiguous()
        ):
            raise TypeError(f"{label} rows must be an exact contiguous CPU int64 tensor")
        if selected_rows.ndim != 1 or selected_rows.numel() == 0:
            raise ValueError(f"{label} row role is empty or malformed")
        if selected_rows.min() < 0 or selected_rows.max() >= self.rows.shape[0]:
            raise ValueError(f"{label} row role contains an out-of-range index")
        if torch.unique(selected_rows).numel() != selected_rows.numel():
            raise ValueError(f"{label} row role contains a duplicate index")
        return selected_rows.clone()

    def _batches(self, selected_rows: torch.Tensor):
        for start in range(0, selected_rows.numel(), self.batch_size):
            indices = selected_rows[start : start + self.batch_size]
            real = int(indices.numel())
            if real < self.batch_size:
                indices = torch.cat((indices, indices[:1].expand(self.batch_size - real)))
            batch = self.rows[indices]
            yield start, real, batch[:, :-1].to(self.device), batch[:, 1:].to(self.device)

    @torch.no_grad()
    def _forward(
        self,
        tokens: torch.Tensor,
        *,
        source_component: str | None = None,
        direction: torch.Tensor | None = None,
        capture: Callable[[str, torch.Tensor], None] | None = None,
        intervention_event: tuple[str, str, int] | None = None,
    ) -> torch.Tensor:
        self._require_hook_free()
        attention_sites: list[int] = []
        mlp_sites: list[int] = []
        source = None if source_component is None else parse_component(
            source_component, layer_count=self.layer_count
        )
        direction_device = None if direction is None else torch.as_tensor(
            direction, dtype=torch.float32, device=self.device
        )
        if (source is None) != (direction_device is None):
            raise ValueError("source component and direction must be supplied together")
        if (source is None) != (intervention_event is None):
            raise ValueError("projection event identity must accompany every intervention")
        if direction_device is not None and (
            direction_device.shape != (self.width,) or not torch.isfinite(direction_device).all()
        ):
            raise ValueError("projection direction is malformed")

        def project(write: torch.Tensor) -> torch.Tensor:
            flat = write.float().reshape(-1, self.width)
            replaced = flat - (flat @ direction_device).unsqueeze(1) * direction_device
            return replaced.reshape_as(write).to(write.dtype)

        def attention(event: facade.AttentionEvent):
            attention_sites.append(event.site)
            write, next_value = event.block.attn(event.state, event.first_value)
            key = f"a{event.site}"
            if capture is not None:
                capture(key, write)
            if source == ("a", event.site):
                write = project(write)
                phase, source_tag, batch_index = intervention_event
                self.ledger.record_projection(
                    phase=phase,
                    source_tag=source_tag,
                    component=key,
                    batch_index=batch_index,
                )
            return write, next_value

        def mlp(event: facade.EarlyMLPEvent):
            mlp_sites.append(event.site)
            write = event.block.mlp(event.state)
            key = f"m{event.site}"
            if capture is not None:
                capture(key, write)
            if source == ("m", event.site):
                write = project(write)
                phase, source_tag, batch_index = intervention_event
                self.ledger.record_projection(
                    phase=phase,
                    source_tag=source_tag,
                    component=key,
                    batch_index=batch_index,
                )
            return write

        logits = facade.forward_with_dispatch(
            self.model,
            tokens,
            attention,
            mlp,
            require_production=self.require_production,
        )
        self.ledger.record(
            attention_sites, mlp_sites, layer_count=self.layer_count
        )
        self._require_hook_free()
        return logits

    @torch.no_grad()
    def _fit_directions(self, fit_rows: torch.Tensor) -> dict[str, object]:
        local_member = {
            spec.tag: local_mask_from_global(
                spec.member_mask, fit_rows, positions_per_row=self.positions
            ).reshape(-1, self.positions)
            for spec in self.specs
        }
        local_off = {
            spec.tag: local_mask_from_global(
                ~torch.as_tensor(spec.slice_mask, dtype=torch.bool),
                fit_rows,
                positions_per_row=self.positions,
            ).reshape(-1, self.positions)
            for spec in self.specs
        }
        specs_by_component: dict[str, list[CircuitSpec]] = {}
        for spec in self.specs:
            specs_by_component.setdefault(spec.component, []).append(spec)
        sums = {
            spec.tag: {
                "member": torch.zeros(self.width, dtype=torch.float64),
                "off": torch.zeros(self.width, dtype=torch.float64),
                "member_count": 0,
                "off_count": 0,
            }
            for spec in self.specs
        }

        for start, real, tokens, _targets in self._batches(fit_rows):
            batch_index = start // self.batch_size
            def capture(component: str, write: torch.Tensor) -> None:
                if component not in specs_by_component:
                    return
                values = write[:real].detach().float().cpu().reshape(-1, self.width)
                for spec in specs_by_component[component]:
                    member = local_member[spec.tag][start : start + real].reshape(-1)
                    off = local_off[spec.tag][start : start + real].reshape(-1)
                    sums[spec.tag]["member"] += values[member].double().sum(0)
                    sums[spec.tag]["off"] += values[off].double().sum(0)
                    sums[spec.tag]["member_count"] += int(member.sum())
                    sums[spec.tag]["off_count"] += int(off.sum())
                self.ledger.record_capture(
                    component=component, batch_index=batch_index
                )

            self._forward(tokens, capture=capture)

        full_master: dict[str, torch.Tensor] = {}
        full: dict[str, torch.Tensor] = {}
        full_norms: dict[str, float] = {}
        fit_counts: dict[str, dict[str, int]] = {}
        fit_write_statistics: dict[str, dict[str, torch.Tensor]] = {}
        for spec in self.specs:
            item = sums[spec.tag]
            if item["member_count"] <= 0 or item["off_count"] <= 0:
                raise RuntimeError("FIT role lacks member/off support")
            vector = item["member"] / item["member_count"] - item["off"] / item["off_count"]
            norm = vector.norm()
            if not torch.isfinite(vector).all() or norm <= 1e-12:
                raise RuntimeError("FIT direction is zero or nonfinite")
            full_master[spec.tag] = vector / norm
            full[spec.tag] = full_master[spec.tag].float()
            full_norms[spec.tag] = float(norm)
            fit_counts[spec.tag] = {
                "member_count": item["member_count"],
                "off_count": item["off_count"],
            }
            fit_write_statistics[spec.tag] = {
                "member_sum": item["member"].clone(),
                "off_sum": item["off"].clone(),
                "member_mean": (item["member"] / item["member_count"]).clone(),
                "off_mean": (item["off"] / item["off_count"]).clone(),
            }

        shared: dict[str, torch.Tensor] = {}
        residual: dict[str, torch.Tensor] = {}
        singular_spectra: dict[str, torch.Tensor] = {}
        relative_singular_gaps: dict[str, float] = {}
        residual_norms: dict[str, float] = {}
        for component, component_specs in specs_by_component.items():
            matrix = torch.stack(
                [full_master[spec.tag] for spec in component_specs]
            )
            shared_master, singular_values = leading_shared_direction(matrix)
            shared[component] = shared_master.float()
            singular_spectra[component] = singular_values.clone()
            relative_singular_gaps[component] = float(
                (singular_values[0] - singular_values[1]) / singular_values[0]
            )
            for spec in component_specs:
                remainder = full_master[spec.tag] - (
                    full_master[spec.tag] @ shared_master
                ) * shared_master
                relative_norm = float(remainder.norm())
                if not math.isfinite(relative_norm) or relative_norm <= 1e-6:
                    raise RuntimeError("residual direction is numerically absent")
                residual_norms[spec.tag] = relative_norm
                residual[spec.tag] = (remainder / remainder.norm()).float()
        return {
            "full": full,
            "residual": residual,
            "shared": shared,
            "fit_counts": fit_counts,
            "fit_write_statistics": fit_write_statistics,
            "full_direction_norms": full_norms,
            "singular_spectra": singular_spectra,
            "relative_singular_gaps": relative_singular_gaps,
            "residual_norms": residual_norms,
        }

    @torch.no_grad()
    def _ce_vector(
        self,
        selected_rows: torch.Tensor,
        *,
        source_component: str | None = None,
        direction: torch.Tensor | None = None,
        phase: str | None = None,
        source_tag: str | None = None,
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        if (source_component is None) != (phase is None or source_tag is None):
            raise ValueError("intervention CE requires phase and source-tag identity")
        for start, real, tokens, targets in self._batches(selected_rows):
            event = None
            if source_component is not None:
                event = (phase, source_tag, start // self.batch_size)
            logits = self._forward(
                tokens,
                source_component=source_component,
                direction=direction,
                intervention_event=event,
            )
            ce = F.cross_entropy(
                logits[:real].reshape(-1, logits.shape[-1]),
                targets[:real].reshape(-1),
                reduction="none",
            )
            values.append(ce.detach().double().cpu())
            del logits, ce
        return torch.cat(values)

    def _claim_once(self) -> None:
        if self._spent:
            raise RuntimeError("response collector is already spent")
        self._spent = True

    def _validate_directions(
        self, directions: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        ordered_tags = [spec.tag for spec in self.specs]
        if set(directions) != set(PHASES):
            raise ValueError("direction phases do not match the frozen phases")
        owned: dict[str, dict[str, torch.Tensor]] = {}
        for phase in PHASES:
            phase_directions = directions[phase]
            if list(phase_directions) != ordered_tags:
                raise ValueError("direction tags do not match the frozen order")
            owned[phase] = {}
            for tag in ordered_tags:
                value = phase_directions[tag]
                if type(value) is not torch.Tensor or value.dtype != torch.float32 or (
                    value.device.type != "cpu"
                    or not value.is_contiguous()
                    or value.shape != (self.width,)
                    or not torch.isfinite(value).all()
                ):
                    raise TypeError("directions must be owned CPU float32 vectors")
                norm = float(value.double().norm())
                if not math.isfinite(norm) or abs(norm - 1.0) > 1e-5:
                    raise ValueError("direction is not unit normalized")
                owned[phase][tag] = value.clone()
        return owned

    @torch.no_grad()
    def _collect_response_role(
        self,
        selected_rows: torch.Tensor,
        directions: Mapping[str, Mapping[str, torch.Tensor]],
        *,
        role: str,
    ) -> dict[str, object]:
        document_ids, position_documents = document_position_index(
            self.row_document_ids,
            selected_rows,
            positions_per_row=self.positions,
        )
        member_masks = {
            spec.tag: local_mask_from_global(
                spec.member_mask, selected_rows, positions_per_row=self.positions
            )
            for spec in self.specs
        }
        off_masks = {
            spec.tag: local_mask_from_global(
                ~torch.as_tensor(spec.slice_mask, dtype=torch.bool),
                selected_rows,
                positions_per_row=self.positions,
            )
            for spec in self.specs
        }
        zeros = torch.zeros(selected_rows.numel() * self.positions, dtype=torch.float64)
        static = aggregate_document_responses(
            zeros,
            position_documents,
            member_masks,
            off_masks,
            document_count=document_ids.numel(),
        )
        source_count = target_count = len(self.specs)
        shape = (len(PHASES), source_count, target_count, document_ids.numel())
        statistics = {
            name: torch.zeros(shape, dtype=torch.float64) for name in STATISTIC_NAMES
        }
        baseline = self._ce_vector(selected_rows)
        for phase_index, phase in enumerate(PHASES):
            phase_directions = directions[phase]
            for source_index, source in enumerate(self.specs):
                intervened = self._ce_vector(
                    selected_rows,
                    source_component=source.component,
                    direction=phase_directions[source.tag],
                    phase=phase,
                    source_tag=source.tag,
                )
                dce = intervened - baseline
                aggregate = aggregate_document_responses(
                    dce,
                    position_documents,
                    member_masks,
                    off_masks,
                    document_count=document_ids.numel(),
                )
                for name in STATISTIC_NAMES:
                    statistics[name][phase_index, source_index] = aggregate[name]

        validation = validate_response_tensors(
            statistics,
            static["member_count"],
            static["off_count"],
            expected_prefix=(len(PHASES), source_count, target_count),
            tolerance=1e-8,
        )
        return {
            "schema": "causal_response_tensor_v1_role_preimage",
            "role": role,
            "row_indices": selected_rows.clone(),
            "document_ids": document_ids,
            "member_count": static["member_count"],
            "off_count": static["off_count"],
            "statistics": statistics,
            "baseline_ce_mean": float(baseline.mean()),
            "validation": validation,
        }

    def _require_ledger(
        self,
        *,
        expected_outer: int,
        batches: int,
        expect_capture: bool,
    ) -> None:
        if self.ledger.outer_forwards != expected_outer:
            raise RuntimeError("outer forward ledger does not close")
        for site in range(self.layer_count):
            if self.ledger.attention_native_calls.count(site) != expected_outer or (
                self.ledger.mlp_native_calls.count(site) != expected_outer
            ):
                raise RuntimeError("per-site native call ledger does not close")
        expected_projection_events = {
            projection_event_key(phase, spec.tag, spec.component, batch_index)
            for phase in PHASES
            for spec in self.specs
            for batch_index in range(batches)
        }
        if set(self.ledger.projection_events) != expected_projection_events or any(
            count != 1 for count in self.ledger.projection_events.values()
        ):
            raise RuntimeError("structured projection event ledger does not close")
        components = {spec.component for spec in self.specs}
        expected_capture_events = (
            {
                capture_event_key(component, batch_index)
                for component in components
                for batch_index in range(batches)
            }
            if expect_capture else set()
        )
        if set(self.ledger.capture_events) != expected_capture_events or any(
            count != 1 for count in self.ledger.capture_events.values()
        ):
            raise RuntimeError("structured capture event ledger does not close")

    @torch.no_grad()
    def fit_stage(self, fit_rows: torch.Tensor) -> dict[str, object]:
        """Return an unpublished FIT preimage for the lifecycle owner.

        This is not a serializable program capability.  The lifecycle must publish and
        semantically reload it before a fresh EVAL process can mint a program.
        """
        self._claim_once()
        fit_rows = self._validate_role(fit_rows, label="FIT")
        if self.require_production and (
            fit_rows.numel() != 496
            or torch.unique(self.row_document_ids[fit_rows]).numel() != 343
        ):
            raise ValueError("production FIT role must be 496 rows from 343 documents")
        directions = self._fit_directions(fit_rows)
        ordered_tags = [spec.tag for spec in self.specs]
        direction_preimage = {
            phase: {
                tag: directions[phase][tag].clone() for tag in ordered_tags
            }
            for phase in PHASES
        }
        response = self._collect_response_role(
            fit_rows, direction_preimage, role="FIT"
        )
        batches = math.ceil(fit_rows.numel() / self.batch_size)
        self._require_ledger(
            expected_outer=batches * (2 + len(PHASES) * len(self.specs)),
            batches=batches,
            expect_capture=True,
        )
        return {
            "schema": "causal_response_tensor_v1_fit_preimage",
            "claim_boundary": (
                "Unpublished internal FIT preimage. It is not an EVAL capability; "
                "a create-only lifecycle must serialize and semantically reload it."
            ),
            "sign_convention": "dCE = rank-one-projection intervention CE - native CE",
            "off_mask": "exact complement of the target circuit's frozen slice mask",
            "phases": list(PHASES),
            "source_tags": ordered_tags,
            "source_components": [spec.component for spec in self.specs],
            "target_tags": ordered_tags,
            "model_layer_count": self.layer_count,
            "model_width": self.width,
            "batch_size": self.batch_size,
            "spec_order_sha256": hashlib.sha256("".join(
                f"{spec.component}\t{spec.tag}\n" for spec in self.specs
            ).encode()).hexdigest(),
            "support_hashes": {
                spec.tag: {
                    "member_mask_sha256": tensor_sha256(spec.member_mask),
                    "slice_mask_sha256": tensor_sha256(spec.slice_mask),
                }
                for spec in self.specs
            },
            "_direction_preimage": direction_preimage,
            "directions": torch.stack([
                torch.stack([direction_preimage[phase][tag] for tag in ordered_tags])
                for phase in PHASES
            ]),
            "shared_directions": {
                component: value.clone()
                for component, value in directions["shared"].items()
            },
            "fit_counts": directions["fit_counts"],
            "fit_write_statistics": directions["fit_write_statistics"],
            "full_direction_norms": directions["full_direction_norms"],
            "singular_spectra": directions["singular_spectra"],
            "relative_singular_gaps": directions["relative_singular_gaps"],
            "residual_norms": directions["residual_norms"],
            "fit_response": response,
            "call_ledger": self.ledger.structured_payload(
                self.specs, batches=batches, include_capture=True
            ),
        }

    @torch.no_grad()
    def _evaluate_stage_preimage(
        self,
        eval_rows: torch.Tensor,
        *,
        direction_preimage: Mapping[str, Mapping[str, torch.Tensor]],
        fit_document_ids: torch.Tensor,
    ) -> dict[str, object]:
        """Internal EVAL computation surface; only a semantic loader may call this."""
        self._claim_once()
        eval_rows = self._validate_role(eval_rows, label="EVAL")
        if type(fit_document_ids) is not torch.Tensor or (
            fit_document_ids.dtype != torch.int64
            or fit_document_ids.device.type != "cpu"
            or fit_document_ids.ndim != 1
            or not fit_document_ids.is_contiguous()
            or torch.unique(fit_document_ids).numel() != fit_document_ids.numel()
        ):
            raise TypeError("FIT document IDs are not a sealed CPU int64 vector")
        eval_documents = set(self.row_document_ids[eval_rows].tolist())
        if self.require_production and (
            eval_rows.numel() != 504
            or len(eval_documents) != 345
            or fit_document_ids.numel() != 343
        ):
            raise ValueError("production EVAL role must be 504 rows from 345 documents")
        if set(fit_document_ids.tolist()) & eval_documents:
            raise ValueError("FIT/EVAL source documents overlap")
        directions = self._validate_directions(direction_preimage)
        response = self._collect_response_role(
            eval_rows, directions, role="EVAL"
        )
        batches = math.ceil(eval_rows.numel() / self.batch_size)
        self._require_ledger(
            expected_outer=batches * (1 + len(PHASES) * len(self.specs)),
            batches=batches,
            expect_capture=False,
        )
        return {
            "schema": "causal_response_tensor_v1_eval_preimage",
            "claim_boundary": (
                "Internal EVAL preimage; publication remains lifecycle-owned."
            ),
            "source_tags": [spec.tag for spec in self.specs],
            "source_components": [spec.component for spec in self.specs],
            "eval_response": response,
            "call_ledger": self.ledger.structured_payload(
                self.specs, batches=batches, include_capture=False
            ),
        }

    def collect(self, *_args, **_kwargs):
        raise RuntimeError(
            "combined FIT/EVAL collection is retired; use the sealed two-stage lifecycle"
        )
