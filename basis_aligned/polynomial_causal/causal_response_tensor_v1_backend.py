"""Source-closed observed-model backend for causal-response tensor collection.

No historical census helper is imported.  Every model call goes through the typed
``bilin18_observed_model_facade`` dispatch surface; component capture and rank-one
projection are synchronous, hook-free, and accounted in an exact physical call
ledger.  Inputs and masks are supplied by an authority-bound outer transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Callable, Sequence

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


def canonicalize_sign(vector: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(vector)
    pivot = int(vector.abs().argmax())
    return -vector if vector[pivot] < 0 else vector


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
        self.model = model
        self.rows = torch.as_tensor(rows, dtype=torch.int64).cpu()
        self.row_document_ids = torch.as_tensor(row_document_ids, dtype=torch.int64).cpu()
        self.specs = tuple(specs)
        self.batch_size = batch_size
        self.require_production = require_production
        self.layer_count = len(model.transformer.h)
        self.positions = self.rows.shape[1] - 1
        self.width = model.config.n_embd
        self.device = next(model.parameters()).device
        self.ledger = PhysicalCallLedger()
        if self.rows.ndim != 2 or self.positions <= 0 or (
            self.row_document_ids.shape != (self.rows.shape[0],)
        ):
            raise ValueError("rows/document IDs are malformed")
        if require_production and self.positions != 256:
            raise ValueError("production rows must contain 256 predictions")
        if not self.specs or len({spec.tag for spec in self.specs}) != len(self.specs):
            raise ValueError("circuit specs must be nonempty with unique tags")
        grid_size = self.rows.shape[0] * self.positions
        for spec in self.specs:
            spec.validate(grid_size=grid_size, layer_count=self.layer_count)

    def _batches(self, selected_rows: torch.Tensor):
        selected_rows = torch.as_tensor(selected_rows, dtype=torch.int64).cpu()
        if selected_rows.ndim != 1 or selected_rows.numel() == 0:
            raise ValueError("selected row role is empty")
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
    ) -> torch.Tensor:
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
                self.ledger.projection_calls[key] = self.ledger.projection_calls.get(key, 0) + 1
            return write, next_value

        def mlp(event: facade.EarlyMLPEvent):
            mlp_sites.append(event.site)
            write = event.block.mlp(event.state)
            key = f"m{event.site}"
            if capture is not None:
                capture(key, write)
            if source == ("m", event.site):
                write = project(write)
                self.ledger.projection_calls[key] = self.ledger.projection_calls.get(key, 0) + 1
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
        return logits

    @torch.no_grad()
    def fit_directions(self, fit_rows: torch.Tensor) -> dict[str, object]:
        fit_rows = torch.as_tensor(fit_rows, dtype=torch.int64).cpu()
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
                self.ledger.capture_calls[component] = (
                    self.ledger.capture_calls.get(component, 0) + 1
                )

            self._forward(tokens, capture=capture)

        full: dict[str, torch.Tensor] = {}
        fit_counts: dict[str, dict[str, int]] = {}
        for spec in self.specs:
            item = sums[spec.tag]
            if item["member_count"] <= 0 or item["off_count"] <= 0:
                raise RuntimeError("FIT role lacks member/off support")
            vector = item["member"] / item["member_count"] - item["off"] / item["off_count"]
            norm = vector.norm()
            if not torch.isfinite(vector).all() or norm <= 1e-12:
                raise RuntimeError("FIT direction is zero or nonfinite")
            full[spec.tag] = (vector / norm).float()
            fit_counts[spec.tag] = {
                "member_count": item["member_count"],
                "off_count": item["off_count"],
            }

        shared: dict[str, torch.Tensor] = {}
        residual: dict[str, torch.Tensor] = {}
        for component, component_specs in specs_by_component.items():
            matrix = torch.stack([full[spec.tag] for spec in component_specs]).double()
            shared_direction = canonicalize_sign(
                torch.linalg.svd(matrix, full_matrices=False).Vh[0]
            ).float()
            shared[component] = shared_direction
            for spec in component_specs:
                remainder = full[spec.tag] - (
                    full[spec.tag] @ shared_direction
                ) * shared_direction
                relative_norm = float(remainder.norm())
                if not math.isfinite(relative_norm) or relative_norm <= 1e-6:
                    raise RuntimeError("residual direction is numerically absent")
                residual[spec.tag] = (remainder / remainder.norm()).float()
        return {
            "full": full,
            "residual": residual,
            "shared": shared,
            "fit_counts": fit_counts,
        }

    @torch.no_grad()
    def _ce_vector(
        self,
        selected_rows: torch.Tensor,
        *,
        source_component: str | None = None,
        direction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for _start, real, tokens, targets in self._batches(selected_rows):
            logits = self._forward(
                tokens, source_component=source_component, direction=direction
            )
            ce = F.cross_entropy(
                logits[:real].reshape(-1, logits.shape[-1]),
                targets[:real].reshape(-1),
                reduction="none",
            )
            values.append(ce.detach().double().cpu())
            del logits, ce
        return torch.cat(values)

    @torch.no_grad()
    def collect(
        self,
        fit_rows: torch.Tensor,
        eval_rows: torch.Tensor,
    ) -> dict[str, object]:
        fit_rows = torch.as_tensor(fit_rows, dtype=torch.int64).cpu()
        eval_rows = torch.as_tensor(eval_rows, dtype=torch.int64).cpu()
        if set(fit_rows.tolist()) & set(eval_rows.tolist()):
            raise ValueError("FIT/EVAL row roles overlap")
        fit_documents = set(self.row_document_ids[fit_rows].tolist())
        eval_documents_set = set(self.row_document_ids[eval_rows].tolist())
        if fit_documents & eval_documents_set:
            raise ValueError("FIT/EVAL source documents overlap")

        directions = self.fit_directions(fit_rows)
        document_ids, position_documents = document_position_index(
            self.row_document_ids,
            eval_rows,
            positions_per_row=self.positions,
        )
        member_masks = {
            spec.tag: local_mask_from_global(
                spec.member_mask, eval_rows, positions_per_row=self.positions
            )
            for spec in self.specs
        }
        off_masks = {
            spec.tag: local_mask_from_global(
                ~torch.as_tensor(spec.slice_mask, dtype=torch.bool),
                eval_rows,
                positions_per_row=self.positions,
            )
            for spec in self.specs
        }
        zeros = torch.zeros(eval_rows.numel() * self.positions, dtype=torch.float64)
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
        baseline = self._ce_vector(eval_rows)
        for phase_index, phase in enumerate(PHASES):
            phase_directions = directions[phase]
            for source_index, source in enumerate(self.specs):
                intervened = self._ce_vector(
                    eval_rows,
                    source_component=source.component,
                    direction=phase_directions[source.tag],
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
        expected_fit_batches = math.ceil(fit_rows.numel() / self.batch_size)
        expected_eval_batches = math.ceil(eval_rows.numel() / self.batch_size)
        expected_outer = expected_fit_batches + expected_eval_batches * (
            1 + len(PHASES) * source_count
        )
        if self.ledger.outer_forwards != expected_outer:
            raise RuntimeError("outer forward ledger does not close")
        for site in range(self.layer_count):
            if self.ledger.attention_native_calls.count(site) != expected_outer or (
                self.ledger.mlp_native_calls.count(site) != expected_outer
            ):
                raise RuntimeError("per-site native call ledger does not close")

        ordered_tags = [spec.tag for spec in self.specs]
        return {
            "schema": "causal_response_tensor_v1_payload",
            "sign_convention": "dCE = rank-one-projection intervention CE - native CE",
            "off_mask": "exact complement of the target circuit's frozen slice mask",
            "phases": list(PHASES),
            "source_tags": ordered_tags,
            "source_components": [spec.component for spec in self.specs],
            "target_tags": ordered_tags,
            "fit_row_indices": fit_rows,
            "eval_row_indices": eval_rows,
            "eval_document_ids": document_ids,
            "directions": {
                phase: torch.stack([directions[phase][tag] for tag in ordered_tags])
                for phase in PHASES
            },
            "shared_directions": {
                component: value for component, value in directions["shared"].items()
            },
            "fit_counts": directions["fit_counts"],
            "member_count": static["member_count"],
            "off_count": static["off_count"],
            "statistics": statistics,
            "baseline_eval_ce_mean": float(baseline.mean()),
            "validation": validation,
            "call_ledger": self.ledger.payload(),
        }
