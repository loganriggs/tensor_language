"""Thin bilin18 execution backend for the frozen early-MLP/context cross.

The expensive, immutable section-1786 program is constructed by the already tested
parent backend.  This module owns new mask descriptors and call ledgers because the
old cut-rank descriptors always compiled attention0/MLP0 and are scientifically
incompatible with this registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import compilation_mask_cut_rank_v1_bilin18_backend as parent
import compilation_mask_cut_rank_v1_gpu_adapter as adapter
import early_mlp_context_cross_v1 as cross
import early_mlp_context_cross_v1_measurements as measurement
import early_mlp_context_cross_v1_statistics as statistics


SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/early_mlp_context_cross_v1_bilin18_backend.py",
    *parent.SOURCE_PATHS,
)
EXECUTION_MODE = "native_module_executes_then_exact_output_substitution"
GAIN_POLICY = "identity_gains_no_mask_specific_refitting"


def _logical_sha256(value: Any) -> str:
    return measurement._logical_sha256(value)


def _canonical_sites(sites: Sequence[cross.Site]) -> tuple[cross.Site, ...]:
    selected = set(sites)
    if len(selected) != len(sites) or any(site not in adapter.ALL_NATIVE_SITES for site in sites):
        raise ValueError("compiled sites are duplicated or outside the model")
    return tuple(site for site in adapter.ALL_NATIVE_SITES if site in selected)


@dataclass(frozen=True, slots=True)
class ProgramDescriptor:
    ordinal: int
    request_sha256: str
    installed_compiled_sites: tuple[cross.Site, ...]
    shared_program_sha256: str
    execution_mode: str = EXECUTION_MODE
    gain_policy: str = GAIN_POLICY

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 64 or self.request_sha256 != (
            measurement.REQUESTS[self.ordinal].sha256
        ) or self.installed_compiled_sites != _canonical_sites(
            measurement.REQUESTS[self.ordinal].sites
        ) or len(self.shared_program_sha256) != 64 or self.execution_mode != (
            EXECUTION_MODE
        ) or self.gain_policy != GAIN_POLICY:
            raise ValueError("cross program descriptor changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class PreparedBank:
    model: adapter.ModelBinding
    programs: tuple[ProgramDescriptor, ...]
    shared_program_sha256: str
    evaluation_role_row_sha256s: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, adapter.ModelBinding) or len(self.programs) != 64 or tuple(
            program.ordinal for program in self.programs
        ) != tuple(range(64)) or any(
            program.shared_program_sha256 != self.shared_program_sha256
            for program in self.programs
        ) or tuple(role for role, _ in self.evaluation_role_row_sha256s) != (
            statistics.ROLE_NAMES
        ) or any(len(value) != 64 for _, value in self.evaluation_role_row_sha256s):
            raise ValueError("prepared cross bank changed")

    @property
    def sha256(self) -> str:
        return _logical_sha256({
            "model": asdict(self.model),
            "shared_program_sha256": self.shared_program_sha256,
            "program_sha256s": [program.sha256 for program in self.programs],
            "evaluation_role_row_sha256s": self.evaluation_role_row_sha256s,
        })


@dataclass(frozen=True, slots=True)
class CellCallLedger:
    ordinal: int
    request_sha256: str
    program_sha256: str
    row_count: int
    scored_token_count: int
    batch_count: int
    outer_forward_count: int
    outer_returned_count: int
    native_module_calls: tuple[tuple[cross.Site, int], ...]
    substitution_calls: tuple[tuple[cross.Site, int], ...]
    hook_order: str = "native_count_registered_before_substitution"
    execution_mode: str = EXECUTION_MODE
    fitter_calls: int = 0
    retained_logits: int = 0

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 64 or self.request_sha256 != (
            measurement.REQUESTS[self.ordinal].sha256
        ) or len(self.program_sha256) != 64 or any(type(value) is not int for value in (
            self.row_count, self.scored_token_count, self.batch_count,
            self.outer_forward_count, self.outer_returned_count,
            self.fitter_calls, self.retained_logits,
        )) or self.row_count <= 0 or self.scored_token_count != (
            self.row_count * measurement.SCORED_TOKENS_PER_ROW
        ) or self.batch_count <= 0 or self.outer_forward_count != self.batch_count or (
            self.outer_returned_count != self.batch_count
        ) or self.hook_order != "native_count_registered_before_substitution" or (
            self.execution_mode != EXECUTION_MODE
        ) or self.fitter_calls != 0 or self.retained_logits != 0:
            raise ValueError("cell call ledger changed")

    def validate(self, descriptor: ProgramDescriptor) -> None:
        expected_native = tuple(
            (site, self.batch_count) for site in adapter.ALL_NATIVE_SITES
        )
        expected_substitution = tuple(
            (site, self.batch_count) for site in descriptor.installed_compiled_sites
        )
        if descriptor.ordinal != self.ordinal or descriptor.sha256 != self.program_sha256 or (
            self.native_module_calls != expected_native
        ) or self.substitution_calls != expected_substitution:
            raise RuntimeError("cell physical call census differs from descriptor")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class BackendCellResult:
    statistics: measurement.RowCellStatistics
    call_ledger: CellCallLedger
    component_tree_before_sha256: str
    component_tree_after_sha256: str
    shared_program_before_sha256: str
    shared_program_after_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.statistics, measurement.RowCellStatistics) or not isinstance(
            self.call_ledger, CellCallLedger
        ) or any(len(value) != 64 for value in (
            self.component_tree_before_sha256, self.component_tree_after_sha256,
            self.shared_program_before_sha256, self.shared_program_after_sha256,
        )):
            raise ValueError("backend cell result changed")


class Bilin18ContextCrossBackend(parent.Bilin18CutRankBackend):
    """Lazy stateful production backend for the new exact mask registry."""

    source_paths = SOURCE_PATHS

    def __init__(self, *args, expected_shared_program_sha256: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cross_bank: PreparedBank | None = None
        if expected_shared_program_sha256 is None and self.dimensions == (
            parent.PRODUCTION_DIMENSIONS
        ):
            expected_shared_program_sha256 = measurement.SHARED_PROGRAM_SHA256
        if expected_shared_program_sha256 is not None and len(
            expected_shared_program_sha256
        ) != 64:
            raise ValueError("expected shared program hash is malformed")
        self.expected_shared_program_sha256 = expected_shared_program_sha256

    def prepare(
        self, role_rows: Mapping[str, torch.Tensor],
        requests: tuple[measurement.MeasurementRequest, ...],
    ) -> PreparedBank:
        if self._closed or self._model is not None or self._program is not None or (
            self._cross_bank is not None
        ) or tuple(requests) != measurement.REQUESTS or not isinstance(
            role_rows, Mapping
        ) or tuple(role_rows) != statistics.ROLE_NAMES:
            raise RuntimeError("context-cross backend prepare is non-pristine or malformed")
        checked_role_rows: list[tuple[str, str]] = []
        for role in statistics.ROLE_NAMES:
            rows = role_rows[role]
            if not torch.is_tensor(rows) or rows.dtype != torch.long or (
                rows.device.type != "cpu"
            ) or rows.ndim != 2 or rows.shape[1] != adapter.TARGET_STOP or not (
                rows.is_contiguous()
            ) or len(rows) == 0:
                raise RuntimeError("context-cross role rows are malformed")
            checked_role_rows.append((role, statistics.tensor_sha256(rows)))
        loader = self._model_loader or self._production_model_loader
        model, binding = loader()
        if not isinstance(model, torch.nn.Module) or not isinstance(
            binding, adapter.ModelBinding
        ):
            raise RuntimeError("model loader returned an untyped realization")
        self._model = model
        self._model_binding = binding
        self._model_guard = parent.ModelTreeGuard(
            model, initial_sha256=binding.component_tree_sha256,
        )
        fit_wave = self._load_fit_wave()
        if not isinstance(fit_wave, adapter.RowWave):
            raise RuntimeError("fit loader returned an untyped row wave")
        parent_source = adapter.file_sha256(Path(parent.__file__).resolve())
        if self.dimensions == parent.PRODUCTION_DIMENSIONS and parent_source != (
            "738f4988fe5b87a7329f833bc7117cc417adcb9834da06533b00bc8b320c18e0"
        ):
            raise RuntimeError("reusable parent program builder source changed")
        builder = self._program_builder or self._build_program
        program = builder(fit_wave, parent_source)
        if not isinstance(program, parent.SharedProgram) or program.dimensions != (
            self.dimensions
        ) or program.manifest["gain_policy"] != GAIN_POLICY or program.manifest[
            "model_realization_sha256"
        ] != binding.model_realization_sha256:
            raise RuntimeError("program builder returned a mismatched realization")
        if self.expected_shared_program_sha256 is not None and program.sha256 != (
            self.expected_shared_program_sha256
        ):
            raise RuntimeError("reconstructed shared program differs from amendment")
        self._program = program
        programs = tuple(
            ProgramDescriptor(
                ordinal=request.ordinal, request_sha256=request.sha256,
                installed_compiled_sites=_canonical_sites(request.sites),
                shared_program_sha256=program.sha256,
            )
            for request in requests
        )
        self._cross_bank = PreparedBank(
            model=binding, programs=programs, shared_program_sha256=program.sha256,
            evaluation_role_row_sha256s=tuple(checked_role_rows),
        )
        return self._cross_bank

    @torch.no_grad()
    def execute_cell(
        self, role: str, request: measurement.MeasurementRequest,
        rows: torch.Tensor, descriptor: ProgramDescriptor,
    ) -> BackendCellResult:
        if self._closed or self._model is None or self._model_guard is None or (
            self._program is None or self._cross_bank is None
        ) or not isinstance(request, measurement.MeasurementRequest) or not isinstance(
            descriptor, ProgramDescriptor
        ) or descriptor != self._cross_bank.programs[request.ordinal] or not torch.is_tensor(
            rows
        ) or rows.dtype != torch.long or rows.device.type != "cpu" or not (
            rows.is_contiguous()
        ) or role not in statistics.ROLE_NAMES or statistics.tensor_sha256(rows) != dict(
            self._cross_bank.evaluation_role_row_sha256s
        )[role] or self._active_handles:
            raise RuntimeError("cell request/descriptor/rows/state differs from prepare")
        component_before = self._model_guard.verify_metadata()
        program_before = self._program.verify_metadata()
        native_counts = {site: 0 for site in adapter.ALL_NATIVE_SITES}
        substitution_counts = {
            site: 0 for site in descriptor.installed_compiled_sites
        }
        current_tokens: torch.Tensor | None = None

        def count_native(site: cross.Site):
            def hook(_module, _arguments, _output):
                native_counts[site] += 1
                return None
            return hook

        def substitute(site: cross.Site):
            table = self._program.rows_for(site)

            def hook(_module, _arguments, output):
                nonlocal current_tokens
                if current_tokens is None:
                    raise RuntimeError("substitution fired outside a bound batch")
                value = output[0] if isinstance(output, tuple) else output
                replacement = table[current_tokens.reshape(-1)].reshape(value.shape).to(
                    value.dtype
                )
                substitution_counts[site] += 1
                return (replacement, *output[1:]) if isinstance(output, tuple) else replacement
            return hook

        # Registration order is part of the physical contract: the native counter
        # observes the original module return before the later substitution hook.
        handles = [
            self._module(site).register_forward_hook(count_native(site))
            for site in adapter.ALL_NATIVE_SITES
        ]
        handles.extend(
            self._module(site).register_forward_hook(substitute(site))
            for site in descriptor.installed_compiled_sites
        )
        self._active_handles = handles
        top1 = torch.empty(len(rows), dtype=torch.long)
        ce = torch.empty(len(rows), dtype=torch.float64)
        returned = 0
        try:
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start:start + self.batch_size]
                current_tokens = batch[:, :adapter.INPUT_STOP].to(self.device).contiguous()
                target = batch[:, adapter.TARGET_START:adapter.TARGET_STOP].to(self.device)
                logits = self._forward_logits(current_tokens)[
                    :, adapter.SCORE_START:adapter.SCORE_STOP
                ]
                prediction = logits.argmax(-1)
                top1[start:start + len(batch)] = (prediction == target).sum(1).cpu().long()
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                    reduction="none",
                ).reshape(target.shape).double()
                ce[start:start + len(batch)] = losses.sum(1).cpu()
                returned += 1
                current_tokens = None
        finally:
            current_tokens = None
            for handle in reversed(handles):
                handle.remove()
            self._active_handles.clear()
        batch_count = math_ceil_div(len(rows), self.batch_size)
        if returned != batch_count or native_counts != {
            site: batch_count for site in adapter.ALL_NATIVE_SITES
        } or substitution_counts != {
            site: batch_count for site in descriptor.installed_compiled_sites
        }:
            raise RuntimeError("physical hook call census changed")
        component_after = self._model_guard.verify_metadata()
        program_after = self._program.verify_metadata()
        values = measurement.RowCellStatistics(
            top1_correct=top1.contiguous(), ce_sum=ce.contiguous(),
            row_token_count=torch.full(
                (len(rows),), measurement.SCORED_TOKENS_PER_ROW, dtype=torch.long,
            ).contiguous(),
        )
        ledger = CellCallLedger(
            ordinal=request.ordinal, request_sha256=request.sha256,
            program_sha256=descriptor.sha256, row_count=len(rows),
            scored_token_count=len(rows) * measurement.SCORED_TOKENS_PER_ROW,
            batch_count=batch_count, outer_forward_count=batch_count,
            outer_returned_count=returned,
            native_module_calls=tuple(
                (site, native_counts[site]) for site in adapter.ALL_NATIVE_SITES
            ),
            substitution_calls=tuple(
                (site, substitution_counts[site])
                for site in descriptor.installed_compiled_sites
            ),
        )
        ledger.validate(descriptor)
        return BackendCellResult(
            statistics=values, call_ledger=ledger,
            component_tree_before_sha256=component_before,
            component_tree_after_sha256=component_after,
            shared_program_before_sha256=program_before,
            shared_program_after_sha256=program_after,
        )

    def close(self) -> str:
        self._cross_bank = None
        return super().close()


def math_ceil_div(numerator: int, denominator: int) -> int:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("ceil-div inputs must be positive")
    return (numerator + denominator - 1) // denominator


def create_backend() -> Bilin18ContextCrossBackend:
    """Return a lazy production backend; no external state is touched here."""

    return Bilin18ContextCrossBackend()
