"""Small model-facing executor for the reusable FIT causal screen.

Import and dry-run are CPU-only.  The real torch/checkpoint path is loaded only
when ``run_science`` is called without an injected backend.  Model adapters
receive exact per-row semantic positions; no intervention replaces a whole
sequence or assumes recipient and donor lengths are equal.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import statistics
import time
from typing import Callable, Literal, Mapping, Protocol, Sequence

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_spec as screen_spec


Side = Literal["base", "donor"]


@dataclass(frozen=True)
class ModelBatch:
    row_ids: tuple[str, ...]
    side: Side
    token_rows: tuple[tuple[int, ...], ...]
    answer_ids: tuple[int, ...]
    foil_ids: tuple[int, ...]
    semantic_positions: tuple[int, ...]


@dataclass(frozen=True)
class BatchOutput:
    answer_foil: tuple[tuple[float, float], ...]
    captured: Mapping[tuple[str, str], object]


class ExecutionBackend(Protocol):
    """Minimal injectable interface used by real and fake model executors."""

    def native(self, batch: ModelBatch, *, capture: bool) -> BatchOutput: ...

    def patched(
        self,
        batch: ModelBatch,
        *,
        site: kernel.SiteRef,
        donor_cache: Mapping[tuple[str, str], object],
    ) -> BatchOutput: ...


@dataclass(frozen=True)
class NativeLogitEvidence:
    row_id: str
    family: kernel.Family
    side: Side
    answer_logit: float
    foil_logit: float

    @property
    def margin(self) -> float:
        return self.answer_logit - self.foil_logit


@dataclass(frozen=True)
class InterventionLogitEvidence:
    row_id: str
    family: kernel.Family
    site: kernel.SiteRef
    answer_logit: float
    foil_logit: float


@dataclass(frozen=True)
class CapabilityCell:
    family: kernel.Family
    cell_id: str
    recipient_answer_id: int
    donor_answer_id: int
    base_correct_count: int
    donor_correct_count: int
    row_count: int
    base_accuracy: float
    donor_accuracy: float
    correct_count: int
    expected_count: int
    accuracy: float
    minimum_accuracy: float
    passed: bool


@dataclass(frozen=True)
class RunTiming:
    forward_calls: int
    example_evaluations: int
    seconds: float


@dataclass(frozen=True)
class FastScreenRun:
    terminal: kernel.Terminal
    reason: str
    selected_site: kernel.SiteRef | None
    head_stage: str
    capability_cells: tuple[CapabilityCell, ...]
    native_logits: tuple[NativeLogitEvidence, ...]
    intervention_logits: tuple[InterventionLogitEvidence, ...]
    site_results: tuple[kernel.SiteScreenResult, ...]
    ranking: tuple[kernel.RankedSite, ...]
    timing: RunTiming


class ProducerError(ValueError):
    """The execution or evidence contract is invalid."""


def compile_dryrun(
    spec: screen_spec.CircuitFastScreenSpec,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compile the exact screen closure without importing model dependencies."""

    return screen_spec.compile_dryrun(spec, rows)


def replace_declared_positions(
    recipient_rows: Sequence[Sequence[object]],
    recipient_positions: Sequence[int],
    donor_rows: Sequence[Sequence[object]],
    donor_positions: Sequence[int],
) -> tuple[tuple[object, ...], ...]:
    """Pure reference gather/scatter: replace exactly one position per row."""

    if not (len(recipient_rows) == len(recipient_positions)
            == len(donor_rows) == len(donor_positions)):
        raise ProducerError("row and semantic-position counts differ")
    output = []
    for recipient, recipient_position, donor, donor_position in zip(
        recipient_rows, recipient_positions, donor_rows, donor_positions
    ):
        if type(recipient_position) is not int or type(donor_position) is not int \
                or not 0 <= recipient_position < len(recipient) \
                or not 0 <= donor_position < len(donor):
            raise ProducerError("semantic position is outside its row")
        changed = list(recipient)
        changed[recipient_position] = donor[donor_position]
        output.append(tuple(changed))
    return tuple(output)


def _site(site_id: str) -> kernel.SiteRef:
    if site_id.startswith("resid:"):
        return kernel.SiteRef("residual", site_id)
    if ":head:" in site_id:
        return kernel.SiteRef("head", site_id)
    if site_id.startswith(("attn:", "mlp:")):
        return kernel.SiteRef("module", site_id)
    raise ProducerError(f"unknown site ID: {site_id}")


def _batch(
    spec: screen_spec.CircuitFastScreenSpec,
    rows: Sequence[Mapping[str, object]],
    side: Side,
) -> ModelBatch:
    fields = spec.fields
    sequence_field = (
        fields.recipient_sequence_field if side == "base" else fields.donor_sequence_field
    )
    answer_field = fields.recipient_answer_field if side == "base" else fields.donor_answer_field
    foil_field = fields.recipient_foil_field if side == "base" else fields.donor_foil_field
    position_field = "_recipient_position" if side == "base" else "_donor_position"
    return ModelBatch(
        row_ids=tuple(str(row[spec.task.row_id_field]) for row in rows),
        side=side,
        token_rows=tuple(tuple(int(token) for token in row[sequence_field]) for row in rows),
        answer_ids=tuple(int(row[answer_field]) for row in rows),
        foil_ids=tuple(int(row[foil_field]) for row in rows),
        semantic_positions=tuple(int(row[position_field]) for row in rows),
    )


def _chunks(values: Sequence[Mapping[str, object]], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _finite_pair(pair: tuple[float, float]) -> tuple[float, float]:
    if len(pair) != 2 or any(type(value) not in {int, float}
                             or not math.isfinite(float(value)) for value in pair):
        raise ProducerError("backend returned invalid answer/foil evidence")
    return float(pair[0]), float(pair[1])


def _family_threshold(family: kernel.Family) -> float:
    return {
        "A1": kernel.MIN_A1_CAPABILITY_ACCURACY,
        "A2": kernel.MIN_A2_CAPABILITY_ACCURACY,
        "P": kernel.MIN_P_CAPABILITY_ACCURACY,
        "C": kernel.MIN_C_CAPABILITY_ACCURACY,
    }[family]


def _capability(
    spec: screen_spec.CircuitFastScreenSpec,
    rows: Sequence[Mapping[str, object]],
    native: Mapping[tuple[str, Side], NativeLogitEvidence],
) -> tuple[tuple[CapabilityCell, ...], kernel.CapabilityEvidence]:
    # Construction plus ordered answer-token pair is the minimum generic cell:
    # opposite directions can never rescue each other by pooling.
    grouped: dict[
        tuple[kernel.Family, str, int, int], list[tuple[bool, bool]]
    ] = {}
    for row in rows:
        row_id = str(row[spec.task.row_id_field])
        family = str(row["transform_id"])
        if family not in screen_spec.TRANSFORMS:
            raise ProducerError("authority contains an unknown family")
        typed_family: kernel.Family = family  # type: ignore[assignment]
        cell_id = row.get("capability_cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ProducerError("authority lacks a nonempty capability_cell_id")
        key = (
            typed_family,
            cell_id,
            int(row[spec.fields.recipient_answer_field]),
            int(row[spec.fields.donor_answer_field]),
        )
        outcome = (
            native[(row_id, "base")].margin > 0.0,
            native[(row_id, "donor")].margin > 0.0,
        )
        grouped.setdefault(key, []).append(outcome)
    cells = tuple(
        CapabilityCell(
            family=family,
            cell_id=cell_id,
            recipient_answer_id=recipient,
            donor_answer_id=donor,
            base_correct_count=sum(base for base, _donor in outcomes),
            donor_correct_count=sum(donor for _base, donor in outcomes),
            row_count=len(outcomes),
            base_accuracy=sum(base for base, _donor in outcomes) / len(outcomes),
            donor_accuracy=sum(donor for _base, donor in outcomes) / len(outcomes),
            correct_count=sum(base + donor for base, donor in outcomes),
            expected_count=2 * len(outcomes),
            accuracy=sum(base + donor for base, donor in outcomes) / (2 * len(outcomes)),
            minimum_accuracy=_family_threshold(family),
            passed=(
                sum(base for base, _donor in outcomes) / len(outcomes)
                >= _family_threshold(family)
                and sum(donor for _base, donor in outcomes) / len(outcomes)
                >= _family_threshold(family)
            ),
        )
        for (family, cell_id, recipient, donor), outcomes in sorted(grouped.items())
    )
    aggregate = kernel.CapabilityEvidence(tuple(
        kernel.FamilyCapabilityEvidence(
            cell.family,
            cell.correct_count,
            cell.expected_count,
            cell.expected_count,
            cell_id=cell.cell_id,
        )
        for cell in cells
    ))
    return cells, aggregate


class Bilin18TorchBackend:
    """Lazy real backend with the checkpoint's exact manual forward."""

    def __init__(self, torch_module: object, model: object, device: str) -> None:
        self.torch = torch_module
        self.F = importlib.import_module("torch.nn.functional")
        self.model = model
        self.device = device
        blocks = tuple(model.transformer.h)
        if len(blocks) != 18 or model.config.n_embd != 1152 or model.config.n_head != 9:
            raise ProducerError("model is not the exact 18-layer, 9-head bilin18 structure")

    @classmethod
    def load(cls, device: str = "cuda") -> "Bilin18TorchBackend":
        torch_module = importlib.import_module("torch")
        fastload = importlib.import_module("fastload")
        if device.startswith("cuda") and not torch_module.cuda.is_available():
            raise ProducerError("CUDA science path requested but unavailable")
        model = fastload.load_model_fast().to(device).eval()
        return cls(torch_module, model, device)

    def _tensor_batch(self, batch: ModelBatch):
        maximum = max(len(row) for row in batch.token_rows)
        padded = [list(row) + [0] * (maximum - len(row)) for row in batch.token_rows]
        tokens = self.torch.tensor(padded, dtype=self.torch.long, device=self.device)
        return tokens, tuple(len(row) for row in batch.token_rows)

    def _save(self, cache, batch: ModelBatch, site_id: str, value) -> None:
        for index, (row_id, position) in enumerate(
            zip(batch.row_ids, batch.semantic_positions)
        ):
            cache[(row_id, site_id)] = value[index, position].detach().clone()

    def _replace(self, value, batch: ModelBatch, site_id: str, donor_cache) -> object:
        changed = value.clone()
        for index, (row_id, position) in enumerate(
            zip(batch.row_ids, batch.semantic_positions)
        ):
            replacement = donor_cache.get((row_id, site_id))
            if replacement is None:
                raise ProducerError(f"donor cache lacks {row_id}/{site_id}")
            changed[index, position] = replacement.to(device=value.device, dtype=value.dtype)
        return changed

    def _replace_head(self, value, batch: ModelBatch, site_id: str, donor_cache) -> object:
        head = int(site_id.rsplit(":", 1)[1])
        width = self.model.config.n_embd // self.model.config.n_head
        start, stop = head * width, (head + 1) * width
        changed = value.clone()
        for index, (row_id, position) in enumerate(
            zip(batch.row_ids, batch.semantic_positions)
        ):
            replacement = donor_cache.get((row_id, site_id))
            if replacement is None or tuple(replacement.shape) != (width,):
                raise ProducerError(f"donor head cache lacks exact slice {row_id}/{site_id}")
            changed[index, position, start:stop] = replacement.to(
                device=value.device, dtype=value.dtype
            )
        return changed

    def _forward(
        self,
        batch: ModelBatch,
        *,
        capture: bool,
        patch_site: kernel.SiteRef | None = None,
        donor_cache: Mapping[tuple[str, str], object] | None = None,
    ) -> BatchOutput:
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        captured: dict[tuple[str, str], object] = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0, v1 = x, None
            if capture:
                self._save(captured, batch, "resid:00", x)
            if patch_site is not None and patch_site.site_id == "resid:00":
                x = self._replace(x, batch, patch_site.site_id, donor_cache or {})
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                preprojection: dict[str, object] = {}

                def c_proj_pre(_module, arguments):
                    value = arguments[0]
                    preprojection["value"] = value
                    prefix = f"attn:{layer:02d}:head:"
                    if patch_site is None or not patch_site.site_id.startswith(prefix):
                        return None
                    return (
                        self._replace_head(
                            value, batch, patch_site.site_id, donor_cache or {}
                        ),
                    ) + tuple(arguments[1:])

                handle = block.attn.c_proj.register_forward_pre_hook(c_proj_pre)
                try:
                    attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                finally:
                    handle.remove()
                if capture:
                    self._save(captured, batch, f"attn:{layer:02d}", attention)
                    value = preprojection["value"]
                    width = model.config.n_embd // model.config.n_head
                    for head in range(model.config.n_head):
                        self._save(
                            captured, batch, f"attn:{layer:02d}:head:{head:02d}",
                            value[..., head * width:(head + 1) * width],
                        )
                if patch_site is not None and patch_site.site_id == f"attn:{layer:02d}":
                    attention = self._replace(
                        attention, batch, patch_site.site_id, donor_cache or {}
                    )
                x = live + attention
                mlp = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if capture:
                    self._save(captured, batch, f"mlp:{layer:02d}", mlp)
                if patch_site is not None and patch_site.site_id == f"mlp:{layer:02d}":
                    mlp = self._replace(mlp, batch, patch_site.site_id, donor_cache or {})
                x = x + mlp
                if capture:
                    self._save(captured, batch, f"resid:{layer + 1:02d}", x)
                if patch_site is not None \
                        and patch_site.site_id == f"resid:{layer + 1:02d}":
                    x = self._replace(x, batch, patch_site.site_id, donor_cache or {})
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0
            )
            values = tuple(
                (
                    float(logits[index, length - 1, batch.answer_ids[index]].float()),
                    float(logits[index, length - 1, batch.foil_ids[index]].float()),
                )
                for index, length in enumerate(lengths)
            )
        return BatchOutput(values, captured)

    def native(self, batch: ModelBatch, *, capture: bool) -> BatchOutput:
        return self._forward(batch, capture=capture)

    def patched(
        self,
        batch: ModelBatch,
        *,
        site: kernel.SiteRef,
        donor_cache: Mapping[tuple[str, str], object],
    ) -> BatchOutput:
        return self._forward(
            batch, capture=False, patch_site=site, donor_cache=donor_cache
        )


def run_science(
    spec: screen_spec.CircuitFastScreenSpec,
    rows: Sequence[Mapping[str, object]],
    *,
    backend: ExecutionBackend | None = None,
    device: str = "cuda",
    clock: Callable[[], float] = time.perf_counter,
) -> FastScreenRun:
    """Execute native capability, 55 sites, and a conditional nine-head stage."""

    started = clock()
    native_records: list[NativeLogitEvidence] = []
    intervention_records: list[InterventionLogitEvidence] = []
    site_results: list[kernel.SiteScreenResult] = []
    cells: tuple[CapabilityCell, ...] = ()
    forward_calls = evaluations = 0

    def finish(terminal, reason, selected=None, head_stage="not_opened"):
        ranking = kernel.rank_sites(site_results) if site_results else ()
        return FastScreenRun(
            terminal, reason, selected, head_stage, cells, tuple(native_records),
            tuple(intervention_records), tuple(site_results), ranking,
            RunTiming(forward_calls, evaluations, clock() - started),
        )

    try:
        screen_spec.compile_screen(spec, rows)
        by_id = screen_spec.validate_fit_authority(spec, rows)
        materialized = tuple(by_id.values())
        executor = backend if backend is not None else Bilin18TorchBackend.load(device)
        donor_cache: dict[tuple[str, str], object] = {}
        native: dict[tuple[str, Side], NativeLogitEvidence] = {}
        for side in ("base", "donor"):
            for family in screen_spec.TRANSFORMS:
                family_rows = [row for row in materialized if row["transform_id"] == family]
                for chunk in _chunks(family_rows, spec.batch_size):
                    batch = _batch(spec, chunk, side)
                    output = executor.native(batch, capture=side == "donor")
                    forward_calls += 1
                    evaluations += len(chunk)
                    if len(output.answer_foil) != len(chunk):
                        raise ProducerError("backend output count differs from its batch")
                    donor_cache.update(output.captured)
                    for row_id, pair in zip(batch.row_ids, output.answer_foil):
                        answer, foil = _finite_pair(pair)
                        record = NativeLogitEvidence(
                            row_id, family, side, answer, foil  # type: ignore[arg-type]
                        )
                        native_records.append(record)
                        native[(row_id, side)] = record
        cells, capability = _capability(spec, materialized, native)
        if not all(cell.passed for cell in cells):
            return finish("null", "native_behavior_incapable", head_stage="capability_stop")

        denominators = []
        for row in materialized:
            family = str(row["transform_id"])
            if family not in {"A1", "A2", "C"}:
                continue
            row_id = str(row[spec.task.row_id_field])
            base = -native[(row_id, "base")].margin
            donor = native[(row_id, "donor")].margin
            kernel.signed_pairwise_donor_recovery(base, donor, base)
            if family in {"A1", "A2"}:
                denominators.append(donor - base)
        target_scale = statistics.median(denominators)
        if not math.isfinite(target_scale) or target_scale <= kernel.MIN_DONOR_DENOMINATOR:
            raise ProducerError("target-family native scale is invalid")

        def evaluate_site(site: kernel.SiteRef) -> kernel.SiteScreenResult:
            evidence = []
            for family in screen_spec.TRANSFORMS:
                family_rows = [row for row in materialized if row["transform_id"] == family]
                for chunk in _chunks(family_rows, spec.batch_size):
                    nonlocal forward_calls, evaluations
                    batch = _batch(spec, chunk, "base")
                    output = executor.patched(batch, site=site, donor_cache=donor_cache)
                    forward_calls += 1
                    evaluations += len(chunk)
                    if len(output.answer_foil) != len(chunk):
                        raise ProducerError("backend output count differs from its batch")
                    for row, pair in zip(chunk, output.answer_foil):
                        answer, foil = _finite_pair(pair)
                        row_id = str(row[spec.task.row_id_field])
                        typed_family: kernel.Family = family  # type: ignore[assignment]
                        intervention_records.append(InterventionLogitEvidence(
                            row_id, typed_family, site, answer, foil,
                        ))
                        native_base = native[(row_id, "base")].margin
                        if family == "P":
                            base_score, donor_score = native_base, None
                            intervened, scale = answer - foil, target_scale
                        else:
                            base_score = -native_base
                            donor_score = native[(row_id, "donor")].margin
                            intervened, scale = -(answer - foil), None
                        evidence.append(kernel.ScalarInterventionEvidence(
                            record_id=f"{site.site_id}|{row_id}", pair_id=row_id,
                            family=typed_family, evidence_kind=site.evidence_kind,
                            site_id=site.site_id, base_score=base_score,
                            donor_score=donor_score, intervened_score=intervened,
                            effect_scale=scale,
                        ))
            return kernel.score_site(
                site, evidence=tuple(evidence),
                expected_record_ids=tuple(record.record_id for record in evidence),
                capability=capability,
            )

        for site_id in screen_spec.CEILING_SITE_IDS:
            site_results.append(evaluate_site(_site(site_id)))
        if any(result.terminal == "invalid" for result in site_results):
            return finish("invalid", "evidence_invalid")
        ranked = kernel.rank_sites(site_results)
        passing = [item.result for item in ranked if item.result.terminal == "screen"]
        if not passing:
            return finish("null", "no_selective_causal_site", head_stage="skipped_no_parent")
        parent = passing[0]
        head_stage = "skipped_parent_not_attention"
        if parent.site.evidence_kind == "module" and parent.site.site_id.startswith("attn:"):
            layer = int(parent.site.site_id.split(":")[1])
            head_stage = "expanded"
            for head in range(9):
                site_results.append(evaluate_site(_site(
                    f"attn:{layer:02d}:head:{head:02d}"
                )))
            if any(result.terminal == "invalid" for result in site_results):
                return finish("invalid", "evidence_invalid", head_stage=head_stage)
        final_ranking = kernel.rank_sites(site_results)
        selected = next(
            (item.result.site for item in final_ranking
             if item.result.terminal == "screen"),
            None,
        )
        if selected is None:  # pragma: no cover - passing parent remains present
            return finish("null", "no_selective_causal_site", head_stage=head_stage)
        return finish("screen", "selective_causal_site", selected, head_stage)
    except Exception:
        return finish("invalid", "execution_invalid")
