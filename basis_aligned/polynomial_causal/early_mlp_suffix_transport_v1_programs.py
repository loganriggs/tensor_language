"""Validation selection and deterministic program freezing for suffix transport v1.

This module has no row loader, model loader, or authority to score validation/final
data.  It consumes fit candidates plus independently collected validation receipts,
applies the frozen admissibility and tie-breaking rules, and emits canonical CPU
program records.  Keeping selection pure makes it possible to test the decision rule
before validation rows exist and prevents a numerical launcher from inventing a
different selector after seeing outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_runtime as runtime


VALIDATION_ROWS = 192
VALIDATION_SCORED_TOKENS = VALIDATION_ROWS * (
    runtime.SCORE_STOP - runtime.SCORE_START
)
ZERO_NATIVE_CALLS = ((0, 0), (1, 0), (2, 0))
SELECTABLE_ROUTES = ("L", "R", "S0", "S1", "T")
LOCAL_ROUTES = frozenset({"L"})
SUFFIX_ROUTES = frozenset({"R", "S0", "S1", "T"})
METRIC_BY_ROUTE = {
    "L": "local_normalized_mse",
    "R": "oon_teacher_kl",
    "S0": "oon_teacher_kl",
    "S1": "oon_teacher_kl",
    "T": "oon_teacher_kl",
}


def _sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def tensor_tree_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and exact bytes in stable key order."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("program tensor state is empty")
    hashes = {}
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name or not torch.is_tensor(value) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("program tensor state is malformed")
        hashes[name] = runtime.tensor_identity_sha256(value)
    return runtime.logical_identity_sha256(hashes)


def restore_fit_candidate(candidate: fit.FitCandidate) -> runtime.JointAffineProgram:
    """Rebuild a fit candidate and prove its exact dense snapshot binding."""

    if not isinstance(candidate, fit.FitCandidate) or candidate.route not in (
        fit.ALL_TRUE_FIT_ROUTES
    ):
        raise TypeError("selection requires a true-row fit candidate")
    state = candidate.state_dict
    exact = {
        "site0.mean", "site0.scale", "site0.weight", "site0.bias",
        "site1.mean", "site1.scale", "site1.weight", "site1.bias",
    }
    if candidate.route == "T":
        exact.add("cross")
    if set(state) != exact:
        raise RuntimeError("fit candidate dense state schema changed")
    sites = {}
    for site in (0, 1):
        sites[site] = runtime.AffineCodeProgram(
            mean=state[f"site{site}.mean"], scale=state[f"site{site}.scale"],
            weight=state[f"site{site}.weight"], bias=state[f"site{site}.bias"],
        )
    program = runtime.JointAffineProgram(sites[0], sites[1], route=candidate.route)
    if candidate.route == "T":
        if program.cross is None or tuple(state["cross"].shape) != (
            runtime.CODE_DIM, runtime.CODE_DIM
        ) or not bool(torch.isfinite(state["cross"]).all()):
            raise RuntimeError("transport fit candidate cross state is malformed")
        with torch.no_grad():
            program.cross.copy_(state["cross"])
    if runtime.program_snapshot_sha256(program) != candidate.final_program_sha256:
        raise RuntimeError("fit candidate state differs from its terminal snapshot")
    return program


@dataclass(frozen=True)
class ValidationScore:
    """Minimal immutable selector receipt for one candidate on all validation rows."""

    route: str
    trial: int
    learning_rate: float
    program_sha256: str
    metric_name: str
    primary_metric: float
    copy_worsening: float
    scored_token_count: int
    common_support_sha256: str
    sufficient_statistics_sha256: str
    student_original_calls: tuple[tuple[int, int], ...]
    hook_restored: bool
    hook_inert: bool

    def __post_init__(self) -> None:
        if self.route not in SELECTABLE_ROUTES or self.metric_name != METRIC_BY_ROUTE[
            self.route
        ] or type(self.trial) is not int or self.trial not in range(
            len(runtime.LEARNING_RATES)
        ) or self.learning_rate != runtime.LEARNING_RATES[self.trial]:
            raise ValueError("validation route/trial/metric identity changed")
        if not _sha256(self.program_sha256) or not _sha256(
            self.common_support_sha256
        ) or not _sha256(self.sufficient_statistics_sha256):
            raise ValueError("validation hash binding is malformed")
        if type(self.scored_token_count) is not int or self.scored_token_count != (
            VALIDATION_SCORED_TOKENS
        ):
            raise ValueError("validation support count changed")
        if self.student_original_calls != ZERO_NATIVE_CALLS:
            raise ValueError("validation student executed an original early MLP")
        if self.hook_restored is not True or self.hook_inert is not True:
            raise ValueError("validation hook did not restore inertly")
        if not math.isfinite(self.primary_metric) or self.primary_metric < 0 or not (
            math.isfinite(self.copy_worsening)
        ):
            raise ValueError("validation metrics must be finite and nonnegative where required")

    @property
    def admissible(self) -> bool:
        return self.copy_worsening <= 0.01


def _row_vector(name: str, value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != (VALIDATION_ROWS,) or (
        value.dtype != dtype
    ) or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
        raise ValueError(f"validation statistic {name} changed shape, dtype, or finiteness")
    return value.detach().cpu().contiguous().clone()


@dataclass(frozen=True)
class ValidationSufficientStatistics:
    """Raw per-row selector statistics; ratios are intentionally not stored."""

    route: str
    program_sha256: str
    common_support_sha256: str
    row_primary_sum: torch.Tensor
    row_primary_count: torch.Tensor
    row_ce_sum: torch.Tensor
    row_ce_count: torch.Tensor
    row_copy_ce_sum: torch.Tensor
    row_copy_count: torch.Tensor
    baseline_row_copy_ce_sum: torch.Tensor
    baseline_row_copy_count: torch.Tensor
    student_original_calls: tuple[tuple[int, int], ...]
    hook_restored: bool
    hook_inert: bool

    def __post_init__(self) -> None:
        if self.route not in SELECTABLE_ROUTES or not _sha256(
            self.program_sha256
        ) or not _sha256(self.common_support_sha256):
            raise ValueError("validation sufficient-statistic identity is malformed")
        float_names = (
            "row_primary_sum", "row_ce_sum", "row_copy_ce_sum",
            "baseline_row_copy_ce_sum",
        )
        count_names = (
            "row_primary_count", "row_ce_count", "row_copy_count",
            "baseline_row_copy_count",
        )
        for name in float_names:
            value = _row_vector(name, getattr(self, name), dtype=torch.float64)
            if bool((value < 0).any()):
                raise ValueError(f"validation statistic {name} is negative")
            object.__setattr__(self, name, value)
        for name in count_names:
            value = _row_vector(name, getattr(self, name), dtype=torch.long)
            if bool((value < 0).any()):
                raise ValueError(f"validation statistic {name} is negative")
            object.__setattr__(self, name, value)
        expected = torch.full(
            (VALIDATION_ROWS,), runtime.SCORE_STOP - runtime.SCORE_START,
            dtype=torch.long,
        )
        if not torch.equal(self.row_primary_count, expected) or not torch.equal(
            self.row_ce_count, expected
        ) or not torch.equal(self.row_copy_count, self.baseline_row_copy_count) or int(
            self.row_copy_count.sum()
        ) <= 0:
            raise ValueError("validation primary/CE/copy supports changed")
        if self.student_original_calls != ZERO_NATIVE_CALLS or self.hook_restored is not True \
                or self.hook_inert is not True:
            raise ValueError("validation sufficient statistics lack a clean student closure")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "route": self.route,
            "program_sha256": self.program_sha256,
            "common_support_sha256": self.common_support_sha256,
            **{
                name: runtime.tensor_identity_sha256(getattr(self, name))
                for name in (
                    "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
                    "row_copy_ce_sum", "row_copy_count", "baseline_row_copy_ce_sum",
                    "baseline_row_copy_count",
                )
            },
            "student_original_calls": self.student_original_calls,
            "hook_restored": self.hook_restored,
            "hook_inert": self.hook_inert,
        })


def validation_score_from_statistics(
    candidate: fit.FitCandidate, statistics: ValidationSufficientStatistics,
) -> ValidationScore:
    """Recompute the only two scalar selector inputs from immutable raw row sums."""

    if not isinstance(candidate, fit.FitCandidate) or not isinstance(
        statistics, ValidationSufficientStatistics
    ) or candidate.route != statistics.route or candidate.final_program_sha256 != (
        statistics.program_sha256
    ):
        raise ValueError("validation statistics differ from their fit candidate")
    primary = float(
        statistics.row_primary_sum.sum() / statistics.row_primary_count.sum()
    )
    candidate_copy = statistics.row_copy_ce_sum.sum() / statistics.row_copy_count.sum()
    baseline_copy = statistics.baseline_row_copy_ce_sum.sum() / (
        statistics.baseline_row_copy_count.sum()
    )
    return ValidationScore(
        route=candidate.route, trial=candidate.trial,
        learning_rate=candidate.learning_rate,
        program_sha256=candidate.final_program_sha256,
        metric_name=METRIC_BY_ROUTE[candidate.route], primary_metric=primary,
        copy_worsening=float(candidate_copy - baseline_copy),
        scored_token_count=int(statistics.row_primary_count.sum()),
        common_support_sha256=statistics.common_support_sha256,
        sufficient_statistics_sha256=statistics.sha256,
        student_original_calls=statistics.student_original_calls,
        hook_restored=statistics.hook_restored, hook_inert=statistics.hook_inert,
    )


def local_primary_rows(
    predictions: Sequence[torch.Tensor], labels: Sequence[torch.Tensor],
    denominators: Sequence[torch.Tensor | float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row sums whose pooled mean equals registered normalized local loss."""

    if len(predictions) != 2 or len(labels) != 2 or len(denominators) != 2:
        raise ValueError("local validation requires exactly two sites")
    row_sum = None
    for prediction, label, denominator in zip(
        predictions, labels, denominators, strict=True,
    ):
        prediction = runtime._canonical_code_support(prediction).float()
        target = runtime._canonical_code_support(label).detach().to(
            device=prediction.device, dtype=torch.float32,
        )
        if prediction.shape != target.shape:
            raise ValueError("local validation prediction/label shapes differ")
        scale = torch.as_tensor(denominator, dtype=torch.float64)
        if scale.numel() != 1 or not bool(torch.isfinite(scale)) or float(scale) <= 0:
            raise ValueError("local validation denominator is invalid")
        contribution = (
            (prediction - target).double().square().sum(dim=-1)
            / (runtime.CODE_DIM * scale)
        ).sum(dim=1)
        row_sum = contribution if row_sum is None else row_sum + contribution
    assert row_sum is not None
    count = torch.full(
        (len(row_sum),), runtime.SCORE_STOP - runtime.SCORE_START, dtype=torch.long,
    )
    return row_sum.detach().cpu().double().contiguous(), count


def suffix_kl_rows(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row token KL sums on exact positions 64:256."""

    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3:
        raise ValueError("validation KL logits must be same-shaped rank-three tensors")
    if teacher_logits.shape[1] == runtime.SEQUENCE_LENGTH:
        teacher_logits = runtime.scored_positions(teacher_logits)
        student_logits = runtime.scored_positions(student_logits)
    if teacher_logits.shape[1] != runtime.SCORE_STOP - runtime.SCORE_START or (
        teacher_logits.shape[-1] <= 1
    ) or not bool(torch.isfinite(teacher_logits).all()) or not bool(
        torch.isfinite(student_logits).all()
    ):
        raise ValueError("validation KL support or logits are malformed")
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    per_token = torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), dim=-1)
    count = torch.full(
        (len(per_token),), runtime.SCORE_STOP - runtime.SCORE_START, dtype=torch.long,
    )
    return per_token.double().sum(dim=1).detach().cpu().contiguous(), count


def copy_mask(role_rows: torch.Tensor) -> torch.Tensor:
    """Frozen 64-token-history copy mask on shifted targets 64:256."""

    if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or role_rows.ndim != 2 \
            or role_rows.shape[1] < runtime.SEQUENCE_LENGTH + 1:
        raise ValueError("copy mask requires frozen rows with at least 257 tokens")
    inputs = role_rows[:, :runtime.SEQUENCE_LENGTH]
    targets = role_rows[:, 1:runtime.SEQUENCE_LENGTH + 1]
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for lag in range(64):
        past = torch.roll(inputs, lag, dims=1)
        if lag:
            past[:, :lag] = -1
        mask |= past == targets
    return mask[:, runtime.SCORE_START:runtime.SCORE_STOP].contiguous()


def ce_and_copy_rows(
    logits: torch.Tensor, role_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-row global/copy CE sums and counts from the frozen shifted targets."""

    if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[0] != len(
        role_rows
    ) or logits.shape[-1] <= 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("validation CE logits are malformed")
    if logits.shape[1] == runtime.SEQUENCE_LENGTH:
        logits = runtime.scored_positions(logits)
    if logits.shape[1] != runtime.SCORE_STOP - runtime.SCORE_START:
        raise ValueError("validation CE logits use the wrong positional support")
    targets = role_rows[
        :, 1:runtime.SEQUENCE_LENGTH + 1
    ][:, runtime.SCORE_START:runtime.SCORE_STOP].to(logits.device)
    if bool((targets < 0).any()) or bool((targets >= logits.shape[-1]).any()):
        raise ValueError("validation targets exceed the supplied logit vocabulary")
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        reduction="none",
    ).view(len(role_rows), -1).double()
    mask = copy_mask(role_rows).to(logits.device)
    return (
        losses.sum(dim=1).detach().cpu().contiguous(),
        torch.full(
            (len(role_rows),), runtime.SCORE_STOP - runtime.SCORE_START,
            dtype=torch.long,
        ),
        (losses * mask).sum(dim=1).detach().cpu().contiguous(),
        mask.sum(dim=1).detach().cpu().long().contiguous(),
    )


@dataclass(frozen=True)
class ScoredCandidate:
    fit_candidate: fit.FitCandidate
    validation: ValidationScore

    def __post_init__(self) -> None:
        candidate = self.fit_candidate
        score = self.validation
        if candidate.route != score.route or candidate.trial != score.trial or (
            candidate.learning_rate != score.learning_rate
        ) or candidate.final_program_sha256 != score.program_sha256:
            raise ValueError("validation receipt differs from its fit candidate")
        # Reconstructing here catches mutated tensor values before they can influence
        # either selection or its final tensor-hash tie-break.
        restore_fit_candidate(candidate)

    @property
    def tensor_sha256(self) -> str:
        return tensor_tree_sha256(self.fit_candidate.state_dict)


def select_candidate(
    candidates: Sequence[ScoredCandidate], *, route: str,
) -> ScoredCandidate:
    """Apply metric, smaller-LR, then lexical-tensor-hash selection exactly."""

    if route not in SELECTABLE_ROUTES or not candidates:
        raise ValueError("selection route/candidate bank is malformed")
    if any(candidate.validation.route != route for candidate in candidates):
        raise ValueError("selection bank mixes program routes")
    trials = [candidate.validation.trial for candidate in candidates]
    if sorted(trials) != list(range(len(runtime.LEARNING_RATES))):
        raise ValueError("selection bank must contain each learning-rate trial exactly once")
    eligible = [candidate for candidate in candidates if candidate.validation.admissible]
    if not eligible:
        raise RuntimeError("no validation candidate satisfies the copy bound")
    return min(eligible, key=lambda candidate: (
        candidate.validation.primary_metric,
        candidate.validation.learning_rate,
        candidate.tensor_sha256,
    ))


def _canonical_svd(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    if not torch.is_tensor(weight) or tuple(weight.shape) != (
        runtime.D_MODEL, runtime.CODE_DIM
    ) or not bool(torch.isfinite(weight).all()):
        raise ValueError("canonical SVD requires one finite dense affine weight")
    original = weight.detach().cpu().float().contiguous()
    u, singular, vh = torch.linalg.svd(original.double(), full_matrices=False)
    for column in range(runtime.CODE_DIM):
        pivot = int(torch.argmax(torch.abs(u[:, column])))
        if float(u[pivot, column]) < 0:
            u[:, column].neg_()
            vh[column].neg_()
    left = (u * singular.unsqueeze(0)).float().contiguous()
    right = vh.float().contiguous()
    replay = left @ right
    error = float(torch.max(torch.abs(replay - original)))
    if not math.isfinite(error) or error > 2e-6:
        raise RuntimeError(f"canonical SVD replay failed: max error {error}")
    # Check the sign convention after storage conversion, where it matters.
    normalized_u = left.double() / torch.linalg.vector_norm(left.double(), dim=0).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    for column in range(runtime.CODE_DIM):
        pivot = int(torch.argmax(torch.abs(normalized_u[:, column])))
        if float(normalized_u[pivot, column]) < 0:
            raise RuntimeError("canonical SVD sign convention failed after storage conversion")
    return left, right, error


def _factored_site(program: runtime.AffineCodeProgram) -> tuple[Mapping[str, Any], float]:
    left, right, error = _canonical_svd(program.weight)
    return MappingProxyType({
        "grammar": "affine",
        "interface": "state_complete_p",
        "mean": program.mean.detach().cpu().float().contiguous().clone(),
        "scale": program.scale.detach().cpu().float().contiguous().clone(),
        "left": left,
        "right": right,
        "bias": program.bias.detach().cpu().float().contiguous().clone(),
    }), error


@dataclass(frozen=True)
class FrozenProgram:
    """Canonical deployable representation of one selected validation candidate."""

    route: str
    trial: int
    learning_rate: float
    validation_metric_name: str
    validation_metric: float
    validation_sufficient_statistics_sha256: str
    source_program_sha256: str
    source_tensor_sha256: str
    canonical_tensor_sha256: str
    site_states: Mapping[int, Mapping[str, Any]]
    cross: torch.Tensor | None
    svd_max_errors: tuple[float, float]

    def make_program(self) -> runtime.JointAffineProgram:
        program = runtime.JointAffineProgram.from_v21_states(
            self.site_states, route=self.route,
        )
        if self.route == "T":
            if self.cross is None or program.cross is None:
                raise RuntimeError("frozen T program lost its cross map")
            with torch.no_grad():
                program.cross.copy_(self.cross)
        elif self.cross is not None:
            raise RuntimeError("non-T frozen program acquired a cross map")
        canonical = tensor_tree_sha256(program.state_dict())
        if canonical != self.canonical_tensor_sha256:
            raise RuntimeError("frozen program tensors changed after canonicalization")
        return program


def freeze_selected(candidate: ScoredCandidate) -> FrozenProgram:
    if not candidate.validation.admissible:
        raise RuntimeError("copy-inadmissible candidate cannot be frozen")
    program = restore_fit_candidate(candidate.fit_candidate)
    states = {}
    errors = []
    for site, affine in ((0, program.site0), (1, program.site1)):
        states[site], error = _factored_site(affine)
        errors.append(error)
    canonical_program = runtime.JointAffineProgram.from_v21_states(
        states, route=candidate.validation.route,
    )
    cross = None
    if candidate.validation.route == "T":
        if program.cross is None or canonical_program.cross is None:
            raise RuntimeError("selected T candidate lacks its cross map")
        cross = program.cross.detach().cpu().float().contiguous().clone()
        with torch.no_grad():
            canonical_program.cross.copy_(cross)
    return FrozenProgram(
        route=candidate.validation.route,
        trial=candidate.validation.trial,
        learning_rate=candidate.validation.learning_rate,
        validation_metric_name=candidate.validation.metric_name,
        validation_metric=candidate.validation.primary_metric,
        validation_sufficient_statistics_sha256=(
            candidate.validation.sufficient_statistics_sha256
        ),
        source_program_sha256=candidate.fit_candidate.final_program_sha256,
        source_tensor_sha256=candidate.tensor_sha256,
        canonical_tensor_sha256=tensor_tree_sha256(canonical_program.state_dict()),
        site_states=MappingProxyType(states),
        cross=cross,
        svd_max_errors=(errors[0], errors[1]),
    )


def select_and_freeze_routes(
    candidates: Mapping[str, Sequence[ScoredCandidate]],
) -> Mapping[str, FrozenProgram]:
    """Freeze the four objective-route families without observing T or final rows."""

    required = set(fit.TRUE_FIT_ROUTES)
    if set(candidates) != required:
        raise ValueError("objective selection requires exactly L/R/S0/S1 banks")
    return MappingProxyType({
        route: freeze_selected(select_candidate(candidates[route], route=route))
        for route in fit.TRUE_FIT_ROUTES
    })


def make_transport_initialization(selected_l: FrozenProgram) -> runtime.JointAffineProgram:
    """Freeze selected L0/L1 and add the registered zero, cross-only T parameter."""

    if not isinstance(selected_l, FrozenProgram) or selected_l.route != "L":
        raise ValueError("transport initialization requires the selected L program")
    local = selected_l.make_program()
    transport = local.independent_clone(route="T")
    transport.require_exact_trainability()
    if transport.cross is None or int(torch.count_nonzero(transport.cross)) != 0 or (
        transport.trainable_parameter_names != ("cross",)
    ):
        raise RuntimeError("transport did not initialize as exact zero cross-only T")
    for name, value in local.state_dict().items():
        if name == "cross" or not torch.equal(value, transport.state_dict()[name]):
            raise RuntimeError("transport initialization changed selected L tensors")
    return transport
