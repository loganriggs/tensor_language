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
