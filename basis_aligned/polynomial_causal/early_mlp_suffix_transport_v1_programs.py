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

import early_mlp_suffix_transport_v1 as contract
import early_mlp_suffix_transport_v1_capabilities as capabilities
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
GAUGE_SIGNED_SEEDS = tuple(2026082801 + index for index in range(4))
GAUGE_HAAR_SEEDS = tuple(2026082810 + index for index in range(4))
INTERVENTION_AMPLITUDES = (0.01, 0.03, 0.1, 0.3, 1.0)
CALIBRATION_BAND = (0.01, 0.20)
FIT_INTERVENTION_CODE_COUNT = capabilities.FIT_ROW_COUNT * (
    runtime.SCORE_STOP - runtime.SCORE_START
)


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


def restore_mapped_fit_candidate(
    candidate: fit.MappedFitCandidate,
) -> runtime.JointAffineProgram:
    """Rebuild a negative-control fit without erasing its separate scientific type."""

    if not isinstance(candidate, fit.MappedFitCandidate) or not valid_mapped_control(
        candidate.control, candidate.route
    ):
        raise TypeError("mapped selection requires a registered negative-control candidate")
    proxy = fit.FitCandidate(
        route=candidate.route, trial=candidate.trial,
        learning_rate=candidate.learning_rate,
        completed_steps=candidate.completed_steps, loss_sum=candidate.loss_sum,
        loss_min=candidate.loss_min, loss_max=candidate.loss_max,
        final_program_sha256=candidate.final_program_sha256,
        transaction_history_sha256=candidate.transaction_history_sha256,
        state_dict=candidate.state_dict,
    )
    return restore_fit_candidate(proxy)


def valid_mapped_control(control: str, route: str) -> bool:
    """Recognize only the prospectively registered mapped control/route pairs."""

    if control == "document_shuffle":
        return route in fit.DOCUMENT_SHUFFLE_ROUTES
    if not isinstance(control, str) or not control.startswith("A_null_") or route != "T":
        return False
    suffix = control.removeprefix("A_null_")
    return len(suffix) == 2 and suffix.isdigit() and 0 <= int(suffix) < 20 and (
        control == f"A_null_{int(suffix):02d}"
    )


def make_validation_identity(
    *, context: capabilities.ValidationRunContext,
    program: runtime.JointAffineProgram, inputs: torch.Tensor,
    indices: Sequence[int], route: str, control: str, trial: int,
    batch_ordinal: int,
) -> runtime.TraceIdentity:
    """Bind one true-row selection batch while preserving fit-control provenance."""

    legal_control = control == "true" or valid_mapped_control(control, route)
    if not isinstance(context, capabilities.ValidationRunContext) or route not in (
        SELECTABLE_ROUTES
    ) or not legal_control or not isinstance(program, runtime.JointAffineProgram) or (
        program.route != route
    ) or type(trial) is not int or trial not in range(len(runtime.LEARNING_RATES)) or (
        type(batch_ordinal) is not int
    ) or not 0 <= batch_ordinal < capabilities.VALIDATION_BATCH_COUNT:
        raise ValueError("validation execution identity is malformed")
    expected = tuple(range(
        batch_ordinal * runtime.BATCH_SIZE,
        (batch_ordinal + 1) * runtime.BATCH_SIZE,
    ))
    if tuple(indices) != expected:
        raise RuntimeError("validation batch is not in canonical role order")
    return runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=indices,
        source_commit=context.source_commit,
        inherited_snapshot_sha256=context.inherited_snapshot_sha256,
        rows_receipt_sha256=context.rows_receipt_sha256,
        # Trace schema v1 keeps this historical name; the validation role gives it
        # the unambiguous meaning of validation-role tensor hash.
        fit_role_tensor_sha256=context.validation_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=context.identity_teacher_mapping_sha256,
        role="early_mlp_suffix_transport_v1_validation", phase="validation",
        route=route, control=control,
        teacher_kind="coordinate_labels" if route == "L" else "oon_logits",
        trial=trial, epoch=0, optimizer_step=batch_ordinal,
        batch_ordinal=batch_ordinal,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )


@dataclass(frozen=True)
class ValidationBaselineIdentity:
    """Exact deployed-N/N validation batch identity, independent of any candidate."""

    source_commit: str
    inherited_snapshot_sha256: str
    rows_receipt_sha256: str
    validation_role_tensor_sha256: str
    common_support_sha256: str
    ordered_batch_indices_sha256: str
    ordered_input_tokens_sha256: str
    batch_ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_commit, str) or len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ) or any(not _sha256(value) for value in (
            self.inherited_snapshot_sha256, self.rows_receipt_sha256,
            self.validation_role_tensor_sha256, self.common_support_sha256,
            self.ordered_batch_indices_sha256, self.ordered_input_tokens_sha256,
        )) or type(self.batch_ordinal) is not int or not (
            0 <= self.batch_ordinal < capabilities.VALIDATION_BATCH_COUNT
        ):
            raise ValueError("validation baseline identity is malformed")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            field: getattr(self, field) for field in self.__dataclass_fields__
        })

    def require_batch(
        self, role_rows: torch.Tensor, ordered_row_indices: Sequence[int],
    ) -> None:
        if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or tuple(
            role_rows.shape
        ) != (runtime.BATCH_SIZE, 513) or role_rows.device.type != "cpu":
            raise RuntimeError("validation baseline requires one CPU role-row batch")
        start = self.batch_ordinal * runtime.BATCH_SIZE
        expected = tuple(range(start, start + runtime.BATCH_SIZE))
        if tuple(ordered_row_indices) != expected or runtime.logical_identity_sha256(
            list(expected)
        ) != self.ordered_batch_indices_sha256 or runtime.tensor_identity_sha256(
            role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous()
        ) != self.ordered_input_tokens_sha256:
            raise RuntimeError("validation baseline batch differs from its frozen identity")


def make_validation_baseline_identity(
    *, context: capabilities.ValidationRunContext, role_rows: torch.Tensor,
    batch_ordinal: int,
) -> ValidationBaselineIdentity:
    """Mint the N/N baseline identity from the complete frozen validation role."""

    if not isinstance(context, capabilities.ValidationRunContext) or not torch.is_tensor(
        role_rows
    ) or role_rows.dtype != torch.long or tuple(role_rows.shape) != (
        VALIDATION_ROWS, 513
    ) or role_rows.device.type != "cpu" or runtime.tensor_identity_sha256(role_rows) != (
        context.validation_role_tensor_sha256
    ) or type(batch_ordinal) is not int or not (
        0 <= batch_ordinal < capabilities.VALIDATION_BATCH_COUNT
    ):
        raise RuntimeError("validation baseline role/context binding changed")
    start = batch_ordinal * runtime.BATCH_SIZE
    indices = tuple(range(start, start + runtime.BATCH_SIZE))
    inputs = role_rows[
        start:start + runtime.BATCH_SIZE, :runtime.SEQUENCE_LENGTH
    ].contiguous()
    return ValidationBaselineIdentity(
        source_commit=context.source_commit,
        inherited_snapshot_sha256=context.inherited_snapshot_sha256,
        rows_receipt_sha256=context.rows_receipt_sha256,
        validation_role_tensor_sha256=context.validation_role_tensor_sha256,
        common_support_sha256=validation_common_support_sha256(role_rows),
        ordered_batch_indices_sha256=runtime.logical_identity_sha256(list(indices)),
        ordered_input_tokens_sha256=runtime.tensor_identity_sha256(inputs),
        batch_ordinal=batch_ordinal,
    )


def mapped_control_key(control: str, route: str) -> str:
    if not valid_mapped_control(control, route):
        raise ValueError("mapped control/route pair is not registered")
    return f"{control}/{route}"


def required_mapped_control_keys() -> tuple[str, ...]:
    return tuple(
        mapped_control_key("document_shuffle", route)
        for route in fit.DOCUMENT_SHUFFLE_ROUTES
    ) + tuple(mapped_control_key(f"A_null_{index:02d}", "T") for index in range(20))


def orthogonal_gauge_bank() -> Mapping[str, torch.Tensor]:
    """Construct the four signed-permutation and four Haar replay gauges."""

    gauges: dict[str, torch.Tensor] = {}
    identity = torch.eye(runtime.CODE_DIM, dtype=torch.float64)
    for index, seed in enumerate(GAUGE_SIGNED_SEEDS):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(runtime.CODE_DIM, generator=generator)
        signs = 2 * torch.randint(
            0, 2, (runtime.CODE_DIM,), generator=generator, dtype=torch.long,
        ) - 1
        gauge = identity[:, permutation] * signs.double().unsqueeze(0)
        contract.validate_orthogonal_gauge(f"signed_permutation_{index}", gauge)
        gauges[f"signed_permutation_{index}"] = gauge.contiguous()
    for index, seed in enumerate(GAUGE_HAAR_SEEDS):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        normal = torch.randn(
            runtime.CODE_DIM, runtime.CODE_DIM, generator=generator,
            dtype=torch.float64,
        )
        gauge, upper = torch.linalg.qr(normal, mode="reduced")
        signs = torch.where(
            torch.diagonal(upper) < 0,
            torch.tensor(-1.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
        gauge = (gauge * signs.unsqueeze(0)).contiguous()
        if bool((torch.diagonal(upper) * signs < 0).any()):
            raise RuntimeError("Haar gauge R-diagonal convention failed")
        contract.validate_orthogonal_gauge(f"haar_{index}", gauge)
        gauges[f"haar_{index}"] = gauge
    expected = tuple(
        [f"signed_permutation_{index}" for index in range(4)]
        + [f"haar_{index}" for index in range(4)]
    )
    if tuple(gauges) != expected:
        raise RuntimeError("orthogonal gauge bank order changed")
    return MappingProxyType(gauges)


def intervention_assignments(role: str) -> Mapping[str, torch.Tensor]:
    """Freeze one position and balanced base-direction assignment for a fresh role."""

    seeds = {
        "validation": (2026083240, 2026083241),
        "final": (2026083250, 2026083251),
    }
    if role not in seeds:
        raise ValueError("intervention role must be validation or final")
    position_seed, permutation_seed = seeds[role]
    positions = torch.randint(
        runtime.SCORE_START, runtime.SCORE_STOP, (VALIDATION_ROWS,),
        generator=torch.Generator(device="cpu").manual_seed(position_seed),
        dtype=torch.long,
    )
    permutation = torch.randperm(
        VALIDATION_ROWS,
        generator=torch.Generator(device="cpu").manual_seed(permutation_seed),
    )
    direction_indices = torch.empty(VALIDATION_ROWS, dtype=torch.long)
    direction_indices[permutation] = torch.arange(VALIDATION_ROWS) % 32
    if not torch.equal(
        torch.bincount(direction_indices, minlength=32),
        torch.full((32,), VALIDATION_ROWS // 32, dtype=torch.long),
    ) or VALIDATION_ROWS != 192:
        raise RuntimeError("intervention directions are not exactly balanced")
    return MappingProxyType({
        "positions": positions.contiguous(),
        "row_permutation": permutation.contiguous(),
        "direction_indices": direction_indices.contiguous(),
    })


@dataclass(frozen=True)
class TransportInterventionGeometry:
    """Frozen selected-L0 fit covariance and its 32 normalized edit directions."""

    selected_l_program_sha256: str
    fit_role_tensor_sha256: str
    code_trajectory_sha256: str
    code_count: int
    mean: torch.Tensor
    covariance: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    clipped_eigenvalues: torch.Tensor
    clip_floor: float
    natural_rms: float
    raw_rademacher_signs: torch.Tensor
    normalized_directions: torch.Tensor

    def __post_init__(self) -> None:
        if any(not _sha256(value) for value in (
            self.selected_l_program_sha256, self.fit_role_tensor_sha256,
            self.code_trajectory_sha256,
        )) or type(self.code_count) is not int or self.code_count != (
            FIT_INTERVENTION_CODE_COUNT
        ) or not math.isfinite(self.clip_floor) or self.clip_floor <= 0 or not math.isfinite(
            self.natural_rms
        ) or self.natural_rms <= 0:
            raise ValueError("transport intervention geometry identity/scalars changed")
        shapes = {
            "mean": (runtime.CODE_DIM,),
            "covariance": (runtime.CODE_DIM, runtime.CODE_DIM),
            "eigenvalues": (runtime.CODE_DIM,),
            "eigenvectors": (runtime.CODE_DIM, runtime.CODE_DIM),
            "clipped_eigenvalues": (runtime.CODE_DIM,),
            "raw_rademacher_signs": (32, runtime.CODE_DIM),
            "normalized_directions": (32, runtime.CODE_DIM),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            expected_dtype = torch.long if name == "raw_rademacher_signs" else torch.float64
            if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != (
                expected_dtype
            ) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"transport intervention geometry {name} is malformed")
            object.__setattr__(self, name, value.detach().cpu().contiguous().clone())
        if not bool(((self.raw_rademacher_signs == -1) | (
            self.raw_rademacher_signs == 1
        )).all()) or bool((self.clipped_eigenvalues < self.clip_floor).any()) or bool((
            self.eigenvalues[1:] < self.eigenvalues[:-1]
        ).any()):
            raise ValueError("transport intervention signs or spectrum changed")
        identity = torch.eye(runtime.CODE_DIM, dtype=torch.float64)
        if float(torch.max(torch.abs(self.eigenvectors.T @ self.eigenvectors - identity))) > (
            2e-12
        ) or float(torch.max(torch.abs(self.covariance - self.covariance.T))) > 2e-12:
            raise ValueError("transport intervention eigensystem is not orthogonal/symmetric")
        replay = self.eigenvectors @ torch.diag(self.eigenvalues) @ self.eigenvectors.T
        if not torch.allclose(replay, self.covariance, rtol=2e-11, atol=2e-12):
            raise ValueError("transport intervention eigensystem does not replay covariance")
        direction_rms = torch.sqrt(torch.mean(self.normalized_directions.square(), dim=1))
        if not torch.allclose(
            direction_rms, torch.ones(32, dtype=torch.float64), rtol=2e-12, atol=2e-12,
        ):
            raise ValueError("transport intervention directions are not RMS normalized")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "selected_l_program_sha256": self.selected_l_program_sha256,
            "fit_role_tensor_sha256": self.fit_role_tensor_sha256,
            "code_trajectory_sha256": self.code_trajectory_sha256,
            "code_count": self.code_count, "clip_floor": self.clip_floor,
            "natural_rms": self.natural_rms,
            **{
                name: runtime.tensor_identity_sha256(getattr(self, name))
                for name in (
                    "mean", "covariance", "eigenvalues", "eigenvectors",
                    "clipped_eigenvalues", "raw_rademacher_signs",
                    "normalized_directions",
                )
            },
        })


def build_transport_intervention_geometry(
    selected_l0_codes: torch.Tensor, *, selected_l_program_sha256: str,
    fit_role_tensor_sha256: str,
) -> TransportInterventionGeometry:
    """Compute the preregistered covariance-shaped edit bank from selected-L0 codes."""

    if not _sha256(selected_l_program_sha256) or not _sha256(fit_role_tensor_sha256) or (
        not torch.is_tensor(selected_l0_codes)
    ) or selected_l0_codes.ndim != 3 or selected_l0_codes.shape[0] != (
        capabilities.FIT_ROW_COUNT
    ) or selected_l0_codes.shape[-1] != runtime.CODE_DIM or selected_l0_codes.shape[1] not in {
        runtime.SEQUENCE_LENGTH, runtime.SCORE_STOP - runtime.SCORE_START,
    } or not bool(torch.isfinite(selected_l0_codes).all()):
        raise ValueError("selected-L0 fit code trajectory is malformed")
    codes = runtime._canonical_code_support(selected_l0_codes).detach().cpu().double().contiguous()
    flat = codes.reshape(-1, runtime.CODE_DIM)
    if len(flat) != FIT_INTERVENTION_CODE_COUNT:
        raise RuntimeError("selected-L0 fit code support count changed")
    mean = flat.mean(dim=0)
    centered = flat - mean
    covariance = (centered.T @ centered / (len(flat) - 1)).contiguous()
    natural_rms = float(torch.sqrt(torch.mean(centered.square())))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    for column in range(runtime.CODE_DIM):
        pivot = int(torch.argmax(torch.abs(eigenvectors[:, column])))
        if float(eigenvectors[pivot, column]) < 0:
            eigenvectors[:, column].neg_()
    trace = float(torch.trace(covariance))
    clip_floor = 1e-12 * trace / runtime.CODE_DIM
    if not math.isfinite(natural_rms) or natural_rms <= 0 or not math.isfinite(
        clip_floor
    ) or clip_floor <= 0:
        raise RuntimeError("selected-L0 fit covariance is degenerate")
    clipped = torch.clamp(eigenvalues, min=clip_floor)
    covariance_sqrt = (
        eigenvectors @ torch.diag(torch.sqrt(clipped)) @ eigenvectors.T
    ).contiguous()
    signs = []
    for index in range(32):
        draw = torch.randint(
            0, 2, (runtime.CODE_DIM,), dtype=torch.long,
            generator=torch.Generator(device="cpu").manual_seed(2026083200 + index),
        )
        signs.append(2 * draw - 1)
    raw_signs = torch.stack(signs).contiguous()
    directions = raw_signs.double() @ covariance_sqrt
    rms = torch.sqrt(torch.mean(directions.square(), dim=1, keepdim=True))
    if bool((rms <= 0).any()) or not bool(torch.isfinite(rms).all()):
        raise RuntimeError("covariance-shaped intervention direction is degenerate")
    directions = (directions / rms).contiguous()
    return TransportInterventionGeometry(
        selected_l_program_sha256=selected_l_program_sha256,
        fit_role_tensor_sha256=fit_role_tensor_sha256,
        code_trajectory_sha256=runtime.tensor_identity_sha256(codes),
        code_count=len(flat), mean=mean.contiguous(), covariance=covariance,
        eigenvalues=eigenvalues.contiguous(), eigenvectors=eigenvectors.contiguous(),
        clipped_eigenvalues=clipped.contiguous(), clip_floor=clip_floor,
        natural_rms=natural_rms, raw_rademacher_signs=raw_signs,
        normalized_directions=directions,
    )


def select_teacher_calibration(
    teacher_median_kls: Mapping[float, float],
) -> Mapping[str, Any]:
    """Choose amplitude from teacher-only validation KL under the frozen rule."""

    if set(teacher_median_kls) != set(INTERVENTION_AMPLITUDES) or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        or float(value) < 0
        for value in teacher_median_kls.values()
    ):
        raise ValueError("teacher calibration must contain five finite nonnegative KLs")
    center = math.sqrt(CALIBRATION_BAND[0] * CALIBRATION_BAND[1])
    eligible = [
        amplitude for amplitude in INTERVENTION_AMPLITUDES
        if CALIBRATION_BAND[0] <= float(teacher_median_kls[amplitude]) <= (
            CALIBRATION_BAND[1]
        )
    ]
    pool = eligible if eligible else list(INTERVENTION_AMPLITUDES)
    selected = min(
        pool,
        key=lambda amplitude: (
            abs(float(teacher_median_kls[amplitude]) - center), amplitude,
        ),
    )
    return MappingProxyType({
        "selected_amplitude_multiplier": selected,
        "selected_teacher_median_kl": float(teacher_median_kls[selected]),
        "geometric_center": center,
        "calibration_passed": bool(eligible),
        "teacher_median_kls": MappingProxyType({
            amplitude: float(teacher_median_kls[amplitude])
            for amplitude in INTERVENTION_AMPLITUDES
        }),
    })


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


@dataclass(frozen=True)
class ValidationBaselineSufficientStatistics:
    """Raw deployed-N/N validation CE and copy baseline on the common support."""

    common_support_sha256: str
    row_ce_sum: torch.Tensor
    row_ce_count: torch.Tensor
    row_copy_ce_sum: torch.Tensor
    row_copy_count: torch.Tensor
    literal_early_mlp_calls: tuple[tuple[int, int], ...]
    native_guard_restored: bool
    native_guard_inert: bool

    def __post_init__(self) -> None:
        if not _sha256(self.common_support_sha256):
            raise ValueError("validation baseline common support is malformed")
        for name in ("row_ce_sum", "row_copy_ce_sum"):
            value = _row_vector(name, getattr(self, name), dtype=torch.float64)
            if bool((value < 0).any()):
                raise ValueError(f"validation baseline {name} is negative")
            object.__setattr__(self, name, value)
        for name in ("row_ce_count", "row_copy_count"):
            value = _row_vector(name, getattr(self, name), dtype=torch.long)
            if bool((value < 0).any()):
                raise ValueError(f"validation baseline {name} is negative")
            object.__setattr__(self, name, value)
        expected = torch.full(
            (VALIDATION_ROWS,), runtime.SCORE_STOP - runtime.SCORE_START,
            dtype=torch.long,
        )
        if not torch.equal(self.row_ce_count, expected) or int(self.row_copy_count.sum()) <= 0:
            raise ValueError("validation baseline CE/copy support changed")
        if self.literal_early_mlp_calls != ZERO_NATIVE_CALLS or (
            self.native_guard_restored is not True or self.native_guard_inert is not True
        ):
            raise ValueError("validation baseline did not use a clean deployed-N/N path")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "common_support_sha256": self.common_support_sha256,
            **{
                name: runtime.tensor_identity_sha256(getattr(self, name))
                for name in (
                    "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
                )
            },
            "literal_early_mlp_calls": self.literal_early_mlp_calls,
            "native_guard_restored": self.native_guard_restored,
            "native_guard_inert": self.native_guard_inert,
        })


@dataclass(frozen=True)
class MappedValidationSufficientStatistics:
    """Raw validation statistics additionally bound to one immutable row map."""

    control: str
    mapping_sha256: str
    base: ValidationSufficientStatistics

    def __post_init__(self) -> None:
        if not isinstance(self.base, ValidationSufficientStatistics) or not (
            valid_mapped_control(self.control, self.base.route)
        ) or not _sha256(self.mapping_sha256):
            raise ValueError("mapped validation sufficient-statistic identity changed")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "control": self.control,
            "mapping_sha256": self.mapping_sha256,
            "base_sha256": self.base.sha256,
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


@dataclass(frozen=True)
class MappedValidationScore:
    """Selector receipt whose map identity cannot be dropped or exchanged."""

    control: str
    mapping_sha256: str
    base: ValidationScore
    mapped_sufficient_statistics_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.base, ValidationScore) or not valid_mapped_control(
            self.control, self.base.route
        ) or not _sha256(self.mapping_sha256) or not _sha256(
            self.mapped_sufficient_statistics_sha256
        ):
            raise ValueError("mapped validation score identity changed")

    @property
    def route(self) -> str:
        return self.base.route

    @property
    def trial(self) -> int:
        return self.base.trial

    @property
    def primary_metric(self) -> float:
        return self.base.primary_metric

    @property
    def learning_rate(self) -> float:
        return self.base.learning_rate

    @property
    def program_sha256(self) -> str:
        return self.base.program_sha256

    @property
    def admissible(self) -> bool:
        return self.base.admissible


def mapped_validation_score_from_statistics(
    candidate: fit.MappedFitCandidate,
    statistics: MappedValidationSufficientStatistics,
) -> MappedValidationScore:
    """Recompute a mapped selector score while retaining control provenance."""

    if not isinstance(candidate, fit.MappedFitCandidate) or not isinstance(
        statistics, MappedValidationSufficientStatistics
    ) or candidate.control != statistics.control or candidate.mapping_sha256 != (
        statistics.mapping_sha256
    ) or candidate.route != statistics.base.route or candidate.final_program_sha256 != (
        statistics.base.program_sha256
    ):
        raise ValueError("mapped validation statistics differ from their fit candidate")
    # Prove the candidate state before allowing any scalar to enter selection.
    restore_mapped_fit_candidate(candidate)
    base = statistics.base
    primary = float(base.row_primary_sum.sum() / base.row_primary_count.sum())
    candidate_copy = base.row_copy_ce_sum.sum() / base.row_copy_count.sum()
    baseline_copy = base.baseline_row_copy_ce_sum.sum() / (
        base.baseline_row_copy_count.sum()
    )
    score = ValidationScore(
        route=candidate.route, trial=candidate.trial,
        learning_rate=candidate.learning_rate,
        program_sha256=candidate.final_program_sha256,
        metric_name=METRIC_BY_ROUTE[candidate.route], primary_metric=primary,
        copy_worsening=float(candidate_copy - baseline_copy),
        scored_token_count=int(base.row_primary_count.sum()),
        common_support_sha256=base.common_support_sha256,
        sufficient_statistics_sha256=base.sha256,
        student_original_calls=base.student_original_calls,
        hook_restored=base.hook_restored, hook_inert=base.hook_inert,
    )
    return MappedValidationScore(
        control=candidate.control, mapping_sha256=candidate.mapping_sha256,
        base=score, mapped_sufficient_statistics_sha256=statistics.sha256,
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

    if teacher_logits.ndim != 3 or student_logits.ndim != 3:
        raise ValueError("validation KL logits must be rank-three tensors")
    if teacher_logits.shape[1] == runtime.SEQUENCE_LENGTH:
        teacher_logits = runtime.scored_positions(teacher_logits)
    if student_logits.shape[1] == runtime.SEQUENCE_LENGTH:
        student_logits = runtime.scored_positions(student_logits)
    if teacher_logits.shape != student_logits.shape or teacher_logits.shape[1] != (
        runtime.SCORE_STOP - runtime.SCORE_START
    ) or (
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


def validation_common_support_sha256(role_rows: torch.Tensor) -> str:
    """Bind exact validation rows, shifted targets, copy mask, and score positions."""

    if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or role_rows.ndim != 2 \
            or role_rows.shape[0] != VALIDATION_ROWS or role_rows.shape[1] < (
        runtime.SEQUENCE_LENGTH + 1
    ):
        raise ValueError("common validation support requires all frozen role rows")
    rows = role_rows.detach().cpu().contiguous()
    targets = rows[:, 1:runtime.SEQUENCE_LENGTH + 1][
        :, runtime.SCORE_START:runtime.SCORE_STOP
    ].contiguous()
    mask = copy_mask(rows).contiguous()
    return runtime.logical_identity_sha256({
        "role": "early_mlp_suffix_transport_v1_validation",
        "row_count": VALIDATION_ROWS,
        "sequence_length": runtime.SEQUENCE_LENGTH,
        "score_start": runtime.SCORE_START,
        "score_stop": runtime.SCORE_STOP,
        "role_rows_sha256": runtime.tensor_identity_sha256(rows),
        "shifted_targets_sha256": runtime.tensor_identity_sha256(targets),
        "copy_mask_sha256": runtime.tensor_identity_sha256(mask),
    })


def _validation_batch_vector(
    name: str, value: torch.Tensor, *, dtype: torch.dtype,
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != (runtime.BATCH_SIZE,) or (
        value.dtype != dtype
    ) or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
        raise ValueError(f"validation batch statistic {name} changed shape, dtype, or finiteness")
    result = value.detach().cpu().contiguous().clone()
    if bool((result < 0).any()):
        raise ValueError(f"validation batch statistic {name} is negative")
    return result


class ValidationBaselineCollector:
    """Exactly-once deployed-N/N CE/copy assembly before candidate selection."""

    __slots__ = (
        "__common_support_sha256", "__next_batch", "__sealed", "__spent", "__vectors",
    )

    def __init__(self, *, common_support_sha256: str) -> None:
        object.__setattr__(self, "_ValidationBaselineCollector__sealed", False)
        if not _sha256(common_support_sha256) or VALIDATION_ROWS <= 0 or (
            VALIDATION_ROWS % runtime.BATCH_SIZE
        ):
            raise ValueError("validation baseline collector identity changed")
        self.__common_support_sha256 = common_support_sha256
        self.__next_batch = 0
        self.__spent = False
        self.__vectors = {
            name: [] for name in ("ce_sum", "ce_count", "copy_ce_sum", "copy_count")
        }
        object.__setattr__(self, "_ValidationBaselineCollector__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_ValidationBaselineCollector__sealed", False):
            raise AttributeError("validation baseline collector is sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("validation baseline collectors cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("validation baseline collectors cannot be copied")

    def __reduce__(self):
        raise RuntimeError("validation baseline collectors cannot be serialized")

    @property
    def completed_rows(self) -> int:
        return self.__next_batch * runtime.BATCH_SIZE

    def require_identity(self, *, common_support_sha256: str) -> None:
        if self.__spent or common_support_sha256 != self.__common_support_sha256:
            raise RuntimeError("validation baseline collector differs from the frozen support")

    def add_batch(
        self, *, batch_ordinal: int, ordered_row_indices: Sequence[int],
        row_ce_sum: torch.Tensor, row_ce_count: torch.Tensor,
        row_copy_ce_sum: torch.Tensor, row_copy_count: torch.Tensor,
        literal_early_mlp_calls: tuple[tuple[int, int], ...],
        native_guard_restored: bool, native_guard_inert: bool,
    ) -> None:
        if self.__spent:
            raise RuntimeError("validation baseline collector was already finalized")
        if type(batch_ordinal) is not int or batch_ordinal != self.__next_batch:
            raise RuntimeError("validation baseline batch is duplicated, missing, or out of order")
        start = batch_ordinal * runtime.BATCH_SIZE
        expected_indices = tuple(range(start, start + runtime.BATCH_SIZE))
        if tuple(ordered_row_indices) != expected_indices or expected_indices[-1] >= (
            VALIDATION_ROWS
        ):
            raise RuntimeError("validation baseline row identity changed")
        if literal_early_mlp_calls != ZERO_NATIVE_CALLS or native_guard_restored is not True or (
            native_guard_inert is not True
        ):
            raise RuntimeError("validation baseline native guard did not close cleanly")
        supplied = {
            "ce_sum": row_ce_sum, "ce_count": row_ce_count,
            "copy_ce_sum": row_copy_ce_sum, "copy_count": row_copy_count,
        }
        batch = {
            name: _validation_batch_vector(
                name, value,
                dtype=torch.float64 if name in {"ce_sum", "copy_ce_sum"} else torch.long,
            )
            for name, value in supplied.items()
        }
        expected_count = torch.full(
            (runtime.BATCH_SIZE,), runtime.SCORE_STOP - runtime.SCORE_START,
            dtype=torch.long,
        )
        if not torch.equal(batch["ce_count"], expected_count):
            raise RuntimeError("validation baseline CE support changed")
        for name, value in batch.items():
            self.__vectors[name].append(value)
        object.__setattr__(
            self, "_ValidationBaselineCollector__next_batch", self.__next_batch + 1,
        )

    def finalize(self) -> ValidationBaselineSufficientStatistics:
        if self.__spent:
            raise RuntimeError("validation baseline collector was already finalized")
        expected_batches = VALIDATION_ROWS // runtime.BATCH_SIZE
        if self.__next_batch != expected_batches or any(
            len(values) != expected_batches for values in self.__vectors.values()
        ):
            raise RuntimeError("validation baseline cannot finalize with missing batches")
        object.__setattr__(self, "_ValidationBaselineCollector__spent", True)
        joined = {
            name: torch.cat(values).contiguous() for name, values in self.__vectors.items()
        }
        for values in self.__vectors.values():
            values.clear()
        return ValidationBaselineSufficientStatistics(
            common_support_sha256=self.__common_support_sha256,
            row_ce_sum=joined["ce_sum"], row_ce_count=joined["ce_count"],
            row_copy_ce_sum=joined["copy_ce_sum"], row_copy_count=joined["copy_count"],
            literal_early_mlp_calls=ZERO_NATIVE_CALLS,
            native_guard_restored=True, native_guard_inert=True,
        )


class ValidationStatisticsCollector:
    """Exactly-once ordered assembly of raw validation batches.

    The collector sees only per-row reductions, never logits, labels, activations, or
    fit parameters.  A separately computed native baseline is frozen at construction;
    every candidate batch must reproduce its exact copy-mask counts.  Finalization is
    possible only after all 192 rows have been consumed in canonical order.
    """

    __slots__ = (
        "__baseline_copy_count", "__baseline_copy_sum", "__common_support_sha256",
        "__next_batch", "__program_sha256", "__route", "__sealed", "__spent",
        "__vectors",
    )

    _FLOAT_FIELDS = ("primary_sum", "ce_sum", "copy_ce_sum")
    _COUNT_FIELDS = ("primary_count", "ce_count", "copy_count")

    def __init__(
        self, *, route: str, program_sha256: str, common_support_sha256: str,
        baseline: ValidationBaselineSufficientStatistics,
    ) -> None:
        object.__setattr__(self, "_ValidationStatisticsCollector__sealed", False)
        if route not in SELECTABLE_ROUTES or not _sha256(program_sha256) or not _sha256(
            common_support_sha256
        ) or VALIDATION_ROWS <= 0 or VALIDATION_ROWS % runtime.BATCH_SIZE:
            raise ValueError("validation collector identity or batch partition changed")
        if not isinstance(baseline, ValidationBaselineSufficientStatistics) or (
            baseline.common_support_sha256 != common_support_sha256
        ):
            raise ValueError("validation collector baseline binding changed")
        baseline_sum = baseline.row_copy_ce_sum.clone()
        baseline_count = baseline.row_copy_count.clone()
        if bool((baseline_sum < 0).any()) or bool((baseline_count < 0).any()) or int(
            baseline_count.sum()
        ) <= 0:
            raise ValueError("validation copy baseline is negative or empty")
        self.__route = route
        self.__program_sha256 = program_sha256
        self.__common_support_sha256 = common_support_sha256
        self.__baseline_copy_sum = baseline_sum
        self.__baseline_copy_count = baseline_count
        self.__vectors = {
            name: [] for name in (*self._FLOAT_FIELDS, *self._COUNT_FIELDS)
        }
        self.__next_batch = 0
        self.__spent = False
        object.__setattr__(self, "_ValidationStatisticsCollector__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_ValidationStatisticsCollector__sealed", False):
            raise AttributeError("validation collector is sealed")
        object.__setattr__(self, name, value)

    def __copy__(self):
        raise RuntimeError("validation collectors cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("validation collectors cannot be copied")

    def __reduce__(self):
        raise RuntimeError("validation collectors cannot be serialized")

    @property
    def completed_rows(self) -> int:
        return self.__next_batch * runtime.BATCH_SIZE

    def require_identity(self, *, route: str, program_sha256: str) -> None:
        if self.__spent or route != self.__route or program_sha256 != self.__program_sha256:
            raise RuntimeError("validation collector differs from the executing program")

    def add_batch(
        self, *, batch_ordinal: int, ordered_row_indices: Sequence[int],
        row_primary_sum: torch.Tensor, row_primary_count: torch.Tensor,
        row_ce_sum: torch.Tensor, row_ce_count: torch.Tensor,
        row_copy_ce_sum: torch.Tensor, row_copy_count: torch.Tensor,
        student_original_calls: tuple[tuple[int, int], ...],
        hook_restored: bool, hook_inert: bool,
    ) -> None:
        if self.__spent:
            raise RuntimeError("validation collector was already finalized")
        if type(batch_ordinal) is not int or batch_ordinal != self.__next_batch:
            raise RuntimeError("validation batch is duplicated, missing, or out of order")
        start = batch_ordinal * runtime.BATCH_SIZE
        expected_indices = tuple(range(start, start + runtime.BATCH_SIZE))
        if tuple(ordered_row_indices) != expected_indices or expected_indices[-1] >= (
            VALIDATION_ROWS
        ):
            raise RuntimeError("validation batch row identity changed")
        if student_original_calls != ZERO_NATIVE_CALLS or hook_restored is not True or (
            hook_inert is not True
        ):
            raise RuntimeError("validation batch lacks a clean poisoned-student closure")
        supplied = {
            "primary_sum": row_primary_sum, "primary_count": row_primary_count,
            "ce_sum": row_ce_sum, "ce_count": row_ce_count,
            "copy_ce_sum": row_copy_ce_sum, "copy_count": row_copy_count,
        }
        batch = {
            name: _validation_batch_vector(
                name, supplied[name],
                dtype=torch.float64 if name in self._FLOAT_FIELDS else torch.long,
            )
            for name in supplied
        }
        expected_primary = torch.full(
            (runtime.BATCH_SIZE,), runtime.SCORE_STOP - runtime.SCORE_START,
            dtype=torch.long,
        )
        if not torch.equal(batch["primary_count"], expected_primary) or not torch.equal(
            batch["ce_count"], expected_primary
        ) or not torch.equal(
            batch["copy_count"], self.__baseline_copy_count[start:start + runtime.BATCH_SIZE]
        ):
            raise RuntimeError("validation batch primary/CE/copy support changed")
        for name, value in batch.items():
            self.__vectors[name].append(value)
        object.__setattr__(
            self, "_ValidationStatisticsCollector__next_batch", self.__next_batch + 1,
        )

    def finalize(self) -> ValidationSufficientStatistics:
        if self.__spent:
            raise RuntimeError("validation collector was already finalized")
        expected_batches = VALIDATION_ROWS // runtime.BATCH_SIZE
        if self.__next_batch != expected_batches or any(
            len(values) != expected_batches for values in self.__vectors.values()
        ):
            raise RuntimeError("validation collector cannot finalize with missing batches")
        object.__setattr__(self, "_ValidationStatisticsCollector__spent", True)
        joined = {
            name: torch.cat(values).contiguous() for name, values in self.__vectors.items()
        }
        for values in self.__vectors.values():
            values.clear()
        return ValidationSufficientStatistics(
            route=self.__route, program_sha256=self.__program_sha256,
            common_support_sha256=self.__common_support_sha256,
            row_primary_sum=joined["primary_sum"],
            row_primary_count=joined["primary_count"],
            row_ce_sum=joined["ce_sum"], row_ce_count=joined["ce_count"],
            row_copy_ce_sum=joined["copy_ce_sum"],
            row_copy_count=joined["copy_count"],
            baseline_row_copy_ce_sum=self.__baseline_copy_sum,
            baseline_row_copy_count=self.__baseline_copy_count,
            student_original_calls=ZERO_NATIVE_CALLS,
            hook_restored=True, hook_inert=True,
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


@dataclass(frozen=True)
class MappedScoredCandidate:
    """A negative-control candidate kept ineligible for true-route selection."""

    fit_candidate: fit.MappedFitCandidate
    validation: MappedValidationScore

    def __post_init__(self) -> None:
        candidate, score = self.fit_candidate, self.validation
        if not isinstance(candidate, fit.MappedFitCandidate) or not isinstance(
            score, MappedValidationScore
        ) or candidate.control != score.control or candidate.mapping_sha256 != (
            score.mapping_sha256
        ) or candidate.route != score.route or candidate.trial != score.trial or (
            candidate.learning_rate != score.learning_rate
        ) or candidate.final_program_sha256 != score.program_sha256:
            raise ValueError("mapped validation receipt differs from its fit candidate")
        restore_mapped_fit_candidate(candidate)

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


def select_mapped_candidate(
    candidates: Sequence[MappedScoredCandidate], *, control: str, route: str,
) -> MappedScoredCandidate:
    """Apply the registered selector within, never across, one negative-control map."""

    if not valid_mapped_control(control, route) or not candidates or any(
        candidate.validation.control != control
        or candidate.validation.route != route
        for candidate in candidates
    ):
        raise ValueError("mapped selection bank mixes controls, mappings, or routes")
    mappings = {candidate.validation.mapping_sha256 for candidate in candidates}
    if len(mappings) != 1:
        raise ValueError("mapped selection bank mixes controls, mappings, or routes")
    trials = [candidate.validation.trial for candidate in candidates]
    if sorted(trials) != list(range(len(runtime.LEARNING_RATES))):
        raise ValueError("mapped selection bank must contain each trial exactly once")
    eligible = [candidate for candidate in candidates if candidate.validation.admissible]
    if not eligible:
        raise RuntimeError("no mapped validation candidate satisfies the copy bound")
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
    validation_copy_worsening: float
    validation_common_support_sha256: str
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


@dataclass(frozen=True)
class FrozenMappedProgram:
    """Canonical deployable negative control with its row-map identity intact."""

    control: str
    mapping_sha256: str
    mapped_sufficient_statistics_sha256: str
    program: FrozenProgram

    def __post_init__(self) -> None:
        if not isinstance(self.program, FrozenProgram) or not valid_mapped_control(
            self.control, self.program.route
        ) or not _sha256(self.mapping_sha256) or not _sha256(
            self.mapped_sufficient_statistics_sha256
        ):
            raise ValueError("frozen mapped program identity changed")

    @property
    def key(self) -> str:
        return mapped_control_key(self.control, self.program.route)

    def make_program(self) -> runtime.JointAffineProgram:
        return self.program.make_program()


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
        validation_copy_worsening=candidate.validation.copy_worsening,
        validation_common_support_sha256=candidate.validation.common_support_sha256,
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


def freeze_mapped_selected(candidate: MappedScoredCandidate) -> FrozenMappedProgram:
    """Canonicalize a selected mapped control without converting it to a true arm."""

    if not isinstance(candidate, MappedScoredCandidate) or not candidate.validation.admissible:
        raise RuntimeError("mapped copy-inadmissible candidate cannot be frozen")
    mapped_candidate = candidate.fit_candidate
    proxy = fit.FitCandidate(
        route=mapped_candidate.route, trial=mapped_candidate.trial,
        learning_rate=mapped_candidate.learning_rate,
        completed_steps=mapped_candidate.completed_steps,
        loss_sum=mapped_candidate.loss_sum, loss_min=mapped_candidate.loss_min,
        loss_max=mapped_candidate.loss_max,
        final_program_sha256=mapped_candidate.final_program_sha256,
        transaction_history_sha256=mapped_candidate.transaction_history_sha256,
        state_dict=mapped_candidate.state_dict,
    )
    true_score = candidate.validation.base
    frozen = freeze_selected(ScoredCandidate(proxy, true_score))
    return FrozenMappedProgram(
        control=mapped_candidate.control,
        mapping_sha256=mapped_candidate.mapping_sha256,
        mapped_sufficient_statistics_sha256=(
            candidate.validation.mapped_sufficient_statistics_sha256
        ),
        program=frozen,
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


def select_and_freeze_mapped_controls(
    candidates: Mapping[str, Sequence[MappedScoredCandidate]],
) -> Mapping[str, FrozenMappedProgram]:
    """Require and freeze all four shuffled routes and all twenty A-null/T maps."""

    required = required_mapped_control_keys()
    if set(candidates) != set(required):
        raise ValueError("mapped control bank must contain exactly 24 registered families")
    frozen = {}
    document_mappings = set()
    all_mappings = set()
    for key in required:
        control, route = key.split("/", 1)
        selected = select_mapped_candidate(
            candidates[key], control=control, route=route,
        )
        value = freeze_mapped_selected(selected)
        if value.key != key:
            raise RuntimeError("frozen mapped control key changed")
        frozen[key] = value
        all_mappings.add(value.mapping_sha256)
        if control == "document_shuffle":
            document_mappings.add(value.mapping_sha256)
    if len(document_mappings) != 1 or len(all_mappings) != 21:
        raise RuntimeError("mapped control plans are duplicated or inconsistent")
    return MappingProxyType(frozen)


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


def required_validation_candidate_keys() -> tuple[str, ...]:
    true = tuple(
        f"true/{route}/trial{trial}"
        for route in SELECTABLE_ROUTES for trial in range(len(runtime.LEARNING_RATES))
    )
    mapped = tuple(
        f"{key}/trial{trial}"
        for key in required_mapped_control_keys()
        for trial in range(len(runtime.LEARNING_RATES))
    )
    return (*true, *mapped)


@dataclass(frozen=True)
class ValidationExecutionManifest:
    """Completeness commitment for all 87 candidate and 48 baseline evaluations."""

    validation_role_tensor_sha256: str
    common_support_sha256: str
    baseline_statistics_sha256: str
    baseline_batch_receipt_sha256s: tuple[str, ...]
    candidate_batch_receipt_sha256s: Mapping[str, tuple[str, ...]]
    candidate_statistics_sha256s: Mapping[str, str]
    broker_ledger_sha256s: Mapping[str, str]

    def __post_init__(self) -> None:
        if any(not _sha256(value) for value in (
            self.validation_role_tensor_sha256, self.common_support_sha256,
            self.baseline_statistics_sha256,
        )) or len(self.baseline_batch_receipt_sha256s) != (
            capabilities.VALIDATION_BATCH_COUNT
        ) or any(not _sha256(value) for value in self.baseline_batch_receipt_sha256s):
            raise ValueError("validation execution manifest role/baseline changed")
        required = required_validation_candidate_keys()
        required_set = set(required)
        if set(self.candidate_batch_receipt_sha256s) != required_set or set(
            self.candidate_statistics_sha256s
        ) != required_set or set(self.broker_ledger_sha256s) != required_set:
            raise ValueError("validation execution manifest candidate bank is incomplete")
        receipts = {}
        statistics = {}
        ledgers = {}
        for key in required:
            values = tuple(self.candidate_batch_receipt_sha256s[key])
            if len(values) != capabilities.VALIDATION_BATCH_COUNT or any(
                not _sha256(value) for value in values
            ) or not _sha256(self.candidate_statistics_sha256s[key]) or not _sha256(
                self.broker_ledger_sha256s[key]
            ):
                raise ValueError("validation execution manifest hash/count changed")
            receipts[key] = values
            statistics[key] = self.candidate_statistics_sha256s[key]
            ledgers[key] = self.broker_ledger_sha256s[key]
        object.__setattr__(
            self, "candidate_batch_receipt_sha256s", MappingProxyType(receipts),
        )
        object.__setattr__(
            self, "candidate_statistics_sha256s", MappingProxyType(statistics),
        )
        object.__setattr__(self, "broker_ledger_sha256s", MappingProxyType(ledgers))

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "validation_role_tensor_sha256": self.validation_role_tensor_sha256,
            "common_support_sha256": self.common_support_sha256,
            "baseline_statistics_sha256": self.baseline_statistics_sha256,
            "baseline_batch_receipt_sha256s": list(self.baseline_batch_receipt_sha256s),
            "candidate_batch_receipt_sha256s": {
                key: list(self.candidate_batch_receipt_sha256s[key])
                for key in required_validation_candidate_keys()
            },
            "candidate_statistics_sha256s": dict(self.candidate_statistics_sha256s),
            "broker_ledger_sha256s": dict(self.broker_ledger_sha256s),
        })


def _frozen_program_payload(value: FrozenProgram) -> dict[str, Any]:
    if not isinstance(value, FrozenProgram):
        raise TypeError("canonical bank requires frozen programs")
    value.make_program()
    return {
        "route": value.route, "trial": value.trial,
        "learning_rate": value.learning_rate,
        "validation_metric_name": value.validation_metric_name,
        "validation_metric": value.validation_metric,
        "validation_copy_worsening": value.validation_copy_worsening,
        "validation_common_support_sha256": value.validation_common_support_sha256,
        "validation_sufficient_statistics_sha256": (
            value.validation_sufficient_statistics_sha256
        ),
        "source_program_sha256": value.source_program_sha256,
        "source_tensor_sha256": value.source_tensor_sha256,
        "canonical_tensor_sha256": value.canonical_tensor_sha256,
        "site_states": {
            str(site): {
                name: item.detach().cpu().contiguous().clone() if torch.is_tensor(item) else item
                for name, item in state.items()
            }
            for site, state in value.site_states.items()
        },
        "cross": None if value.cross is None else value.cross.detach().cpu().contiguous().clone(),
        "svd_max_errors": list(value.svd_max_errors),
    }


def _geometry_payload(value: TransportInterventionGeometry) -> dict[str, Any]:
    if not isinstance(value, TransportInterventionGeometry):
        raise TypeError("canonical bank requires transport intervention geometry")
    return {
        "selected_l_program_sha256": value.selected_l_program_sha256,
        "fit_role_tensor_sha256": value.fit_role_tensor_sha256,
        "code_trajectory_sha256": value.code_trajectory_sha256,
        "code_count": value.code_count, "clip_floor": value.clip_floor,
        "natural_rms": value.natural_rms, "geometry_sha256": value.sha256,
        **{
            name: getattr(value, name).detach().cpu().contiguous().clone()
            for name in (
                "mean", "covariance", "eigenvalues", "eigenvectors",
                "clipped_eigenvalues", "raw_rademacher_signs", "normalized_directions",
            )
        },
    }


def _payload_identity(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"tensor_sha256": runtime.tensor_identity_sha256(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical program payload mappings require string keys")
        return {key: _payload_identity(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_payload_identity(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"canonical program payload contains unsupported {type(value).__name__}")


def build_canonical_program_bank(
    *, true_programs: Mapping[str, FrozenProgram],
    mapped_programs: Mapping[str, FrozenMappedProgram],
    validation_baseline: ValidationBaselineSufficientStatistics,
    validation_execution: ValidationExecutionManifest,
    transport_geometry: TransportInterventionGeometry,
    teacher_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete tensor payload which must be frozen before final loading."""

    if set(true_programs) != set(SELECTABLE_ROUTES) or tuple(mapped_programs) != (
        required_mapped_control_keys()
    ) or not isinstance(validation_baseline, ValidationBaselineSufficientStatistics) or (
        not isinstance(validation_execution, ValidationExecutionManifest)
    ) or validation_baseline.sha256 != validation_execution.baseline_statistics_sha256 or (
        validation_baseline.common_support_sha256 != validation_execution.common_support_sha256
    ):
        raise ValueError("canonical program bank is incomplete or support-mixed")
    common_support = validation_execution.common_support_sha256
    for route in SELECTABLE_ROUTES:
        value = true_programs[route]
        if not isinstance(value, FrozenProgram) or value.route != route or (
            value.validation_common_support_sha256 != common_support
        ):
            raise ValueError("canonical true program route/support changed")
        key = f"true/{route}/trial{value.trial}"
        if validation_execution.candidate_statistics_sha256s[key] != (
            value.validation_sufficient_statistics_sha256
        ):
            raise RuntimeError("selected true program is absent from validation manifest")
    for key in required_mapped_control_keys():
        value = mapped_programs[key]
        if not isinstance(value, FrozenMappedProgram) or value.key != key or (
            value.program.validation_common_support_sha256 != common_support
        ):
            raise ValueError("canonical mapped program route/support changed")
        manifest_key = f"{key}/trial{value.program.trial}"
        if validation_execution.candidate_statistics_sha256s[manifest_key] != (
            value.mapped_sufficient_statistics_sha256
        ):
            raise RuntimeError("selected mapped program is absent from validation manifest")
    document_mappings = {
        mapped_programs[f"document_shuffle/{route}"].mapping_sha256
        for route in fit.DOCUMENT_SHUFFLE_ROUTES
    }
    all_mappings = {value.mapping_sha256 for value in mapped_programs.values()}
    if len(document_mappings) != 1 or len(all_mappings) != 21:
        raise RuntimeError("canonical mapped control plans are duplicated or inconsistent")
    if transport_geometry.selected_l_program_sha256 != true_programs[
        "L"
    ].canonical_tensor_sha256:
        raise RuntimeError("transport geometry was not built from selected L")
    if not isinstance(teacher_calibration, Mapping) or set(teacher_calibration) != {
        "selected_amplitude_multiplier", "selected_teacher_median_kl",
        "geometric_center", "calibration_passed", "teacher_median_kls",
    } or not isinstance(teacher_calibration["teacher_median_kls"], Mapping):
        raise RuntimeError("teacher calibration schema differs from the frozen selection rule")
    calibration = select_teacher_calibration(teacher_calibration["teacher_median_kls"])
    if any(
        teacher_calibration[key] != calibration[key]
        for key in calibration if key != "teacher_median_kls"
    ) or dict(teacher_calibration["teacher_median_kls"]) != dict(
        calibration["teacher_median_kls"]
    ):
        raise RuntimeError("teacher calibration differs from the frozen selection rule")
    gauges = orthogonal_gauge_bank()
    assignments = {
        role: intervention_assignments(role) for role in ("validation", "final")
    }
    body = {
        "schema_version": 1,
        "kind": "early_mlp_suffix_transport_v1_canonical_program_bank",
        "true_programs": {
            route: _frozen_program_payload(true_programs[route]) for route in SELECTABLE_ROUTES
        },
        "mapped_programs": {
            key: {
                "control": mapped_programs[key].control,
                "mapping_sha256": mapped_programs[key].mapping_sha256,
                "mapped_sufficient_statistics_sha256": (
                    mapped_programs[key].mapped_sufficient_statistics_sha256
                ),
                "program": _frozen_program_payload(mapped_programs[key].program),
            }
            for key in required_mapped_control_keys()
        },
        "validation_baseline": {
            "common_support_sha256": validation_baseline.common_support_sha256,
            "baseline_sha256": validation_baseline.sha256,
            **{
                name: getattr(validation_baseline, name).clone()
                for name in (
                    "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
                )
            },
        },
        "validation_execution": {
            "manifest_sha256": validation_execution.sha256,
            "validation_role_tensor_sha256": (
                validation_execution.validation_role_tensor_sha256
            ),
            "common_support_sha256": common_support,
            "baseline_statistics_sha256": validation_execution.baseline_statistics_sha256,
            "baseline_batch_receipt_sha256s": list(
                validation_execution.baseline_batch_receipt_sha256s
            ),
            "candidate_batch_receipt_sha256s": {
                key: list(validation_execution.candidate_batch_receipt_sha256s[key])
                for key in required_validation_candidate_keys()
            },
            "candidate_statistics_sha256s": dict(
                validation_execution.candidate_statistics_sha256s
            ),
            "broker_ledger_sha256s": dict(validation_execution.broker_ledger_sha256s),
        },
        "transport_geometry": _geometry_payload(transport_geometry),
        "gauge_bank": {name: value.clone() for name, value in gauges.items()},
        "intervention_assignments": {
            role: {name: value.clone() for name, value in assignments[role].items()}
            for role in ("validation", "final")
        },
        "teacher_calibration": {
            key: value for key, value in calibration.items() if key != "teacher_median_kls"
        } | {
            "teacher_median_kls": {
                str(amplitude): calibration["teacher_median_kls"][amplitude]
                for amplitude in INTERVENTION_AMPLITUDES
            },
        },
    }
    return {**body, "payload_sha256": runtime.logical_identity_sha256(_payload_identity(body))}


def _exact_payload_keys(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"canonical {name} payload schema changed")
    return value


def _restore_frozen_program_payload(
    value: Any, *, expected_route: str,
) -> FrozenProgram:
    payload = _exact_payload_keys(value, {
        "route", "trial", "learning_rate", "validation_metric_name",
        "validation_metric", "validation_copy_worsening",
        "validation_common_support_sha256",
        "validation_sufficient_statistics_sha256", "source_program_sha256",
        "source_tensor_sha256", "canonical_tensor_sha256", "site_states", "cross",
        "svd_max_errors",
    }, "frozen program")
    trial = payload["trial"]
    if payload["route"] != expected_route or expected_route not in SELECTABLE_ROUTES or (
        type(trial) is not int
    ) or trial not in range(len(runtime.LEARNING_RATES)) or payload["learning_rate"] != (
        runtime.LEARNING_RATES[trial]
    ) or payload["validation_metric_name"] != METRIC_BY_ROUTE[expected_route] or any(
        not isinstance(payload[name], (int, float)) or not math.isfinite(float(payload[name]))
        for name in ("validation_metric", "validation_copy_worsening")
    ) or float(payload["validation_metric"]) < 0 or float(
        payload["validation_copy_worsening"]
    ) > 0.01 or any(not _sha256(payload[name]) for name in (
        "validation_common_support_sha256",
        "validation_sufficient_statistics_sha256", "source_program_sha256",
        "source_tensor_sha256", "canonical_tensor_sha256",
    )):
        raise RuntimeError("canonical frozen program identity changed")
    states_payload = _exact_payload_keys(payload["site_states"], {"0", "1"}, "site states")
    states: dict[int, Mapping[str, Any]] = {}
    for site in (0, 1):
        state = _exact_payload_keys(states_payload[str(site)], {
            "grammar", "interface", "mean", "scale", "left", "right", "bias",
        }, f"site{site} state")
        states[site] = MappingProxyType({
            name: item.detach().cpu().contiguous().clone() if torch.is_tensor(item) else item
            for name, item in state.items()
        })
    errors = payload["svd_max_errors"]
    if not isinstance(errors, (tuple, list)) or len(errors) != 2 or any(
        not isinstance(error, (int, float)) or not math.isfinite(float(error))
        or not 0 <= float(error) <= 2e-6 for error in errors
    ):
        raise RuntimeError("canonical frozen program SVD receipt changed")
    cross = payload["cross"]
    if cross is not None:
        if not torch.is_tensor(cross) or tuple(cross.shape) != (
            runtime.CODE_DIM, runtime.CODE_DIM
        ) or cross.dtype != torch.float32 or not bool(torch.isfinite(cross).all()):
            raise RuntimeError("canonical frozen program cross tensor changed")
        cross = cross.detach().cpu().contiguous().clone()
    if (expected_route == "T") != (cross is not None):
        raise RuntimeError("canonical frozen program cross route changed")
    frozen = FrozenProgram(
        route=expected_route, trial=trial,
        learning_rate=float(payload["learning_rate"]),
        validation_metric_name=payload["validation_metric_name"],
        validation_metric=float(payload["validation_metric"]),
        validation_copy_worsening=float(payload["validation_copy_worsening"]),
        validation_common_support_sha256=payload["validation_common_support_sha256"],
        validation_sufficient_statistics_sha256=(
            payload["validation_sufficient_statistics_sha256"]
        ),
        source_program_sha256=payload["source_program_sha256"],
        source_tensor_sha256=payload["source_tensor_sha256"],
        canonical_tensor_sha256=payload["canonical_tensor_sha256"],
        site_states=MappingProxyType(states), cross=cross,
        svd_max_errors=(float(errors[0]), float(errors[1])),
    )
    frozen.make_program()
    return frozen


def validate_canonical_program_bank_payload(value: Any) -> Mapping[str, Any]:
    """Reconstruct and replay every typed invariant after artifact deserialization."""

    payload = _exact_payload_keys(value, {
        "schema_version", "kind", "true_programs", "mapped_programs",
        "validation_baseline", "validation_execution", "transport_geometry",
        "gauge_bank", "intervention_assignments", "teacher_calibration",
        "payload_sha256",
    }, "program bank")
    if payload["schema_version"] != 1 or payload["kind"] != (
        "early_mlp_suffix_transport_v1_canonical_program_bank"
    ) or not _sha256(payload["payload_sha256"]):
        raise RuntimeError("canonical program bank header changed")
    body = {key: payload[key] for key in payload if key != "payload_sha256"}
    observed_identity = runtime.logical_identity_sha256(_payload_identity(body))
    if observed_identity != payload["payload_sha256"]:
        raise RuntimeError("canonical program bank payload hash changed")

    true_payload = _exact_payload_keys(
        payload["true_programs"], set(SELECTABLE_ROUTES), "true program bank",
    )
    true_programs = MappingProxyType({
        route: _restore_frozen_program_payload(true_payload[route], expected_route=route)
        for route in SELECTABLE_ROUTES
    })
    mapped_payload = _exact_payload_keys(
        payload["mapped_programs"], set(required_mapped_control_keys()),
        "mapped program bank",
    )
    mapped_programs: dict[str, FrozenMappedProgram] = {}
    for key in required_mapped_control_keys():
        control, route = key.split("/", 1)
        entry = _exact_payload_keys(mapped_payload[key], {
            "control", "mapping_sha256", "mapped_sufficient_statistics_sha256", "program",
        }, "mapped program")
        if entry["control"] != control:
            raise RuntimeError("canonical mapped program control changed")
        mapped_programs[key] = FrozenMappedProgram(
            control=control, mapping_sha256=entry["mapping_sha256"],
            mapped_sufficient_statistics_sha256=(
                entry["mapped_sufficient_statistics_sha256"]
            ),
            program=_restore_frozen_program_payload(entry["program"], expected_route=route),
        )
    mapped_programs_proxy = MappingProxyType(mapped_programs)

    baseline_payload = _exact_payload_keys(payload["validation_baseline"], {
        "common_support_sha256", "baseline_sha256", "row_ce_sum", "row_ce_count",
        "row_copy_ce_sum", "row_copy_count",
    }, "validation baseline")
    baseline = ValidationBaselineSufficientStatistics(
        common_support_sha256=baseline_payload["common_support_sha256"],
        row_ce_sum=baseline_payload["row_ce_sum"],
        row_ce_count=baseline_payload["row_ce_count"],
        row_copy_ce_sum=baseline_payload["row_copy_ce_sum"],
        row_copy_count=baseline_payload["row_copy_count"],
        literal_early_mlp_calls=ZERO_NATIVE_CALLS,
        native_guard_restored=True, native_guard_inert=True,
    )
    if baseline.sha256 != baseline_payload["baseline_sha256"]:
        raise RuntimeError("canonical validation baseline hash changed")

    execution_payload = _exact_payload_keys(payload["validation_execution"], {
        "manifest_sha256", "validation_role_tensor_sha256", "common_support_sha256",
        "baseline_statistics_sha256", "baseline_batch_receipt_sha256s",
        "candidate_batch_receipt_sha256s", "candidate_statistics_sha256s",
        "broker_ledger_sha256s",
    }, "validation execution")
    execution = ValidationExecutionManifest(
        validation_role_tensor_sha256=execution_payload["validation_role_tensor_sha256"],
        common_support_sha256=execution_payload["common_support_sha256"],
        baseline_statistics_sha256=execution_payload["baseline_statistics_sha256"],
        baseline_batch_receipt_sha256s=tuple(
            execution_payload["baseline_batch_receipt_sha256s"]
        ),
        candidate_batch_receipt_sha256s={
            key: tuple(execution_payload["candidate_batch_receipt_sha256s"][key])
            for key in required_validation_candidate_keys()
        },
        candidate_statistics_sha256s=execution_payload["candidate_statistics_sha256s"],
        broker_ledger_sha256s=execution_payload["broker_ledger_sha256s"],
    )
    if execution.sha256 != execution_payload["manifest_sha256"]:
        raise RuntimeError("canonical validation execution manifest hash changed")

    geometry_payload = _exact_payload_keys(payload["transport_geometry"], {
        "selected_l_program_sha256", "fit_role_tensor_sha256",
        "code_trajectory_sha256", "code_count", "clip_floor", "natural_rms",
        "geometry_sha256", "mean", "covariance", "eigenvalues", "eigenvectors",
        "clipped_eigenvalues", "raw_rademacher_signs", "normalized_directions",
    }, "transport geometry")
    geometry = TransportInterventionGeometry(
        selected_l_program_sha256=geometry_payload["selected_l_program_sha256"],
        fit_role_tensor_sha256=geometry_payload["fit_role_tensor_sha256"],
        code_trajectory_sha256=geometry_payload["code_trajectory_sha256"],
        code_count=geometry_payload["code_count"], mean=geometry_payload["mean"],
        covariance=geometry_payload["covariance"],
        eigenvalues=geometry_payload["eigenvalues"],
        eigenvectors=geometry_payload["eigenvectors"],
        clipped_eigenvalues=geometry_payload["clipped_eigenvalues"],
        clip_floor=geometry_payload["clip_floor"], natural_rms=geometry_payload["natural_rms"],
        raw_rademacher_signs=geometry_payload["raw_rademacher_signs"],
        normalized_directions=geometry_payload["normalized_directions"],
    )
    if geometry.sha256 != geometry_payload["geometry_sha256"]:
        raise RuntimeError("canonical transport geometry hash changed")

    gauge_payload = _exact_payload_keys(
        payload["gauge_bank"], set(orthogonal_gauge_bank()), "gauge bank",
    )
    for name, expected in orthogonal_gauge_bank().items():
        if not torch.is_tensor(gauge_payload[name]) or not torch.equal(
            gauge_payload[name], expected
        ):
            raise RuntimeError("canonical gauge bank changed")
    assignment_payload = _exact_payload_keys(
        payload["intervention_assignments"], {"validation", "final"},
        "intervention assignments",
    )
    for role in ("validation", "final"):
        observed = _exact_payload_keys(
            assignment_payload[role], {"positions", "row_permutation", "direction_indices"},
            f"{role} intervention assignment",
        )
        expected = intervention_assignments(role)
        if any(not torch.is_tensor(observed[name]) or not torch.equal(
            observed[name], expected[name]
        ) for name in expected):
            raise RuntimeError("canonical intervention assignment changed")

    calibration_payload = _exact_payload_keys(payload["teacher_calibration"], {
        "selected_amplitude_multiplier", "selected_teacher_median_kl",
        "geometric_center", "calibration_passed", "teacher_median_kls",
    }, "teacher calibration")
    median_payload = _exact_payload_keys(
        calibration_payload["teacher_median_kls"],
        {str(amplitude) for amplitude in INTERVENTION_AMPLITUDES},
        "teacher median KL",
    )
    calibration = select_teacher_calibration({
        amplitude: median_payload[str(amplitude)] for amplitude in INTERVENTION_AMPLITUDES
    })
    rebuilt = build_canonical_program_bank(
        true_programs=true_programs, mapped_programs=mapped_programs_proxy,
        validation_baseline=baseline, validation_execution=execution,
        transport_geometry=geometry, teacher_calibration={
            **{key: calibration_payload[key] for key in calibration_payload if key != (
                "teacher_median_kls"
            )},
            "teacher_median_kls": {
                amplitude: median_payload[str(amplitude)]
                for amplitude in INTERVENTION_AMPLITUDES
            },
        },
    )
    if _payload_identity(rebuilt) != _payload_identity(payload):
        raise RuntimeError("canonical program bank did not replay after deserialization")
    return MappingProxyType({
        "true_programs": true_programs,
        "mapped_programs": mapped_programs_proxy,
        "validation_baseline": baseline,
        "validation_execution": execution,
        "transport_geometry": geometry,
        "teacher_calibration": calibration,
        "payload_sha256": payload["payload_sha256"],
    })
