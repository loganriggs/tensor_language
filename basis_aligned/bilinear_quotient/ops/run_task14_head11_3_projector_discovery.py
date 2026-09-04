#!/usr/bin/env python3
"""Program A for the frozen Task 14 head-11.3 projector experiment.

This module owns the leakage boundary, rank ladder, health/selection gates, and
create-only receipt format.  Model-specific collection and finite fitting are
deliberately behind ``ProgramABackend`` so the complete lifecycle can be tested
without loading a checkpoint.  No production backend is installed yet; the CLI
therefore supports only an import-safe ``--dry-run``.

Program A never parses the token-bearing Task 14 authority.  It hash-checks that
file as an opaque dependency, then materializes only the preregistered DISCOVERY
relations from the donor metadata.  A later production backend must consume a
separate DISCOVERY-only endpoint shard and is never given the outer partition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch

import task14_causal_spectral_rank_one as spectral
import task14_head11_3_projector_adapter as adapter


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "task14_head11_3_projector_program_a_v1"
EXPERIMENT_ID = "task14-head11-3-causal-projector-program-a-v1"
SITE_ID = adapter.SITE_ID

PREREG_PATH = ROOT.parent / "polynomial_causal/TASK14_HEAD11_3_CAUSAL_PROJECTOR_PREREGISTRATION.md"
EXECUTION_ADDENDUM_PATH = ROOT.parent / (
    "polynomial_causal/"
    "TASK14_HEAD11_3_CAUSAL_PROJECTOR_EXECUTION_ADDENDUM_2026-09-04_1645.md"
)
IMPLEMENTATION_AMENDMENT_PATH = ROOT.parent / (
    "polynomial_causal/"
    "TASK14_HEAD11_3_CAUSAL_PROJECTOR_IMPLEMENTATION_AMENDMENT_2026-09-04_1705.md"
)
AUDIT_AMENDMENT_PATH = ROOT.parent / (
    "polynomial_causal/"
    "TASK14_HEAD11_3_CAUSAL_PROJECTOR_AUDIT_AMENDMENT_2026-09-04_1720.md"
)
AUTHORITY_PATH = ROOT / "ops/circuit_battery_task14_agreement_fit_authority.json"
PARTITION_PATH = ROOT / "ops/circuit_battery_task14_fit_localization_partition_v2.json"
DONORS_PATH = ROOT / "ops/circuit_battery_task14_fit_localization_donors_v2.json"
ADAPTER_PATH = ROOT / "ops/task14_head11_3_projector_adapter.py"
SPECTRAL_PATH = ROOT / "ops/task14_causal_spectral_rank_one.py"
DISCOVERY_SHARD_PATH = ROOT / "ops/task14_projector_discovery_endpoint_shard_v1.json"
BACKEND_PATH = ROOT / "ops/task14_program_a_torch_backend.py"
FACADE_PATH = ROOT.parent / "polynomial_causal/bilin18_observed_model_facade.py"
DAS_LIBRARY_PATH = ROOT / "ops/das_shared_private_lib.py"
PROGRAM_A_RECEIPT_PATH = ROOT / "circuits/followups/task14_head11_3_causal_projector_program_a_v1_receipt.json"
PROGRAM_A_BUNDLE_PATH = ROOT / "circuits/followups/task14_head11_3_causal_projector_program_a_v1_bundle.pt"

SOURCE_PATHS = {
    "preregistration": PREREG_PATH,
    "execution_addendum": EXECUTION_ADDENDUM_PATH,
    "implementation_amendment": IMPLEMENTATION_AMENDMENT_PATH,
    "audit_amendment": AUDIT_AMENDMENT_PATH,
    "authority_opaque": AUTHORITY_PATH,
    "partition": PARTITION_PATH,
    "donors": DONORS_PATH,
    "projector_adapter": ADAPTER_PATH,
    "causal_spectral": SPECTRAL_PATH,
    "discovery_endpoint_shard": DISCOVERY_SHARD_PATH,
    "production_backend": BACKEND_PATH,
    "observed_model_facade": FACADE_PATH,
    "das_shared_private_library": DAS_LIBRARY_PATH,
}
EXPECTED_SOURCE_SHA256 = {
    "preregistration": "dc0749a48c6c21cf00115d44804d7a8e24d411c7f202765ac84f41a1e5ce24ac",
    "execution_addendum": "32e25dc298a80203a689666e407215b0e997989ad37d8eefbd84dc9e9ae085e7",
    "implementation_amendment": "5121f191ff4d616ee796e838eda844284d07b94e4ece88faf20582a403fadc7e",
    "audit_amendment": "17982c8da1d881d48a53a99cc46c3b4c41a9509996f6f4375b3492da2459d14b",
    "authority_opaque": "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f",
    "partition": "1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3",
    "donors": "ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a",
    "projector_adapter": "70602b0589aa8e0125ec26362a8c4f7ec42308c0c9042438ece589e451c0a2c2",
    "causal_spectral": "667dc100d7a936f85ed36557da333f02965f3dc300a1bbcc3520550f667aea40",
    "discovery_endpoint_shard": "1e3b9a204c08a9c6af4ea7f5668abba719fd1943a8a7e7df0dc488f3183f4e1b",
    "production_backend": "df91d8e06d9b0b1df21add8bb492b64a01c1a6a7203dd81275c16a21e871e4c8",
    "observed_model_facade": "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    "das_shared_private_library": "edcf3d750e8fbdcb2ae479bcc6e68bd7ccc5078217b62cf981570656b6a773e4",
}

FIT_GROUP_NUMBERS = frozenset((0, 9, 10, 11, 16, 25, 26, 27))
SELECT_GROUP_NUMBERS = frozenset((1, 4, 6, 15, 17, 20, 22, 31))
EXPECTED_GROUP_IDS = {
    "FIT": frozenset((
        "FIT:3b8a8a2defa026c4b1db", "FIT:954c958556f0f60294a2",
        "FIT:dbcd436adab06f21d17f", "FIT:016b1dd5d94969200cee",
        "FIT:8213eb1bb96931cd6c90", "FIT:feb3e9f6c7f3710fb3c6",
        "FIT:f7ff6ba750fc31662220", "FIT:b5e90891726c976e6fc4",
    )),
    "SELECT": frozenset((
        "FIT:79ef6e8ea73eba41ed03", "FIT:63cb303e93945a616f95",
        "FIT:4ae7a1e95552fd10549e", "FIT:28084a8c3b51605d5f97",
        "FIT:e5f5ad8619988eb26a6b", "FIT:ff577036cc6749d665f0",
        "FIT:6a8a707d9bdbbf9f3339", "FIT:f94e02b80c73de0e29a2",
    )),
}
EXPECTED_ORDINAL_SHA256 = {
    "FIT": "5c24f97e98de6ff351514e19586d6ec4e72b5d1af6a3d5971d3d0b7d1b2267db",
    "SELECT": "4b7de6802d6f6fd23c669cde5276e57987ba552dd2b1ac67068c21fea9c0f823",
}
EXPECTED_COUNTS = {
    "FIT": {"relations": 153, "targets": 116, "controls": 37, "endpoints": 64},
    "SELECT": {"relations": 145, "targets": 106, "controls": 39, "endpoints": 64},
}

PRIMARY_RANKS = (1, 2, 4)
PRIMARY_STARTS = (0, 1, 2)
CONFIRMATION_STARTS = (3, 4)
PERMUTATION_IDS = (0, 1)
HAAR_SEEDS = tuple(range(141300, 141316))
UPDATES = 100
BATCH_SIZE = 32
MAX_ORTHONORMALITY_ERROR = 1e-5
MIN_PROJECTOR_MOVEMENT = 0.02
MIN_FIT_IMPROVEMENT = 0.05
MIN_HEALTHY_PER_RANK = 2
MIN_FINAL_PASSING_STARTS = 4
MIN_TARGET_HEAD_FRACTION = 0.80
MIN_TARGET_DIRECTION_FRACTION = 0.75
MIN_TARGET_NATIVE_RECOVERY = 0.40
MIN_COORDINATED_NATIVE_RECOVERY = 0.35
MAX_CONTROL_MOVEMENT = 0.10
MAX_CONTROL_ABOVE_FULL_HEAD = 0.025
MAX_CONTROL_VOCAB_RMS = 0.10
MIN_RANDOM_ADVANTAGE = 0.10
MAX_EFFECT_STABILITY_MEDIAN = 0.10
MAX_EFFECT_STABILITY_P90 = 0.20

PRIMARY_PRICE = {
    "forward_calls": 1206,
    "backward_calls": 902,
    "example_evaluations": 37700,
    "stored_frame_bytes": 141824,
}
RANK8_INCREMENTAL_PRICE = {
    "forward_calls": 395,
    "backward_calls": 300,
    "example_evaluations": 12355,
    "stored_frame_bytes": 12288,
}
CONDITIONAL_PRICE = {
    "forward_calls": 420,
    "backward_calls": 400,
    "example_evaluations": 13380,
}


class ProgramAError(ValueError):
    """Program A's immutable authority or staged lifecycle is invalid."""


@dataclass(frozen=True)
class Relation:
    ordinal: int
    record_id: str
    target_endpoint_id: str
    donor_endpoint_id: str
    cell_key: str
    role: str


@dataclass(frozen=True)
class DiscoveryPlan:
    fit: tuple[Relation, ...]
    select: tuple[Relation, ...]
    source_sha256: Mapping[str, str]
    ordinal_sha256: Mapping[str, str]


@dataclass(frozen=True)
class SpectralInputs:
    ordinals: tuple[int, ...]
    cell_keys: tuple[str, ...]
    head_deltas: torch.Tensor
    downstream_gradients: torch.Tensor
    full_head_effects: torch.Tensor
    source_partitions: tuple[str, ...] = ("DISCOVERY",)
    validation_records_seen: int = 0
    validation_token_sequences_seen: int = 0
    model_counts: Mapping[str, int] | None = None


@dataclass(frozen=True)
class FitHealth:
    finite: bool
    model_parameter_gradients_absent: bool
    checkpoint_hash_unchanged: bool
    hook_exact: bool
    replay_rank0_exact: bool
    replay_rank128_exact: bool
    orthonormality_error: float
    normalized_projector_movement: float
    first20_objective: float
    final20_objective: float
    schedule_updates: int


@dataclass(frozen=True)
class FitObjectiveConfig:
    target_coefficient: float = 1.0
    control_coefficient: float = 1.0
    huber_transition: float = 0.5
    denominator_floor: float = 1e-6
    control_normalizer: str = "detached_median_positive_fit_target_full_head_effect"
    full_vocabulary_size: int = 50304
    target_draws_per_update: int = 16
    control_draws_per_update: int = 16
    sampling: str = "uniform_cell_then_uniform_relation_with_replacement"
    learning_rate: float = 0.03
    learning_rate_schedule: str = "0.03*(1+cos(pi*t/99))/2"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    orthogonal_parametrization: str = "torch_householder"
    schedule_seed_base: int = 14114000
    permutation_rule: str = "within_cell_sha256_balanced_plus_minus_target_labels"


FIT_OBJECTIVE = FitObjectiveConfig()


@dataclass(frozen=True)
class TargetCellScore:
    full_head_fraction: float
    direction_fraction: float
    native_donor_recovery: float
    coordinated_subject_cell: bool = False


@dataclass(frozen=True)
class ControlCellScore:
    normalized_margin_movement: float
    full_head_normalized_movement: float
    normalized_full_vocabulary_rms: float


@dataclass(frozen=True)
class FitResult:
    rank: int
    start: int
    frame: torch.Tensor
    health: FitHealth
    target_cells: Mapping[str, TargetCellScore]
    control_cells: Mapping[str, ControlCellScore]
    normalized_row_effects: tuple[float, ...]
    model_counts: Mapping[str, int]
    source_partitions: tuple[str, ...] = ("DISCOVERY",)
    validation_records_seen: int = 0
    validation_token_sequences_seen: int = 0
    scored_ordinals: tuple[int, ...] = ()
    normalized_row_effect_ordinals: tuple[int, ...] = ()


class ProgramABackend(Protocol):
    """Model boundary. Implementations receive DISCOVERY relation metadata only."""

    def collect_spectral_inputs(self, relations: Sequence[Relation]) -> SpectralInputs: ...

    def fit_and_score(
        self,
        *,
        fit_relations: Sequence[Relation],
        select_relations: Sequence[Relation],
        rank: int,
        start: int,
        initial_frame: torch.Tensor,
        updates: int,
        batch_size: int,
        objective: FitObjectiveConfig,
        permutation_id: int | None,
    ) -> FitResult: ...

    def score_fixed_frame(
        self, *, select_relations: Sequence[Relation], frame: torch.Tensor, control_id: str
    ) -> FitResult: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise ProgramAError(f"source is not a JSON object: {path}")
    return value


def _check_sources() -> dict[str, str]:
    observed = {name: _sha256(path) for name, path in SOURCE_PATHS.items()}
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        if observed[name] != expected:
            raise ProgramAError(
                f"immutable source changed for {name}: expected={expected}, observed={observed[name]}"
            )
    return observed


def compile_discovery_plan() -> DiscoveryPlan:
    """Compile the inner split without parsing any token-bearing authority rows."""

    source_hashes = _check_sources()
    partition = _load_json(PARTITION_PATH)
    donors = _load_json(DONORS_PATH)
    partition_records = partition.get("records")
    donor_records = donors.get("records")
    endpoints = donors.get("endpoints")
    if not isinstance(partition_records, list) or not isinstance(donor_records, list) \
            or not isinstance(endpoints, list):
        raise ProgramAError("partition/donor metadata lacks canonical records")

    discovery_groups = {
        int(row["group_number"]): str(row["group_id"])
        for row in partition_records
        if isinstance(row, dict) and row.get("partition") == "DISCOVERY"
    }
    if set(discovery_groups) != FIT_GROUP_NUMBERS | SELECT_GROUP_NUMBERS:
        raise ProgramAError("DISCOVERY group census changed")
    for split, numbers in (("FIT", FIT_GROUP_NUMBERS), ("SELECT", SELECT_GROUP_NUMBERS)):
        if {discovery_groups[n] for n in numbers} != EXPECTED_GROUP_IDS[split]:
            raise ProgramAError(f"{split} group IDs changed")

    # Keep only endpoint metadata for the two DISCOVERY halves.  Prompt text and
    # token IDs are not present in this donor manifest and never enter Program A.
    endpoint_meta: dict[str, tuple[int, int]] = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            raise ProgramAError("malformed endpoint metadata")
        number = int(endpoint["group_number"])
        if number in discovery_groups:
            endpoint_meta[str(endpoint["endpoint_id"])] = (
                number, int(endpoint["subject_state"])
            )
    if len(endpoint_meta) != 128:
        raise ProgramAError("DISCOVERY endpoint census changed")

    materialized: dict[str, list[Relation]] = {"FIT": [], "SELECT": []}
    for record in donor_records:
        if not isinstance(record, dict) or record.get("partition") != "DISCOVERY":
            continue
        target_id, donor_id = str(record["target_endpoint_id"]), str(record["donor_endpoint_id"])
        if target_id not in endpoint_meta or donor_id not in endpoint_meta:
            raise ProgramAError("DISCOVERY relation references a non-DISCOVERY endpoint")
        target_group, target_state = endpoint_meta[target_id]
        donor_group, _ = endpoint_meta[donor_id]
        split = None
        if target_group in FIT_GROUP_NUMBERS and donor_group in FIT_GROUP_NUMBERS:
            split = "FIT"
        elif target_group in SELECT_GROUP_NUMBERS and donor_group in SELECT_GROUP_NUMBERS:
            split = "SELECT"
        if split is None:
            continue
        expected_relation = str(record["expected_relation"])
        if expected_relation == "opposite_subject_toward_donor":
            role = "target"
        elif expected_relation == "same_subject_zero_projected_effect":
            role = "control"
        else:
            raise ProgramAError(f"unknown relation role: {expected_relation}")
        cell_key = "|".join((
            str(record["arm"]), str(record["family"]), str(record["matching"]), str(target_state)
        ))
        materialized[split].append(Relation(
            ordinal=int(record["ordinal"]), record_id=str(record["record_id"]),
            target_endpoint_id=target_id, donor_endpoint_id=donor_id,
            cell_key=cell_key, role=role,
        ))

    for split in ("FIT", "SELECT"):
        rows = sorted(materialized[split], key=lambda row: row.ordinal)
        materialized[split] = rows
        ordinals = [row.ordinal for row in rows]
        if _canonical_sha(ordinals) != EXPECTED_ORDINAL_SHA256[split]:
            raise ProgramAError(f"{split} retained ordinal hash changed")
        counts = {
            "relations": len(rows),
            "targets": sum(row.role == "target" for row in rows),
            "controls": sum(row.role == "control" for row in rows),
            "endpoints": len({value for row in rows for value in (
                row.target_endpoint_id, row.donor_endpoint_id
            )}),
        }
        if counts != EXPECTED_COUNTS[split]:
            raise ProgramAError(f"{split} counts changed: {counts}")
    fit_endpoints = {x for row in materialized["FIT"] for x in (
        row.target_endpoint_id, row.donor_endpoint_id
    )}
    select_endpoints = {x for row in materialized["SELECT"] for x in (
        row.target_endpoint_id, row.donor_endpoint_id
    )}
    if fit_endpoints & select_endpoints:
        raise ProgramAError("FIT and SELECT endpoints overlap")
    return DiscoveryPlan(
        fit=tuple(materialized["FIT"]), select=tuple(materialized["SELECT"]),
        source_sha256=source_hashes, ordinal_sha256=dict(EXPECTED_ORDINAL_SHA256),
    )


def _require_discovery_access(
    *, source_partitions: Sequence[str], validation_records_seen: int,
    validation_token_sequences_seen: int,
) -> None:
    if tuple(source_partitions) != ("DISCOVERY",):
        raise ProgramAError("Program A backend accessed a non-DISCOVERY partition")
    if validation_records_seen != 0 or validation_token_sequences_seen != 0:
        raise ProgramAError("Program A backend accessed VALIDATION records or token sequences")


def _equal_cell_spectral_operator(inputs: SpectralInputs) -> torch.Tensor:
    _require_discovery_access(
        source_partitions=inputs.source_partitions,
        validation_records_seen=inputs.validation_records_seen,
        validation_token_sequences_seen=inputs.validation_token_sequences_seen,
    )
    if len(inputs.ordinals) != len(inputs.cell_keys):
        raise ProgramAError("spectral ordinal/cell alignment changed")
    if len(set(inputs.ordinals)) != len(inputs.ordinals):
        raise ProgramAError("spectral ordinals are not unique")
    cells = sorted(set(inputs.cell_keys))
    if not cells:
        raise ProgramAError("spectral input has no target cells")
    cell_counts = {cell: inputs.cell_keys.count(cell) for cell in cells}
    weights = torch.tensor(
        [1.0 / (len(cells) * cell_counts[cell]) for cell in inputs.cell_keys],
        dtype=torch.float64,
    )
    operator = spectral.causal_spectral_rank_one(
        inputs.head_deltas,
        inputs.downstream_gradients,
        inputs.full_head_effects,
        sample_weights=weights,
    ).operator
    if tuple(operator.shape) != (128, 128) or not bool(torch.isfinite(operator).all()):
        raise ProgramAError("cell-balanced spectral operator is malformed")
    return operator


def top_algebraic_frame(operator: torch.Tensor, rank: int) -> torch.Tensor:
    """Return the largest algebraic eigenvectors, with deterministic signs."""
    if rank not in (1, 2, 4, 8) or tuple(operator.shape) != (128, 128):
        raise ProgramAError("spectral operator/rank is outside the frozen ladder")
    values, vectors = torch.linalg.eigh(operator.to(dtype=torch.float64, device="cpu"))
    if not bool(torch.isfinite(values).all() and torch.isfinite(vectors).all()):
        raise ProgramAError("cell-balanced spectral eigendecomposition is nonfinite")
    frame = vectors[:, -rank:].clone()
    for column in range(rank):
        frame[:, column] = spectral.canonicalize_direction_sign(frame[:, column])
    adapter.validate_head_frame(frame)
    return frame


def deterministic_initial_frame(
    analytic_frame: torch.Tensor, *, rank: int, start: int
) -> torch.Tensor:
    if rank not in (1, 2, 4, 8) or start not in range(5):
        raise ProgramAError("rank/start is outside the frozen ladder")
    adapter.validate_head_frame(analytic_frame)
    if tuple(analytic_frame.shape) != (128, rank):
        raise ProgramAError("analytic frame does not match the opened rank")
    generator = torch.Generator(device="cpu").manual_seed(14113100 + 100 * rank + start)
    perturbation = torch.randn(128, rank, generator=generator, dtype=torch.float64)
    raw = analytic_frame.to(dtype=torch.float64) + perturbation * 1e-2
    frame, triangular = torch.linalg.qr(raw, mode="reduced")
    signs = torch.where(torch.diag(triangular) < 0, -1.0, 1.0)
    frame = frame * signs[None, :]
    adapter.validate_head_frame(frame)
    return frame


def deterministic_haar_frames(rank: int) -> tuple[torch.Tensor, ...]:
    frames = []
    for seed in HAAR_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed + 1000 * rank)
        raw = torch.randn(128, rank, generator=generator, dtype=torch.float64)
        frame, triangular = torch.linalg.qr(raw, mode="reduced")
        signs = torch.where(torch.diag(triangular) < 0, -1.0, 1.0)
        frames.append(frame * signs[None, :])
    return tuple(frames)


def fit_is_healthy(result: FitResult) -> bool:
    h = result.health
    return (
        h.finite and h.model_parameter_gradients_absent and h.checkpoint_hash_unchanged
        and h.hook_exact and h.replay_rank0_exact and h.replay_rank128_exact
        and math.isfinite(h.orthonormality_error)
        and h.orthonormality_error <= MAX_ORTHONORMALITY_ERROR
        and math.isfinite(h.normalized_projector_movement)
        and h.normalized_projector_movement >= MIN_PROJECTOR_MOVEMENT
        and math.isfinite(h.first20_objective) and math.isfinite(h.final20_objective)
        and h.first20_objective - h.final20_objective >= MIN_FIT_IMPROVEMENT
        and h.schedule_updates == UPDATES
    )


def target_cells_pass(result: FitResult) -> bool:
    if not result.target_cells:
        return False
    for value in result.target_cells.values():
        floor = MIN_COORDINATED_NATIVE_RECOVERY if value.coordinated_subject_cell \
            else MIN_TARGET_NATIVE_RECOVERY
        if not (
            math.isfinite(value.full_head_fraction)
            and value.full_head_fraction >= MIN_TARGET_HEAD_FRACTION
            and math.isfinite(value.direction_fraction)
            and value.direction_fraction >= MIN_TARGET_DIRECTION_FRACTION
            and math.isfinite(value.native_donor_recovery)
            and value.native_donor_recovery >= floor
        ):
            return False
    return True


def control_cells_pass(result: FitResult) -> bool:
    if not result.control_cells:
        return False
    for value in result.control_cells.values():
        if not (
            math.isfinite(value.normalized_margin_movement)
            and value.normalized_margin_movement <= MAX_CONTROL_MOVEMENT
            and math.isfinite(value.full_head_normalized_movement)
            and value.normalized_margin_movement
                <= value.full_head_normalized_movement + MAX_CONTROL_ABOVE_FULL_HEAD
            and math.isfinite(value.normalized_full_vocabulary_rms)
            and value.normalized_full_vocabulary_rms <= MAX_CONTROL_VOCAB_RMS
        ):
            return False
    return True


def _random_bar_passes(candidate: FitResult, random_scores: Sequence[FitResult]) -> bool:
    if len(random_scores) != 16 or not candidate.target_cells:
        return False
    candidate_worst = min(x.full_head_fraction for x in candidate.target_cells.values())
    random_worst = [
        min(x.full_head_fraction for x in score.target_cells.values())
        for score in random_scores if score.target_cells
    ]
    if len(random_worst) != 16 or not all(math.isfinite(x) for x in random_worst):
        return False
    return candidate_worst >= float(np.quantile(random_worst, 0.95)) + MIN_RANDOM_ADVANTAGE


def _effect_stability(passing: Sequence[FitResult]) -> tuple[float, float, bool]:
    differences: list[float] = []
    for i, left in enumerate(passing):
        for right in passing[i + 1:]:
            if len(left.normalized_row_effects) != len(right.normalized_row_effects):
                raise ProgramAError("seed effect fingerprints have different lengths")
            differences.extend(abs(a - b) for a, b in zip(
                left.normalized_row_effects, right.normalized_row_effects
            ))
    if not differences:
        return math.inf, math.inf, False
    median, p90 = float(np.median(differences)), float(np.quantile(differences, 0.90))
    return median, p90, (
        median <= MAX_EFFECT_STABILITY_MEDIAN and p90 <= MAX_EFFECT_STABILITY_P90
    )


def _rank8_licensed(
    rank4: Sequence[FitResult], random_scores: Sequence[FitResult]
) -> bool:
    """Open rank 8 only for a control-only or stability-only rank-4 miss."""
    target_random = [
        result for result in rank4
        if fit_is_healthy(result) and target_cells_pass(result)
        and _random_bar_passes(result, random_scores)
    ]
    if any(not control_cells_pass(result) for result in target_random):
        return True
    otherwise_passing = [result for result in target_random if control_cells_pass(result)]
    return len(otherwise_passing) >= 2 and not _effect_stability(otherwise_passing)[2]


def _expected_score_cells(
    relations: Sequence[Relation],
) -> tuple[set[str], set[str], dict[str, bool]]:
    targets = {row.cell_key for row in relations if row.role == "target"}
    controls = {row.cell_key for row in relations if row.role == "control"}
    coordinated = {
        cell: cell.startswith("C_to_ordinary_singular|C|") for cell in targets
    }
    return targets, controls, coordinated


def _validate_model_counts(counts: Mapping[str, int]) -> None:
    required = {"forward_calls", "backward_calls", "example_evaluations"}
    if set(counts) != required or any(
        type(counts[key]) is not int or counts[key] < 0 for key in required
    ):
        raise ProgramAError("backend model counts are not exact nonnegative integers")


def _check_score_provenance(
    result: FitResult,
    *,
    expected_relations: Sequence[Relation],
    expected_rank: int,
) -> None:
    _require_discovery_access(
        source_partitions=result.source_partitions,
        validation_records_seen=result.validation_records_seen,
        validation_token_sequences_seen=result.validation_token_sequences_seen,
    )
    expected_ordinals = tuple(row.ordinal for row in expected_relations)
    if result.scored_ordinals != expected_ordinals:
        raise ProgramAError("backend score does not cover the exact inner SELECT ordinals")
    target_ordinals = tuple(row.ordinal for row in expected_relations if row.role == "target")
    if result.normalized_row_effect_ordinals != target_ordinals \
            or len(result.normalized_row_effects) != len(target_ordinals) \
            or not all(math.isfinite(value) for value in result.normalized_row_effects):
        raise ProgramAError("backend row effects do not cover the exact ordered SELECT targets")
    target_cells, control_cells, coordinated = _expected_score_cells(expected_relations)
    if set(result.target_cells) != target_cells or set(result.control_cells) != control_cells:
        raise ProgramAError("backend score omits or invents an exact SELECT cell")
    if any(
        result.target_cells[cell].coordinated_subject_cell != expected
        for cell, expected in coordinated.items()
    ):
        raise ProgramAError("backend coordinated-subject cell flags changed")
    if result.rank != expected_rank or tuple(result.frame.shape) != (128, expected_rank):
        raise ProgramAError("backend score has the wrong rank or frame shape")
    adapter.validate_head_frame(result.frame)
    _validate_model_counts(result.model_counts)


def _fit_one(
    backend: ProgramABackend, plan: DiscoveryPlan, analytic_operator: torch.Tensor,
    *, rank: int, start: int, permutation_id: int | None = None,
) -> FitResult:
    result = backend.fit_and_score(
        fit_relations=plan.fit, select_relations=plan.select, rank=rank, start=start,
        initial_frame=deterministic_initial_frame(
            top_algebraic_frame(analytic_operator, rank), rank=rank, start=start
        ),
        updates=UPDATES, batch_size=BATCH_SIZE, permutation_id=permutation_id,
        objective=FIT_OBJECTIVE,
    )
    if result.rank != rank or result.start != start or tuple(result.frame.shape) != (128, rank):
        raise ProgramAError("backend returned a misidentified fit")
    _check_score_provenance(
        result,
        expected_relations=plan.select,
        expected_rank=rank,
    )
    return result


def _score_fixed(
    backend: ProgramABackend,
    plan: DiscoveryPlan,
    *,
    frame: torch.Tensor,
    control_id: str,
) -> FitResult:
    result = backend.score_fixed_frame(
        select_relations=plan.select,
        frame=frame,
        control_id=control_id,
    )
    _check_score_provenance(
        result,
        expected_relations=plan.select,
        expected_rank=frame.shape[1],
    )
    return result


def execute_program_a(
    backend: ProgramABackend, *, receipt_path: Path, bundle_path: Path
) -> dict[str, object]:
    """Run the frozen lifecycle against an injected backend and create outputs."""

    plan = compile_discovery_plan()
    inputs = backend.collect_spectral_inputs(tuple(x for x in plan.fit if x.role == "target"))
    expected = tuple(x.ordinal for x in plan.fit if x.role == "target")
    if inputs.ordinals != expected or any("VALIDATION" in key for key in inputs.cell_keys):
        raise ProgramAError("backend spectral inputs escaped the exact FIT targets")
    analytic_operator = _equal_cell_spectral_operator(inputs)

    fits: dict[int, list[FitResult]] = {}
    random_scores: dict[int, list[FitResult]] = {}
    for rank in PRIMARY_RANKS:
        fits[rank] = [_fit_one(backend, plan, analytic_operator, rank=rank, start=start)
                      for start in PRIMARY_STARTS]
        random_scores[rank] = [_score_fixed(
            backend, plan, frame=frame, control_id=f"haar-r{rank}-{index}"
        ) for index, frame in enumerate(deterministic_haar_frames(rank))]

    opened_rank8 = _rank8_licensed(fits[4], random_scores[4])
    if opened_rank8:
        fits[8] = [_fit_one(backend, plan, analytic_operator, rank=8, start=start)
                   for start in PRIMARY_STARTS]
        random_scores[8] = [_score_fixed(
            backend, plan, frame=frame, control_id=f"haar-r8-{index}"
        ) for index, frame in enumerate(deterministic_haar_frames(8))]

    provisional_rank = None
    for rank in sorted(fits):
        initial = [x for x in fits[rank] if fit_is_healthy(x)]
        passing = [x for x in initial if target_cells_pass(x) and control_cells_pass(x)
                   and _random_bar_passes(x, random_scores[rank])]
        if len(initial) >= MIN_HEALTHY_PER_RANK and len(passing) >= 2:
            provisional_rank = rank
            break

    permutation_fits: list[FitResult] = []
    permutations_fail: bool | None = None
    selected_rank = None
    selected_start = None
    stability = (math.inf, math.inf, False)
    final_passing: list[FitResult] = []
    instrument_invalid_reasons = [
        f"rank_{rank}_has_fewer_than_{MIN_HEALTHY_PER_RANK}_healthy_primary_fits"
        for rank in sorted(fits)
        if sum(fit_is_healthy(result) for result in fits[rank]) < MIN_HEALTHY_PER_RANK
    ]
    if provisional_rank is not None:
        fits[provisional_rank].extend(
            _fit_one(backend, plan, analytic_operator, rank=provisional_rank, start=start)
            for start in CONFIRMATION_STARTS
        )
        final_passing = [x for x in fits[provisional_rank] if fit_is_healthy(x)
                         and target_cells_pass(x) and control_cells_pass(x)
                         and _random_bar_passes(x, random_scores[provisional_rank])]
        stability = _effect_stability(final_passing)
        permutation_fits = [
            _fit_one(backend, plan, analytic_operator, rank=provisional_rank, start=index,
                     permutation_id=permutation_id)
            for index, permutation_id in enumerate(PERMUTATION_IDS)
        ]
        if sum(fit_is_healthy(result) for result in fits[provisional_rank]) \
                < MIN_FINAL_PASSING_STARTS:
            instrument_invalid_reasons.append(
                f"rank_{provisional_rank}_has_fewer_than_{MIN_FINAL_PASSING_STARTS}_healthy_total_fits"
            )
        if not all(fit_is_healthy(result) for result in permutation_fits):
            instrument_invalid_reasons.append("permutation_fit_health_failed")
        permutations_fail = all(fit_is_healthy(x) and not (
            target_cells_pass(x) and control_cells_pass(x)
            and _random_bar_passes(x, random_scores[provisional_rank])
        ) for x in permutation_fits)
        if not instrument_invalid_reasons and len(final_passing) >= MIN_FINAL_PASSING_STARTS \
                and stability[2] and permutations_fail:
            selected_rank = provisional_rank
            selected_start = min(x.start for x in final_passing)

    ordinary_fits = [x for rank in sorted(fits) for x in fits[rank]]
    all_fits = ordinary_fits + permutation_fits
    frames = {
        "analytic_equal_cell_operator": analytic_operator.detach().cpu().to(dtype=torch.float64)
    }
    frames.update({
        f"rank{result.rank}_start{result.start}_permnone":
            result.frame.detach().cpu().to(dtype=torch.float32)
        for result in ordinary_fits
    })
    for permutation_id, result in zip(PERMUTATION_IDS, permutation_fits):
        frames[f"rank{result.rank}_start{result.start}_perm{permutation_id}"] = (
            result.frame.detach().cpu().to(dtype=torch.float32)
        )
    counted_results = all_fits + [x for rank in sorted(random_scores) for x in random_scores[rank]]
    counts = {key: sum(int(x.model_counts.get(key, 0)) for x in counted_results)
              for key in ("forward_calls", "backward_calls", "example_evaluations")}
    if inputs.model_counts is not None:
        _validate_model_counts(inputs.model_counts)
        counts = {key: counts[key] + int(inputs.model_counts.get(key, 0)) for key in counts}
    expected_price = dict(PRIMARY_PRICE)
    if opened_rank8:
        expected_price = {key: expected_price[key] + RANK8_INCREMENTAL_PRICE[key]
                          for key in expected_price}
    if provisional_rank is not None:
        for key, value in CONDITIONAL_PRICE.items():
            expected_price[key] += value
        expected_price["stored_frame_bytes"] += 2048 * provisional_rank
    if any(counts[key] > expected_price[key] for key in counts):
        raise ProgramAError("backend model counts exceed the compatible execution ceiling")

    projector_overlaps = []
    for index, left in enumerate(final_passing):
        for right in final_passing[index + 1:]:
            rank = left.rank
            raw = float(torch.linalg.matrix_norm(left.frame.T @ right.frame).square() / rank)
            chance = rank / 128.0
            corrected = (raw - chance) / (1.0 - chance)
            if not math.isfinite(corrected):
                raise ProgramAError("chance-corrected projector overlap is nonfinite")
            projector_overlaps.append({
                "left_start": left.start, "right_start": right.start,
                "raw_overlap": raw, "chance_corrected_overlap": corrected,
            })

    def summarize(result: FitResult) -> dict[str, object]:
        return {
            "rank": result.rank, "start": result.start,
            "frame_sha256": _tensor_sha(result.frame),
            "healthy": fit_is_healthy(result), "health": asdict(result.health),
            "target_cells": {key: asdict(value) for key, value in result.target_cells.items()},
            "control_cells": {key: asdict(value) for key, value in result.control_cells.items()},
            "target_cells_pass": target_cells_pass(result),
            "control_cells_pass": control_cells_pass(result),
            "scored_ordinals_sha256": _canonical_sha(list(result.scored_ordinals)),
            "model_counts": dict(result.model_counts),
        }

    nonidentification_reasons: list[str] = []
    if provisional_rank is not None and selected_rank is None \
            and not instrument_invalid_reasons:
        if len(final_passing) < MIN_FINAL_PASSING_STARTS:
            nonidentification_reasons.append("confirmation_fits_did_not_pass")
        if not stability[2]:
            nonidentification_reasons.append("causal_effects_not_stable_across_starts")
        if permutations_fail is False:
            nonidentification_reasons.append("permutation_control_not_rejected")
        if not nonidentification_reasons:
            nonidentification_reasons.append("registered_selection_gate_failed")
    terminal = (
        "instrument_invalid" if instrument_invalid_reasons else
        "program_a_selected" if selected_rank is not None else
        "program_a_not_identified" if provisional_rank is not None else
        "small_linear_subspace_null"
    )
    receipt: dict[str, object] = {
        "schema": SCHEMA, "experiment_id": EXPERIMENT_ID, "site_id": SITE_ID,
        "terminal": terminal,
        # pred_a: every registered source, replay, coverage, optimization-health,
        # and execution-count check must hold; otherwise the instrument is invalid.
        "pred_a_instrument_health": terminal != "instrument_invalid",
        # pred_b: a causally selective projector is identified at the smallest
        # licensed rank passing every finite SELECT/control/random/stability gate.
        "pred_b_causal_projector_selected": terminal == "program_a_selected",
        # pred_c: if all licensed healthy ranks miss, the registered small-linear-
        # subspace hypothesis is null rather than extended into another rank sweep.
        "pred_c_small_subspace_null": terminal == "small_linear_subspace_null",
        "instrument_invalid_reasons": instrument_invalid_reasons,
        "nonidentification_reasons": nonidentification_reasons,
        "program_b_opened": False, "validation_rows_loaded": 0,
        "source_sha256": dict(plan.source_sha256),
        "ordinal_sha256": dict(plan.ordinal_sha256),
        "counts": {split: dict(EXPECTED_COUNTS[split]) for split in ("FIT", "SELECT")},
        "primary_ranks": list(PRIMARY_RANKS), "rank8_opened": opened_rank8,
        "provisional_rank": provisional_rank, "selected_rank": selected_rank,
        "selected_start": selected_start,
        "analytic_operator_sha256": _tensor_sha(analytic_operator),
        "effect_stability": {
            "median": stability[0] if math.isfinite(stability[0]) else None,
            "p90": stability[1] if math.isfinite(stability[1]) else None,
            "passed": stability[2],
        },
        "projector_overlap_pairs": projector_overlaps,
        "literal_price": expected_price, "backend_reported_model_counts": counts,
        "fit_objective_constants": asdict(FIT_OBJECTIVE),
        "fit_receipts": {str(rank): [summarize(x) for x in fits[rank]] for rank in sorted(fits)},
        "random_control_verdicts": {
            str(rank): {
                "count": len(random_scores[rank]),
                "passing_candidate_starts": [
                    x.start for x in fits[rank] if _random_bar_passes(x, random_scores[rank])
                ],
            } for rank in sorted(random_scores)
        },
        "permutation_receipts": [summarize(x) for x in permutation_fits],
        "registered_execution_counts_exactly_reconciled": (
            counts == {key: expected_price[key] for key in counts}
        ),
        "registered_execution_counts_within_ceiling": all(
            counts[key] <= expected_price[key] for key in counts
        ),
        "remaining_production_functions": ["independent_receipt_and_bundle_audit"],
    }
    _write_create_only(receipt_path, bundle_path, receipt, frames)
    return receipt


def _write_create_only(
    receipt_path: Path, bundle_path: Path, receipt: Mapping[str, object],
    frames: Mapping[str, torch.Tensor],
) -> None:
    if receipt_path.exists() or bundle_path.exists():
        raise FileExistsError("Program A outputs are create-only")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_buffer = io.BytesIO()
    torch.save(dict(frames), bundle_buffer)
    with bundle_path.open("xb") as handle:
        handle.write(bundle_buffer.getvalue())
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(dict(receipt), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise


def compile_dryrun() -> dict[str, object]:
    plan = compile_discovery_plan()
    return {
        "schema": f"{SCHEMA}_dryrun", "experiment_id": EXPERIMENT_ID,
        "site_id": SITE_ID, "source_sha256": dict(plan.source_sha256),
        "ordinal_sha256": dict(plan.ordinal_sha256),
        "counts": {split: dict(EXPECTED_COUNTS[split]) for split in ("FIT", "SELECT")},
        "primary_ranks": list(PRIMARY_RANKS), "conditional_rank": 8,
        "starts_per_primary_rank": len(PRIMARY_STARTS), "updates": UPDATES,
        "batch_size": BATCH_SIZE, "haar_controls_per_opened_rank": len(HAAR_SEEDS),
        "primary_price": dict(PRIMARY_PRICE), "rank8_incremental_price": dict(RANK8_INCREMENTAL_PRICE),
        "conditional_price": dict(CONDITIONAL_PRICE),
        "conditional_stored_frame_bytes": "2048 * selected_rank",
        "fit_objective_constants": asdict(FIT_OBJECTIVE),
        "fit_objective_constants_blocking": False,
        "authority_parsed": False, "validation_rows_loaded": 0,
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "production_backend_available": True,
        "remaining_production_functions": ["independent_receipt_and_bundle_audit"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise ProgramAError(f"{name} must be absent or exactly 1")
    if args.dry_run or any(
        os.environ.get(name) == "1" for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")
    ):
        print(json.dumps(compile_dryrun(), sort_keys=True, indent=2))
        return 0

    import task14_program_a_torch_backend as production

    backend = production.Task14ProgramATorchBackend.load_production(device="cuda")
    receipt = execute_program_a(
        backend, receipt_path=PROGRAM_A_RECEIPT_PATH,
        bundle_path=PROGRAM_A_BUNDLE_PATH,
    )
    print(json.dumps({
        "terminal": receipt["terminal"],
        "selected_rank": receipt["selected_rank"],
        "literal_price": receipt["literal_price"],
        "receipt_path": str(PROGRAM_A_RECEIPT_PATH),
        "bundle_path": str(PROGRAM_A_BUNDLE_PATH),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
