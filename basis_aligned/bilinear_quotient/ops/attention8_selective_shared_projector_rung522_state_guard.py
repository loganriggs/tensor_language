"""Pure protocol state guard for the rung-522 scientific runner.

This module is intentionally separate from the rung-522 statistical helpers.
It imports no tensor/model/data code.  The eventual runner should route every
fit authorization and split access through :class:`ProtocolState`, making the
pre-TEST freeze and the fourth-target exclusion executable constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from types import MappingProxyType
from typing import Literal, Sequence


FITTED_TARGETS = ("r.2.0.2", "r.2.1.1", "r.2.2.1")
RESERVED_FOURTH_TARGET = "r.2.0.1"
REAL_SEEDS = tuple(range(52200, 52205))
PERMUTATION_SEEDS = tuple(range(52300, 52316))
SHARED_FAMILIES = frozenset({
    "real_leave_one_out", "recovery_only", "label_null", "all_three"
})

REGISTERED_PRICE = MappingProxyType({
    "frame_count_before_test": 103,
    "updates_per_fit": 200,
    "optimization_forward_events": 20_600,
    "optimization_backward_events": 20_600,
    "optimization_combined_events": 41_200,
    "optimization_hard_ceiling": 45_000,
    "registered_worst_case_inference_forwards": 9_422,
    "inference_forward_ceiling": 12_000,
    "removal_inference_ceiling": 2_000,
    "rank": 4,
    "ambient_dimension": 1_152,
    "largest_retained_frame_values": 4_608,
})

INFERENCE_LEDGER = MappingProxyType({
    "native_capture": 167,
    "native_replay": 167,
    "self_donor": 3,
    "fit_d0_full_attention8": 95,
    "fit_health": 206,
    "full_attention8_comparator": 72,
    "prediction_a": 5_976,
    "recovery_only": 1_080,
    "haar": 1_440,
    "all_three_selection_and_test": 216,
})

REGISTERED_PREDICTIONS = MappingProxyType({
    "A": MappingProxyType({
        "passing_seeds_per_fold": 4,
        "seeds_per_fold": 5,
        "minimum_signed_cosine": 0.75,
        "maximum_scaled_relative_residual": 0.55,
        "minimum_oracle_recovery_fraction": 0.50,
        "minimum_member_rms_nat": 0.02,
        "maximum_control_to_member_rms": 0.25,
        "minimum_concentration": 4.0,
        "minimum_full_attention8_concentration_improvement": 1.0,
        "row_bootstraps": 2_000,
        "fourfold_margin_lower_95_strictly_positive": True,
        "permutation_retrains_per_fold": 16,
        "projector_stability_real_values": 5,
        "projector_stability_null_values": 16,
        "higher_interpolation_quantile": 0.95,
        "oracle_minimum_aligned_recovery": 0.05,
        "exact_token_subset_minimum_pairs": 32,
    }),
    "B": MappingProxyType({
        "passing_paired_seeds_per_fold": 4,
        "minimum_recovery_only_concentration_improvement": 0.5,
        "maximum_signed_cosine_loss": 0.05,
        "row_bootstraps": 2_000,
        "bounded_selectivity_improvement_lower_95_strictly_positive": True,
        "paired_seed_sign_flips": 32,
        "mean_improvement_above_sign_flip_q95": True,
        "joint_statistic": "minimum_selectivity_times_minimum_aligned_recovery",
        "strictly_beats_haar_maximum": True,
        "strictly_beats_label_permutation_q95": True,
    }),
    "C": MappingProxyType({
        "fingerprint_cells": 4,
        "fingerprint_permutations_per_cell": 20_000,
        "outside_union_to_smallest_quartet_rms_maximum": 0.25,
        "fingerprint_statistic": "min_quartet_coordinate_minus_max_nonquartet_coordinate",
        "fingerprint_statistic_strictly_positive": True,
        "fingerprint_strictly_above_higher_q95": True,
    }),
    "D": MappingProxyType({
        "quartet_to_median_nonquartet_effect_minimum": 2.0,
        "requires_member_minus_control_sign_preservation": True,
        "requires_quartet_negative_order_preservation": True,
        "requires_max_statistic_permutation_q95": True,
        "removal_inference_ceiling": 2_000,
    }),
})

_HASH = re.compile(r"^[0-9a-f]{64}$")


class ProtocolViolation(RuntimeError):
    """Raised before a registered sequencing or isolation rule is violated."""


@dataclass(frozen=True)
class FrameSpec:
    frame_id: str
    family: Literal[
        "real_leave_one_out", "recovery_only", "target_oracle", "label_null", "all_three"
    ]
    seed: int
    training_targets: tuple[str, ...]
    health_targets: tuple[str, ...]
    omitted_target: str | None = None
    oracle_target: str | None = None


@dataclass(frozen=True)
class FrozenFrame:
    spec: FrameSpec
    frame_sha256: str
    scheduler_sha256: str
    frozen: bool = True


@dataclass(frozen=True)
class PretestFreeze:
    frame_manifest_sha256: str
    scheduler_manifest_sha256: str
    validation_decisions_sha256: str
    medoid_selection_sha256: str
    fingerprint_definition_sha256: str
    test_sweep_plan_sha256: str
    registered_contract_sha256: str
    selected_final_frame_id: str
    eligible_all_three_frame_ids: tuple[str, ...]
    selection_targets: tuple[str, ...]
    validation_provisional_gates_passed: bool
    medoid_selection_rule: str
    test_sweep_plan_frozen: bool


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _valid_hash(value: str) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def expected_frame_specs() -> tuple[FrameSpec, ...]:
    """Construct the literal 15+15+20+48+5 pre-TEST inventory."""
    frames: list[FrameSpec] = []
    for family in ("real_leave_one_out", "recovery_only"):
        for omitted in FITTED_TARGETS:
            trained = tuple(target for target in FITTED_TARGETS if target != omitted)
            for seed in REAL_SEEDS:
                frames.append(FrameSpec(
                    f"{family}:{omitted}:{seed}", family, seed, trained, trained, omitted
                ))
    for target in (RESERVED_FOURTH_TARGET,) + FITTED_TARGETS:
        for seed in REAL_SEEDS:
            frames.append(FrameSpec(
                f"target_oracle:{target}:{seed}", "target_oracle", seed,
                (target,), (target,), oracle_target=target,
            ))
    for null_seed in PERMUTATION_SEEDS:
        for omitted in FITTED_TARGETS:
            trained = tuple(target for target in FITTED_TARGETS if target != omitted)
            frames.append(FrameSpec(
                f"label_null:{null_seed}:{omitted}", "label_null", null_seed,
                trained, trained, omitted,
            ))
    for seed in REAL_SEEDS:
        frames.append(FrameSpec(
            f"all_three:{seed}", "all_three", seed,
            FITTED_TARGETS, FITTED_TARGETS,
        ))
    if len(frames) != 103 or len({frame.frame_id for frame in frames}) != 103:
        raise AssertionError("internal rung522 frame census is not 103 unique objects")
    return tuple(frames)


EXPECTED_FRAME_SPECS = MappingProxyType({
    frame.frame_id: frame for frame in expected_frame_specs()
})


def registered_contract_sha256() -> str:
    return _sha256_json({
        "fitted_targets": FITTED_TARGETS,
        "reserved_fourth_target": RESERVED_FOURTH_TARGET,
        "price": dict(REGISTERED_PRICE),
        "inference_ledger": dict(INFERENCE_LEDGER),
        "predictions": {
            name: dict(values) for name, values in REGISTERED_PREDICTIONS.items()
        },
        "frames": [
            asdict(EXPECTED_FRAME_SPECS[name]) for name in sorted(EXPECTED_FRAME_SPECS)
        ],
    })


class ProtocolState:
    """One-way FIT/VALIDATION -> frozen manifest -> one TEST sweep state."""

    def __init__(self) -> None:
        self._frames: dict[str, FrozenFrame] = {}
        self._pretest: PretestFreeze | None = None
        self._test_open = False
        self._test_closed = False
        self._optimization_forwards = 0
        self._optimization_backwards = 0
        self._inference_forwards = 0
        self._removal_forwards = 0

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def test_open(self) -> bool:
        return self._test_open and not self._test_closed

    def authorize_training(
        self,
        frame_id: str,
        *,
        split: str,
        training_targets: Sequence[str],
        health_targets: Sequence[str],
    ) -> FrameSpec:
        if self._pretest is not None or self._test_open or self._test_closed:
            raise ProtocolViolation("training is permanently closed after pre-TEST freeze")
        if split != "FIT":
            raise ProtocolViolation("gradients may use FIT only")
        expected = EXPECTED_FRAME_SPECS.get(frame_id)
        if expected is None:
            raise ProtocolViolation(f"unregistered frame {frame_id!r}")
        if tuple(training_targets) != expected.training_targets:
            raise ProtocolViolation("training target identities differ from registration")
        if tuple(health_targets) != expected.health_targets:
            raise ProtocolViolation("health target identities differ from registration")
        if expected.family in SHARED_FAMILIES and RESERVED_FOURTH_TARGET in (
            tuple(training_targets) + tuple(health_targets)
        ):
            raise ProtocolViolation("reserved fourth target entered shared fitting or health")
        return expected

    def register_frozen_frame(self, frame: FrozenFrame) -> None:
        if self._pretest is not None or self._test_open or self._test_closed:
            raise ProtocolViolation("cannot register a frame after pre-TEST freeze")
        expected = EXPECTED_FRAME_SPECS.get(frame.spec.frame_id)
        if expected is None or frame.spec != expected:
            raise ProtocolViolation("frame metadata differs from registration")
        if frame.spec.frame_id in self._frames:
            raise ProtocolViolation("duplicate frame identifier")
        if not frame.frozen:
            raise ProtocolViolation("frame is not frozen")
        if not _valid_hash(frame.frame_sha256) or not _valid_hash(frame.scheduler_sha256):
            raise ProtocolViolation("frame and scheduler require lowercase SHA-256 hashes")
        self._frames[frame.spec.frame_id] = frame

    def record_optimization_events(self, forwards: int, backwards: int) -> None:
        if self._pretest is not None or self._test_open or self._test_closed:
            raise ProtocolViolation("optimization is forbidden after pre-TEST freeze")
        if not isinstance(forwards, int) or not isinstance(backwards, int) or min(
            forwards, backwards
        ) < 0:
            raise ProtocolViolation("optimization counts must be nonnegative integers")
        self._optimization_forwards += forwards
        self._optimization_backwards += backwards
        if self._optimization_forwards + self._optimization_backwards > 45_000:
            raise ProtocolViolation("optimization hard ceiling exceeded")

    def record_inference_events(self, count: int, *, removal: bool = False) -> None:
        if not isinstance(count, int) or count < 0:
            raise ProtocolViolation("inference count must be a nonnegative integer")
        if removal:
            self._removal_forwards += count
            if self._removal_forwards > 2_000:
                raise ProtocolViolation("removal inference ceiling exceeded")
        else:
            self._inference_forwards += count
            if self._inference_forwards > 12_000:
                raise ProtocolViolation("inference ceiling exceeded")

    def frame_manifest_sha256(self) -> str:
        return _sha256_json([
            {
                "spec": asdict(self._frames[name].spec),
                "frame_sha256": self._frames[name].frame_sha256,
                "scheduler_sha256": self._frames[name].scheduler_sha256,
                "frozen": self._frames[name].frozen,
            }
            for name in sorted(self._frames)
        ])

    def scheduler_manifest_sha256(self) -> str:
        return _sha256_json([
            (name, self._frames[name].scheduler_sha256) for name in sorted(self._frames)
        ])

    def freeze_pretest(self, freeze: PretestFreeze) -> None:
        if self._pretest is not None or self._test_open or self._test_closed:
            raise ProtocolViolation("pre-TEST state was already frozen")
        missing = sorted(set(EXPECTED_FRAME_SPECS) - set(self._frames))
        extra = sorted(set(self._frames) - set(EXPECTED_FRAME_SPECS))
        if missing or extra:
            raise ProtocolViolation(
                f"TEST requires all 103 frames; missing={missing[:3]}, extra={extra[:3]}"
            )
        if self._optimization_forwards != 20_600 or self._optimization_backwards != 20_600:
            raise ProtocolViolation("TEST requires exactly 20,600 forward and backward fit events")
        hashes = (
            freeze.frame_manifest_sha256,
            freeze.scheduler_manifest_sha256,
            freeze.validation_decisions_sha256,
            freeze.medoid_selection_sha256,
            freeze.fingerprint_definition_sha256,
            freeze.test_sweep_plan_sha256,
            freeze.registered_contract_sha256,
        )
        if not all(_valid_hash(value) for value in hashes):
            raise ProtocolViolation("all pre-TEST artifacts require lowercase SHA-256 hashes")
        if freeze.frame_manifest_sha256 != self.frame_manifest_sha256():
            raise ProtocolViolation("frame manifest hash does not match the 103 frames")
        if freeze.scheduler_manifest_sha256 != self.scheduler_manifest_sha256():
            raise ProtocolViolation("scheduler manifest hash does not match the 103 frames")
        if freeze.registered_contract_sha256 != registered_contract_sha256():
            raise ProtocolViolation("runner price/prediction contract differs from registration")
        if not freeze.validation_provisional_gates_passed:
            raise ProtocolViolation("provisional VALIDATION gates failed")
        if freeze.medoid_selection_rule != "grassmann_medoid_lower_seed_tiebreak":
            raise ProtocolViolation("all-three selection is not the frozen geometry-only medoid")
        if not freeze.test_sweep_plan_frozen:
            raise ProtocolViolation("TEST sweep plan is not frozen")
        if tuple(freeze.selection_targets) != FITTED_TARGETS:
            raise ProtocolViolation("fourth target entered shared medoid selection")
        all_three = {
            frame_id for frame_id, spec in EXPECTED_FRAME_SPECS.items()
            if spec.family == "all_three"
        }
        eligible = tuple(freeze.eligible_all_three_frame_ids)
        if not eligible or len(set(eligible)) != len(eligible) or not set(eligible) <= all_three:
            raise ProtocolViolation("medoid eligibility is not a nonempty all-three subset")
        if freeze.selected_final_frame_id not in eligible:
            raise ProtocolViolation("selected medoid is not eligible")
        self._pretest = freeze

    def authorize_split_access(self, split: str) -> None:
        if split not in {"FIT", "VALIDATION", "TEST"}:
            raise ProtocolViolation(f"unknown split {split!r}")
        if split == "TEST" and not self.test_open:
            raise ProtocolViolation("TEST access attempted before the frozen one-shot sweep")

    def open_test_once(self) -> None:
        if self._test_open or self._test_closed:
            raise ProtocolViolation("TEST may open only once")
        if self._pretest is None:
            raise ProtocolViolation("TEST cannot open before pre-TEST freeze")
        self._test_open = True

    def close_test(self) -> None:
        if not self.test_open:
            raise ProtocolViolation("no TEST sweep is open")
        self._test_closed = True


__all__ = [
    "EXPECTED_FRAME_SPECS", "FITTED_TARGETS", "FrameSpec", "FrozenFrame",
    "INFERENCE_LEDGER", "PERMUTATION_SEEDS", "PretestFreeze", "ProtocolState",
    "ProtocolViolation", "REAL_SEEDS", "REGISTERED_PREDICTIONS", "REGISTERED_PRICE",
    "RESERVED_FOURTH_TARGET", "SHARED_FAMILIES", "expected_frame_specs",
    "registered_contract_sha256",
]
