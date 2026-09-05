#!/usr/bin/env python3
# BQLANE: cpu
"""Outcome-blind factory for later-phase Task14 cross-syntax authorities.

The factory constructs a prospective cross-noun relation without consulting
model outcomes.  Within each grammatical stratum, every target group receives
the cyclically next group's opposite-syntax donor endpoint.  TEST stratifies
by subject number and attractor number; OOD additionally stratifies by the
second attractor.  It reads only the deterministic Task14 stimulus generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import circuit_battery_task14 as task14


TASK_ID = "subject_verb.number_agreement"
PARTITION = "HELD_OUT"
SITE_IDS = ("attn:11", "attn:11:head:03")
BATCH_SIZE = 32
MIN_NATIVE_CELL_ACCURACY = 0.85
MIN_CELL_DIRECTION_FRACTION = 0.75
MIN_CELL_MEAN_RECOVERY = 0.40
MIN_DONOR_DENOMINATOR = 1.0e-6

TASK14_GENERATOR_SHA256 = (
    "33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94"
)
TASK14_FULL_AUTHORITY_SHA256 = (
    "1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1"
)


class PhaseCrossSyntaxAuthorityError(ValueError):
    """A later-phase stimulus or derived matched-noun relation changed."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PhaseCrossSyntaxConfig:
    phase: str
    schema: str
    validation_scope: str
    expected_phase_records_sha256: str
    correction: str
    donor_rule: str
    site_ids: tuple[str, ...] = SITE_IDS

    def __post_init__(self) -> None:
        if self.phase not in {"TEST", "OOD"}:
            raise PhaseCrossSyntaxAuthorityError(
                "the later-phase factory accepts only TEST or OOD"
            )
        for label, value in (
            ("schema", self.schema),
            ("validation_scope", self.validation_scope),
            ("correction", self.correction),
        ):
            if not isinstance(value, str) or not value:
                raise PhaseCrossSyntaxAuthorityError(f"{label} must be nonempty text")
        digest = self.expected_phase_records_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise PhaseCrossSyntaxAuthorityError(
                "expected phase-record digest must be a lowercase SHA-256"
            )
        expected_rule = {
            "TEST": "cyclic_cross_noun_by_subject_and_attractor",
            "OOD": "cyclic_cross_noun_by_subject_and_two_attractors",
        }[self.phase]
        if self.donor_rule != expected_rule:
            raise PhaseCrossSyntaxAuthorityError(
                f"{self.phase} donor rule must be {expected_rule}"
            )
        if not self.site_ids or len(set(self.site_ids)) != len(self.site_ids) \
                or any(site not in SITE_IDS for site in self.site_ids):
            raise PhaseCrossSyntaxAuthorityError("site IDs must be a nonempty targeted subset")


class PhaseCrossSyntaxCandidate:
    """Module-like candidate object accepted by the targeted shared runner."""

    TASK_ID = TASK_ID
    PARTITION = PARTITION
    SITE_IDS = SITE_IDS
    BATCH_SIZE = BATCH_SIZE
    MIN_NATIVE_CELL_ACCURACY = MIN_NATIVE_CELL_ACCURACY
    MIN_CELL_DIRECTION_FRACTION = MIN_CELL_DIRECTION_FRACTION
    MIN_CELL_MEAN_RECOVERY = MIN_CELL_MEAN_RECOVERY
    MIN_DONOR_DENOMINATOR = MIN_DONOR_DENOMINATOR

    def __init__(self, config: PhaseCrossSyntaxConfig) -> None:
        self.config = config
        self.PHASE = config.phase
        self.SCHEMA = config.schema
        self.VALIDATION_SCOPE = config.validation_scope
        self.SITE_IDS = config.site_ids
        self.EXPECTED_SOURCE_SHA256 = {
            "task14_generator_file": TASK14_GENERATOR_SHA256,
            "task14_full_authority": TASK14_FULL_AUTHORITY_SHA256,
            "task14_phase_records": config.expected_phase_records_sha256,
        }

    def _source_rows(self) -> list[dict]:
        generator_path = Path(task14.__file__)
        if hashlib.sha256(generator_path.read_bytes()).hexdigest() != \
                TASK14_GENERATOR_SHA256:
            raise PhaseCrossSyntaxAuthorityError("Task14 generator file changed")
        authority, authority_sha256 = task14.build_authority()
        if authority_sha256 != TASK14_FULL_AUTHORITY_SHA256:
            raise PhaseCrossSyntaxAuthorityError("Task14 full authority changed")
        rows, rows_sha256 = task14.split_rows(authority, self.PHASE)
        if rows_sha256 != self.config.expected_phase_records_sha256:
            raise PhaseCrossSyntaxAuthorityError(
                f"Task14 {self.PHASE} records changed"
            )
        return rows

    @staticmethod
    def _endpoint(row: Mapping[str, object], side: str) -> dict:
        if side not in {"base", "donor"}:
            raise PhaseCrossSyntaxAuthorityError("endpoint side is invalid")
        answer_id = int(row[f"{side}_answer_id"])
        if answer_id not in {318, 389}:
            raise PhaseCrossSyntaxAuthorityError(
                "answer left the frozen is/are vocabulary"
            )
        return {
            "ids": list(row[f"{side}_ids"]),
            "answer_id": answer_id,
            "foil_id": 389 if answer_id == 318 else 318,
            "position": int(row[f"{side}_prediction_position"]),
            "text": str(row[f"{side}_text"]),
            "subject_number": str(row[f"{side}_subject_number"]),
            "endpoint_id": f"{row['row_id']}:{side}",
        }

    def _build_rows_unvalidated(self) -> list[dict]:
        by_group: dict[str, dict[str, dict]] = {}
        for row in self._source_rows():
            by_group.setdefault(str(row["group_id"]), {})[
                str(row["transform_id"])
            ] = row
        output = []
        panels = sorted(
            by_group.items(), key=lambda item: int(item[1]["A1"]["group_number"]),
        )
        strata: dict[tuple[object, ...], list[tuple[str, dict[str, dict]]]] = {}
        for group_id, panel in panels:
            if set(panel) != {"A1", "A2", "P", "C"}:
                raise PhaseCrossSyntaxAuthorityError(
                    f"{self.PHASE} panel is incomplete"
                )
            a1 = panel["A1"]
            key: tuple[object, ...] = (
                str(a1["base_subject_number"]), bool(a1["base_attractor_plural"]),
            )
            if self.config.donor_rule == \
                    "cyclic_cross_noun_by_subject_and_two_attractors":
                key += (bool(a1["base_second_attractor_plural"]),)
            strata.setdefault(key, []).append((group_id, panel))
        donor_panel_by_group: dict[str, tuple[str, dict[str, dict]]] = {}
        for members in strata.values():
            ordered = sorted(
                members, key=lambda item: int(item[1]["A1"]["group_number"]),
            )
            if len(ordered) < 2:
                raise PhaseCrossSyntaxAuthorityError("cross-noun donor stratum is singleton")
            for index, (group_id, _panel) in enumerate(ordered):
                donor_panel_by_group[group_id] = ordered[(index + 1) % len(ordered)]

        for group_id, panel in panels:
            donor_group_id, donor_panel = donor_panel_by_group[group_id]
            if donor_group_id == group_id:
                raise PhaseCrossSyntaxAuthorityError("cross-noun donor reused target group")
            for target_family, donor_family in (("A1", "A2"), ("A2", "A1")):
                target = self._endpoint(panel[target_family], "base")
                donor = self._endpoint(donor_panel[donor_family], "donor")
                target_syntax = "pp" if target_family == "A1" else "relative"
                donor_syntax = "pp" if donor_family == "A1" else "relative"
                cell_id = (
                    f"{target_syntax}_{target['subject_number']}_to_"
                    f"{donor_syntax}_{donor['subject_number']}"
                )
                identity = [
                    self.SCHEMA, self.config.donor_rule, group_id, donor_group_id,
                    target_family, donor_family,
                ]
                output.append({
                    "schema": self.SCHEMA,
                    "task_id": self.TASK_ID,
                    "split": self.PHASE,
                    "partition": self.PARTITION,
                    "validation_scope": self.VALIDATION_SCOPE,
                    "row_id": canonical_sha256(identity),
                    "group_id": group_id,
                    "donor_group_id": donor_group_id,
                    "donor_rule": self.config.donor_rule,
                    "cell_id": cell_id,
                    "target_family": target_family,
                    "donor_family": donor_family,
                    "target_endpoint_id": target["endpoint_id"],
                    "donor_endpoint_id": donor["endpoint_id"],
                    "base_ids": target["ids"],
                    "donor_ids": donor["ids"],
                    "base_text": target["text"],
                    "donor_text": donor["text"],
                    "base_answer_id": target["answer_id"],
                    "base_foil_id": target["foil_id"],
                    "donor_answer_id": donor["answer_id"],
                    "donor_foil_id": donor["foil_id"],
                    "base_semantic_position": target["position"],
                    "donor_semantic_position": donor["position"],
                    "base_subject_number": target["subject_number"],
                    "donor_subject_number": donor["subject_number"],
                    "expected_effect": "toward_opposite_number_cross_syntax_donor",
                })
        return output

    def build_rows(self) -> list[dict]:
        output = self._build_rows_unvalidated()
        self.validate_rows(output)
        return output

    def validate_rows(self, rows: Sequence[Mapping[str, object]]) -> str:
        materialized = [dict(row) for row in rows]
        if len(materialized) != 64 \
                or len({row.get("row_id") for row in materialized}) != 64:
            raise PhaseCrossSyntaxAuthorityError(
                f"{self.PHASE} authority must contain 64 unique rows"
            )
        cells: dict[str, int] = {}
        for row in materialized:
            if row.get("schema") != self.SCHEMA \
                    or row.get("task_id") != self.TASK_ID \
                    or row.get("split") != self.PHASE \
                    or row.get("partition") != self.PARTITION \
                    or row.get("validation_scope") != self.VALIDATION_SCOPE:
                raise PhaseCrossSyntaxAuthorityError(
                    f"{self.PHASE} row identity changed"
                )
            if {row.get("target_family"), row.get("donor_family")} != {"A1", "A2"}:
                raise PhaseCrossSyntaxAuthorityError("row is not cross-syntax")
            if row.get("donor_rule") != self.config.donor_rule \
                    or row.get("group_id") == row.get("donor_group_id"):
                raise PhaseCrossSyntaxAuthorityError("row is not cross-noun by the frozen rule")
            if row.get("base_subject_number") == row.get("donor_subject_number"):
                raise PhaseCrossSyntaxAuthorityError("row does not reverse subject number")
            if row.get("base_answer_id") != row.get("donor_foil_id") \
                    or row.get("base_foil_id") != row.get("donor_answer_id"):
                raise PhaseCrossSyntaxAuthorityError(
                    "answer orientation is not reversed"
                )
            if row.get("base_semantic_position") != len(row.get("base_ids", ())) - 1 \
                    or row.get("donor_semantic_position") != len(row.get("donor_ids", ())) - 1:
                raise PhaseCrossSyntaxAuthorityError("semantic position is not final")
            cells[str(row["cell_id"])] = cells.get(str(row["cell_id"]), 0) + 1
        expected = {
            "pp_singular_to_relative_plural",
            "pp_plural_to_relative_singular",
            "relative_singular_to_pp_plural",
            "relative_plural_to_pp_singular",
        }
        if set(cells) != expected or set(cells.values()) != {16}:
            raise PhaseCrossSyntaxAuthorityError(
                f"direction-cell balance changed: {cells}"
            )
        canonical = canonical_sha256(materialized)
        if canonical != canonical_sha256(self._build_rows_unvalidated()):
            raise PhaseCrossSyntaxAuthorityError(
                f"rows differ from exact regenerated {self.PHASE} authority"
            )
        return canonical

    def authority_sha256(self) -> str:
        return self.validate_rows(self.build_rows())

    def compile_plan(
        self, rows: Sequence[Mapping[str, object]] | None = None,
    ) -> dict:
        materialized = self.build_rows() if rows is None else [dict(row) for row in rows]
        digest = self.validate_rows(materialized)
        calls = []
        for side in ("base", "donor"):
            for start in range(0, len(materialized), self.BATCH_SIZE):
                calls.append({
                    "kind": "native",
                    "side": side,
                    "capture": True,
                    "row_ids": [
                        row["row_id"]
                        for row in materialized[start:start + self.BATCH_SIZE]
                    ],
                })
        for site_id in self.SITE_IDS:
            for start in range(0, len(materialized), self.BATCH_SIZE):
                calls.append({
                    "kind": "exact_single_position_interchange",
                    "site_id": site_id,
                    "row_ids": [
                        row["row_id"]
                        for row in materialized[start:start + self.BATCH_SIZE]
                    ],
                })
            for operation in ("zero_removal", "native_head_replay"):
                for start in range(0, len(materialized), self.BATCH_SIZE):
                    calls.append({
                        "kind": operation,
                        "site_id": site_id,
                        "row_ids": [
                            row["row_id"]
                            for row in materialized[start:start + self.BATCH_SIZE]
                        ],
                    })
        plan = {
            "schema": "task14_targeted_cross_syntax_plan_v1",
            "phase": self.PHASE,
            "partition": self.PARTITION,
            "validation_scope": self.VALIDATION_SCOPE,
            "authority_sha256": digest,
            "source_sha256": dict(self.EXPECTED_SOURCE_SHA256),
            "site_ids": list(self.SITE_IDS),
            "row_count": len(materialized),
            "batch_size": self.BATCH_SIZE,
            "calls": calls,
            "price": {
                "forward_calls": len(calls),
                "example_evaluations": len(materialized) * (
                    2 + 3 * len(self.SITE_IDS)
                ),
                "backward_calls": 0,
                "model_updates": 0,
                "raw_numeric_evidence_bytes": (
                    8 * len(materialized) * (2 + 3 * len(self.SITE_IDS))
                ),
            },
            "score": {
                "minimum_native_cell_accuracy": self.MIN_NATIVE_CELL_ACCURACY,
                "minimum_cell_direction_fraction": self.MIN_CELL_DIRECTION_FRACTION,
                "minimum_cell_mean_recovery": self.MIN_CELL_MEAN_RECOVERY,
                "minimum_cell_median_normalized_removal_damage": 0.25,
                "minimum_cell_positive_removal_fraction": 0.65,
                "maximum_native_head_replay_absolute_logit_error": 1.0e-4,
            },
            "correction": self.config.correction,
        }
        plan["compiled_sha256"] = canonical_sha256(plan)
        return plan


def make_candidate(config: PhaseCrossSyntaxConfig) -> PhaseCrossSyntaxCandidate:
    """Create one fixed-phase, outcome-blind candidate for a thin facade."""
    return PhaseCrossSyntaxCandidate(config)
