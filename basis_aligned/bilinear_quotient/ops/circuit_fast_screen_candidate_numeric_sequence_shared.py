#!/usr/bin/env python3
# BQLANE: cpu
"""Shared frozen-row adapter for numeric-sequence fast-screen hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import tiktoken

import circuit_battery_integration_contract as battery


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
ROWS_PATH = ROOT / "increment_two_hypothesis_rows_rung567.json"
ROWS_SHA256 = "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053"
CONTROL_PATH = ROOT / "pending_opener_controls_rung537.json"
SCHEMA = "circuit_fast_screen_candidate_v1"
HYPOTHESIS = "numeric_sequence_continuation"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260904
SPLIT = "FIT"
ENCODING = tiktoken.get_encoding("gpt2")
WORD_PREV = {" eleven": " ten", " twelve": " eleven"}


class NumericSequenceCandidateError(RuntimeError):
    pass


@dataclass(frozen=True)
class NumericSequenceCandidateConfig:
    task_id: str
    prior_art_name: str
    family_map: Mapping[str, tuple[str, str]]
    transforms: tuple[battery.TransformSpec, ...]

    def __post_init__(self) -> None:
        if len(self.family_map) != 4 or sorted(value[0] for value in self.family_map.values()) != [
            "A1", "A2", "C", "P",
        ]:
            raise NumericSequenceCandidateError("family map must contain exactly one A1/A2/P/C family")
        if tuple(spec.transform_id for spec in self.transforms) != ("A1", "A2", "P", "C"):
            raise NumericSequenceCandidateError("transforms must be ordered A1/A2/P/C")

    @property
    def prior_art(self) -> Path:
        return ROOT / "circuits" / self.prior_art_name

    @property
    def task_spec(self) -> battery.BatteryTaskSpec:
        return battery.BatteryTaskSpec(
            task_id=self.task_id,
            generator_role="adapt_frozen_numeric_sequence_panels_to_linked_cross_construction_fit_rows",
            answer_role="score_jointly_tokenized_final_label_plus_one_vs_final_label",
            transforms=self.transforms,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _foil_id(answer_text: str) -> int:
    """Return the registered competitor: the previous numeric value."""
    previous = WORD_PREV.get(answer_text)
    if previous is None:
        try:
            previous = str(int(answer_text) - 1)
        except ValueError as error:
            raise NumericSequenceCandidateError(f"non-numeric answer {answer_text!r}") from error
    ids = ENCODING.encode(previous)
    if len(ids) != 1:
        raise NumericSequenceCandidateError(f"foil {previous!r} is not one token")
    return ids[0]


def _declared_changes(config: NumericSequenceCandidateConfig, transform_id: str) -> bool:
    for spec in config.transforms:
        if spec.transform_id == transform_id:
            return bool(spec.answer_changes)
    raise NumericSequenceCandidateError(f"unknown transform {transform_id}")


def _checks(config, row, base_foil, donor_foil, transform_id):
    answer_changes = row["base_answer_id"] != row["donor_answer_id"]
    return {
        "single_token_answers": all(
            len(ENCODING.encode(row[key])) == 1 for key in ("base_answer", "donor_answer")
        ),
        "joint_answer_tokenization": (
            ENCODING.encode(row["base_text"] + row["base_answer"])
            == row["base_ids"] + [row["base_answer_id"]]
            and ENCODING.encode(row["donor_text"] + row["donor_answer"])
            == row["donor_ids"] + [row["donor_answer_id"]]
        ),
        "prompt_roundtrip": (
            ENCODING.decode(row["base_ids"]) == row["base_text"]
            and ENCODING.decode(row["donor_ids"]) == row["donor_text"]
        ),
        "distinct_prompts": row["base_text"] != row["donor_text"],
        "paired_answer_foil_alignment": (
            {row["base_answer_id"], base_foil} == {row["donor_answer_id"], donor_foil}
            and row["base_answer_id"] != base_foil
            and row["donor_answer_id"] != donor_foil
        ),
        "answer_change_matches_transform": answer_changes == _declared_changes(config, transform_id),
        "foil_is_the_registered_competitor": (
            True if (answer_changes or transform_id == "C")
            else base_foil == _foil_id(row["base_answer"])
        ),
    }


def build_rows(
    config: NumericSequenceCandidateConfig,
    task_id: str,
    groups: int = DEFAULT_GROUPS,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    if task_id != config.task_id:
        raise NumericSequenceCandidateError(f"unknown task_id {task_id!r}")
    if _sha256(ROWS_PATH) != ROWS_SHA256:
        raise NumericSequenceCandidateError("frozen stimulus rows changed; refusing to build")
    source = json.loads(ROWS_PATH.read_text())["rows"]
    controls = json.loads(CONTROL_PATH.read_text())["rows"]
    picked: dict[str, list[dict]] = {}
    for family, (transform_id, _construction_id) in config.family_map.items():
        pool = controls if family == "nonopener_punctuation_substitution" else source
        rows = [
            row for row in pool
            if row.get("family_id") == family and row.get("split") == SPLIT
        ][:groups]
        if len(rows) < groups:
            raise NumericSequenceCandidateError(
                f"family {family} has {len(rows)} rows, need {groups}"
            )
        picked[transform_id] = rows

    output: list[dict] = []
    for direction_id in ("base_to_donor", "donor_to_base"):
        for index in range(groups):
            group_id = f"{config.task_id}:{direction_id}:{index:03d}"
            for family, (transform_id, construction_id) in config.family_map.items():
                row = picked[transform_id][index]
                punctuation_control = family == "nonopener_punctuation_substitution"
                if punctuation_control:
                    answer = int(row["answer_id"])
                    row = dict(
                        row,
                        base_answer_id=answer,
                        donor_answer_id=answer,
                        base_answer=ENCODING.decode([answer]),
                        donor_answer=ENCODING.decode([answer]),
                    )
                answer_changes = row["base_answer_id"] != row["donor_answer_id"]
                if punctuation_control:
                    base_foil = donor_foil = 1 if row["base_answer_id"] == 8 else 8
                else:
                    base_foil = (
                        row["donor_answer_id"] if answer_changes else _foil_id(row["base_answer"])
                    )
                    donor_foil = (
                        row["base_answer_id"] if answer_changes else _foil_id(row["donor_answer"])
                    )
                checks = _checks(config, row, base_foil, donor_foil, transform_id)
                failed = sorted(name for name, held in checks.items() if not held)
                if failed:
                    raise NumericSequenceCandidateError(
                        f"row {row.get('row_id')} in {family} failed: {failed}"
                    )
                output.append({
                    "schema": SCHEMA,
                    "task_id": config.task_id,
                    "seed": seed,
                    "hypothesis_id": row.get("hypothesis_id"),
                    "family_id": family,
                    "transform_id": transform_id,
                    "construction_id": construction_id,
                    "direction_id": direction_id,
                    "capability_cell_id": (
                        f"{construction_id}/{direction_id}"
                        f"/a{row['base_answer_id']}_{row['donor_answer_id']}"
                    ),
                    "group_id": group_id,
                    "row_id": f"{row.get('row_id')}:{direction_id}",
                    "split": row.get("split"),
                    "base_text": row["base_text"],
                    "donor_text": row["donor_text"],
                    "base_ids": row["base_ids"],
                    "donor_ids": row["donor_ids"],
                    "base_answer": row["base_answer"],
                    "donor_answer": row["donor_answer"],
                    "base_semantic_position": len(row["base_ids"]) - 1,
                    "donor_semantic_position": len(row["donor_ids"]) - 1,
                    "base_answer_id": row["base_answer_id"],
                    "donor_answer_id": row["donor_answer_id"],
                    "base_foil_id": base_foil,
                    "donor_foil_id": donor_foil,
                    "answer_changes": answer_changes,
                    "construction_checks": checks,
                })
    return output


def validate_rows(
    config: NumericSequenceCandidateConfig,
    rows,
    *,
    task_id: str,
    groups: int = DEFAULT_GROUPS,
    seed: int = DEFAULT_SEED,
) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != build_rows(config, task_id, groups, seed):
        raise NumericSequenceCandidateError("rows differ from the deterministic authority")
    try:
        digest = battery.validate_rows(config.task_spec, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise NumericSequenceCandidateError(str(error)) from error
    cells: dict[tuple[str, str], int] = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise NumericSequenceCandidateError("a stored construction check is false")
        key = (row["transform_id"], row["direction_id"])
        cells[key] = cells.get(key, 0) + 1
    for transform in ("A1", "A2", "P", "C"):
        for direction in ("base_to_donor", "donor_to_base"):
            if cells.get((transform, direction)) != groups:
                raise NumericSequenceCandidateError(
                    f"{transform}/{direction} is unbalanced: {cells.get((transform, direction))}"
                )
    return digest


def authority_sha256(
    config: NumericSequenceCandidateConfig,
    task_id: str,
    groups: int = DEFAULT_GROUPS,
    seed: int = DEFAULT_SEED,
) -> str:
    rows = build_rows(config, task_id, groups, seed)
    return validate_rows(config, rows, task_id=task_id, groups=groups, seed=seed)
