"""Candidate: numeric_sequence.cross_construction_sufficiency.

Does a single site carry the numeric-sequence state across BOTH surface constructions -- digit ("8, 9, 10,"
-> " 11") and word ("eight, nine, ten," -> " eleven")?

Prior art (task_numeric_sequence_continuation.json): native capability HELD (r570, audited r571);
`numeric_sequence_cached_value_reuse.r576` NULL at final_label_l0_value_through_l8h3_h7;
`numeric_sequence_complete_state_factor_localization.r577` NULL on its factor ladder, audited HELD (r583).
ADJACENCY STATED PLAINLY: r577 already looked for the complete state on a factor ladder and found none. This
is not that test -- it is natural donor interchange over the engine's ceiling site set, and its cross-
construction axis (digit vs word) is not present in any recorded event -- but the two questions are close
enough that a null here would sharpen r577 rather than surprise anyone.

Stimuli are NOT invented: the frozen, validated `increment_two_hypothesis_rows_rung567.json`, sha-pinned.

This module only BUILDS AND VALIDATES rows. It loads no model and runs nothing.
"""
# BQGATE: ANALYSIS  pred_a_rows_build_and_validate

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import tiktoken

import circuit_battery_integration_contract as battery

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
ROWS_PATH = ROOT / "increment_two_hypothesis_rows_rung567.json"
ROWS_SHA256 = "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053"
PRIOR_ART = ROOT / "circuits/fast_screen_numeric_sequence_cross_construction_prior_art.json"

SCHEMA = "circuit_fast_screen_candidate_v1"
TASK_ID = "numeric_sequence.cross_construction_sufficiency"
HYPOTHESIS = "numeric_sequence_continuation"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260904
SPLIT = "FIT"
ENCODING = tiktoken.get_encoding("gpt2")

# family -> (transform_id, construction_id)
FAMILY_MAP = {
    # A1/A2 are the SAME state in two genuinely different surface constructions:
    #   digit  "8, 9, 10,"                  -> " 11"   (token 1367)
    #   word   "eight, nine, ten,"          -> " eleven" (token 22216)
    # so pred_b (cross-construction transfer) is a real test rather than two spellings of one arm.
    "sequence_digit_state_shift": ("A1", "digit_format"),
    "sequence_word_state_shift": ("A2", "word_format"),
    # P is answer-preserving and DIGIT-only on purpose: the synthesised competitor foil is "the previous
    # numeral", which is computable for digits and not for word numerals.
    "sequence_digit_surface_preserved": ("P", "digit_surface_preserved"),
    # C is `registered_active`: answer-changing on an UNRELATED behaviour that shares numeral endpoints --
    # here the numbered-list index successor, the mirror of the choice made for the list screen.
    "list_two_line_state_shift": ("C", "numbered_list_control"),
}


TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="adapt_frozen_numeric_sequence_panels_to_linked_cross_construction_fit_rows",
    answer_role="score_jointly_tokenized_final_label_plus_one_vs_final_label",
    transforms=(
        battery.TransformSpec("A1", "digit_sequence_state_shift", True, "toward_donor"),
        battery.TransformSpec("A2", "word_sequence_state_shift", True, "toward_donor"),
        battery.TransformSpec("P", "digit_surface_preserved_rewrite", False, "invariant"),
        battery.TransformSpec("C", "numbered_list_active_control", True, "registered_active"),
    ),
)


class NumberedListSufficiencyError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _foil_id(answer_text: str) -> int:
    """The competitor endpoint: the final label itself, i.e. one below the successor."""
    try:
        value = int(answer_text)
    except ValueError as exc:
        raise NumberedListSufficiencyError(f"non-numeric answer {answer_text!r}") from exc
    ids = ENCODING.encode(str(value - 1))
    if len(ids) != 1:
        raise NumberedListSufficiencyError(f"foil {value - 1} is not a single token")
    return ids[0]


def _checks(row, base_foil, donor_foil, transform_id):
    answer_changes = row["base_answer_id"] != row["donor_answer_id"]
    return {
        "single_token_answers": all(
            len(ENCODING.encode(row[k])) == 1 for k in ("base_answer", "donor_answer")),
        "joint_answer_tokenization": (
            ENCODING.encode(row["base_text"] + row["base_answer"])
            == row["base_ids"] + [row["base_answer_id"]]
            and ENCODING.encode(row["donor_text"] + row["donor_answer"])
            == row["donor_ids"] + [row["donor_answer_id"]]),
        "prompt_roundtrip": (
            ENCODING.decode(row["base_ids"]) == row["base_text"]
            and ENCODING.decode(row["donor_ids"]) == row["donor_text"]),
        "distinct_prompts": row["base_text"] != row["donor_text"],
        "paired_answer_foil_alignment": (
            {row["base_answer_id"], base_foil} == {row["donor_answer_id"], donor_foil}
            and row["base_answer_id"] != base_foil and row["donor_answer_id"] != donor_foil),
        # Under the battery protocol only P preserves the answer: A1/A2 are `toward_donor` and C is
        # `registered_active`, all answer-changing. An earlier version of this check assumed C was
        # answer-preserving and rejected every C row.
        "answer_change_matches_transform": answer_changes == (transform_id != "P"),
        # Only meaningful where the foil is SYNTHESISED (the invariance families). For an answer-changing
        # row the foil is the paired answer, which `paired_answer_foil_alignment` already pins; asserting
        # the competitor rule there compared 24 against 22 and failed every A1 row.
        "foil_is_the_registered_competitor": (
            True if answer_changes else base_foil == _foil_id(row["base_answer"])),
    }


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED):
    if task_id != TASK_ID:
        raise NumberedListSufficiencyError(f"unknown task_id {task_id!r}")
    if _sha256(ROWS_PATH) != ROWS_SHA256:
        raise NumberedListSufficiencyError("frozen stimulus rows changed; refusing to build")
    source = json.loads(ROWS_PATH.read_text())["rows"]
    # One GROUP is a linked panel: exactly one A1, A2, P and C row, all in the same ordered direction.
    # The frozen file's own group_id groups differently (by stimulus family), so panels are built here.
    picked = {}
    for family, (transform_id, _construction_id) in FAMILY_MAP.items():
        rows_for_family = [r for r in source
                           if r.get("family_id") == family and r.get("split") == SPLIT][:groups]
        if len(rows_for_family) < groups:
            raise NumberedListSufficiencyError(
                f"family {family} has {len(rows_for_family)} rows, need {groups}")
        picked[transform_id] = rows_for_family
    out = []
    for direction_id in ("base_to_donor", "donor_to_base"):
        for index in range(groups):
            group_id = f"{TASK_ID}:{direction_id}:{index:03d}"
            for transform_id, (family, construction_id) in (
                    (t, (f, c)) for f, (t, c) in FAMILY_MAP.items()):
                row = picked[transform_id][index]
                answer_changes = row["base_answer_id"] != row["donor_answer_id"]
                base_foil = row["donor_answer_id"] if answer_changes else _foil_id(row["base_answer"])
                donor_foil = row["base_answer_id"] if answer_changes else _foil_id(row["donor_answer"])
                checks = _checks(row, base_foil, donor_foil, transform_id)
                failed = sorted(k for k, ok in checks.items() if not ok)
                if failed:
                    raise NumberedListSufficiencyError(
                        f"row {row.get('row_id')} in {family} failed: {failed}")
                out.append({
                    "schema": SCHEMA, "task_id": TASK_ID, "seed": seed,
                    "hypothesis_id": row.get("hypothesis_id"), "family_id": family,
                    "transform_id": transform_id, "construction_id": construction_id,
                    "direction_id": direction_id,
                    # Two constraints, both read out of the engine rather than guessed:
                    # (1) no transform prefix -- the harness composes "{transform_id}/{capability_cell_id}",
                    #     so embedding it produced "A1/A1/two_line/..." and invalidated the 20:48 run;
                    # (2) the ENDPOINT PAIR must be part of the cell -- `producer` groups capability by
                    #     (family, cell_id, recipient_answer_id, donor_answer_id) but aggregates keyed only
                    #     on (family, cell_id), so a cell spanning two answer pairs emits two entries with
                    #     the same key and the kernel rejects it as a duplicate. Codex's pronoun screen has
                    #     exactly ONE answer pair per cell (he/she); a numbered list ends at a different
                    #     index in every panel, so the pair has to be named. That invalidated the 21:07 run.
                    "capability_cell_id": (
                        f"{construction_id}/{direction_id}"
                        f"/a{row['base_answer_id']}_{row['donor_answer_id']}"),
                    "group_id": group_id,
                    "row_id": f"{row.get('row_id')}:{direction_id}",
                    "split": row.get("split"),
                    "base_text": row["base_text"], "donor_text": row["donor_text"],
                    "base_ids": row["base_ids"], "donor_ids": row["donor_ids"],
                    "base_answer": row["base_answer"], "donor_answer": row["donor_answer"],
                    "base_semantic_position": len(row["base_ids"]) - 1,
                    "donor_semantic_position": len(row["donor_ids"]) - 1,
                    "base_answer_id": row["base_answer_id"],
                    "donor_answer_id": row["donor_answer_id"],
                    "base_foil_id": base_foil, "donor_foil_id": donor_foil,
                    "answer_changes": answer_changes,
                    "construction_checks": checks,
                })
    return out


def validate_rows(rows, *, task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
                  seed: int = DEFAULT_SEED) -> str:
    """Recompute the authority from the frozen source and return its canonical digest."""
    materialized = [dict(r) for r in rows]
    if materialized != build_rows(task_id, groups, seed):
        raise NumberedListSufficiencyError("rows differ from the deterministic authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise NumberedListSufficiencyError(str(error)) from error
    cells = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise NumberedListSufficiencyError("a stored construction check is false")
        key = (row["transform_id"], row["direction_id"])
        cells[key] = cells.get(key, 0) + 1
    for transform in ("A1", "A2", "P", "C"):
        for direction in ("base_to_donor", "donor_to_base"):
            if cells.get((transform, direction)) != groups:
                raise NumberedListSufficiencyError(
                    f"{transform}/{direction} is unbalanced: {cells.get((transform, direction))}")
    return digest


def authority_sha256(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    return validate_rows(build_rows(task_id, groups, seed), task_id=task_id, groups=groups, seed=seed)


if __name__ == "__main__":
    rows = build_rows()
    cells = sorted({r["capability_cell_id"] for r in rows})
    print(f"{len(rows)} rows, {len(cells)} capability cells, authority {authority_sha256()[:16]}")
    for c in cells:
        n = sum(1 for r in rows if r["capability_cell_id"] == c)
        print(f"   {c:<44} {n}")
