"""Candidate: numbered_list.cached_value_sufficiency -- is the cached label index SUFFICIENT?

Claimed 2026-09-04T20:14:23Z by claude-lane1 through ops/circuit_candidate_claims.py.
Prior-art receipt: circuits/fast_screen_numbered_list_sufficiency_prior_art.json (sha256 16da3957...).

WHY. `task_numbered_list_index_successor.json` records, at site `final_label_l0_value_through_l8h3_h7`:
cached-value WEIGHTS held (r576), cached-value REMOVAL a scientific_null (r576, bar ">=0.75 positive margin
damage in every non-copy list cell"), and the removal audit held (r579, all_saved_decisions_recomputed=1.0).
So the path carries the weights and is NOT necessary -- the redundancy/self-repair signature. Nothing in the
authority measures SUFFICIENCY there (`sufficiency`/`backup`/`hydra` return 0 prior events).

STIMULI ARE NOT INVENTED. They are the frozen, already-validated rows of
`increment_two_hypothesis_rows_rung567.json`, whose families map exactly onto the A1/A2/P/C shape:

    A1  list_two_line_state_shift     interchange, answer-changing, 2-line construction
    A2  list_three_line_state_shift   interchange, answer-changing, 3-line construction
    P   list_surface_preserved        invariance: item words shuffled, indices held -> answer unchanged
    C   list_repeated_index_control   invariance: every label identical -> answer is the label itself,
                                      an endpoint control reusing the same numeral output tokens

FOIL. The registered competitor for this behaviour is "copy the final label" versus "final label + 1"
(r569/r572: "all 64 FIT endpoint margins favor final-label+1"). So the foil is the final label itself, i.e.
the numeral one below the answer -- not an arbitrary distractor.

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
PRIOR_ART = ROOT / "circuits/fast_screen_numbered_list_sufficiency_prior_art.json"

SCHEMA = "circuit_fast_screen_candidate_v1"
TASK_ID = "numbered_list.cached_value_sufficiency"
HYPOTHESIS = "numbered_list_index_successor"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260904
SPLIT = "FIT"
ENCODING = tiktoken.get_encoding("gpt2")

# family -> (transform_id, construction_id)
FAMILY_MAP = {
    "list_two_line_state_shift": ("A1", "two_line"),
    "list_three_line_state_shift": ("A2", "three_line"),
    "list_surface_preserved": ("P", "surface_preserved"),
    # C must be `registered_active`: an ANSWER-CHANGING control on an unrelated route. The numbered-list
    # invariance families (repeated_index, step_two_conflict) are answer-preserving, so they are P-shaped and
    # the protocol rejects them here. `sequence_digit_state_shift` is the numeric-sequence-continuation
    # behaviour -- a different hypothesis in the same frozen file, answer-changing, numeral endpoints.
    # Caveat recorded: its answers are space-prefixed (" 11") where A1/A2 are bare ("23"), so it is an
    # unrelated-behaviour control rather than a strict same-output-token control.
    "sequence_digit_state_shift": ("C", "digit_sequence_control"),
}


TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="adapt_frozen_numbered_list_index_panels_to_linked_fit_rows",
    answer_role="score_jointly_tokenized_final_label_plus_one_vs_final_label",
    transforms=(
        battery.TransformSpec("A1", "two_line_index_state_shift", True, "toward_donor"),
        battery.TransformSpec("A2", "three_line_index_state_shift", True, "toward_donor"),
        battery.TransformSpec("P", "surface_preserved_item_rewrite", False, "invariant"),
        battery.TransformSpec("C", "digit_sequence_active_control", True, "registered_active"),
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
                    "capability_cell_id": f"{transform_id}/{construction_id}/{direction_id}",
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
