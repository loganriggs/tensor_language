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
PRIOR_ART = ROOT / "circuits/fast_screen_a2_family_prior_art.json"
CONTROL_PATH = ROOT / "pending_opener_controls_rung537.json"

SCHEMA = "circuit_fast_screen_candidate_v1"
TASK_ID = "numeric_sequence.a2_family_discriminator"
HYPOTHESIS = "numeric_sequence_continuation"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260904
SPLIT = "FIT"
ENCODING = tiktoken.get_encoding("gpt2")

# family -> (transform_id, construction_id)
FAMILY_MAP = {
    # A1/A2/P identical to the 22:46Z screen; ONLY C changes, numeral -> non-numeral. The 02:15Z
    # numbered-list discriminator flipped that screen's verdict by exactly this substitution.
    "sequence_digit_state_shift": ("A1", "digit_format"),
    # A2 SWAPPED, and that is the whole experiment. The 03:15Z run used `sequence_word_state_shift`
    # (within-format, word). `sequence_cross_format_shift` interchanges ACROSS formats -- digit base,
    # word donor -- so it is a different operationalisation of "the same state, another construction".
    # C was varied twice and P once, each time overturning a verdict; A2 has never been varied and the
    # surviving attn:08 claim depends on it.
    "sequence_cross_format_shift": ("A2", "cross_format"),
    # P SWAPPED, and that is the whole experiment. The 02:45Z run used `sequence_digit_surface_preserved`
    # and attn:08 failed P_invariance at 0.2517 against a 0.2 bar, which is my headline asymmetry against
    # numbered-list (0.0224 there). Having just found that a CONTROL family choice can drive a verdict, the
    # same doubt applies to the P family. `sequence_word_surface_preserved` is the same behaviour, a
    # different surface family: if the P failure persists it is about the SITE, if it vanishes it was about
    # the P family.
    "sequence_word_surface_preserved": ("P", "word_surface_preserved"),
    "nonopener_punctuation_substitution": ("C", "bracket_punctuation_control"),
}


TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="adapt_frozen_numeric_sequence_panels_to_linked_cross_construction_fit_rows",
    answer_role="score_jointly_tokenized_final_label_plus_one_vs_final_label",
    transforms=(
        battery.TransformSpec("A1", "digit_sequence_state_shift", True, "toward_donor"),
        battery.TransformSpec("A2", "cross_format_sequence_state_shift", True, "toward_donor"),
        battery.TransformSpec("P", "digit_surface_preserved_rewrite", False, "invariant"),
        battery.TransformSpec("C", "bracket_nonopener_punctuation_control", False, "registered_active"),
    ),
)


def _declared_changes(transform_id: str) -> bool:
    for spec in TASK_SPEC.transforms:
        if spec.transform_id == transform_id:
            return bool(spec.answer_changes)
    raise NumberedListSufficiencyError(f"unknown transform {transform_id}")


class NumberedListSufficiencyError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The word-format P family answers with " eleven" / " twelve". Its competitor, matching the digit rule
# ("the previous value"), is the preceding word numeral.
WORD_PREV = {" eleven": " ten", " twelve": " eleven"}


def _foil_id(answer_text: str) -> int:
    """The competitor endpoint: the final label itself, i.e. one below the successor."""
    if answer_text in WORD_PREV:
        ids = ENCODING.encode(WORD_PREV[answer_text])
        if len(ids) != 1:
            raise NumberedListSufficiencyError(f"word foil {WORD_PREV[answer_text]!r} is not one token")
        return ids[0]
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
        "answer_change_matches_transform": answer_changes == _declared_changes(transform_id),
        # Only meaningful where the foil is SYNTHESISED (the invariance families). For an answer-changing
        # row the foil is the paired answer, which `paired_answer_foil_alignment` already pins; asserting
        # the competitor rule there compared 24 against 22 and failed every A1 row.
        "foil_is_the_registered_competitor": (
            True if (answer_changes or transform_id == "C")
            else base_foil == _foil_id(row["base_answer"])),
    }


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED):
    if task_id != TASK_ID:
        raise NumberedListSufficiencyError(f"unknown task_id {task_id!r}")
    if _sha256(ROWS_PATH) != ROWS_SHA256:
        raise NumberedListSufficiencyError("frozen stimulus rows changed; refusing to build")
    source = json.loads(ROWS_PATH.read_text())["rows"]
    ctrl = json.loads(CONTROL_PATH.read_text())["rows"]
    # One GROUP is a linked panel: exactly one A1, A2, P and C row, all in the same ordered direction.
    # The frozen file's own group_id groups differently (by stimulus family), so panels are built here.
    picked = {}
    for family, (transform_id, _construction_id) in FAMILY_MAP.items():
        pool = ctrl if transform_id == "C" else source
        rows_for_family = [r for r in pool
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
                if transform_id == "C":
                    b_ans = int(row["answer_id"])
                    row = dict(row, base_answer_id=b_ans, donor_answer_id=b_ans,
                               base_answer=ENCODING.decode([b_ans]), donor_answer=ENCODING.decode([b_ans]))
                answer_changes = row["base_answer_id"] != row["donor_answer_id"]
                if transform_id == "C":
                    base_foil = donor_foil = 1 if row["base_answer_id"] == 8 else 8
                else:
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
