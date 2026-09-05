"""Candidate: bracket_pending_opener.instrument_positive_control.

WHY THIS RUNS AT ALL. Two fast screens from this lane returned `no_selective_causal_site`
(numbered-list 21:15Z, numeric-sequence 22:46Z). The engine has never produced a positive, so those nulls
have an unexamined alternative reading: the fixed bars may simply be unreachable. R538 established that
`resid8` DOES transfer for `task.bracket.pending_opener` under full-state interchange, so this behaviour is
a **positive control on the instrument**. If the screen recovers a transferring site, the two nulls stand as
evidence about those behaviours; if it cannot, they say more about the bars than about the model.

Rediscovering resid8 is therefore NOT the claim -- it is the expected outcome and the point.

STIMULI ARE NOT INVENTED. Two frozen, already-validated files, both sha-pinned:
  pending_opener_multifamily_rows_rung537.json   A1/A2/P, 48 FIT rows per family
  pending_opener_controls_rung537.json           C, 48 FIT rows

  A1  opener_type_substitution              8 -> 1   substitute the opener type
  A2  closed_then_reopened_type             1 -> 8   a DIFFERENT route to the same variable
  P   pending_state_preserved_surface_edit  8 -> 8   answer-preserving surface edit
  C   nonopener_punctuation_substitution    answer-preserving, on the controls file

The endpoint vocabulary is exactly {")" = 8, '"' = 1}, so the foil is always the other closer -- the same
contrast R540 named (W_U[")"] - W_U['"']). C is answer-preserving, which the contract permits: only P is
constrained on `answer_changes`; C is constrained on its `expected_effect` label alone.

This module only BUILDS AND VALIDATES rows. It loads no model and runs nothing.
"""
# BQGATE: ANALYSIS  pred_a_rows_build_and_validate

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken

import circuit_battery_integration_contract as battery

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
ROWS_PATH = ROOT / "pending_opener_multifamily_rows_rung537.json"
CONTROL_PATH = ROOT / "pending_opener_controls_rung537.json"
PRIOR_ART = ROOT / "circuits/fast_screen_bracket_positive_control_prior_art.json"

SCHEMA = "circuit_fast_screen_candidate_v1"
TASK_ID = "bracket_pending_opener.instrument_positive_control"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260905
SPLIT = "FIT"
CLOSERS = (8, 1)                      # ")" and '"' -- the whole endpoint vocabulary
ENCODING = tiktoken.get_encoding("gpt2")

FAMILY_MAP = {
    "opener_type_substitution": ("A1", "opener_substitution"),
    "closed_then_reopened_type": ("A2", "closed_then_reopened"),
    "pending_state_preserved_surface_edit": ("P", "surface_preserved"),
    "nonopener_punctuation_substitution": ("C", "nonopener_punctuation"),
}

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="adapt_frozen_pending_opener_panels_to_linked_fit_rows",
    answer_role="score_jointly_tokenized_close_paren_vs_quote",
    transforms=(
        battery.TransformSpec("A1", "opener_type_substitution", True, "toward_donor"),
        battery.TransformSpec("A2", "closed_then_reopened_type", True, "toward_donor"),
        battery.TransformSpec("P", "pending_state_preserved_surface_edit", False, "invariant"),
        battery.TransformSpec("C", "nonopener_punctuation_substitution", False, "registered_active"),
    ),
)


class BracketControlError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _other_closer(answer_id: int) -> int:
    if answer_id not in CLOSERS:
        raise BracketControlError(f"answer {answer_id} is not one of {CLOSERS}")
    return CLOSERS[1] if answer_id == CLOSERS[0] else CLOSERS[0]


def _checks(base_ids, donor_ids, base_text, donor_text, b_ans, d_ans, b_foil, d_foil, transform_id):
    answer_changes = b_ans != d_ans
    return {
        "answers_in_closer_vocabulary": all(x in CLOSERS for x in (b_ans, d_ans, b_foil, d_foil)),
        "prompt_roundtrip": (ENCODING.decode(base_ids) == base_text
                             and ENCODING.decode(donor_ids) == donor_text),
        "joint_answer_tokenization": (
            ENCODING.encode(base_text + ENCODING.decode([b_ans])) == base_ids + [b_ans]
            and ENCODING.encode(donor_text + ENCODING.decode([d_ans])) == donor_ids + [d_ans]),
        "distinct_prompts": base_text != donor_text,
        "paired_answer_foil_alignment": ({b_ans, b_foil} == {d_ans, d_foil}
                                         and b_ans != b_foil and d_ans != d_foil),
        "answer_change_matches_transform": answer_changes == (transform_id in ("A1", "A2")),
        "foil_is_the_other_closer": b_foil == _other_closer(b_ans) and d_foil == _other_closer(d_ans),
    }


def _pick(source, family, groups):
    rows = [r for r in source if r.get("family_id") == family and r.get("split") == SPLIT][:groups]
    if len(rows) < groups:
        raise BracketControlError(f"family {family} has {len(rows)} FIT rows, need {groups}")
    return rows


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED):
    if task_id != TASK_ID:
        raise BracketControlError(f"unknown task_id {task_id!r}")
    main = json.loads(ROWS_PATH.read_text())["rows"]
    ctrl = json.loads(CONTROL_PATH.read_text())["rows"]
    picked = {}
    for family, (transform_id, _c) in FAMILY_MAP.items():
        picked[transform_id] = _pick(ctrl if transform_id == "C" else main, family, groups)
    out = []
    for direction_id in ("base_to_donor", "donor_to_base"):
        for index in range(groups):
            group_id = f"{TASK_ID}:{direction_id}:{index:03d}"
            for family, (transform_id, construction_id) in FAMILY_MAP.items():
                row = picked[transform_id][index]
                if transform_id == "C":
                    # the controls file carries a single `answer_id`, identical on both sides
                    b_ans = d_ans = int(row["answer_id"])
                else:
                    b_ans, d_ans = int(row["base_answer_id"]), int(row["donor_answer_id"])
                b_foil, d_foil = _other_closer(b_ans), _other_closer(d_ans)
                checks = _checks(row["base_ids"], row["donor_ids"], row["base_text"], row["donor_text"],
                                 b_ans, d_ans, b_foil, d_foil, transform_id)
                failed = sorted(k for k, ok in checks.items() if not ok)
                if failed:
                    raise BracketControlError(f"{family} row {index} failed: {failed}")
                out.append({
                    "schema": SCHEMA, "task_id": TASK_ID, "seed": seed,
                    "family_id": family, "transform_id": transform_id,
                    "construction_id": construction_id, "direction_id": direction_id,
                    "capability_cell_id": f"{construction_id}/{direction_id}/a{b_ans}_{d_ans}",
                    "group_id": group_id,
                    "row_id": f"{family}:{index:03d}:{direction_id}",
                    "split": SPLIT,
                    "base_text": row["base_text"], "donor_text": row["donor_text"],
                    "base_ids": row["base_ids"], "donor_ids": row["donor_ids"],
                    "base_answer": ENCODING.decode([b_ans]), "donor_answer": ENCODING.decode([d_ans]),
                    "base_answer_id": b_ans, "donor_answer_id": d_ans,
                    "base_foil_id": b_foil, "donor_foil_id": d_foil,
                    "base_semantic_position": len(row["base_ids"]) - 1,
                    "donor_semantic_position": len(row["donor_ids"]) - 1,
                    "answer_changes": b_ans != d_ans,
                    "construction_checks": checks,
                })
    return out


def validate_rows(rows, *, task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    materialized = [dict(r) for r in rows]
    if materialized != build_rows(task_id, groups, seed):
        raise BracketControlError("rows differ from the deterministic authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise BracketControlError(str(error)) from error
    cells = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise BracketControlError("a stored construction check is false")
        key = (row["transform_id"], row["direction_id"])
        cells[key] = cells.get(key, 0) + 1
    for transform in ("A1", "A2", "P", "C"):
        for direction in ("base_to_donor", "donor_to_base"):
            if cells.get((transform, direction)) != groups:
                raise BracketControlError(f"{transform}/{direction} unbalanced: {cells.get((transform, direction))}")
    return digest


def authority_sha256(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    return validate_rows(build_rows(task_id, groups, seed), task_id=task_id, groups=groups, seed=seed)


if __name__ == "__main__":
    rows = build_rows()
    print(f"{len(rows)} rows, {len({(r['transform_id'], r['capability_cell_id']) for r in rows})} cells, "
          f"sources sha {_sha256(ROWS_PATH)[:12]} / {_sha256(CONTROL_PATH)[:12]}")
    print("authority", authority_sha256())
