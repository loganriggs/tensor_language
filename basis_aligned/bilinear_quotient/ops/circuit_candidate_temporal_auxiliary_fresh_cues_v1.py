"""Fresh Later/Previously authority for the identified will/had cue path."""

from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidate_temporal_auxiliary as original
import circuit_fast_screen_candidates as lex


_head = lambda index: lex._REPORTERS[index][1]
_alternate = lambda index: lex._REPORTERS[index][0]
_object_a = lambda index: lex._OBJECTS[(index + 11) % len(lex._OBJECTS)]
_object_b = lambda index: lex._OBJECTS[(index + 23) % len(lex._OBJECTS)]
_cue = lambda future: "Later" if future else "Previously"

SPEC = bs.BehaviourSpec(
    task_id="temporal_auxiliary.will_vs_had.fresh_later_previously",
    vocabulary=(" will", " had"),
    generator_role="generate_fresh_temporal_auxiliary_path_panels",
    answer_role="score_jointly_tokenized_will_versus_had",
    a1=bs.Family(
        "fresh_bare_frame",
        "fresh_later_previously_swap",
        lambda index, future: f"{_cue(future)} the {_head(index)} beyond the {_object_a(index)}",
    ),
    a2=bs.Family(
        "fresh_note_frame",
        "fresh_note_later_previously_swap",
        lambda index, future: f"The note reads: {_cue(future)} the {_head(index)} across the {_object_b(index)}",
    ),
    p_donor=lambda index, future: (
        f"{_cue(future)} the {_alternate(index)} beyond the {_object_a(index)}"
    ),
    a1_suffix=lambda index: f" {_object_a(index)}",
    a2_suffix=lambda index: f" {_object_b(index)}",
    directions=("future_to_anterior", "anterior_to_future"),
    kinds=("future", "anterior"),
    p_generator_role="fresh_head_noun_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
_build_rows, _validate_rows, authority_sha256 = SPEC.api()


def validate_rows(rows, *, task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=bs.DEFAULT_SEED):
    digest = _validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    targets = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    old_text = {
        row[key]
        for row in original.build_rows()
        if row["transform_id"] in {"A1", "A2"}
        for key in ("base_text", "donor_text")
    }
    if len(targets) != 2 * groups:
        raise bs.CandidateBankError("fresh target population changed")
    if any(row[key] in old_text for row in targets for key in ("base_text", "donor_text")):
        raise bs.CandidateBankError("fresh target text overlaps discovery authority")
    for row in targets:
        base_ids, donor_ids = row["base_ids"], row["donor_ids"]
        differences = [index for index, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]
        if len(base_ids) != len(donor_ids) or len(differences) != 1:
            raise bs.CandidateBankError("fresh pair is not aligned at exactly one cue token")
    return digest


def build_rows(task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=bs.DEFAULT_SEED):
    rows = _build_rows(task_id, groups, seed)
    validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    return rows


if __name__ == "__main__":
    materialized = build_rows()
    print("authority:", validate_rows(materialized))
    for family in ("A1", "A2"):
        row = next(item for item in materialized if item["transform_id"] == family)
        print(family, row["base_text"], "->", row["base_answer"])
        print(family, row["donor_text"], "->", row["donor_answer"])
