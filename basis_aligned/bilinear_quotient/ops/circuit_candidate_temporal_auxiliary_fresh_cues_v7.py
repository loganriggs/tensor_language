"""Seventh, capability-first Next-week/Last-week temporal authority."""

from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_candidate_temporal_auxiliary_fresh_cues_v6 as previous


lex = previous.lex
_head = lambda index: lex._REPORTERS[(index + 53) % len(lex._REPORTERS)][1]
_alternate = lambda index: lex._REPORTERS[(index + 53) % len(lex._REPORTERS)][0]
_object_a = lambda index: lex._OBJECTS[(index + 163) % len(lex._OBJECTS)]
_object_b = lambda index: lex._OBJECTS[(index + 179) % len(lex._OBJECTS)]
_cue = lambda future: "Next week" if future else "Last week"

SPEC = bs.BehaviourSpec(
    task_id="temporal_auxiliary.will_vs_had.fresh_next_last_week",
    vocabulary=(" will", " had"),
    generator_role="generate_seventh_fresh_temporal_auxiliary_panels",
    answer_role="score_jointly_tokenized_will_versus_had",
    a1=bs.Family(
        "seventh_fresh_bare_frame", "fresh_next_last_week_swap",
        lambda index, future: f"{_cue(future)} the {_head(index)} alongside the {_object_a(index)}",
    ),
    a2=bs.Family(
        "seventh_fresh_schedule_frame", "fresh_schedule_next_last_week_swap",
        lambda index, future: (
            f"The schedule says: {_cue(future)} the {_head(index)} near the {_object_b(index)}"
        ),
    ),
    p_donor=lambda index, future: (
        f"{_cue(future)} the {_alternate(index)} alongside the {_object_a(index)}"
    ),
    a1_suffix=lambda index: f" {_object_a(index)}",
    a2_suffix=lambda index: f" {_object_b(index)}",
    directions=("future_to_anterior", "anterior_to_future"),
    kinds=("future", "anterior"),
    p_generator_role="seventh_fresh_head_noun_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
_build_rows, _validate_rows, authority_sha256 = SPEC.api()


def validate_rows(rows, *, task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=20261030):
    digest = _validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    targets = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    prior_rows = previous.build_rows()
    old_text = {row[key] for row in prior_rows if row["transform_id"] in {"A1", "A2"}
                for key in ("base_text", "donor_text")}
    if len(targets) != 2 * groups:
        raise bs.CandidateBankError("seventh fresh target population changed")
    if any(row[key] in old_text for row in targets for key in ("base_text", "donor_text")):
        raise bs.CandidateBankError("seventh fresh target text overlaps v6")
    for row in targets:
        differences = [index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                       if pair[0] != pair[1]]
        if len(row["base_ids"]) != len(row["donor_ids"]) or len(differences) != 1:
            raise bs.CandidateBankError("seventh fresh pair is not aligned at exactly one cue token")
    return digest


def build_rows(task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=20261030):
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
