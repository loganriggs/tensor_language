"""Fresh registry/field-report temporal authority for response-program confirmation."""

from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as previous


lex = previous.lex
_head = lambda index: lex._REPORTERS[(index + 157) % len(lex._REPORTERS)][1]
_alternate = lambda index: lex._REPORTERS[(index + 157) % len(lex._REPORTERS)][0]
_object_a = lambda index: lex._OBJECTS[(index + 337) % len(lex._OBJECTS)]
_object_b = lambda index: lex._OBJECTS[(index + 359) % len(lex._OBJECTS)]
_cue = lambda future: "Next year" if future else "Last year"

SPEC = bs.BehaviourSpec(
    task_id="temporal_auxiliary.will_vs_had.fresh_registry_field_report",
    vocabulary=(" will", " had"),
    generator_role="generate_twelfth_fresh_temporal_auxiliary_panels",
    answer_role="score_jointly_tokenized_will_versus_had",
    a1=bs.Family(
        "twelfth_fresh_registry_frame", "fresh_registry_next_last_year_across_swap",
        lambda index, future: (
            f"According to the registry, {_cue(future)} the {_head(index)} across from the {_object_a(index)}"
        ),
    ),
    a2=bs.Family(
        "twelfth_fresh_field_report_frame", "fresh_field_report_next_last_year_beside_swap",
        lambda index, future: (
            f"The field report notes: {_cue(future)} the {_head(index)} beside the {_object_b(index)}"
        ),
    ),
    p_donor=lambda index, future: (
        f"According to the registry, {_cue(future)} the {_alternate(index)} across from the {_object_a(index)}"
    ),
    a1_suffix=lambda index: f" {_object_a(index)}",
    a2_suffix=lambda index: f" {_object_b(index)}",
    directions=("future_to_anterior", "anterior_to_future"),
    kinds=("future", "anterior"),
    p_generator_role="twelfth_fresh_head_noun_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
_build_rows, _validate_rows, authority_sha256 = SPEC.api()


def validate_rows(rows, *, task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=20261107):
    digest = _validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    targets = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    old_text = {row[key] for row in previous.build_rows() if row["transform_id"] in {"A1", "A2"}
                for key in ("base_text", "donor_text")}
    if len(targets) != 2 * groups:
        raise bs.CandidateBankError("twelfth fresh target population changed")
    if any(row[key] in old_text for row in targets for key in ("base_text", "donor_text")):
        raise bs.CandidateBankError("twelfth fresh target text overlaps v11")
    for row in targets:
        differences = [index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                       if pair[0] != pair[1]]
        if len(row["base_ids"]) != len(row["donor_ids"]) or len(differences) != 1:
            raise bs.CandidateBankError("twelfth fresh pair is not aligned at exactly one cue token")
    return digest


def build_rows(task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=20261107):
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
