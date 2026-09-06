"""Fifth, capability-first Tomorrow/Yesterday temporal authority."""

from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidate_temporal_auxiliary as original
import circuit_fast_screen_candidates as lex
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fresh_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as fresh_v2
import circuit_candidate_temporal_auxiliary_fresh_cues_v3 as fresh_v3
import circuit_candidate_temporal_auxiliary_fresh_cues_v4 as fresh_v4


_head = lambda index: lex._REPORTERS[(index + 31) % len(lex._REPORTERS)][1]
_alternate = lambda index: lex._REPORTERS[(index + 31) % len(lex._REPORTERS)][0]
_object_a = lambda index: lex._OBJECTS[(index + 109) % len(lex._OBJECTS)]
_object_b = lambda index: lex._OBJECTS[(index + 127) % len(lex._OBJECTS)]
_cue = lambda future: "Tomorrow" if future else "Yesterday"

SPEC = bs.BehaviourSpec(
    task_id="temporal_auxiliary.will_vs_had.fresh_tomorrow_yesterday",
    vocabulary=(" will", " had"),
    generator_role="generate_fifth_fresh_temporal_auxiliary_panels",
    answer_role="score_jointly_tokenized_will_versus_had",
    a1=bs.Family(
        "fifth_fresh_bare_frame", "fresh_tomorrow_yesterday_swap",
        lambda index, future: f"{_cue(future)} the {_head(index)} beyond the {_object_a(index)}",
    ),
    a2=bs.Family(
        "fifth_fresh_forecast_frame", "fresh_forecast_tomorrow_yesterday_swap",
        lambda index, future: (
            f"The forecast says: {_cue(future)} the {_head(index)} near the {_object_b(index)}"
        ),
    ),
    p_donor=lambda index, future: (
        f"{_cue(future)} the {_alternate(index)} beyond the {_object_a(index)}"
    ),
    a1_suffix=lambda index: f" {_object_a(index)}",
    a2_suffix=lambda index: f" {_object_b(index)}",
    directions=("future_to_anterior", "anterior_to_future"),
    kinds=("future", "anterior"),
    p_generator_role="fifth_fresh_head_noun_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
_build_rows, _validate_rows, authority_sha256 = SPEC.api()


def validate_rows(rows, *, task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=20261028):
    digest = _validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    targets = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    prior_rows = [*original.build_rows(), *fresh_v1.build_rows(), *fresh_v2.build_rows(),
                  *fresh_v3.build_rows(), *fresh_v4.build_rows()]
    old_text = {row[key] for row in prior_rows if row["transform_id"] in {"A1", "A2"}
                for key in ("base_text", "donor_text")}
    if len(targets) != 2 * groups:
        raise bs.CandidateBankError("fifth fresh target population changed")
    if any(row[key] in old_text for row in targets for key in ("base_text", "donor_text")):
        raise bs.CandidateBankError("fifth fresh target text overlaps an earlier authority")
    for row in targets:
        differences = [index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                       if pair[0] != pair[1]]
        if len(row["base_ids"]) != len(row["donor_ids"]) or len(differences) != 1:
            raise bs.CandidateBankError("fifth fresh pair is not aligned at exactly one cue token")
    return digest


def build_rows(task_id=TASK_ID, groups=bs.DEFAULT_GROUPS, seed=20261028):
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
