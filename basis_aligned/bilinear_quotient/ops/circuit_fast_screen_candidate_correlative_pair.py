"""correlative_pair.both_vs_neither -- a second correlative, testing generality across PAIRS.

`correlative_state` established that `either` obliges `or`. This is a different correlative with
a different second element and a disjoint answer vocabulary: `both` obliges ` and`, `neither`
obliges ` nor`.

It also differs in kind. `either` and `neither` both open a two-way choice; `both` and `neither`
differ in POLARITY over a conjunction, so what the model must carry is a positive-versus-negative
distinction rather than a choice between two disjuncts.

    A1 bare frame
      "The leader praised both the clerk"    -> " and"
      "The leader praised neither the clerk" -> " nor"
    A2 report frame, a second matrix verb
      "In the notes the leader named both the clerk"    -> " and"
      "In the notes the leader named neither the clerk" -> " nor"

Recipient and donor differ only in the correlative word and end on the same noun token. P swaps
the AGENT, safe here because the object noun is final (see the design invariants in
ops/README.md).

Control. Canonical same-answer control v2, so this C joins the comparable set directly.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_canonical_control_v2 as canonical
import circuit_fast_screen_candidates as lex

canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED

TASK_ID = "correlative_pair.both_vs_neither"
PAIRED = (" and", " nor")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_correlative_pair_fit_panels",
    answer_role="score_jointly_tokenized_and_versus_nor",
    transforms=(
        battery.TransformSpec("A1", "bare_frame_correlative_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_frame_correlative_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_AGENTS_ALT = tuple(pair[1] for pair in lex._REPORTERS)
# Objects are rotated off the agent table so the agent and the object are never the same
# noun -- v1 of this module produced "The leader praised both the leader".
_A1_PARTICIPLES = tuple(lex._REPORTERS[(i + 11) % len(lex._REPORTERS)][0]
                        for i in range(len(lex._REPORTERS)))
_A2_PARTICIPLES = tuple(lex._REPORTERS[(i + 19) % len(lex._REPORTERS)][0]
                        for i in range(len(lex._REPORTERS)))
if not (len(_AGENTS) == len(_A1_PARTICIPLES) == len(_A2_PARTICIPLES) == DEFAULT_GROUPS):
    raise RuntimeError("voice tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _bare(agent: str, obj: str, positive: bool) -> str:
    return f"The {agent} praised {'both' if positive else 'neither'} the {obj}"


def _report(agent: str, obj: str, positive: bool) -> str:
    return f"In the notes the {agent} named {'both' if positive else 'neither'} the {obj}"


def _answer(positive: bool) -> str:
    return PAIRED[0] if positive else PAIRED[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "both_to_neither" if forward else "neither_to_both"
    agent, alt_agent = _AGENTS[case_index], _AGENTS_ALT[case_index]
    part1, part2 = _A1_PARTICIPLES[case_index], _A2_PARTICIPLES[case_index]
    # P preserves the answer by swapping the SUBJECT while holding the voice.
    phrase = f" {part1}"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("both" if base_more else "neither",
             "both" if donor_more else "neither")
    common = dict(common_no_vocab, vocabulary=PAIRED)
    return [
        builder._row(**common, transform_id="A1", construction_id="bare_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_bare(agent, part1, base_more),
                     donor_text=_bare(agent, part1, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="report_frame",
                     direction_id=direction, matched_suffix=f" {part2}",
                     base_text=_report(agent, part2, base_more),
                     donor_text=_report(agent, part2, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"bare_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_bare(agent, part1, base_more),
                     donor_text=_bare(alt_agent, part1, base_more),
                     base_answer=_answer(base_more), donor_answer=_answer(base_more),
                     sentence_types=(kinds[0], kinds[0])),
        builder._row(**common_no_vocab, transform_id="C",
                     **canonical.row_kwargs(case_index, forward)),
    ]


def _build(groups: int, seed: int) -> list[dict[str, Any]]:
    order = lex._permutation(seed)
    return [row for group_number in range(groups)
            for row in _panel(seed, group_number, order[group_number])]


def _validate(rows, groups: int, seed: int) -> str:
    if type(groups) is not int or not 2 <= groups <= DEFAULT_GROUPS or groups % 2:
        raise CandidateBankError("groups must be an even integer from 2 through 32")
    materialized = [dict(row) for row in rows]
    if materialized != _build(groups, seed):
        raise CandidateBankError("rows differ from the deterministic semantic authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    if len({str(row["row_id"]) for row in materialized}) != len(materialized):
        raise CandidateBankError("row IDs are not unique")
    cells: dict[tuple[str, str], int] = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise CandidateBankError("a stored construction check is false")
        key = (str(row["transform_id"]), str(row["direction_id"]))
        cells[key] = cells.get(key, 0) + 1
    half = groups // 2
    for transform in ("A1", "A2"):
        if cells.get((transform, "both_to_neither")) != half \
                or cells.get((transform, "neither_to_both")) != half:
            raise CandidateBankError(f"{transform} ordered directions are unbalanced")
    if cells.get(("C", "base_to_donor")) != half or cells.get(("C", "donor_to_base")) != half:
        raise CandidateBankError("C ordered directions are unbalanced")
    return digest


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rows = _build(groups, seed)
    _validate(rows, groups, seed)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
                  groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    return _validate(rows, groups, seed)


def authority_sha256(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
                     seed: int = DEFAULT_SEED) -> str:
    return _validate(build_rows(task_id, groups, seed), groups, seed)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for family in ("A1", "A2", "P", "C"):
        r = next(x for x in rows if x["family"] == family)
        print(f"  {family} [{r['direction_id']}] changes={r['answer_changes']}")
        print(f"    base : {r['base_text']!r} -> {r['base_answer']!r}")
        print(f"    donor: {r['donor_text']!r} -> {r['donor_answer']!r}")
