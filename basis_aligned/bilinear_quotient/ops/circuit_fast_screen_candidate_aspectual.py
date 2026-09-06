"""aspectual_anchor.has_vs_had -- ASPECT, which the corpus did not cover.

A temporal preposition fixes the anchor point: `since` opens an interval reaching the present
and takes the present perfect, `by` fixes a past deadline and takes the past perfect. The cue is
a function word two phrases before the target, and the target is an auxiliary.

    A1 fronted temporal
      "Since last spring the clerk" -> " has"
      "By last spring the clerk"    -> " had"
    A2 report-embedded
      "The record shows that since last spring the clerk" -> " has"
      "The record shows that by last spring the clerk"    -> " had"

Recipient and donor differ only in that preposition and end on the same subject token.

The corpus has temporal FRAME (narrative_tense, was/is), conditional MOOD (modal_remoteness,
would/will) and NUMBER on an auxiliary (existential, were/was) -- but no aspect. Note also that
the prediction site here follows a content noun, which the six-screen agreement investigation
just established is the configuration that works; a capability stop would therefore be
informative rather than a stimulus artefact.

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

TASK_ID = "aspectual_anchor.has_vs_had"
ASPECT = (" has", " had")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_aspectual_anchor_fit_panels",
    answer_role="score_jointly_tokenized_has_versus_had",
    transforms=(
        battery.TransformSpec("A1", "fronted_temporal_aspect_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_embedded_aspect_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_AGENTS_ALT = tuple(pair[1] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = _AGENTS_ALT
# identity labels only; not used in any prompt text
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS
_PERIODS = ("spring", "summer", "autumn", "winter", "season", "term", "quarter", "year",
            "month", "week", "cycle", "session", "harvest", "voyage", "shift", "round",
            "census", "audit", "review", "survey", "inspection", "election", "festival",
            "fair", "market", "contest", "match", "tour", "drill", "rehearsal", "hearing",
            "trial")
if not (len(_AGENTS) == len(_PERIODS) == DEFAULT_GROUPS):
    raise RuntimeError("aspect tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _fronted(agent: str, period: str, present: bool) -> str:
    return f"{'Since' if present else 'By'} last {period} the {agent}"


def _report(agent: str, period: str, present: bool) -> str:
    return f"The record shows that {'since' if present else 'by'} last {period} the {agent}"


def _answer(present: bool) -> str:
    return ASPECT[0] if present else ASPECT[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    agent, alt_agent = _AGENTS[case_index], _AGENTS_ALT[case_index]
    part1 = part2 = _PERIODS[case_index]
    # P preserves the answer by swapping the SUBJECT while holding the voice.
    phrase = f" {agent}"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("present_perfect" if base_more else "past_perfect",
             "present_perfect" if donor_more else "past_perfect")
    common = dict(common_no_vocab, vocabulary=ASPECT)
    return [
        builder._row(**common, transform_id="A1", construction_id="fronted_temporal",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_fronted(agent, part1, base_more),
                     donor_text=_fronted(agent, part1, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="report_embedded",
                     direction_id=direction, matched_suffix=f" {agent}",
                     base_text=_report(agent, part2, base_more),
                     donor_text=_report(agent, part2, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"fronted_temporal_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     # The AGENT is the final token here, so P varies the PERIOD instead --
                     # swapping the agent would break the matched-final-token invariant.
                     base_text=_fronted(agent, part1, base_more),
                     donor_text=_fronted(agent, _PERIODS[(case_index + 7) % DEFAULT_GROUPS],
                                         base_more),
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
        if cells.get((transform, "present_to_past")) != half \
                or cells.get((transform, "past_to_present")) != half:
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
