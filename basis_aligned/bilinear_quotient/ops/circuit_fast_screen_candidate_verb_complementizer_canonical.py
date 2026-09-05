"""verb_complementizer.canonical_control -- verb_complementizer.whether_vs_that scored against the CANONICAL same-answer control.

A1, A2 and P are unchanged from the verb_complementizer.whether_vs_that authority. Only the control changes, to
the fixed control in `ops/circuit_fast_screen_canonical_control.py`, byte-identical to the one
used by every other canonical-control candidate.

Why. Measured at 11:46Z, swapping only a control frame moved C from 0.230 to 0.141 with verdict,
selected site, passing band and P unchanged, so C values from screens with different controls are
not comparable; the dependency-type ordering was retired at 12:34Z on a preregistered criterion.
Holding the control fixed is what makes a cross-behaviour C comparison mean anything.

The prediction registered for the five-behaviour set before these ran: if the control was the
whole story, all five C values fall within a spread of 0.05; a behaviour outside that band is a
real behaviour effect surviving a fixed control.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_canonical_control as canonical
import circuit_fast_screen_candidates as lex

canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED

TASK_ID = "verb_complementizer.canonical_control"
COMPLEMENTIZER = (" whether", " that")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_verb_complementizer_fit_panels",
    answer_role="score_jointly_tokenized_whether_versus_that",
    transforms=(
        battery.TransformSpec("A1", "bare_frame_complementizer_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_frame_complementizer_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_ADVERBS = (
    "aloud", "openly", "briefly", "calmly", "clearly", "firmly", "gently", "loudly",
    "quietly", "sharply", "slowly", "softly", "sternly", "swiftly", "warmly", "wryly",
    "plainly", "politely", "promptly", "quickly", "readily", "sadly", "simply", "sincerely",
    "steadily", "strongly", "sweetly", "tersely", "truly", "vaguely", "wisely", "boldly",
)
if len(_ADVERBS) != DEFAULT_GROUPS:
    raise RuntimeError("adverb table changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _bare(agent: str, adverb: str, interrogative: bool) -> str:
    return f"The {agent} {'wondered' if interrogative else 'remarked'} {adverb}"


def _report(agent: str, adverb: str, interrogative: bool) -> str:
    """The same verbs as A1 in a PP-fronted report frame.

    v1 used a different verb pair here (asked/stated) and A2 missed the capability bar on one
    cell, 0.81 against 0.85: "asked" is ambiguous between selecting a clause and taking a direct
    object. Holding the verbs fixed and varying only the construction is also the cleaner
    contrast -- A1 and A2 then differ in construction alone, which is what cross-construction
    transfer is supposed to test.
    """
    return f"In the report the {agent} {'wondered' if interrogative else 'remarked'} {adverb}"


def _control(agent: str, adverb: str, first: bool) -> str:
    """A third pair of declarative-selecting verbs, both answering " that".

    Unrelated to the wondered/remarked contrast under test, so C probes a different lexical
    licenser instead of repeating P's edit.
    """
    return f"The {agent} {'noted' if first else 'replied'} {adverb}"


def _answer(interrogative: bool) -> str:
    return COMPLEMENTIZER[0] if interrogative else COMPLEMENTIZER[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "interrogative_to_declarative" if forward else "declarative_to_interrogative"
    adverb = _ADVERBS[case_index]
    phrase = f" {adverb}"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("interrogative" if base_more else "declarative",
             "interrogative" if donor_more else "declarative")
    common = dict(common_no_vocab, vocabulary=COMPLEMENTIZER)
    return [
        builder._row(**common, transform_id="A1", construction_id="bare_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_bare(agent, adverb, base_more),
                     donor_text=_bare(agent, adverb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="report_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_report(agent, adverb, base_more),
                     donor_text=_report(agent, adverb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"bare_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_bare(p_agent, adverb, base_more),
                     donor_text=_bare(p_donor_agent, adverb, base_more),
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
        if cells.get((transform, "interrogative_to_declarative")) != half \
                or cells.get((transform, "declarative_to_interrogative")) != half:
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
