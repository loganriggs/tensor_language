"""finiteness_selection.to_vs_that -- the third axis of verb subcategorization.

`verb_complementizer` varies interrogative against declarative, both complements FINITE.
`preposition_selection` varies which preposition a verb takes. This varies FINITENESS itself:
`decided` selects a nonfinite complement and obliges ` to`, `insisted` selects a finite one and
obliges ` that`.

    A1 bare frame
      "The board decided firmly"  -> " to"
      "The board insisted firmly" -> " that"
    A2 report frame, a SECOND verb pair
      "In the notes the board agreed quickly"   -> " to"
      "In the notes the board declared quickly" -> " that"

Recipient and donor differ only in the matrix verb and end on the same adverb token. A2 uses a
different verb pair from A1, so a single lexical item cannot satisfy both constructions -- which
is the point of the A2 hypothesis and is worth doing here because the cue IS a lexical item.

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

TASK_ID = "finiteness_selection.to_vs_that_v2"
FINITENESS = (" to", " that")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_finiteness_fit_panels",
    answer_role="score_jointly_tokenized_to_versus_that",
    transforms=(
        battery.TransformSpec("A1", "bare_frame_finiteness_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_frame_finiteness_swap", True, "toward_donor"),
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
_ADVERBS_A1 = ("firmly", "quietly", "promptly", "calmly", "boldly", "quickly", "plainly",
               "politely", "readily", "steadily", "swiftly", "warmly", "wisely", "openly",
               "clearly", "softly", "sharply", "slowly", "sternly", "tersely", "truly",
               "vaguely", "sadly", "simply", "sincerely", "strongly", "sweetly", "loudly",
               "briskly", "neatly", "gently", "carefully")
_ADVERBS_A2 = ("quickly", "firmly", "calmly", "promptly", "plainly", "boldly", "politely",
               "readily", "steadily", "swiftly", "warmly", "wisely", "openly", "clearly",
               "softly", "sharply", "slowly", "sternly", "tersely", "truly", "vaguely",
               "sadly", "simply", "sincerely", "strongly", "sweetly", "loudly", "briskly",
               "neatly", "gently", "carefully", "quietly")
if not (len(_AGENTS) == len(_ADVERBS_A1) == len(_ADVERBS_A2) == DEFAULT_GROUPS):
    raise RuntimeError("finiteness tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _bare(agent: str, adverb: str, nonfinite: bool) -> str:
    """v2 nonfinite verb.

    v1 used 'decided' and failed one-sided: the finite side scored 1.00 in both constructions
    while the nonfinite side scored 0.50 and 0.56. That is the signature of a collocation broken
    by the intervening adverb, not of a subcategorization failure -- verb_complementizer uses the
    same adverb-intervening design and passes at +4.45. 'refused' takes an adverb far more
    comfortably before its infinitive. The FINITE verbs are unchanged.
    """
    return f"The {agent} {'refused' if nonfinite else 'insisted'} {adverb}"


def _report(agent: str, adverb: str, nonfinite: bool) -> str:
    return f"In the notes the {agent} {'offered' if nonfinite else 'declared'} {adverb}"


def _answer(nonfinite: bool) -> str:
    return FINITENESS[0] if nonfinite else FINITENESS[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "nonfinite_to_finite" if forward else "finite_to_nonfinite"
    agent, alt_agent = _AGENTS[case_index], _AGENTS_ALT[case_index]
    part1, part2 = _ADVERBS_A1[case_index], _ADVERBS_A2[case_index]
    # P preserves the answer by swapping the SUBJECT while holding the voice.
    phrase = f" {part1}"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("nonfinite" if base_more else "finite",
             "nonfinite" if donor_more else "finite")
    common = dict(common_no_vocab, vocabulary=FINITENESS)
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
        if cells.get((transform, "nonfinite_to_finite")) != half \
                or cells.get((transform, "finite_to_nonfinite")) != half:
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
