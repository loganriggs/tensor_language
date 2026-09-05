"""modal_remoteness.would_vs_will -- MOOD, which the corpus did not previously cover.

An `if` plus past-tense antecedent makes its consequent remote and obliges `would`; a `when`
plus present-tense antecedent is factual and takes `will`. The cue is a function word together
with an inflection, spread across a subordinate clause, and the target is an AUXILIARY -- not a
determiner, complementizer or preposition, which is what the rest of the corpus predicts.

    A1 fronted conditional
      "If the harbor closed early, the bright route"   -> " would"
      "When the harbor closes early, the bright route" -> " will"
    A2 report-embedded conditional
      "The report notes that if the harbor closed, the bright route"   -> " would"
      "The report notes that when the harbor closes, the bright route" -> " will"

Recipient and donor differ only in that antecedent and end on the same noun token. P swaps the
place while holding the mood, so it holds the answer.

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

TASK_ID = "modal_remoteness.would_vs_will"
REMOTENESS = (" would", " will")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_modal_remoteness_fit_panels",
    answer_role="score_jointly_tokenized_would_versus_will",
    transforms=(
        battery.TransformSpec("A1", "fronted_conditional_mood_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_embedded_mood_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_PLACES = lex._OBJECTS
_ADJECTIVES = lex._ADJECTIVES
if not (len(_PLACES) == len(_ADJECTIVES) == DEFAULT_GROUPS):
    raise RuntimeError("modal-remoteness tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _fronted(place: str, adjective: str, remote: bool) -> str:
    head = f"If the {place} closed early" if remote else f"When the {place} closes early"
    return f"{head}, the {adjective} route"


def _embedded(place: str, adjective: str, remote: bool) -> str:
    head = f"if the {place} closed" if remote else f"when the {place} closes"
    return f"The report notes that {head}, the {adjective} route"


def _answer(remote: bool) -> str:
    return REMOTENESS[0] if remote else REMOTENESS[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "remote_to_factual" if forward else "factual_to_remote"
    place, adjective = _PLACES[case_index], _ADJECTIVES[case_index]
    # P preserves the answer by swapping the PLACE while holding the mood.
    alt_place = _PLACES[(case_index + 7) % DEFAULT_GROUPS]
    phrase = f" {adjective} route"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("remote" if base_more else "factual",
             "remote" if donor_more else "factual")
    common = dict(common_no_vocab, vocabulary=REMOTENESS)
    return [
        builder._row(**common, transform_id="A1", construction_id="fronted_conditional",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_fronted(place, adjective, base_more),
                     donor_text=_fronted(place, adjective, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="report_embedded",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_embedded(place, adjective, base_more),
                     donor_text=_embedded(place, adjective, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"fronted_conditional_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_fronted(place, adjective, base_more),
                     donor_text=_fronted(alt_place, adjective, base_more),
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
        if cells.get((transform, "remote_to_factual")) != half \
                or cells.get((transform, "factual_to_remote")) != half:
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
