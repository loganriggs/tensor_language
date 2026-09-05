"""interrogative_licensing.question_vs_declarative -- a second LICENSING behaviour, run as a test.

polarity_state showed that negation licenses " any". This uses a DIFFERENT licenser for the same
determiner: subject-auxiliary inversion. A question makes " any" available where the
corresponding declarative leaves " some" natural.

    A1 do-support
      "Did the leader find"  -> " any"
      "The leader did find"  -> " some"
    A2 perfect
      "Has the leader kept"  -> " any"
      "The leader has kept"  -> " some"

Recipient and donor differ only in word order and end on the same verb token, so a transferring
site carries interrogative mood rather than a surface property of the patched position.

Why this behaviour and not another. Across thirteen behaviours the same-answer control statistic
C/A1 has ordered by dependency TYPE: obligations 0.034-0.080, narrative_tense 0.142, and the
single licensing behaviour 0.173. That ordering rests on ONE licensing screen and could be a
property of that stimulus set. The prediction registered in the prior-art receipt before this ran
is that a second licensing behaviour lands above 0.12 and outside the obligation cluster; landing
inside the cluster refutes the ordering and I withdraw it.

Control. C is same-answer -- both sides answer " some" -- under two declarative frames with no
polarity item, so it probes a different licenser rather than repeating P's edit.

Code path. Reuses the row builder pinned by
`circuit_fast_screen_candidate_sentence_terminal_context_control` (digest d0da3cda...).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_candidates as lex

canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
DEFAULT_GROUPS = builder.DEFAULT_GROUPS
DEFAULT_SEED = builder.DEFAULT_SEED

TASK_ID = "interrogative_licensing.question_vs_declarative"
MOOD = (" any", " some")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_interrogative_licensing_fit_panels",
    answer_role="score_jointly_tokenized_any_versus_some",
    transforms=(
        battery.TransformSpec("A1", "do_support_inversion_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "perfect_inversion_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        battery.TransformSpec("C", "declarative_no_polarity_item_frame", False, "registered_active"),
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


# All REGULAR, so the perfect frame's participle is derivable: v1 used irregulars and produced
# "had not manage" -- the modal frame takes a bare infinitive but the perfect takes a participle.
_VERBS = (
    "fetch", "check", "reach", "spot", "gather", "offer", "collect", "supply",
    "locate", "obtain", "recall", "secure", "detect", "provide", "retain", "attract",
    "borrow", "deliver", "identify", "manage", "notice", "produce", "receive", "recover",
    "release", "report", "request", "reserve", "restore", "reveal", "select", "submit",
)
if len(_VERBS) != DEFAULT_GROUPS:
    raise RuntimeError("verb table changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _participle(verb: str) -> str:
    return verb + ("d" if verb.endswith("e") else "ed")


def _do_support(agent: str, verb: str, question: bool) -> str:
    return f"Did the {agent} {verb}" if question else f"The {agent} did {verb}"


def _perfect(agent: str, verb: str, question: bool) -> str:
    part = _participle(verb)
    return f"Has the {agent} {part}" if question else f"The {agent} has {part}"


def _control(agent: str, verb: str, first: bool) -> str:
    """Two declarative frames with no polarity item, both answering " some"."""
    return f"The {agent} {'hoped' if first else 'offered'} to {verb}"


def _answer(question: bool) -> str:
    return MOOD[0] if question else MOOD[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "question_to_declarative" if forward else "declarative_to_question"
    verb = _VERBS[case_index]
    phrase = f" {verb}"
    common = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, vocabulary=MOOD, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("question" if base_more else "declarative",
             "question" if donor_more else "declarative")
    return [
        builder._row(**common, transform_id="A1", construction_id="do_support",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_do_support(agent, verb, base_more),
                     donor_text=_do_support(agent, verb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="perfect",
                     direction_id=direction, matched_suffix=f" {_participle(verb)}",
                     base_text=_perfect(agent, verb, base_more),
                     donor_text=_perfect(agent, verb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"do_support_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_do_support(p_agent, verb, base_more),
                     donor_text=_do_support(p_donor_agent, verb, base_more),
                     base_answer=_answer(base_more), donor_answer=_answer(base_more),
                     sentence_types=(kinds[0], kinds[0])),
        builder._row(**common, transform_id="C",
                     construction_id="declarative_no_polarity_item_frame",
                     direction_id="base_to_donor" if forward else "donor_to_base",
                     matched_suffix=f" {verb}",
                     base_text=_control(agent if forward else alternate, verb, True),
                     donor_text=_control(alternate if forward else agent, verb, False),
                     base_answer=MOOD[1], donor_answer=MOOD[1],
                     sentence_types=("declarative", "declarative")),
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
        if cells.get((transform, "question_to_declarative")) != half \
                or cells.get((transform, "declarative_to_question")) != half:
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
