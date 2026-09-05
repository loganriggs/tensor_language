"""polarity_state.negative_vs_positive -- the corpus's first LICENSING relation.

Every obligatory behaviour so far FORCES its second element: `more` forces `than`, `either`
forces `or`, `wondered` forces an interrogative complement. Negation does not work that way. It
LICENSES ` any` -- makes it available -- while the positive counterpart leaves ` some` as the
natural continuation. So this is a weaker, probabilistic dependency than anything else in the
corpus, and worth having for exactly that reason.

    A1 modal frame
      "The leader could not find"   -> " any"
      "The leader could still find" -> " some"
    A2 perfect frame
      "In the notes the leader had not kept"   -> " any"
      "In the notes the leader had still kept" -> " some"

Recipient and donor are a MINIMAL PAIR differing in exactly one word (not / still) and ending on
the same verb token, so a transferring site carries clause polarity rather than a surface
property of the patched position. P rewrites the agent while holding the polarity.

Control. C is same-answer -- both sides answer " some" -- under two verbs that involve no
polarity item at all, so it probes a different licenser rather than repeating P's edit. Its
measured ceiling in this configuration is 0.034 to 0.142 against a 0.35 bar, so the C clause is
a statistic, not a passed test.

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

TASK_ID = "polarity_state.negative_vs_positive"
POLARITY = (" any", " some")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_polarity_state_fit_panels",
    answer_role="score_jointly_tokenized_any_versus_some",
    transforms=(
        battery.TransformSpec("A1", "modal_frame_polarity_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "perfect_frame_polarity_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        battery.TransformSpec("C", "polarity_free_verb_frame", False, "registered_active"),
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


def _modal(agent: str, verb: str, negative: bool) -> str:
    return f"The {agent} could {'not' if negative else 'still'} {verb}"


def _participle(verb: str) -> str:
    return verb + ("d" if verb.endswith("e") else "ed")


def _perfect(agent: str, verb: str, negative: bool) -> str:
    return (f"In the notes the {agent} had {'not' if negative else 'still'} "
            f"{_participle(verb)}")


def _control(agent: str, verb: str, first: bool) -> str:
    """Two polarity-free frames, both answering " some".

    Neither involves a negative-polarity item, so C probes a different licenser for the same
    answer instead of repeating P's edit.
    """
    return f"The {agent} {'hoped' if first else 'offered'} to {verb}"


def _answer(negative: bool) -> str:
    return POLARITY[0] if negative else POLARITY[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "negative_to_positive" if forward else "positive_to_negative"
    verb = _VERBS[case_index]
    phrase = f" {verb}"
    common = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, vocabulary=POLARITY, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("negative" if base_more else "positive",
             "negative" if donor_more else "positive")
    return [
        builder._row(**common, transform_id="A1", construction_id="modal_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_modal(agent, verb, base_more),
                     donor_text=_modal(agent, verb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="perfect_frame",
                     direction_id=direction, matched_suffix=f" {_participle(verb)}",
                     base_text=_perfect(agent, verb, base_more),
                     donor_text=_perfect(agent, verb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"modal_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_modal(p_agent, verb, base_more),
                     donor_text=_modal(p_donor_agent, verb, base_more),
                     base_answer=_answer(base_more), donor_answer=_answer(base_more),
                     sentence_types=(kinds[0], kinds[0])),
        builder._row(**common, transform_id="C",
                     construction_id="polarity_free_verb_frame",
                     direction_id="base_to_donor" if forward else "donor_to_base",
                     matched_suffix=f" {verb}",
                     base_text=_control(agent if forward else alternate, verb, True),
                     donor_text=_control(alternate if forward else agent, verb, False),
                     base_answer=POLARITY[1], donor_answer=POLARITY[1],
                     sentence_types=("polarity_free", "polarity_free")),
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
        if cells.get((transform, "negative_to_positive")) != half \
                or cells.get((transform, "positive_to_negative")) != half:
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
