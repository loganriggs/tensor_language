"""countability_state.count_vs_mass -- a LEXICAL feature carried to a quantity determiner.

Whether a noun is mass or count is a property of the word itself, not of any phrase structure,
and it is not marked anywhere on the token being predicted. The model has to carry it across the
clause to choose the determiner.

    A1 fronted PP
      "Of the bottles the leader did not take" -> " many"
      "Of the water the leader did not take"   -> " much"
    A2 scarcity
      "The bottles were scarce, so the leader could not find" -> " many"
      "The water was scarce, so the leader could not find"    -> " much"

Recipient and donor differ only in that early noun and end on the same verb token. The nearest
existing behaviour is `subject_verb.number_agreement`, whose variable is a subject's NUMBER on a
copula; this one is a lexical property of a content word.

Control. The canonical same-answer control v2 (" night" against " day"), disjoint from this
behaviour's " many"/" much" vocabulary. Authored against v2 from the start, so this C is directly
comparable with the five-behaviour set without a retrofit.
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

TASK_ID = "countability_state.count_vs_mass"
QUANTITY = (" many", " much")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_countability_fit_panels",
    answer_role="score_jointly_tokenized_many_versus_much",
    transforms=(
        battery.TransformSpec("A1", "fronted_pp_countability_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "scarcity_frame_countability_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_COUNT = ("bottles", "letters", "coins", "stones", "bricks", "ropes", "lamps", "crates",
          "baskets", "planks", "nails", "wheels", "boxes", "cups", "plates", "tools",
          "wires", "tiles", "blankets", "candles", "barrels", "buckets", "sacks", "ladders",
          "benches", "chairs", "doors", "windows", "mirrors", "keys", "locks", "hinges")
_MASS = ("water", "sand", "flour", "salt", "coal", "oil", "milk", "grain",
         "timber", "cloth", "wool", "ash", "dust", "chalk", "paint", "ink",
         "honey", "butter", "wax", "soap", "smoke", "steam", "gravel", "clay",
         "iron", "copper", "silver", "fabric", "leather", "rubber", "powder", "cement")
if not (len(_COUNT) == len(_MASS) == DEFAULT_GROUPS):
    raise RuntimeError("countability tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _fronted(agent: str, noun: str) -> str:
    """Positive frame.

    v1 used "did not take" and A1 came out at chance, 32/64, with the mirror signature: the model
    answered " much" on both sides. After a NEGATED verb " much" is the default continuation in
    English and it overrode the countability cue entirely. A2's negated frame ("could not find")
    was less affected but still missed the bar at 54/64. Both frames are positive here.
    """
    return f"Of the {noun} the {agent} took"


def _scarcity(agent: str, noun: str, plural: bool) -> str:
    return f"The {noun} {'lay' if plural else 'lay'} everywhere, so the {agent} collected"


def _answer(count: bool) -> str:
    return QUANTITY[0] if count else QUANTITY[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "count_to_mass" if forward else "mass_to_count"
    count_noun, mass_noun = _COUNT[case_index], _MASS[case_index]
    base_noun = count_noun if base_more else mass_noun
    donor_noun = count_noun if donor_more else mass_noun
    phrase = " took"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("count" if base_more else "mass",
             "count" if donor_more else "mass")
    common = dict(common_no_vocab, vocabulary=QUANTITY)
    return [
        builder._row(**common, transform_id="A1", construction_id="fronted_pp",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_fronted(agent, base_noun),
                     donor_text=_fronted(agent, donor_noun),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="scarcity_frame",
                     direction_id=direction, matched_suffix=" collected",
                     base_text=_scarcity(agent, base_noun, base_more),
                     donor_text=_scarcity(agent, donor_noun, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"fronted_pp_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_fronted(p_agent, base_noun),
                     donor_text=_fronted(p_donor_agent, base_noun),
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
        if cells.get((transform, "count_to_mass")) != half \
                or cells.get((transform, "mass_to_count")) != half:
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
