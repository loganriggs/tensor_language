"""preposition_selection.on_vs_of -- a verb selecting its PREPOSITION.

`depend` subcategorizes for `on`, `consist` for `of`. The choice is fixed by the verb and is not
marked anywhere on the token being predicted.

    A1 modal frame
      "The outcome will depend entirely"  -> " on"
      "The mixture will consist entirely" -> " of"
    A2 present frame
      "In the notes the outcome depends largely"  -> " on"
      "In the notes the mixture consists largely" -> " of"

Recipient and donor differ in the subject-verb pair and end on the same adverb token.

Why this behaviour. Three lexical-semantic cues have failed cross-construction capability
(pronoun gender, countability, animacy) while six function-word or subcategorization cues have
passed. This is a NEW KIND of subcategorization -- a preposition rather than a clause type,
which is what `verb_complementizer` tested -- so it extends the passing class rather than
repeating a member of it. The prediction registered before the run is that it passes; a
capability stop would be the first grammatical failure and would weaken the boundary badly.

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

TASK_ID = "preposition_selection.on_vs_of"
PREPOSITION = (" on", " of")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_preposition_selection_fit_panels",
    answer_role="score_jointly_tokenized_on_versus_of",
    transforms=(
        battery.TransformSpec("A1", "modal_frame_preposition_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "present_frame_preposition_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_ON_SUBJECTS = ("outcome", "result", "verdict", "decision", "ranking", "schedule", "payment",
                "answer", "rating", "reward", "penalty", "timing", "choice", "amount",
                "balance", "figure", "total", "score", "grade", "wage", "fee", "rate",
                "price", "value", "budget", "quota", "limit", "target", "margin", "yield",
                "return", "share")
_OF_SUBJECTS = ("mixture", "blend", "compound", "solution", "alloy", "batter", "dough",
                "syrup", "paste", "mortar", "varnish", "lacquer", "resin", "ointment",
                "tincture", "emulsion", "slurry", "grout", "plaster", "cement", "glaze",
                "enamel", "pigment", "dye", "brine", "broth", "stew", "sauce", "gravy",
                "custard", "batch", "filling")
_ADVERBS_A1 = ("entirely", "wholly", "largely", "chiefly", "mainly", "mostly", "partly",
               "solely", "purely", "simply", "clearly", "plainly", "greatly", "heavily",
               "strongly", "firmly", "closely", "directly", "broadly", "narrowly", "loosely",
               "roughly", "barely", "hardly", "nearly", "fully", "truly", "really", "wholly",
               "utterly", "totally", "quite")
_ADVERBS_A2 = ("largely", "chiefly", "mainly", "mostly", "entirely", "wholly", "partly",
               "solely", "purely", "simply", "clearly", "plainly", "greatly", "heavily",
               "strongly", "firmly", "closely", "directly", "broadly", "narrowly", "loosely",
               "roughly", "barely", "hardly", "nearly", "fully", "truly", "really", "wholly",
               "utterly", "totally", "quite")
if not (len(_ON_SUBJECTS) == len(_OF_SUBJECTS) == len(_ADVERBS_A1) == len(_ADVERBS_A2) == DEFAULT_GROUPS):
    raise RuntimeError("preposition tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _modal(subject: str, verb: str, adverb: str) -> str:
    return f"The {subject} will {verb} {adverb}"


def _present(subject: str, verb: str, adverb: str) -> str:
    return f"In the notes the {subject} {verb}s {adverb}"


def _answer(on_type: bool) -> str:
    return PREPOSITION[0] if on_type else PREPOSITION[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "on_to_of" if forward else "of_to_on"
    on_subj, of_subj = _ON_SUBJECTS[case_index], _OF_SUBJECTS[case_index]
    adv1, adv2 = _ADVERBS_A1[case_index], _ADVERBS_A2[case_index]
    base_pair = (on_subj, "depend") if base_more else (of_subj, "consist")
    donor_pair = (on_subj, "depend") if donor_more else (of_subj, "consist")
    # P preserves the answer by swapping to a different subject of the SAME verb type.
    alt = (_ON_SUBJECTS if base_more else _OF_SUBJECTS)[(case_index + 7) % DEFAULT_GROUPS]
    p_pair = (alt, base_pair[1])
    phrase = f" {adv1}"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("on_type" if base_more else "of_type",
             "on_type" if donor_more else "of_type")
    common = dict(common_no_vocab, vocabulary=PREPOSITION)
    return [
        builder._row(**common, transform_id="A1", construction_id="modal_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_modal(*base_pair, adv1),
                     donor_text=_modal(*donor_pair, adv1),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="present_frame",
                     direction_id=direction, matched_suffix=f" {adv2}",
                     base_text=_present(*base_pair, adv2),
                     donor_text=_present(*donor_pair, adv2),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"modal_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_modal(*base_pair, adv1),
                     donor_text=_modal(*p_pair, adv1),
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
        if cells.get((transform, "on_to_of")) != half \
                or cells.get((transform, "of_to_on")) != half:
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
