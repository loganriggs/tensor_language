"""dative_alternation.to_vs_for -- a verb selecting its dative preposition.

Verb subcategorization has been screened on three axes -- clause type, preposition, finiteness --
and all pass. This is the preposition axis again but a DIFFERENT alternation: `sent` selects a
recipient with ` to`, `bought` a benefactive with ` for`.

The earlier preposition screen contrasted verbs whose prepositions are lexically idiosyncratic
(`depend on`, `consist of`). Here the two prepositions mark distinct SEMANTIC ROLES, so the cue
is the verb's argument structure rather than an arbitrary lexical requirement — which is what
makes this a generality test rather than a repeat.

    A1 bare frame
      "The clerk quickly sent the parcel"   -> " to"
      "The clerk quickly bought the parcel" -> " for"
    A2 report frame, a second verb pair
      "In the notes the clerk mailed the ledger"  -> " to"
      "In the notes the clerk ordered the ledger" -> " for"

Recipient and donor differ only in the matrix verb and end on the same object token. P swaps the
agent, which is safe here because the OBJECT is the final token (see the design invariants in
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

TASK_ID = "dative_alternation.to_vs_for"
DATIVE = (" to", " for")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_dative_alternation_fit_panels",
    answer_role="score_jointly_tokenized_to_versus_for",
    transforms=(
        battery.TransformSpec("A1", "bare_frame_dative_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_frame_dative_swap", True, "toward_donor"),
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
_A1_PARTICIPLES = ("parcel", "ledger", "notice", "receipt", "permit", "invoice", "summary", "contract",
                   "schedule", "manifest", "warrant", "bulletin", "circular", "docket", "memo",
                   "charter", "roster", "transcript", "abstract", "brochure", "leaflet", "petition",
                   "statement", "voucher", "citation", "diploma", "banner", "poster", "placard",
                   "label", "report", "letter")
_A2_PARTICIPLES = ("parcel", "ledger", "notice", "receipt", "permit", "invoice", "summary", "contract",
                   "schedule", "manifest", "warrant", "bulletin", "circular", "docket", "memo",
                   "charter", "roster", "transcript", "abstract", "brochure", "leaflet", "petition",
                   "statement", "voucher", "citation", "diploma", "banner", "poster", "placard",
                   "label", "report", "letter")
if not (len(_AGENTS) == len(_A1_PARTICIPLES) == len(_A2_PARTICIPLES) == DEFAULT_GROUPS):
    raise RuntimeError("voice tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _bare(agent: str, obj: str, recipient: bool) -> str:
    return f"The {agent} quickly {'sent' if recipient else 'bought'} the {obj}"


def _report(agent: str, obj: str, recipient: bool) -> str:
    return f"In the notes the {agent} {'mailed' if recipient else 'ordered'} the {obj}"


def _answer(recipient: bool) -> str:
    return DATIVE[0] if recipient else DATIVE[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "recipient_to_benefactive" if forward else "benefactive_to_recipient"
    agent, alt_agent = _AGENTS[case_index], _AGENTS_ALT[case_index]
    part1, part2 = _A1_PARTICIPLES[case_index], _A2_PARTICIPLES[case_index]
    # P preserves the answer by swapping the SUBJECT while holding the voice.
    phrase = f" {part1}"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("recipient" if base_more else "benefactive",
             "recipient" if donor_more else "benefactive")
    common = dict(common_no_vocab, vocabulary=DATIVE)
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
        if cells.get((transform, "recipient_to_benefactive")) != half \
                or cells.get((transform, "benefactive_to_recipient")) != half:
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
