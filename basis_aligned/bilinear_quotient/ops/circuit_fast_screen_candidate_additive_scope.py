"""additive_scope.not_only_vs_plain -- a focus particle obliging its continuation.

`not only` takes scope over the clause and commits the sentence to a contrastive continuation;
a plain manner adverb in the same slot leaves the default coordinator.

    A1 bare frame
      "The leader not only wrote the report"  -> " but"
      "The leader carefully wrote the report" -> " and"
    A2 report frame
      "In the notes the leader not only signed the letter"  -> " but"
      "In the notes the leader quickly signed the letter"   -> " and"

Recipient and donor differ only in that adverb slot and end on the same noun token. The nearest
existing behaviour is `correlative_state.either_vs_neither`; that is a disjunction choice where
the first element selects between two continuations, while this is a focus particle scoping over
a whole clause.

Risk recorded before the run. The countability null measured earlier the same day showed a verb
frame's default continuation overriding a cue entirely -- " much" after a negated verb, " many"
after "collected". Here " and" is the default after a direct object, so this screen tests whether
an explicit scope operator is strong enough to beat a frame default. A capability stop on the
not-only side would be that same finding again, not a stimulus defect.

Control. Canonical same-answer control v2 (" night" against " day"), disjoint from " but"/" and",
so this C enters the comparable set with no retrofit.
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

TASK_ID = "additive_scope.not_only_vs_plain"
ADDITIVE = (" but", " and")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_additive_scope_fit_panels",
    answer_role="score_jointly_tokenized_but_versus_and",
    transforms=(
        battery.TransformSpec("A1", "bare_frame_scope_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "report_frame_scope_swap", True, "toward_donor"),
        battery.TransformSpec("P", "noun_lexical_rewrite", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = tuple(pair[0] for pair in lex._REPORTERS)
_ALTERNATE_AGENTS = tuple(pair[1] for pair in lex._REPORTERS)
_ADJECTIVES = lex._ADJECTIVES
_NOUNS = lex._OBJECTS


_OBJECTS = ("report", "letter", "notice", "ledger", "receipt", "permit", "invoice", "summary",
            "contract", "schedule", "manifest", "warrant", "bulletin", "circular", "docket",
            "affidavit", "memo", "charter", "roster", "transcript", "abstract", "brochure",
            "leaflet", "petition", "statement", "voucher", "citation", "diploma", "banner",
            "poster", "placard", "label")
_ADVERBS = ("carefully", "quickly", "quietly", "neatly", "promptly", "calmly", "firmly",
            "gently", "boldly", "briskly", "plainly", "politely", "readily", "steadily",
            "swiftly", "warmly", "wisely", "openly", "clearly", "softly", "sharply", "slowly",
            "sternly", "tersely", "truly", "vaguely", "sadly", "simply", "sincerely",
            "strongly", "sweetly", "loudly")
if not (len(_OBJECTS) == len(_ADVERBS) == DEFAULT_GROUPS):
    raise RuntimeError("additive-scope tables changed size")


def _phrase(adjective: str, noun: str) -> str:
    return f"the {noun}"


def _bare(agent: str, obj: str, adverb: str, scoped: bool) -> str:
    slot = "not only" if scoped else adverb
    return f"The {agent} {slot} wrote the {obj}"


def _report(agent: str, obj: str, adverb: str, scoped: bool) -> str:
    slot = "not only" if scoped else adverb
    return f"In the notes the {agent} {slot} signed the {obj}"


def _answer(scoped: bool) -> str:
    return ADDITIVE[0] if scoped else ADDITIVE[1]


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    agent, alternate = _AGENTS[case_index], _ALTERNATE_AGENTS[case_index]
    adjective, noun = _ADJECTIVES[case_index], _NOUNS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_more, donor_more = (True, False) if forward else (False, True)
    direction = "scoped_to_plain" if forward else "plain_to_scoped"
    obj, adverb = _OBJECTS[case_index], _ADVERBS[case_index]
    phrase = f" the {obj}"
    common_no_vocab = dict(seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
                  reporter=agent, alternate_reporter=alternate, adjective=adjective,
                  object_name=noun, spec=TASK_SPEC)
    p_agent, p_donor_agent = (agent, alternate) if forward else (alternate, agent)
    kinds = ("scoped" if base_more else "plain",
             "scoped" if donor_more else "plain")
    common = dict(common_no_vocab, vocabulary=ADDITIVE)
    return [
        builder._row(**common, transform_id="A1", construction_id="bare_frame",
                     direction_id=direction, matched_suffix=phrase,
                     base_text=_bare(agent, obj, adverb, base_more),
                     donor_text=_bare(agent, obj, adverb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="A2", construction_id="report_frame",
                     direction_id=direction, matched_suffix=f" the {obj}",
                     base_text=_report(agent, obj, adverb, base_more),
                     donor_text=_report(agent, obj, adverb, donor_more),
                     base_answer=_answer(base_more), donor_answer=_answer(donor_more),
                     sentence_types=kinds),
        builder._row(**common, transform_id="P",
                     construction_id=f"bare_frame_{kinds[0]}",
                     direction_id="primary_to_alternative" if forward else "alternative_to_primary",
                     matched_suffix=phrase,
                     base_text=_bare(p_agent, obj, adverb, base_more),
                     donor_text=_bare(p_donor_agent, obj, adverb, base_more),
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
        if cells.get((transform, "scoped_to_plain")) != half \
                or cells.get((transform, "plain_to_scoped")) != half:
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
