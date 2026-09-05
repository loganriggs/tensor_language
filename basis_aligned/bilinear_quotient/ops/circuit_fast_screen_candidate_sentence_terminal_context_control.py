#!/usr/bin/env python3
"""sentence_terminal with a CONTEXT-DRIVEN control in place of the instruction-following one.

Why this exists. The v1 screen
(`circuits/fast_screens/sentence_terminal_semantic_choice_v1_result.json`) is recorded
`native_behavior_incapable`, which reads as "the model cannot do sentence-mode choice". It
cannot: A1, A2 and P are all 64/64 rows correct with answer-vs-foil margins of +4.2 to +5.1.
Only the C family fails, at 16/32, and its two failing cells are exact mirrors --
`base 1.00 / donor 0.00` then `base 0.00 / donor 1.00`.

The cause is visible in the prompt. The v1 control is

    "Copy the visible mark ? after a remarkably bright lantern"

whose answer is carried by an INSTRUCTION ("copy the visible mark"), not by context. GPT-2
small does not follow instructions; it emits the sentence-final period it prefers, so it
scores correct exactly when the shown mark happens to be "." -- which is precisely the
mirrored pattern. The control was unpassable by construction, and it stopped the run before
a single site was screened.

So the control is replaced with one the model performs FROM CONTEXT and that is a different
behaviour from sentence-final punctuation: subject-verb number agreement, whose native
capability already passes repeatedly in `circuits/fast_screen_ledger.jsonl` under
`subject_verb.number_agreement`. Base and donor differ only in the number of a noun early in
the prompt and end on the same final token:

    "Beside the harbor the lantern  that the pilot repaired" -> " is"
    "Beside the harbor the lanterns that the pilot repaired" -> " are"

A1, A2 and P are taken from the v1 authority UNCHANGED, so this varies only the hypothesis
the verdict turned on (standing lesson 2).

Control on the new code path. This module rebuilds every family itself rather than importing
finished rows, so the new path could silently differ from the one that produced the v1
authority. `reproduces_v1_authority()` therefore builds with the ORIGINAL control through
THIS module's code and asserts the digest equals the v1 authority. Run it before trusting any
row this module emits.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidates as v1

canonical_sha256 = v1.canonical_sha256
ENCODING = v1.ENCODING
CandidateBankError = v1.CandidateBankError

SCHEMA = v1.SCHEMA
TASK_ID = "sentence_terminal.semantic_choice_context_control"
DEFAULT_GROUPS = v1.DEFAULT_GROUPS
DEFAULT_SEED = v1.DEFAULT_SEED
SPLIT = v1.SPLIT

# The v1 control's answers; kept so the reproduction control can rebuild it exactly.
PUNCTUATION = v1.PUNCTUATION
# The replacement control's answers. Both are single GPT-2 tokens and both are produced from
# context rather than from an instruction.
AGREEMENT = (" is", " are")

_SUBJECTS = (
    ("lantern", "lanterns"), ("harbor", "harbors"), ("meadow", "meadows"),
    ("river", "rivers"), ("forest", "forests"), ("garden", "gardens"),
    ("bridge", "bridges"), ("beacon", "beacons"), ("marble", "marbles"),
    ("willow", "willows"), ("orchard", "orchards"), ("canyon", "canyons"),
    ("falcon", "falcons"), ("kettle", "kettles"), ("vessel", "vessels"),
    ("cloud", "clouds"), ("valley", "valleys"), ("island", "islands"),
    ("tower", "towers"), ("window", "windows"), ("table", "tables"),
    ("chair", "chairs"), ("house", "houses"), ("road", "roads"),
    ("field", "fields"), ("ocean", "oceans"), ("maple", "maples"),
    ("lemon", "lemons"), ("apple", "apples"), ("basket", "baskets"),
    ("cabin", "cabins"), ("market", "markets"),
)
_VERBS = (
    "repaired", "described", "counted", "praised", "sketched", "measured",
    "cleaned", "labelled", "inspected", "recorded", "painted", "moved",
    "carried", "opened", "closed", "lifted", "guarded", "polished",
    "gathered", "sorted", "packed", "weighed", "mapped", "traced",
    "framed", "listed", "checked", "marked", "handled", "stored",
    "shifted", "tested",
)
if not (len(_SUBJECTS) == len(_VERBS) == DEFAULT_GROUPS):
    raise RuntimeError("context-control lexical tables changed size")


def _agreement_text(subject: str, reporter: str, verb: str) -> str:
    return f"Beside the harbor the {subject} that the {reporter} {verb}"


def _agreement_answer(plural: bool) -> str:
    return " are" if plural else " is"


def _opposite(answer: str, vocabulary: tuple[str, ...]) -> str:
    if answer not in vocabulary:
        raise CandidateBankError("answer lies outside its declared vocabulary")
    return vocabulary[1] if answer == vocabulary[0] else vocabulary[0]


def _row(
    *, seed: int, task_id: str, group_number: int, group_id: str, transform_id: str,
    construction_id: str, direction_id: str, reporter: str, alternate_reporter: str,
    adjective: str, object_name: str, base_text: str, donor_text: str,
    base_answer: str, donor_answer: str, vocabulary: tuple[str, ...],
    matched_suffix: str, spec: battery.BatteryTaskSpec,
    sentence_types: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """One interchange row. Mirrors the v1 checks, with the answer vocabulary made a parameter."""
    transform = next(item for item in spec.transforms if item.transform_id == transform_id)
    base_foil = _opposite(base_answer, vocabulary)
    donor_foil = _opposite(donor_answer, vocabulary)
    base_ids, base_answer_id = v1._joint_token_id(base_text, base_answer)
    donor_ids, donor_answer_id = v1._joint_token_id(donor_text, donor_answer)
    _, base_foil_id = v1._joint_token_id(base_text, base_foil)
    _, donor_foil_id = v1._joint_token_id(donor_text, donor_foil)
    base_position, donor_position = len(base_ids) - 1, len(donor_ids) - 1
    answer_changes = base_answer != donor_answer
    capability_cell_id = f"{transform_id}/{construction_id}/{direction_id}"
    # v1 named this check after its vocabulary; keep that so the reproduction control is exact.
    single_token_key = ("single_token_punctuation" if vocabulary == PUNCTUATION
                        else "single_token_answers")
    checks = {
        single_token_key: all(len(ENCODING.encode(a)) == 1 for a in vocabulary),
        "exact_answer_vocabulary": all(
            a in vocabulary for a in (base_answer, donor_answer, base_foil, donor_foil)
        ),
        "base_joint_answer_tokenization": (
            ENCODING.encode(base_text + base_answer) == base_ids + [base_answer_id]
        ),
        "base_joint_foil_tokenization": (
            ENCODING.encode(base_text + base_foil) == base_ids + [base_foil_id]
        ),
        "donor_joint_answer_tokenization": (
            ENCODING.encode(donor_text + donor_answer) == donor_ids + [donor_answer_id]
        ),
        "donor_joint_foil_tokenization": (
            ENCODING.encode(donor_text + donor_foil) == donor_ids + [donor_foil_id]
        ),
        "prompt_roundtrip": (
            ENCODING.decode(base_ids) == base_text and ENCODING.decode(donor_ids) == donor_text
        ),
        "distinct_prompts": base_text != donor_text,
        "answer_change_matches_transform": answer_changes is transform.answer_changes,
        "paired_answer_foil_alignment": (
            {base_answer_id, base_foil_id} == {donor_answer_id, donor_foil_id}
            and base_answer_id != base_foil_id and donor_answer_id != donor_foil_id
        ),
        "matched_long_final_suffix": (
            base_text.endswith(matched_suffix) and donor_text.endswith(matched_suffix)
        ),
        "matched_final_input_token": base_ids[-1] == donor_ids[-1],
        "semantic_position_is_final_input": (
            base_position == len(base_ids) - 1 and donor_position == len(donor_ids) - 1
        ),
        "no_trailing_space_or_quote": all(
            text and not text[-1].isspace() and text[-1] not in "\"'"
            for text in (base_text, donor_text)
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise CandidateBankError(f"row construction checks failed: {failed}")
    # v1 derived sentence type from the punctuation answer; a control whose answers are not
    # punctuation must state its own, or every row would read as 'interrogative'.
    types = sentence_types or tuple(
        "declarative" if a == "." else "interrogative" for a in (base_answer, donor_answer))
    identity = {
        "schema": SCHEMA,
        "task_id": task_id,
        "split": SPLIT,
        "seed": seed,
        "group_number": group_number,
        "group_id": group_id,
        "transform_id": transform_id,
        "construction_id": construction_id,
        "direction_id": direction_id,
        "capability_cell_id": capability_cell_id,
        "reporter": reporter,
        "alternate_reporter": alternate_reporter,
        "adjective": adjective,
        "object_name": object_name,
        "base_text": base_text,
        "donor_text": donor_text,
        "base_answer": base_answer,
        "donor_answer": donor_answer,
        "base_foil": base_foil,
        "donor_foil": donor_foil,
        "base_prediction_position": base_position,
        "donor_prediction_position": donor_position,
    }
    return {
        **identity,
        "row_id": canonical_sha256(identity),
        "family_id": f"{task_id}/{transform_id}",
        "family": transform_id,
        "role": "interchange",
        "answer_changes": answer_changes,
        "expected_effect": transform.expected_effect,
        "changed_variable": transform.generator_role,
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer_id": base_answer_id,
        "donor_answer_id": donor_answer_id,
        "base_foil_id": base_foil_id,
        "donor_foil_id": donor_foil_id,
        "base_semantic_position": base_position,
        "donor_semantic_position": donor_position,
        "semantic_details": {
            "semantic_role": "final_prediction_position_input_token",
            "construction_id": construction_id,
            "direction_id": direction_id,
            "matched_final_suffix": matched_suffix,
            "base_sentence_type": types[0],
            "donor_sentence_type": types[1],
            "copy_control": transform_id == "C",
        },
        "construction_checks": checks,
    }


def _spec(task_id: str, control: battery.TransformSpec) -> battery.BatteryTaskSpec:
    """A1/A2/P are the v1 transforms verbatim; only the control differs."""
    a1, a2, p = (
        battery.TransformSpec("A1", "reporting_frame_declarative_question_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "direct_declarative_question_swap", True, "toward_donor"),
        battery.TransformSpec("P", "reporter_lexical_rewrite", False, "invariant"),
    )
    return battery.BatteryTaskSpec(
        task_id=task_id,
        generator_role="generate_linked_sentence_terminal_fit_panels",
        answer_role="score_jointly_tokenized_period_vs_question_mark",
        transforms=(a1, a2, p, control),
    )


V1_CONTROL = battery.TransformSpec(
    "C", "explicit_visible_punctuation_copy", True, "registered_active")
CONTEXT_CONTROL = battery.TransformSpec(
    "C", "subject_number_agreement_context", True, "registered_active")

V1_SPEC = _spec(v1.TASK_ID, V1_CONTROL)
TASK_SPEC = _spec(TASK_ID, CONTEXT_CONTROL)


def _control_row(*, kind: str, seed, task_id, group_number, group_id, forward,
                 reporter, alternate, adjective, object_name, suffix, spec):
    """The C row, in whichever control family is selected."""
    if kind == "v1":
        base_mark, donor_mark = ((".", "?") if forward else ("?", "."))
        return _row(
            seed=seed, task_id=task_id, group_number=group_number, group_id=group_id,
            transform_id="C", construction_id="explicit_visible_mark_copy",
            direction_id="period_to_question" if forward else "question_to_period",
            reporter=reporter, alternate_reporter=alternate, adjective=adjective,
            object_name=object_name,
            base_text=v1._copy_text(base_mark, suffix),
            donor_text=v1._copy_text(donor_mark, suffix),
            base_answer=base_mark, donor_answer=donor_mark,
            vocabulary=PUNCTUATION, matched_suffix=suffix, spec=spec,
        )
    singular, plural = _SUBJECTS[_CASE[group_number]]
    verb = _VERBS[_CASE[group_number]]
    base_subject, donor_subject = (singular, plural) if forward else (plural, singular)
    base_text = _agreement_text(base_subject, reporter, verb)
    donor_text = _agreement_text(donor_subject, reporter, verb)
    matched = f"that the {reporter} {verb}"
    return _row(
        seed=seed, task_id=task_id, group_number=group_number, group_id=group_id,
        transform_id="C", construction_id="subject_number_agreement",
        direction_id="singular_to_plural" if forward else "plural_to_singular",
        reporter=reporter, alternate_reporter=alternate, adjective=adjective,
        object_name=object_name, base_text=base_text, donor_text=donor_text,
        base_answer=_agreement_answer(not forward), donor_answer=_agreement_answer(forward),
        vocabulary=AGREEMENT, matched_suffix=matched, spec=spec,
        sentence_types=("declarative", "declarative"),
    )


_CASE: dict[int, int] = {}


def _panel(seed, group_number, case_index, *, kind, task_id, spec):
    reporter, alternate = v1._REPORTERS[case_index]
    adjective, object_name = v1._ADJECTIVES[case_index], v1._OBJECTS[case_index]
    suffix = v1._suffix(adjective, object_name)
    group_id = f"FIT:{canonical_sha256([SCHEMA, task_id, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_type, donor_type = (
        ("declarative", "interrogative") if forward else ("interrogative", "declarative"))
    direction = f"{base_type}_to_{donor_type}"
    _CASE[group_number] = case_index

    p_base_reporter, p_donor_reporter = (
        (reporter, alternate) if forward else (alternate, reporter))
    common = dict(seed=seed, task_id=task_id, group_number=group_number, group_id=group_id,
                  reporter=reporter, alternate_reporter=alternate, adjective=adjective,
                  object_name=object_name, vocabulary=PUNCTUATION, matched_suffix=suffix,
                  spec=spec)
    return [
        _row(**common, transform_id="A1", construction_id="reporting_frame",
             direction_id=direction,
             base_text=v1._reporting_text(reporter, suffix, base_type),
             donor_text=v1._reporting_text(reporter, suffix, donor_type),
             base_answer=v1._terminal(base_type), donor_answer=v1._terminal(donor_type)),
        _row(**common, transform_id="A2", construction_id="direct_question",
             direction_id=direction,
             base_text=v1._direct_text(reporter, suffix, base_type),
             donor_text=v1._direct_text(reporter, suffix, donor_type),
             base_answer=v1._terminal(base_type), donor_answer=v1._terminal(donor_type)),
        _row(**common, transform_id="P", construction_id=f"reporting_frame_{base_type}",
             direction_id="primary_to_alternative" if forward else "alternative_to_primary",
             base_text=v1._reporting_text(p_base_reporter, suffix, base_type),
             donor_text=v1._reporting_text(p_donor_reporter, suffix, base_type),
             base_answer=v1._terminal(base_type), donor_answer=v1._terminal(base_type)),
        _control_row(kind=kind, seed=seed, task_id=task_id, group_number=group_number,
                     group_id=group_id, forward=forward, reporter=reporter,
                     alternate=alternate, adjective=adjective, object_name=object_name,
                     suffix=suffix, spec=spec),
    ]


def _build(groups, seed, *, kind, task_id, spec):
    order = v1._permutation(seed)
    return [row
            for group_number in range(groups)
            for row in _panel(seed, group_number, order[group_number],
                              kind=kind, task_id=task_id, spec=spec)]


def _validate(rows, groups, seed, *, kind, task_id, spec) -> str:
    if type(groups) is not int or not 2 <= groups <= DEFAULT_GROUPS or groups % 2:
        raise CandidateBankError("groups must be an even integer from 2 through 32")
    materialized = [dict(row) for row in rows]
    if materialized != _build(groups, seed, kind=kind, task_id=task_id, spec=spec):
        raise CandidateBankError("rows differ from the deterministic semantic authority")
    try:
        digest = battery.validate_rows(spec, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    row_ids = [str(row["row_id"]) for row in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise CandidateBankError("row IDs are not unique")
    cells: dict[tuple[str, str], int] = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise CandidateBankError("a stored construction check is false")
        cells[(str(row["transform_id"]), str(row["direction_id"]))] = \
            cells.get((str(row["transform_id"]), str(row["direction_id"])), 0) + 1
    half = groups // 2
    for transform in ("A1", "A2"):
        if cells.get((transform, "declarative_to_interrogative")) != half \
                or cells.get((transform, "interrogative_to_declarative")) != half:
            raise CandidateBankError(f"{transform} ordered directions are unbalanced")
    forward_id, reverse_id = (("period_to_question", "question_to_period") if kind == "v1"
                              else ("singular_to_plural", "plural_to_singular"))
    if cells.get(("C", forward_id)) != half or cells.get(("C", reverse_id)) != half:
        raise CandidateBankError("C ordered directions are unbalanced")
    return digest


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rows = _build(groups, seed, kind="context", task_id=TASK_ID, spec=TASK_SPEC)
    _validate(rows, groups, seed, kind="context", task_id=TASK_ID, spec=TASK_SPEC)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
                  groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    return _validate(rows, groups, seed, kind="context", task_id=TASK_ID, spec=TASK_SPEC)


def authority_sha256(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
                     seed: int = DEFAULT_SEED) -> str:
    return validate_rows(build_rows(task_id, groups, seed), groups=groups, seed=seed)


def reproduces_v1_authority() -> tuple[bool, str, str]:
    """Control: build the ORIGINAL control through THIS module and compare digests.

    If the new code path is faithful, selecting the v1 control must reproduce the v1
    authority digest exactly. Any drift here invalidates every row this module emits.
    """
    rows = _build(DEFAULT_GROUPS, DEFAULT_SEED, kind="v1", task_id=v1.TASK_ID, spec=V1_SPEC)
    mine = _validate(rows, DEFAULT_GROUPS, DEFAULT_SEED, kind="v1",
                     task_id=v1.TASK_ID, spec=V1_SPEC)
    theirs = v1.authority_sha256()
    return mine == theirs, mine, theirs


if __name__ == "__main__":
    same, mine, theirs = reproduces_v1_authority()
    print(f"control (v1 rows through new path): {'MATCH' if same else 'DRIFT'}")
    print(f"  this module {mine}\n  v1 authority {theirs}")
    print(f"context-control authority: {authority_sha256()}")
