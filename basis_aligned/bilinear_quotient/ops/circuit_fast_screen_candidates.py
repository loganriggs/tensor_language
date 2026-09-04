#!/usr/bin/env python3
"""Small, deterministic candidate bank for reusable circuit FIT screens.

This module is data construction only.  It neither imports a model nor defines
an executor.  The first candidate tests semantic sentence-terminal choice while
holding the final input phrase fixed across each counterfactual pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import tiktoken

import circuit_battery_integration_contract as battery


SCHEMA = "circuit_fast_screen_candidate_rows_v1"
TASK_ID = "sentence_terminal.semantic_choice"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260904
SPLIT = "FIT"
TRANSFORMS = battery.TRANSFORMS
PUNCTUATION = (".", "?")
ENCODING = tiktoken.get_encoding("gpt2")


class CandidateBankError(ValueError):
    """A candidate request or generated authority violates its semantics."""


@dataclass(frozen=True)
class CandidateSpec:
    task: battery.BatteryTaskSpec
    description: str
    answer_vocabulary: tuple[str, ...]
    default_groups: int
    default_seed: int


TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_sentence_terminal_fit_panels",
    answer_role="score_jointly_tokenized_period_vs_question_mark",
    transforms=(
        battery.TransformSpec(
            "A1", "reporting_frame_declarative_question_swap", True, "toward_donor"
        ),
        battery.TransformSpec(
            "A2", "direct_declarative_question_swap", True, "toward_donor"
        ),
        battery.TransformSpec(
            "P", "reporter_lexical_rewrite", False, "invariant"
        ),
        battery.TransformSpec(
            "C", "explicit_visible_punctuation_copy", True, "registered_active"
        ),
    ),
)

CANDIDATES: Mapping[str, CandidateSpec] = MappingProxyType({
    TASK_ID: CandidateSpec(
        task=TASK_SPEC,
        description=(
            "Choose a period or question mark from sentence construction while "
            "the final lexical suffix is matched; C copies an earlier visible mark."
        ),
        answer_vocabulary=PUNCTUATION,
        default_groups=DEFAULT_GROUPS,
        default_seed=DEFAULT_SEED,
    )
})


# All 32 cases have a distinct proposition.  The reporter alternatives are used
# only by P; A1/A2 retain the primary reporter and proposition within a group.
_REPORTERS = (
    ("pilot", "sailor"), ("teacher", "doctor"), ("farmer", "baker"),
    ("artist", "writer"), ("driver", "rider"), ("guard", "clerk"),
    ("judge", "mayor"), ("nurse", "coach"), ("chef", "miner"),
    ("actor", "singer"), ("parent", "neighbor"), ("student", "scholar"),
    ("captain", "merchant"), ("ranger", "worker"), ("visitor", "owner"),
    ("editor", "author"), ("reader", "speaker"), ("friend", "cousin"),
    ("agent", "broker"), ("leader", "member"), ("expert", "novice"),
    ("buyer", "seller"), ("lawyer", "banker"), ("builder", "planner"),
    ("manager", "helper"), ("officer", "reporter"), ("director", "producer"),
    ("engineer", "designer"), ("scientist", "analyst"), ("owner", "tenant"),
    ("guide", "guest"), ("trainer", "runner"),
)
_ADJECTIVES = (
    "bright", "quiet", "clear", "warm", "cold", "soft", "sharp", "dark",
    "green", "blue", "red", "silver", "calm", "solid", "smooth", "round",
    "small", "large", "tall", "short", "light", "heavy", "plain", "rare",
    "clean", "fresh", "dry", "sweet", "strong", "gentle", "narrow", "wide",
)
_OBJECTS = (
    "lantern", "harbor", "meadow", "river", "forest", "garden", "bridge", "beacon",
    "marble", "willow", "orchard", "canyon", "falcon", "kettle", "vessel", "cloud",
    "valley", "island", "tower", "window", "table", "chair", "house", "road",
    "field", "ocean", "maple", "lemon", "apple", "basket", "cabin", "market",
)
if not (len(_REPORTERS) == len(_ADJECTIVES) == len(_OBJECTS) == DEFAULT_GROUPS):
    raise RuntimeError("sentence-terminal lexical tables changed size")


def canonical_sha256(value: object) -> str:
    """Canonical JSON SHA-256 used for groups, rows, and complete authorities."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _permutation(seed: int) -> tuple[int, ...]:
    material = f"{SCHEMA}|{TASK_ID}|{seed}|lexical-order".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))
    order = list(range(DEFAULT_GROUPS))
    rng.shuffle(order)
    return tuple(order)


def _suffix(adjective: str, object_name: str) -> str:
    return f"a remarkably {adjective} {object_name}"


def _reporting_text(reporter: str, suffix: str, sentence_type: str) -> str:
    if sentence_type == "declarative":
        return f"The {reporter} states: that is {suffix}"
    if sentence_type == "interrogative":
        return f"The {reporter} asks: is that {suffix}"
    raise CandidateBankError("unknown reporting-frame sentence type")


def _direct_text(reporter: str, suffix: str, sentence_type: str) -> str:
    if sentence_type == "declarative":
        return f"The {reporter} describes that object as {suffix}"
    if sentence_type == "interrogative":
        return f"Does the {reporter} describe that object as {suffix}"
    raise CandidateBankError("unknown direct sentence type")


def _copy_text(mark: str, suffix: str) -> str:
    return f"Copy the visible mark {mark} after {suffix}"


def _terminal(sentence_type: str) -> str:
    return "." if sentence_type == "declarative" else "?"


def _joint_token_id(prompt: str, continuation: str) -> tuple[list[int], int]:
    prompt_ids = ENCODING.encode(prompt)
    complete_ids = ENCODING.encode(prompt + continuation)
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise CandidateBankError("prompt is not a stable joint-tokenization prefix")
    suffix = complete_ids[len(prompt_ids):]
    standalone = ENCODING.encode(continuation)
    if len(standalone) != 1 or suffix != standalone:
        raise CandidateBankError("punctuation is not its exact standalone continuation token")
    return prompt_ids, suffix[0]


def _opposite(mark: str) -> str:
    if mark not in PUNCTUATION:
        raise CandidateBankError("answer lies outside the punctuation vocabulary")
    return "?" if mark == "." else "."


def _row(
    *, seed: int, group_number: int, group_id: str, transform_id: str,
    construction_id: str, direction_id: str, reporter: str,
    alternate_reporter: str, adjective: str, object_name: str,
    base_text: str, donor_text: str, base_answer: str, donor_answer: str,
) -> dict[str, Any]:
    transform = next(item for item in TASK_SPEC.transforms
                     if item.transform_id == transform_id)
    base_foil, donor_foil = _opposite(base_answer), _opposite(donor_answer)
    base_ids, base_answer_id = _joint_token_id(base_text, base_answer)
    donor_ids, donor_answer_id = _joint_token_id(donor_text, donor_answer)
    _, base_foil_id = _joint_token_id(base_text, base_foil)
    _, donor_foil_id = _joint_token_id(donor_text, donor_foil)
    suffix = _suffix(adjective, object_name)
    base_position, donor_position = len(base_ids) - 1, len(donor_ids) - 1
    answer_changes = base_answer != donor_answer
    capability_cell_id = f"{transform_id}/{construction_id}/{direction_id}"
    checks = {
        "single_token_punctuation": all(len(ENCODING.encode(mark)) == 1
                                         for mark in PUNCTUATION),
        "exact_answer_vocabulary": (
            base_answer in PUNCTUATION and donor_answer in PUNCTUATION
            and base_foil in PUNCTUATION and donor_foil in PUNCTUATION
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
            base_text.endswith(suffix) and donor_text.endswith(suffix)
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
    identity = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
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
        "family_id": f"{TASK_ID}/{transform_id}",
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
            "matched_final_suffix": suffix,
            "base_sentence_type": "declarative" if base_answer == "." else "interrogative",
            "donor_sentence_type": "declarative" if donor_answer == "." else "interrogative",
            "copy_control": transform_id == "C",
        },
        "construction_checks": checks,
    }


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    reporter, alternate = _REPORTERS[case_index]
    adjective, object_name = _ADJECTIVES[case_index], _OBJECTS[case_index]
    suffix = _suffix(adjective, object_name)
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_type, donor_type = (
        ("declarative", "interrogative") if forward
        else ("interrogative", "declarative")
    )
    direction = f"{base_type}_to_{donor_type}"

    a1_base = _reporting_text(reporter, suffix, base_type)
    a1_donor = _reporting_text(reporter, suffix, donor_type)
    a2_base = _direct_text(reporter, suffix, base_type)
    a2_donor = _direct_text(reporter, suffix, donor_type)

    # P uses the A1 base construction exactly and reverses lexical direction on
    # alternate groups, so the invariant control is not tied to one word order.
    p_base_reporter, p_donor_reporter = (
        (reporter, alternate) if forward else (alternate, reporter)
    )
    p_base = _reporting_text(p_base_reporter, suffix, base_type)
    p_donor = _reporting_text(p_donor_reporter, suffix, base_type)
    p_direction = "primary_to_alternative" if forward else "alternative_to_primary"

    c_base_mark, c_donor_mark = ((".", "?") if forward else ("?", "."))
    c_direction = (
        "period_to_question" if forward else "question_to_period"
    )
    return [
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="A1", construction_id="reporting_frame",
            direction_id=direction, reporter=reporter, alternate_reporter=alternate,
            adjective=adjective, object_name=object_name,
            base_text=a1_base, donor_text=a1_donor,
            base_answer=_terminal(base_type), donor_answer=_terminal(donor_type),
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="A2", construction_id="direct_question",
            direction_id=direction, reporter=reporter, alternate_reporter=alternate,
            adjective=adjective, object_name=object_name,
            base_text=a2_base, donor_text=a2_donor,
            base_answer=_terminal(base_type), donor_answer=_terminal(donor_type),
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="P", construction_id=f"reporting_frame_{base_type}",
            direction_id=p_direction, reporter=reporter, alternate_reporter=alternate,
            adjective=adjective, object_name=object_name,
            base_text=p_base, donor_text=p_donor,
            base_answer=_terminal(base_type), donor_answer=_terminal(base_type),
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="C", construction_id="explicit_visible_mark_copy",
            direction_id=c_direction, reporter=reporter, alternate_reporter=alternate,
            adjective=adjective, object_name=object_name,
            base_text=_copy_text(c_base_mark, suffix),
            donor_text=_copy_text(c_donor_mark, suffix),
            base_answer=c_base_mark, donor_answer=c_donor_mark,
        ),
    ]


def _build_rows(groups: int, seed: int) -> list[dict[str, Any]]:
    order = _permutation(seed)
    return [
        row
        for group_number in range(groups)
        for row in _panel(seed, group_number, order[group_number])
    ]


def _validate_request(task_id: str, groups: int, seed: int) -> None:
    if task_id not in CANDIDATES:
        raise KeyError(f"unknown circuit-screen candidate: {task_id}")
    if type(groups) is not int or not 2 <= groups <= DEFAULT_GROUPS or groups % 2:
        raise CandidateBankError("groups must be an even integer from 2 through 32")
    if type(seed) is not int or seed < 0:
        raise CandidateBankError("seed must be a nonnegative integer")


def validate_rows(
    rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
    groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED,
) -> str:
    """Recompute the complete authority and return its canonical digest."""
    _validate_request(task_id, groups, seed)
    materialized = [dict(row) for row in rows]
    expected = _build_rows(groups, seed)
    if materialized != expected:
        raise CandidateBankError("rows differ from the deterministic semantic authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    row_ids = [str(row["row_id"]) for row in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise CandidateBankError("row IDs are not unique")
    cells: dict[tuple[str, str], int] = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise CandidateBankError("a stored construction check is false")
        key = (str(row["transform_id"]), str(row["direction_id"]))
        cells[key] = cells.get(key, 0) + 1
    half = groups // 2
    for transform in ("A1", "A2"):
        if cells.get((transform, "declarative_to_interrogative")) != half \
                or cells.get((transform, "interrogative_to_declarative")) != half:
            raise CandidateBankError(f"{transform} ordered directions are unbalanced")
    if cells.get(("C", "period_to_question")) != half \
            or cells.get(("C", "question_to_period")) != half:
        raise CandidateBankError("C ordered directions are unbalanced")
    return digest


def build_rows(
    task_id: str, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Build linked, FIT-only A1/A2/P/C panels for a registered candidate."""
    _validate_request(task_id, groups, seed)
    rows = _build_rows(groups, seed)
    validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    return rows


def authority_sha256(
    task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED,
) -> str:
    """Return the integration-contract digest of the deterministic rows."""
    rows = build_rows(task_id, groups=groups, seed=seed)
    return validate_rows(rows, task_id=task_id, groups=groups, seed=seed)

