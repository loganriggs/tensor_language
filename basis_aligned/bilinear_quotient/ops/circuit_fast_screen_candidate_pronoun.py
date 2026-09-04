#!/usr/bin/env python3
"""Deterministic FIT panels for a pronoun-antecedent causal screen.

The answer-changing families select which of two explicitly gendered people
performed an action.  A1 states the action in active voice and A2 in passive
voice.  P changes only a location while preserving actor and answer.  C uses
the same `` he``/`` she`` endpoints in a natural repetition completion, so a
generic endpoint or answer-token handle is not evidence for antecedent state.

This module constructs data only.  It does not import a model or run a GPU.
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
TASK_ID = "pronoun_antecedent.gender_reference"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260904
SPLIT = "FIT"
TRANSFORMS = battery.TRANSFORMS
PRONOUNS = (" he", " she")
ENCODING = tiktoken.get_encoding("gpt2")


class CandidateBankError(ValueError):
    """A request or generated authority violates the candidate semantics."""


@dataclass(frozen=True)
class CandidateSpec:
    task: battery.BatteryTaskSpec
    description: str
    answer_vocabulary: tuple[str, ...]
    default_groups: int
    default_seed: int


TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_pronoun_antecedent_fit_panels",
    answer_role="score_jointly_tokenized_he_vs_she",
    transforms=(
        battery.TransformSpec(
            "A1", "active_voice_actor_swap", True, "toward_donor"
        ),
        battery.TransformSpec(
            "A2", "passive_voice_actor_swap", True, "toward_donor"
        ),
        battery.TransformSpec(
            "P", "irrelevant_location_rewrite", False, "invariant"
        ),
        battery.TransformSpec(
            "C", "explicit_visible_pronoun_copy", True, "registered_active"
        ),
    ),
)

CANDIDATES: Mapping[str, CandidateSpec] = MappingProxyType({
    TASK_ID: CandidateSpec(
        task=TASK_SPEC,
        description=(
            "Choose he or she for the selected antecedent across active and "
            "passive constructions; P changes location and C repeats a visible pronoun."
        ),
        answer_vocabulary=PRONOUNS,
        default_groups=DEFAULT_GROUPS,
        default_seed=DEFAULT_SEED,
    )
})


_OBJECTS = (
    "parcel", "lantern", "folder", "basket", "camera", "ticket", "letter", "bottle",
    "package", "notebook", "painting", "blanket", "suitcase", "vase", "helmet", "map",
    "tablet", "jacket", "backpack", "key", "newspaper", "umbrella", "violin", "box",
    "thermos", "poster", "wallet", "book", "radio", "scarf", "trophy", "envelope",
)
_LOCATIONS = (
    ("station", "museum"), ("library", "theater"), ("harbor", "market"),
    ("garden", "office"), ("school", "clinic"), ("airport", "hotel"),
    ("workshop", "gallery"), ("kitchen", "studio"), ("garage", "bakery"),
    ("warehouse", "farm"), ("park", "plaza"), ("cafe", "lobby"),
    ("factory", "college"), ("hospital", "courthouse"), ("bank", "church"),
    ("stadium", "bookstore"), ("laboratory", "restaurant"), ("hall", "porch"),
    ("bridge", "dock"), ("castle", "tower"), ("depot", "terminal"),
    ("classroom", "auditorium"), ("courtyard", "greenhouse"), ("chapel", "palace"),
    ("mill", "foundry"), ("ranch", "orchard"), ("marina", "village"),
    ("station", "arena"), ("salon", "nursery"), ("mansion", "cottage"),
    ("camp", "resort"), ("zoo", "aquarium"),
)
if not (len(_OBJECTS) == len(_LOCATIONS) == DEFAULT_GROUPS):
    raise RuntimeError("pronoun-antecedent lexical tables changed size")


def canonical_sha256(value: object) -> str:
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


def _target_suffix() -> str:
    return "Answer with he or she. The answer is"


def _copy_suffix() -> str:
    return "The pronoun shown on the card is"


def _introduction(woman_label: str, man_label: str, object_name: str, location: str) -> str:
    return (
        f"{woman_label} was a woman, and {man_label} was a man. "
        f"They inspected the {object_name} near the {location}."
    )


def _active_text(
    woman_label: str, man_label: str, actor: str, object_name: str, location: str,
) -> str:
    return (
        f"{_introduction(woman_label, man_label, object_name, location)} "
        f"{actor} carried it inside. Who carried it? {_target_suffix()}"
    )


def _passive_text(
    woman_label: str, man_label: str, actor: str, object_name: str, location: str,
) -> str:
    return (
        f"{_introduction(woman_label, man_label, object_name, location)} "
        f"It was carried inside by {actor}. Who carried it? {_target_suffix()}"
    )


def _copy_text(pronoun: str, location: str) -> str:
    visible = pronoun.strip()
    return (
        f"A card near the {location} displays the pronoun {visible}. "
        f"{_copy_suffix()}"
    )


def _pronoun_for(actor: str, woman_label: str, man_label: str) -> str:
    if actor == woman_label:
        return " she"
    if actor == man_label:
        return " he"
    raise CandidateBankError("actor is not one of the explicitly introduced people")


def _opposite(pronoun: str) -> str:
    if pronoun not in PRONOUNS:
        raise CandidateBankError("answer lies outside the he/she vocabulary")
    return " she" if pronoun == " he" else " he"


def _joint_token_id(prompt: str, continuation: str) -> tuple[list[int], int]:
    prompt_ids = ENCODING.encode(prompt)
    complete_ids = ENCODING.encode(prompt + continuation)
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise CandidateBankError("prompt is not a stable joint-tokenization prefix")
    suffix = complete_ids[len(prompt_ids):]
    standalone = ENCODING.encode(continuation)
    if len(standalone) != 1 or suffix != standalone:
        raise CandidateBankError("answer is not its exact standalone continuation token")
    return prompt_ids, suffix[0]


def _row(
    *, seed: int, group_number: int, group_id: str, transform_id: str,
    construction_id: str, direction_id: str, woman_label: str, man_label: str,
    object_name: str, primary_location: str, alternate_location: str,
    base_text: str, donor_text: str, base_answer: str, donor_answer: str,
    base_antecedent: str | None, donor_antecedent: str | None,
) -> dict[str, Any]:
    transform = next(item for item in TASK_SPEC.transforms
                     if item.transform_id == transform_id)
    base_foil, donor_foil = _opposite(base_answer), _opposite(donor_answer)
    base_ids, base_answer_id = _joint_token_id(base_text, base_answer)
    donor_ids, donor_answer_id = _joint_token_id(donor_text, donor_answer)
    _, base_foil_id = _joint_token_id(base_text, base_foil)
    _, donor_foil_id = _joint_token_id(donor_text, donor_foil)
    base_position, donor_position = len(base_ids) - 1, len(donor_ids) - 1
    answer_changes = base_answer != donor_answer
    capability_cell_id = f"{transform_id}/{construction_id}/{direction_id}"
    matched_suffix = _copy_suffix() if transform_id == "C" else _target_suffix()
    checks = {
        "single_token_pronouns": all(len(ENCODING.encode(item)) == 1 for item in PRONOUNS),
        "exact_answer_vocabulary": (
            base_answer in PRONOUNS and donor_answer in PRONOUNS
            and base_foil in PRONOUNS and donor_foil in PRONOUNS
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
        "antecedents_are_introduced_or_copy_control": (
            transform_id == "C" or (
                base_antecedent in (woman_label, man_label)
                and donor_antecedent in (woman_label, man_label)
                and f"{woman_label} was a woman" in base_text
                and f"{man_label} was a man" in base_text
                and f"{woman_label} was a woman" in donor_text
                and f"{man_label} was a man" in donor_text
            )
        ),
        "answers_match_antecedents_or_visible_copy": (
            (transform_id == "C" and base_answer.strip() in base_text
             and donor_answer.strip() in donor_text)
            or (transform_id != "C"
                and _pronoun_for(str(base_antecedent), woman_label, man_label) == base_answer
                and _pronoun_for(str(donor_antecedent), woman_label, man_label) == donor_answer)
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
        "woman_label": woman_label,
        "man_label": man_label,
        "object_name": object_name,
        "primary_location": primary_location,
        "alternate_location": alternate_location,
        "base_text": base_text,
        "donor_text": donor_text,
        "base_answer": base_answer,
        "donor_answer": donor_answer,
        "base_foil": base_foil,
        "donor_foil": donor_foil,
        "base_antecedent": base_antecedent,
        "donor_antecedent": donor_antecedent,
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
            "matched_final_suffix": matched_suffix,
            "base_antecedent": base_antecedent,
            "donor_antecedent": donor_antecedent,
            "antecedent_binding_task": transform_id in ("A1", "A2", "P"),
            "location_invariance_control": transform_id == "P",
            "copy_control": transform_id == "C",
        },
        "construction_checks": checks,
    }


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    # Reversing which neutral label is female prevents a fixed A/B-to-pronoun rule.
    woman_label, man_label = (
        ("Person A", "Person B") if case_index % 2 == 0
        else ("Person B", "Person A")
    )
    object_name = _OBJECTS[case_index]
    primary_location, alternate_location = _LOCATIONS[case_index]
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_actor, donor_actor = (
        (woman_label, man_label) if forward else (man_label, woman_label)
    )
    direction = "female_to_male" if forward else "male_to_female"
    base_answer = _pronoun_for(base_actor, woman_label, man_label)
    donor_answer = _pronoun_for(donor_actor, woman_label, man_label)

    # P preserves the A1 base actor and answer while changing only location.
    p_base_location, p_donor_location = (
        (primary_location, alternate_location) if forward
        else (alternate_location, primary_location)
    )
    p_direction = (
        "female_location_primary_to_alternate" if forward
        else "male_location_alternate_to_primary"
    )

    c_base_answer, c_donor_answer = (
        (" she", " he") if forward else (" he", " she")
    )
    c_direction = "she_to_he" if forward else "he_to_she"
    return [
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="A1", construction_id="active_voice",
            direction_id=direction, woman_label=woman_label, man_label=man_label,
            object_name=object_name, primary_location=primary_location,
            alternate_location=alternate_location,
            base_text=_active_text(woman_label, man_label, base_actor, object_name, primary_location),
            donor_text=_active_text(woman_label, man_label, donor_actor, object_name, primary_location),
            base_answer=base_answer, donor_answer=donor_answer,
            base_antecedent=base_actor, donor_antecedent=donor_actor,
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="A2", construction_id="passive_voice",
            direction_id=direction, woman_label=woman_label, man_label=man_label,
            object_name=object_name, primary_location=primary_location,
            alternate_location=alternate_location,
            base_text=_passive_text(woman_label, man_label, base_actor, object_name, primary_location),
            donor_text=_passive_text(woman_label, man_label, donor_actor, object_name, primary_location),
            base_answer=base_answer, donor_answer=donor_answer,
            base_antecedent=base_actor, donor_antecedent=donor_actor,
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="P", construction_id=(
                "active_voice_female_actor" if forward else "active_voice_male_actor"
            ),
            direction_id=p_direction, woman_label=woman_label, man_label=man_label,
            object_name=object_name, primary_location=primary_location,
            alternate_location=alternate_location,
            base_text=_active_text(woman_label, man_label, base_actor, object_name, p_base_location),
            donor_text=_active_text(woman_label, man_label, base_actor, object_name, p_donor_location),
            base_answer=base_answer, donor_answer=base_answer,
            base_antecedent=base_actor, donor_antecedent=base_actor,
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="C", construction_id="explicit_visible_pronoun_copy",
            direction_id=c_direction, woman_label=woman_label, man_label=man_label,
            object_name=object_name, primary_location=primary_location,
            alternate_location=alternate_location,
            base_text=_copy_text(c_base_answer, primary_location),
            donor_text=_copy_text(c_donor_answer, primary_location),
            base_answer=c_base_answer, donor_answer=c_donor_answer,
            base_antecedent=None, donor_antecedent=None,
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
        if cells.get((transform, "female_to_male")) != half \
                or cells.get((transform, "male_to_female")) != half:
            raise CandidateBankError(f"{transform} ordered directions are unbalanced")
    if cells.get(("C", "she_to_he")) != half \
            or cells.get(("C", "he_to_she")) != half:
        raise CandidateBankError("C ordered directions are unbalanced")
    return digest


def build_rows(
    task_id: str, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    _validate_request(task_id, groups, seed)
    rows = _build_rows(groups, seed)
    validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    return rows


def authority_sha256(
    task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED,
) -> str:
    rows = build_rows(task_id, groups=groups, seed=seed)
    return validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
