#!/usr/bin/env python3
"""Deterministic FIT panels for pending-quote-state causal screening.

This is data construction only: it imports no model and runs no intervention.
The target rows differ by one unmatched ASCII quote placed before a shared
description.  A second construction contains an earlier balanced quote pair,
so a transferable site must represent pending-quote state rather than merely
the presence of any quote.  P rewrites the writer while preserving the answer.
C contrasts a completed count statement with an inch-mark continuation without
putting a literal quote in the prompt, so it is an answer-changing endpoint
control unrelated to delimiter parity.
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


SCHEMA = "circuit_fast_screen_quote_parity_rows_v1"
TASK_ID = "quote_parity.pending_close"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260904
SPLIT = "FIT"
TRANSFORMS = battery.TRANSFORMS
ANSWER_VOCABULARY = (".", '"')
ENCODING = tiktoken.get_encoding("gpt2")


class QuoteParityCandidateError(ValueError):
    """A request or generated quote-parity authority is invalid."""


@dataclass(frozen=True)
class CandidateSpec:
    task: battery.BatteryTaskSpec
    description: str
    answer_vocabulary: tuple[str, ...]
    default_groups: int
    default_seed: int


TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_pending_quote_fit_panels",
    answer_role="score_jointly_tokenized_period_vs_ascii_quote",
    transforms=(
        battery.TransformSpec(
            "A1", "single_span_pending_quote_swap", True, "toward_donor"
        ),
        battery.TransformSpec(
            "A2", "balanced_prefix_pending_quote_swap", True, "toward_donor"
        ),
        battery.TransformSpec(
            "P", "writer_lexical_rewrite", False, "invariant"
        ),
        battery.TransformSpec(
            "C", "sentence_end_vs_inch_unit_mark", True, "registered_active"
        ),
    ),
)

CANDIDATES: Mapping[str, CandidateSpec] = MappingProxyType({
    TASK_ID: CandidateSpec(
        task=TASK_SPEC,
        description=(
            "Close one pending ASCII quote or end the corresponding unquoted "
            "description; preserve that choice under a writer rewrite and reject "
            "a generic endpoint direction using a period-versus-inch-mark completion."
        ),
        answer_vocabulary=ANSWER_VOCABULARY,
        default_groups=DEFAULT_GROUPS,
        default_seed=DEFAULT_SEED,
    )
})


_WRITERS = (
    ("editor", "curator"), ("clerk", "keeper"), ("teacher", "tutor"),
    ("author", "reader"), ("pilot", "sailor"), ("doctor", "nurse"),
    ("farmer", "baker"), ("artist", "writer"), ("driver", "rider"),
    ("guard", "warden"), ("judge", "mayor"), ("coach", "trainer"),
    ("chef", "miner"), ("actor", "singer"), ("parent", "neighbor"),
    ("student", "scholar"), ("captain", "merchant"), ("ranger", "worker"),
    ("visitor", "owner"), ("speaker", "listener"), ("friend", "cousin"),
    ("agent", "broker"), ("leader", "member"), ("expert", "novice"),
    ("buyer", "seller"), ("lawyer", "banker"), ("builder", "planner"),
    ("manager", "helper"), ("officer", "reporter"), ("director", "producer"),
    ("engineer", "designer"), ("scientist", "analyst"),
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
if not (len(_WRITERS) == len(_ADJECTIVES) == len(_OBJECTS) == DEFAULT_GROUPS):
    raise RuntimeError("quote-parity lexical tables changed size")


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


def _suffix(adjective: str, object_name: str) -> str:
    return (
        f"a remarkably {adjective} {object_name} in the archived catalog"
    )


def _description_text(writer: str, suffix: str, state: str, construction: str) -> str:
    opener = '"' if state == "pending" else ""
    if construction == "single_span":
        return f"The {writer} classifies this record as {opener}{suffix}"
    if construction == "balanced_prefix":
        return (
            f'The {writer} first files "sample", then classifies this record as '
            f"{opener}{suffix}"
        )
    raise QuoteParityCandidateError("unknown quote-parity construction")


def _control_text(answer: str, number: int) -> str:
    # The double quote is an inch unit here, not a delimiter.  Both continuations
    # attach directly to the final number in ordinary written English.
    if answer == ".":
        return f"The inventory has {number} items; its final count is exactly {number}"
    if answer == '"':
        return (
            f"The board is {number} inches long; in customary notation it measures "
            f"{number}"
        )
    raise QuoteParityCandidateError("unknown control answer")


def _answer(state: str) -> str:
    if state == "outside":
        return "."
    if state == "pending":
        return '"'
    raise QuoteParityCandidateError("unknown quote state")


def _opposite(answer: str) -> str:
    if answer not in ANSWER_VOCABULARY:
        raise QuoteParityCandidateError("answer is outside the frozen vocabulary")
    return '"' if answer == "." else "."


def _joint_token_id(prompt: str, continuation: str) -> tuple[list[int], int]:
    prompt_ids = ENCODING.encode(prompt)
    complete_ids = ENCODING.encode(prompt + continuation)
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise QuoteParityCandidateError("prompt is not a stable joint-tokenization prefix")
    suffix = complete_ids[len(prompt_ids):]
    standalone = ENCODING.encode(continuation)
    if len(standalone) != 1 or suffix != standalone:
        raise QuoteParityCandidateError(
            "answer is not its exact standalone continuation token"
        )
    return prompt_ids, suffix[0]


def _row(
    *, seed: int, group_number: int, group_id: str, transform_id: str,
    construction_id: str, direction_id: str, writer: str,
    alternate_writer: str, adjective: str, object_name: str,
    base_text: str, donor_text: str, base_answer: str, donor_answer: str,
) -> dict[str, Any]:
    transform = next(
        item for item in TASK_SPEC.transforms if item.transform_id == transform_id
    )
    base_foil, donor_foil = _opposite(base_answer), _opposite(donor_answer)
    base_ids, base_answer_id = _joint_token_id(base_text, base_answer)
    donor_ids, donor_answer_id = _joint_token_id(donor_text, donor_answer)
    _, base_foil_id = _joint_token_id(base_text, base_foil)
    _, donor_foil_id = _joint_token_id(donor_text, donor_foil)
    suffix = _suffix(adjective, object_name)
    base_position, donor_position = len(base_ids) - 1, len(donor_ids) - 1
    answer_changes = base_answer != donor_answer
    is_target = transform_id in ("A1", "A2")
    target_parity_valid = (
        (not is_target)
        or (
            (base_text.count('"') % 2 == 1) == (base_answer == '"')
            and (donor_text.count('"') % 2 == 1) == (donor_answer == '"')
            and (base_text + base_answer).count('"') % 2 == 0
            and (donor_text + donor_answer).count('"') % 2 == 0
        )
    )
    capability_cell_id = f"{transform_id}/{construction_id}/{direction_id}"
    checks = {
        "single_token_answers": all(
            len(ENCODING.encode(answer)) == 1 for answer in ANSWER_VOCABULARY
        ),
        "exact_answer_vocabulary": all(
            answer in ANSWER_VOCABULARY
            for answer in (base_answer, donor_answer, base_foil, donor_foil)
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
        "matched_target_long_final_suffix": (
            transform_id == "C"
            or (base_text.endswith(suffix) and donor_text.endswith(suffix))
        ),
        "matched_final_input_token": base_ids[-1] == donor_ids[-1],
        "semantic_position_is_final_input": (
            base_position == len(base_ids) - 1 and donor_position == len(donor_ids) - 1
        ),
        "no_trailing_space_or_quote": all(
            text and not text[-1].isspace() and text[-1] not in "\"'"
            for text in (base_text, donor_text)
        ),
        "target_quote_parity_determines_answer": target_parity_valid,
        "control_has_no_literal_quote": (
            transform_id != "C" or ('"' not in base_text and '"' not in donor_text)
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise QuoteParityCandidateError(f"row construction checks failed: {failed}")
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
        "writer": writer,
        "alternate_writer": alternate_writer,
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
            "matched_final_suffix": (
                suffix if transform_id != "C" else base_text.split()[-1]
            ),
            "base_quote_count": base_text.count('"'),
            "donor_quote_count": donor_text.count('"'),
            "base_pending_quote": is_target and base_answer == '"',
            "donor_pending_quote": is_target and donor_answer == '"',
            "inch_mark_endpoint_control": transform_id == "C",
        },
        "construction_checks": checks,
    }


def _panel(seed: int, group_number: int, case_index: int) -> list[dict[str, Any]]:
    writer, alternate = _WRITERS[case_index]
    adjective, object_name = _ADJECTIVES[case_index], _OBJECTS[case_index]
    suffix = _suffix(adjective, object_name)
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    forward = group_number % 2 == 0
    base_state, donor_state = (
        ("outside", "pending") if forward else ("pending", "outside")
    )
    direction = f"{base_state}_to_{donor_state}"

    a1_base = _description_text(writer, suffix, base_state, "single_span")
    a1_donor = _description_text(writer, suffix, donor_state, "single_span")
    a2_base = _description_text(writer, suffix, base_state, "balanced_prefix")
    a2_donor = _description_text(writer, suffix, donor_state, "balanced_prefix")

    p_base_writer, p_donor_writer = (
        (writer, alternate) if forward else (alternate, writer)
    )
    p_base = _description_text(p_base_writer, suffix, base_state, "single_span")
    p_donor = _description_text(p_donor_writer, suffix, base_state, "single_span")
    p_direction = "primary_to_alternative" if forward else "alternative_to_primary"

    c_base_answer, c_donor_answer = (
        (".", '"') if forward else ('"', ".")
    )
    c_direction = (
        "period_to_quote" if forward else "quote_to_period"
    )
    return [
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="A1", construction_id="single_span",
            direction_id=direction, writer=writer, alternate_writer=alternate,
            adjective=adjective, object_name=object_name,
            base_text=a1_base, donor_text=a1_donor,
            base_answer=_answer(base_state), donor_answer=_answer(donor_state),
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="A2", construction_id="balanced_prefix",
            direction_id=direction, writer=writer, alternate_writer=alternate,
            adjective=adjective, object_name=object_name,
            base_text=a2_base, donor_text=a2_donor,
            base_answer=_answer(base_state), donor_answer=_answer(donor_state),
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="P", construction_id=f"single_span_{base_state}",
            direction_id=p_direction, writer=writer, alternate_writer=alternate,
            adjective=adjective, object_name=object_name,
            base_text=p_base, donor_text=p_donor,
            base_answer=_answer(base_state), donor_answer=_answer(base_state),
        ),
        _row(
            seed=seed, group_number=group_number, group_id=group_id,
            transform_id="C", construction_id="sentence_end_vs_inch_unit_mark",
            direction_id=c_direction, writer=writer, alternate_writer=alternate,
            adjective=adjective, object_name=object_name,
            base_text=_control_text(c_base_answer, 12 + group_number),
            donor_text=_control_text(c_donor_answer, 12 + group_number),
            base_answer=c_base_answer, donor_answer=c_donor_answer,
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
        raise QuoteParityCandidateError("groups must be an even integer from 2 through 32")
    if type(seed) is not int or seed < 0:
        raise QuoteParityCandidateError("seed must be a nonnegative integer")


def validate_rows(
    rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
    groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED,
) -> str:
    """Recompute the exact authority and return its integration digest."""
    _validate_request(task_id, groups, seed)
    materialized = [dict(row) for row in rows]
    expected = _build_rows(groups, seed)
    if materialized != expected:
        raise QuoteParityCandidateError(
            "rows differ from the deterministic quote-parity authority"
        )
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise QuoteParityCandidateError(str(error)) from error
    row_ids = [str(row["row_id"]) for row in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise QuoteParityCandidateError("row IDs are not unique")
    cells: dict[tuple[str, str], int] = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise QuoteParityCandidateError("a stored construction check is false")
        key = (str(row["transform_id"]), str(row["direction_id"]))
        cells[key] = cells.get(key, 0) + 1
    half = groups // 2
    for transform in ("A1", "A2"):
        if cells.get((transform, "outside_to_pending")) != half \
                or cells.get((transform, "pending_to_outside")) != half:
            raise QuoteParityCandidateError(f"{transform} ordered directions are unbalanced")
    if cells.get(("C", "period_to_quote")) != half \
            or cells.get(("C", "quote_to_period")) != half:
        raise QuoteParityCandidateError("C ordered directions are unbalanced")
    return digest


def build_rows(
    task_id: str, groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    _validate_request(task_id, groups, seed)
    rows = _build_rows(groups, seed)
    validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
    return rows


def authority_sha256(
    task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
    seed: int = DEFAULT_SEED,
) -> str:
    rows = build_rows(task_id, groups=groups, seed=seed)
    return validate_rows(rows, task_id=task_id, groups=groups, seed=seed)
