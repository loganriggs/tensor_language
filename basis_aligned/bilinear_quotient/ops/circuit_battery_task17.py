#!/usr/bin/env python3
"""Strict positional-list-retrieval rows for the reusable circuit battery.

This module is CPU-only.  It constructs a complete FIT/SELECT/TEST/OOD
authority before any model call and validates the task against
``circuit_battery_integration_contract``.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable

import tiktoken

import circuit_battery_integration_contract as contract


TASK_ID = "retrieval.positional_list"
SCHEMA = "circuit_battery_positional_list_v1"
ENCODING = tiktoken.get_encoding("gpt2")

# Every value is one GPT-2 token when preceded by the answer-space.  The four
# interleaved slices are separate lexical pools for FIT/SELECT/TEST/OOD.
_CANDIDATES = (
    "red", "blue", "green", "black", "white", "brown", "gray", "pink",
    "gold", "silver", "bronze", "amber", "coral", "teal", "navy", "plum",
    "pearl", "stone", "wood", "glass", "steel", "copper", "iron", "clay",
    "sand", "snow", "rain", "wind", "cloud", "storm", "frost", "mist",
    "apple", "lemon", "mango", "peach", "plum", "grape", "melon", "berry",
    "cedar", "maple", "willow", "birch", "pine", "oak", "moss", "fern",
    "river", "ocean", "valley", "forest", "island", "canyon", "harbor", "meadow",
    "flame", "spark", "ember", "smoke", "light", "shadow", "dawn", "dusk",
    "circle", "square", "arrow", "cross", "heart", "star", "moon", "sun",
    "music", "paper", "clock", "table", "chair", "house", "road", "field",
)


def _single_answer(word: str) -> str:
    answer = " " + word
    if len(ENCODING.encode(answer)) != 1:
        raise RuntimeError(f"candidate is not one GPT-2 token at answer boundary: {answer!r}")
    return answer


VALUES = tuple(dict.fromkeys(word for word in _CANDIDATES if len(ENCODING.encode(" " + word)) == 1))
if len(VALUES) < 48:  # pragma: no cover - import-time version tripwire
    raise RuntimeError("positional-list vocabulary unexpectedly too small")


TASK_SPEC = contract.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_positional_list_panel",
    answer_role="score_jointly_tokenized_single_token_margin",
    transforms=(
        contract.TransformSpec("A1", "change_query_index", True, "toward_donor"),
        contract.TransformSpec("A2", "swap_queried_payload", True, "toward_donor"),
        contract.TransformSpec("P", "replace_unqueried_payload", False, "invariant"),
        # The query changes but the answer stays fixed because the target occurs
        # at both queried locations.  This active control asks whether an
        # intervention moves the index variable even when behavior need not move.
        contract.TransformSpec("C", "duplicate_target_index_control", False, "registered_active"),
    ),
)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rng(seed: int, split: str, group_number: int) -> random.Random:
    material = f"{SCHEMA}|{seed}|{split}|{group_number}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def _pool(split: str) -> tuple[str, ...]:
    return VALUES[contract.PHASES.index(split)::len(contract.PHASES)]


def _prompt(values: Iterable[str], query_index: int) -> str:
    return f"Items: {', '.join(values)}. Item {query_index + 1}:"


def _joint_encoding(prompt: str, answer: str) -> tuple[list[int], list[int]]:
    prompt_ids = ENCODING.encode(prompt)
    complete_ids = ENCODING.encode(prompt + answer)
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not a stable prefix under joint prompt+answer tokenization")
    suffix = complete_ids[len(prompt_ids):]
    if len(suffix) != 1:
        raise ValueError("answer is not exactly one token at the actual continuation boundary")
    return prompt_ids, suffix


def _row(
    *, split: str, group_id: str, transform_id: str,
    base_values: list[str], donor_values: list[str],
    base_query: int, donor_query: int, changed_variable: str,
) -> dict[str, Any]:
    base_text = _prompt(base_values, base_query)
    donor_text = _prompt(donor_values, donor_query)
    base_answer = _single_answer(base_values[base_query])
    donor_answer = _single_answer(donor_values[donor_query])
    base_ids, base_suffix = _joint_encoding(base_text, base_answer)
    donor_ids, donor_suffix = _joint_encoding(donor_text, donor_answer)
    if len(base_ids) != len(donor_ids):
        raise ValueError("base and donor prompts are not position-aligned")
    answer_changes = base_answer != donor_answer
    expected = next(item for item in TASK_SPEC.transforms if item.transform_id == transform_id)
    if answer_changes != expected.answer_changes:
        raise ValueError(f"{transform_id} answer-change semantics are wrong")
    foils = sorted({" " + value for value in base_values + donor_values} - {base_answer})
    if not foils:
        raise ValueError("row has no distinct foil")
    identity = {
        "schema": SCHEMA, "task_id": TASK_ID, "split": split,
        "group_id": group_id, "transform_id": transform_id,
        "base_values": base_values, "donor_values": donor_values,
        "base_query": base_query, "donor_query": donor_query,
    }
    return {
        **identity,
        "row_id": _canonical_sha(identity),
        "answer_changes": answer_changes,
        "expected_effect": expected.expected_effect,
        "changed_variable": changed_variable,
        "base_text": base_text,
        "donor_text": donor_text,
        "base_answer": base_answer,
        "donor_answer": donor_answer,
        "foil_answers": foils,
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer_id": base_suffix[0],
        "donor_answer_id": donor_suffix[0],
        "list_length": len(base_values),
    }


def _panel(split: str, group_number: int, seed: int) -> list[dict[str, Any]]:
    rng = _rng(seed, split, group_number)
    length = 6 if split == "OOD" else 4
    values = rng.sample(_pool(split), length + 2)
    base = values[:length]
    novel = values[length]
    query, alternate, untouched = rng.sample(range(length), 3)
    group_id = f"{split}:{_canonical_sha([SCHEMA, seed, split, group_number])[:20]}"

    a1 = _row(
        split=split, group_id=group_id, transform_id="A1",
        base_values=base, donor_values=list(base), base_query=query,
        donor_query=alternate, changed_variable="query_index",
    )

    swapped = list(base)
    swapped[query], swapped[alternate] = swapped[alternate], swapped[query]
    a2 = _row(
        split=split, group_id=group_id, transform_id="A2",
        base_values=base, donor_values=swapped, base_query=query,
        donor_query=query, changed_variable="payload_at_queried_index",
    )

    preserved = list(base)
    preserved[untouched] = novel
    p = _row(
        split=split, group_id=group_id, transform_id="P",
        base_values=base, donor_values=preserved, base_query=query,
        donor_query=query, changed_variable="unqueried_payload",
    )

    duplicated = list(base)
    duplicated[alternate] = duplicated[query]
    c = _row(
        split=split, group_id=group_id, transform_id="C",
        base_values=duplicated, donor_values=list(duplicated), base_query=query,
        donor_query=alternate, changed_variable="query_index_with_duplicate_target",
    )
    return [a1, a2, p, c]


def _validate_panel_semantics(panel: list[dict[str, Any]]) -> None:
    by_transform = {row["transform_id"]: row for row in panel}
    if set(by_transform) != set(contract.TRANSFORMS) or len(panel) != 4:
        raise contract.BatteryContractError("panel is not exactly A1/A2/P/C")
    a1, a2, p, c = (by_transform[name] for name in contract.TRANSFORMS)
    core = a1["base_values"]
    query = a1["base_query"]
    if not (a2["base_values"] == p["base_values"] == core
            and a2["base_query"] == p["base_query"] == query):
        raise contract.BatteryContractError("A1/A2/P do not share one generated base situation")

    alternate = a1["donor_query"]
    if a1["donor_values"] != core or alternate == query:
        raise contract.BatteryContractError("A1 changes more than the query index")

    changed_a2 = [index for index, values in enumerate(zip(core, a2["donor_values"]))
                  if values[0] != values[1]]
    if (a2["donor_query"] != query or set(changed_a2) != {query, alternate}
            or a2["donor_values"][query] != core[alternate]
            or a2["donor_values"][alternate] != core[query]):
        raise contract.BatteryContractError("A2 is not the registered queried-payload swap")

    changed_p = [index for index, values in enumerate(zip(core, p["donor_values"]))
                 if values[0] != values[1]]
    if (p["donor_query"] != query or len(changed_p) != 1 or changed_p[0] == query
            or p["donor_values"][changed_p[0]] in core):
        raise contract.BatteryContractError("P is not one novel unqueried-payload replacement")

    expected_control = list(core)
    expected_control[alternate] = core[query]
    if (c["base_values"] != expected_control or c["donor_values"] != expected_control
            or c["base_query"] != query or c["donor_query"] != alternate
            or len(set(expected_control)) < 2):
        raise contract.BatteryContractError("C is not the registered duplicate-target index control")


def validate_authority(rows: list[dict[str, Any]]) -> str:
    """Fail closed on task semantics in addition to the shared typed contract."""
    authority_sha = contract.validate_rows(TASK_SPEC, rows)
    phases = set(contract.PHASES)
    prompts_by_phase: dict[str, set[str]] = {phase: set() for phase in phases}
    values_by_phase: dict[str, set[str]] = {phase: set() for phase in phases}
    panels: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        phase = row["split"]
        prompts_by_phase[phase].update((row["base_text"], row["donor_text"]))
        values_by_phase[phase].update(row["base_values"])
        values_by_phase[phase].update(row["donor_values"])
        panels.setdefault((phase, row["group_id"]), []).append(row)
        identity = {
            "schema": SCHEMA, "task_id": TASK_ID, "split": phase,
            "group_id": row["group_id"], "transform_id": row["transform_id"],
            "base_values": row["base_values"], "donor_values": row["donor_values"],
            "base_query": row["base_query"], "donor_query": row["donor_query"],
        }
        if row["row_id"] != _canonical_sha(identity):
            raise contract.BatteryContractError("row identity does not bind its structured authority")
        _joint_encoding(row["base_text"], row["base_answer"])
        _joint_encoding(row["donor_text"], row["donor_answer"])
        if len(row["base_ids"]) != len(row["donor_ids"]):
            raise contract.BatteryContractError("interchange prompts are not position-aligned")
        if row["base_answer"] in row["foil_answers"] or not row["foil_answers"]:
            raise contract.BatteryContractError("answer/foil margin is degenerate")
    for panel in panels.values():
        _validate_panel_semantics(panel)
    for index, phase in enumerate(contract.PHASES):
        for other in contract.PHASES[index + 1:]:
            if prompts_by_phase[phase] & prompts_by_phase[other]:
                raise contract.BatteryContractError("prompt leakage across phases")
            if values_by_phase[phase] & values_by_phase[other]:
                raise contract.BatteryContractError("payload vocabulary leakage across phases")
    return authority_sha


def build_authority(groups_per_phase: int = 24, seed: int = 59317) -> tuple[list[dict[str, Any]], str]:
    if type(groups_per_phase) is not int or groups_per_phase <= 0:
        raise ValueError("groups_per_phase must be a positive integer")
    rows = [
        row
        for split in contract.PHASES
        for group_number in range(groups_per_phase)
        for row in _panel(split, group_number, seed)
    ]
    return rows, validate_authority(rows)


if __name__ == "__main__":
    authority, digest = build_authority()
    print(json.dumps({
        "schema": SCHEMA, "task_id": TASK_ID, "rows": len(authority),
        "groups": len(authority) // 4, "authority_sha256": digest,
    }, sort_keys=True))
