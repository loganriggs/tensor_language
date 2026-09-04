#!/usr/bin/env python3
"""Strict repeated-token-copy rows for the reusable circuit battery.

This CPU-only module creates one complete, linked A1/A2/P/C panel per
situation.  It defines data only: no model, outcome, or localization path is
present.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from typing import Any, Iterable

import tiktoken

import circuit_battery_integration_contract as contract


TASK_ID = "verbatim_repeat.copy"
SCHEMA = "circuit_battery_verbatim_repeat_v1"
ENCODING = tiktoken.get_encoding("gpt2")

_CANDIDATES = (
    "red", "blue", "green", "black", "white", "brown", "gray", "pink",
    "gold", "silver", "bronze", "amber", "coral", "teal", "navy", "plum",
    "pearl", "stone", "wood", "glass", "steel", "copper", "iron", "clay",
    "sand", "snow", "rain", "wind", "cloud", "storm", "frost", "mist",
    "apple", "lemon", "mango", "peach", "grape", "melon", "berry", "cedar",
    "maple", "willow", "birch", "pine", "oak", "moss", "fern", "river",
    "ocean", "valley", "forest", "island", "canyon", "harbor", "meadow", "flame",
    "spark", "ember", "smoke", "light", "shadow", "dawn", "dusk", "circle",
    "square", "arrow", "cross", "heart", "star", "moon", "sun", "music",
    "paper", "clock", "table", "chair", "house", "road", "field", "quiet",
    "bright", "soft", "sharp", "north", "south", "east", "west", "spring",
    "summer", "autumn", "winter", "morning", "evening", "garden", "bridge",
)


def _answer(word: str) -> str:
    answer = " " + word
    if len(ENCODING.encode(answer)) != 1:
        raise RuntimeError(f"copy candidate is not one GPT-2 continuation token: {answer!r}")
    return answer


_USABLE_VALUES = tuple(dict.fromkeys(
    word for word in _CANDIDATES if len(ENCODING.encode(" " + word)) == 1
))
if len(_USABLE_VALUES) < 84:  # pragma: no cover - tokenizer-version tripwire
    raise RuntimeError("verbatim-copy vocabulary unexpectedly too small")
# Freeze an equal 21-token vocabulary for each of the four phases.  Values after
# the first 84 are deliberately unused rather than allowing unequal phase pools.
VALUES = _USABLE_VALUES[:84]


TASK_SPEC = contract.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_verbatim_repeat_panel",
    answer_role="score_jointly_tokenized_single_token_margin",
    transforms=(
        contract.TransformSpec("A1", "replace_entire_trailing_repeat", True, "toward_donor"),
        contract.TransformSpec("A2", "replace_latest_two_token_run", True, "toward_donor"),
        contract.TransformSpec("P", "replace_leading_filler", False, "invariant"),
        contract.TransformSpec("C", "extend_same_target_repeat", False, "registered_active"),
    ),
)

_CHANGED_VARIABLE = {
    "A1": "identity_of_entire_trailing_repeat",
    "A2": "identity_and_onset_of_latest_two_token_run",
    "P": "leading_nonrepeat_filler",
    "C": "same_target_repeat_length",
}


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rng(seed: int, split: str, group_number: int) -> random.Random:
    material = f"{SCHEMA}|{seed}|{split}|{group_number}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def _pool(split: str) -> tuple[str, ...]:
    return VALUES[contract.PHASES.index(split)::len(contract.PHASES)]


def _shape(split: str) -> tuple[str, int, int]:
    if split == "OOD":
        return "Echo the final run:", 3, 4
    return "Repeat exactly:", 2, 3


def _prompt(prefix: str, tokens: Iterable[str]) -> str:
    return prefix + "".join(" " + token for token in tokens)


def _joint_encoding(prompt: str, answer: str) -> tuple[list[int], list[int]]:
    prompt_ids = ENCODING.encode(prompt)
    complete_ids = ENCODING.encode(prompt + answer)
    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not a stable prefix under joint prompt+answer tokenization")
    suffix = complete_ids[len(prompt_ids):]
    if len(suffix) != 1:
        raise ValueError("answer is not exactly one token at the continuation boundary")
    return prompt_ids, suffix


def _trailing_run(tokens: list[str]) -> tuple[str, int, int]:
    target = tokens[-1]
    start = len(tokens) - 1
    while start > 0 and tokens[start - 1] == target:
        start -= 1
    return target, start, len(tokens) - start


def _row(
    *, split: str, group_id: str, transform_id: str, prefix: str,
    base_tokens: list[str], donor_tokens: list[str], changed_variable: str,
) -> dict[str, Any]:
    base_target, base_start, base_count = _trailing_run(base_tokens)
    donor_target, donor_start, donor_count = _trailing_run(donor_tokens)
    base_text = _prompt(prefix, base_tokens)
    donor_text = _prompt(prefix, donor_tokens)
    base_answer = _answer(base_target)
    donor_answer = _answer(donor_target)
    base_ids, base_suffix = _joint_encoding(base_text, base_answer)
    donor_ids, donor_suffix = _joint_encoding(donor_text, donor_answer)
    if len(base_ids) != len(donor_ids):
        raise ValueError("base and donor copy prompts are not position-aligned")
    expected = next(item for item in TASK_SPEC.transforms if item.transform_id == transform_id)
    answer_changes = base_answer != donor_answer
    if answer_changes != expected.answer_changes:
        raise ValueError(f"{transform_id} answer-change semantics are wrong")
    identity = {
        "schema": SCHEMA, "task_id": TASK_ID, "split": split,
        "group_id": group_id, "transform_id": transform_id, "prefix": prefix,
        "base_tokens": base_tokens, "donor_tokens": donor_tokens,
        "base_repeat_start": base_start, "donor_repeat_start": donor_start,
        "base_repeat_count": base_count, "donor_repeat_count": donor_count,
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
        "base_ids": base_ids,
        "donor_ids": donor_ids,
        "base_answer_id": base_suffix[0],
        "donor_answer_id": donor_suffix[0],
        "sequence_words": len(base_tokens),
    }


def _panel(split: str, group_number: int, seed: int) -> list[dict[str, Any]]:
    prefix, filler_count, repeat_count = _shape(split)
    pool = list(_pool(split))
    if len(pool) != 21:
        raise RuntimeError("balanced task authority requires exactly 21 tokens per phase")
    # One SHA-seeded phase permutation plus distinct cyclic offsets is a Latin
    # assignment: across 21 groups every token appears exactly once in every
    # semantic role, while all roles within one group remain distinct.
    _rng(seed, split, -1).shuffle(pool)
    target = pool[group_number % 21]
    alternative = pool[(group_number + 1) % 21]
    novel = pool[(group_number + 2) % 21]
    fillers = [pool[(group_number + 3 + index) % 21] for index in range(filler_count)]
    base = fillers + [target] * repeat_count
    group_id = f"{split}:{_canonical_sha([SCHEMA, seed, split, group_number])[:20]}"

    a1_donor = fillers + [alternative] * repeat_count
    a2_donor = fillers + [target] * (repeat_count - 2) + [alternative, alternative]
    p_donor = list(base)
    p_donor[0] = novel
    c_donor = list(base)
    c_donor[filler_count - 1] = target

    return [
        _row(
            split=split, group_id=group_id, transform_id="A1", prefix=prefix,
            base_tokens=list(base), donor_tokens=a1_donor,
            changed_variable="identity_of_entire_trailing_repeat",
        ),
        _row(
            split=split, group_id=group_id, transform_id="A2", prefix=prefix,
            base_tokens=list(base), donor_tokens=a2_donor,
            changed_variable="identity_and_onset_of_latest_two_token_run",
        ),
        _row(
            split=split, group_id=group_id, transform_id="P", prefix=prefix,
            base_tokens=list(base), donor_tokens=p_donor,
            changed_variable="leading_nonrepeat_filler",
        ),
        _row(
            split=split, group_id=group_id, transform_id="C", prefix=prefix,
            base_tokens=list(base), donor_tokens=c_donor,
            changed_variable="same_target_repeat_length",
        ),
    ]


def _validate_panel_semantics(panel: list[dict[str, Any]]) -> None:
    by_transform = {row["transform_id"]: row for row in panel}
    if set(by_transform) != set(contract.TRANSFORMS) or len(panel) != 4:
        raise contract.BatteryContractError("copy panel is not exactly A1/A2/P/C")
    a1, a2, p, c = (by_transform[name] for name in contract.TRANSFORMS)
    base = a1["base_tokens"]
    if any(row["base_tokens"] != base or row["prefix"] != a1["prefix"] for row in (a2, p, c)):
        raise contract.BatteryContractError("A1/A2/P/C do not share one base copy situation")
    target, start, count = _trailing_run(base)
    fillers = base[:start]
    if count < 3 or len(fillers) < 2 or len(set(fillers + [target])) != len(fillers) + 1:
        raise contract.BatteryContractError("base is not a distinct-filler plus trailing-repeat situation")

    alternative = a1["donor_tokens"][-1]
    if (a1["donor_tokens"][:start] != fillers
            or a1["donor_tokens"][start:] != [alternative] * count
            or alternative == target):
        raise contract.BatteryContractError("A1 is not an entire-repeat identity replacement")

    if (a2["donor_tokens"][:start] != fillers
            or a2["donor_tokens"][start:start + count - 2] != [target] * (count - 2)
            or a2["donor_tokens"][-2:] != [alternative, alternative]):
        raise contract.BatteryContractError("A2 is not the distinct latest-two-token run replacement")

    changed_p = [index for index, pair in enumerate(zip(base, p["donor_tokens"])) if pair[0] != pair[1]]
    if (changed_p != [0] or p["donor_tokens"][start:] != [target] * count
            or p["donor_tokens"][0] in set(base + a1["donor_tokens"])):
        raise contract.BatteryContractError("P is not one novel leading-filler replacement")

    expected_c = list(base)
    expected_c[start - 1] = target
    if (c["donor_tokens"] != expected_c or c["donor_answer"] != c["base_answer"]
            or c["donor_repeat_count"] != count + 1):
        raise contract.BatteryContractError("C is not a same-target repeat-length extension")


def validate_authority(rows: list[dict[str, Any]]) -> str:
    """Recompute row identity, linked edit semantics, phase isolation, and token positions."""
    authority_sha = contract.validate_rows(TASK_SPEC, rows)
    prompts_by_phase = {phase: set() for phase in contract.PHASES}
    values_by_phase = {phase: set() for phase in contract.PHASES}
    panels: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        phase = row["split"]
        prefix, filler_count, repeat_count = _shape(phase)
        transform = next(
            item for item in TASK_SPEC.transforms if item.transform_id == row["transform_id"]
        )
        if (row["schema"] != SCHEMA
                or row["task_id"] != TASK_ID
                or row["prefix"] != prefix
                or row["expected_effect"] != transform.expected_effect
                or row["answer_changes"] != transform.answer_changes
                or row["changed_variable"] != _CHANGED_VARIABLE[row["transform_id"]]
                or row["sequence_words"] != len(row["base_tokens"])
                or len(row["base_tokens"]) != len(row["donor_tokens"])
                or len(row["base_tokens"]) != filler_count + repeat_count
                or row["base_repeat_start"] != filler_count
                or row["base_repeat_count"] != repeat_count):
            raise contract.BatteryContractError("row semantic metadata or phase shape changed")
        prompts_by_phase[phase].update((row["base_text"], row["donor_text"]))
        values_by_phase[phase].update(row["base_tokens"])
        values_by_phase[phase].update(row["donor_tokens"])
        panels.setdefault((phase, row["group_id"]), []).append(row)
        identity = {
            key: row[key] for key in (
                "schema", "task_id", "split", "group_id", "transform_id", "prefix",
                "base_tokens", "donor_tokens", "base_repeat_start", "donor_repeat_start",
                "base_repeat_count", "donor_repeat_count",
            )
        }
        if row["row_id"] != _canonical_sha(identity):
            raise contract.BatteryContractError("row identity does not bind structured copy authority")
        for side in ("base", "donor"):
            target, start, count = _trailing_run(row[f"{side}_tokens"])
            if (row[f"{side}_text"] != _prompt(prefix, row[f"{side}_tokens"])
                    or row[f"{side}_answer"] != _answer(target)
                    or row[f"{side}_repeat_start"] != start
                    or row[f"{side}_repeat_count"] != count):
                raise contract.BatteryContractError("copy answer or trailing-run metadata changed")
            ids, suffix = _joint_encoding(row[f"{side}_text"], row[f"{side}_answer"])
            if ids != row[f"{side}_ids"] or suffix != [row[f"{side}_answer_id"]]:
                raise contract.BatteryContractError("stored continuation tokenization changed")
        if len(row["base_ids"]) != len(row["donor_ids"]):
            raise contract.BatteryContractError("interchange positions are not aligned")
    for panel in panels.values():
        _validate_panel_semantics(panel)
    for phase in contract.PHASES:
        phase_panels = [panel for (split, _), panel in panels.items() if split == phase]
        if len(phase_panels) != 21:
            raise contract.BatteryContractError("phase does not have exactly 21 balanced panels")
        roles: dict[str, Counter[str]] = {
            "target": Counter(), "alternative": Counter(), "novel": Counter(),
        }
        filler_count = _shape(phase)[1]
        roles.update({f"filler_{index}": Counter() for index in range(filler_count)})
        for panel in phase_panels:
            by_transform = {row["transform_id"]: row for row in panel}
            base = by_transform["A1"]["base_tokens"]
            start = by_transform["A1"]["base_repeat_start"]
            roles["target"][base[-1]] += 1
            roles["alternative"][by_transform["A1"]["donor_tokens"][-1]] += 1
            roles["novel"][by_transform["P"]["donor_tokens"][0]] += 1
            for index, value in enumerate(base[:start]):
                roles[f"filler_{index}"][value] += 1
        expected_values = set(_pool(phase))
        if any(set(counts) != expected_values or set(counts.values()) != {1}
               for counts in roles.values()):
            raise contract.BatteryContractError("semantic roles are not exactly phase-balanced")
    for index, phase in enumerate(contract.PHASES):
        for other in contract.PHASES[index + 1:]:
            if prompts_by_phase[phase] & prompts_by_phase[other]:
                raise contract.BatteryContractError("copy prompt leakage across phases")
            if values_by_phase[phase] & values_by_phase[other]:
                raise contract.BatteryContractError("copy vocabulary leakage across phases")
    return authority_sha


def build_authority(groups_per_phase: int = 21, seed: int = 61818) -> tuple[list[dict[str, Any]], str]:
    if groups_per_phase != 21:
        raise ValueError("balanced authority requires exactly 21 groups per phase")
    rows = [
        row
        for split in contract.PHASES
        for group_number in range(groups_per_phase)
        for row in _panel(split, group_number, seed)
    ]
    return rows, validate_authority(rows)


def split_rows(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], str]:
    if split not in contract.PHASES:
        raise ValueError("split is invalid")
    selected = [row for row in rows if row["split"] == split]
    contract.validate_rows(TASK_SPEC, selected, required_phases=(split,))
    return selected, _canonical_sha(selected)


if __name__ == "__main__":
    authority, digest = build_authority()
    print(json.dumps({
        "schema": SCHEMA, "task_id": TASK_ID, "rows": len(authority),
        "groups": len(authority) // 4, "authority_sha256": digest,
        "split_records_sha256": {
            split: split_rows(authority, split)[1] for split in contract.PHASES
        },
    }, sort_keys=True))
