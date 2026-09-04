#!/usr/bin/env python3
"""CPU-only semantic and adversarial tests for task 21 verbatim copy."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import circuit_battery_integration_contract as contract
import circuit_battery_task21 as task21


OPS = Path(__file__).resolve().parent
FULL_SHA = "191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b"
SPLIT_SHA = {
    "FIT": "c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8",
    "SELECT": "c437ebcf8fa4c00e43be26063ee985dacd767e76c41bbf0263ef9bde52638139",
    "TEST": "d780a7e0993422ed0d52aafacb42c7eb3433503d1b01bf1197bffcdd8b8c6d45",
    "OOD": "2ee14e4547291888608f484c43d4b656f65bc5e709625cafbc5cac4de9ab640b",
}
FILE_SHA = {
    "FIT": "69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94",
    "SELECT": "151e50755c9570cf411e614111fe9c5857d5ea13aab7fb7e53d6ce493b8a1f67",
    "TEST": "dc3340c18d7c2efaa460fecf1e0134bc07532f939d1b424016977ecab810c155",
    "OOD": "bf338c34ff0ffe17a56c6c8cb8f3e7c74fcf4c0549c4f9933065bbe8cca16c38",
}


def authority():
    return task21.build_authority()


def _resign(row: dict[str, object]) -> None:
    identity = {
        key: row[key] for key in (
            "schema", "task_id", "split", "group_id", "transform_id", "prefix",
            "base_tokens", "donor_tokens", "base_repeat_start", "donor_repeat_start",
            "base_repeat_count", "donor_repeat_count",
        )
    }
    row["row_id"] = task21._canonical_sha(identity)


def test_exact_authority_panels_and_frozen_digest() -> None:
    rows, digest = authority()
    assert task21.TASK_ID == "verbatim_repeat.copy"
    assert len(rows) == 336 and digest == FULL_SHA
    assert digest == contract.validate_rows(task21.TASK_SPEC, rows)
    panels = defaultdict(list)
    for row in rows:
        panels[(row["split"], row["group_id"])].append(row["transform_id"])
    assert len(panels) == 84
    assert all(sorted(value) == sorted(contract.TRANSFORMS) for value in panels.values())


def test_deterministic_across_python_hash_seeds() -> None:
    command = [sys.executable, "-c", (
        "import circuit_battery_task21 as t; print(t.build_authority()[1])"
    )]
    for hash_seed in ("0", "1", "999"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(OPS))
        assert subprocess.check_output(command, env=env, text=True).strip() == FULL_SHA
    assert task21.build_authority(seed=61819)[1] != FULL_SHA


def test_saved_split_wrappers_are_exact_and_regenerate() -> None:
    rows, full = authority()
    for split in contract.PHASES:
        path = OPS / f"circuit_battery_task21_copy_{split.lower()}_authority.json"
        payload = path.read_bytes()
        value = json.loads(payload)
        selected, digest = task21.split_rows(rows, split)
        assert hashlib.sha256(payload).hexdigest() == FILE_SHA[split]
        assert set(value) == {
            "schema", "task_id", "split", "task21_authority_sha256",
            "split_records_sha256", "groups", "rows",
        }
        assert value["schema"] == "circuit_battery_task21_split_authority_v1"
        assert value["task_id"] == task21.TASK_ID
        assert value["split"] == split
        assert value["task21_authority_sha256"] == full == FULL_SHA
        assert value["split_records_sha256"] == digest == SPLIT_SHA[split]
        assert value["groups"] == 21 and value["rows"] == selected
        assert {row["split"] for row in value["rows"]} == {split}


def test_roles_are_exactly_balanced_and_phases_disjoint() -> None:
    rows, _ = authority()
    values = defaultdict(set)
    prompts = defaultdict(set)
    panels = defaultdict(dict)
    for row in rows:
        values[row["split"]].update(row["base_tokens"] + row["donor_tokens"])
        prompts[row["split"]].update((row["base_text"], row["donor_text"]))
        panels[(row["split"], row["group_id"])][row["transform_id"]] = row
    for phase in contract.PHASES:
        counts = {role: Counter() for role in ("target", "alternative", "novel")}
        filler_count = 3 if phase == "OOD" else 2
        counts.update({f"filler_{index}": Counter() for index in range(filler_count)})
        for (split, _), panel in panels.items():
            if split != phase:
                continue
            a1, p = panel["A1"], panel["P"]
            counts["target"][a1["base_tokens"][-1]] += 1
            counts["alternative"][a1["donor_tokens"][-1]] += 1
            counts["novel"][p["donor_tokens"][0]] += 1
            for index, token in enumerate(a1["base_tokens"][:filler_count]):
                counts[f"filler_{index}"][token] += 1
        assert all(len(count) == 21 and set(count.values()) == {1} for count in counts.values())
    for index, phase in enumerate(contract.PHASES):
        for other in contract.PHASES[index + 1:]:
            assert not values[phase] & values[other]
            assert not prompts[phase] & prompts[other]


def test_joint_tokenization_positions_and_exact_transform_edits() -> None:
    rows, _ = authority()
    for row in rows:
        assert len(row["base_ids"]) == len(row["donor_ids"]) == (13 if row["split"] == "OOD" else 8)
        for side in ("base", "donor"):
            prompt, answer = row[f"{side}_text"], row[f"{side}_answer"]
            prompt_ids = task21.ENCODING.encode(prompt)
            complete = task21.ENCODING.encode(prompt + answer)
            assert prompt_ids == row[f"{side}_ids"]
            assert complete[:-1] == prompt_ids and complete[-1] == row[f"{side}_answer_id"]
            assert answer == " " + row[f"{side}_tokens"][-1]
        base, donor = row["base_tokens"], row["donor_tokens"]
        start, count = row["base_repeat_start"], row["base_repeat_count"]
        if row["transform_id"] == "A1":
            assert donor[:start] == base[:start] and len(set(donor[start:])) == 1
            assert row["base_answer"] != row["donor_answer"]
        elif row["transform_id"] == "A2":
            assert donor[:start + count - 2] == base[:start + count - 2]
            assert donor[-2:] == [donor[-1], donor[-1]] and donor[-1] != base[-1]
        elif row["transform_id"] == "P":
            assert [i for i, pair in enumerate(zip(base, donor)) if pair[0] != pair[1]] == [0]
            assert row["base_answer"] == row["donor_answer"]
        else:
            assert donor[start - 1] == base[-1]
            assert row["donor_repeat_count"] == count + 1
            assert row["base_answer"] == row["donor_answer"]


@pytest.mark.parametrize("attack", ["coherent_text", "effect", "words", "ood_shape", "schema"])
def test_semantic_validator_rejects_resigned_or_unbound_mutations(attack: str) -> None:
    rows, _ = authority()
    if attack == "ood_shape":
        row = next(item for item in rows if item["split"] == "OOD")
        row["prefix"] = "Repeat exactly:"
        for side in ("base", "donor"):
            row[f"{side}_text"] = task21._prompt(row["prefix"], row[f"{side}_tokens"])
            ids, suffix = task21._joint_encoding(row[f"{side}_text"], row[f"{side}_answer"])
            row[f"{side}_ids"] = ids
            row[f"{side}_answer_id"] = suffix[0]
        _resign(row)
    else:
        row = rows[0]
        if attack == "coherent_text":
            row["base_text"] = row["donor_text"]
            row["base_ids"] = list(row["donor_ids"])
        elif attack == "effect":
            row["expected_effect"] = "invariant"
        elif attack == "words":
            row["sequence_words"] = 999
        else:
            for candidate in rows:
                candidate["schema"] = "wrong_schema"
                _resign(candidate)
    with pytest.raises(contract.BatteryContractError):
        task21.validate_authority(rows)


def test_resigned_structural_corruption_and_invalid_group_count_fail() -> None:
    rows, _ = authority()
    a2 = next(row for row in rows if row["split"] == "FIT" and row["transform_id"] == "A2")
    a2["donor_tokens"] = list(a2["base_tokens"])
    target, start, count = task21._trailing_run(a2["donor_tokens"])
    a2["donor_repeat_start"], a2["donor_repeat_count"] = start, count
    _resign(a2)
    with pytest.raises(contract.BatteryContractError):
        task21.validate_authority(rows)
    with pytest.raises(ValueError, match="exactly 21"):
        task21.build_authority(groups_per_phase=20)
