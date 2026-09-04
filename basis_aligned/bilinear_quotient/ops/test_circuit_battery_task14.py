#!/usr/bin/env python3
"""CPU semantic and adversarial tests for task14 subject–verb agreement."""

from __future__ import annotations

from collections import Counter, defaultdict
import os
from pathlib import Path
import subprocess
import sys

import pytest

import circuit_battery_integration_contract as contract
import circuit_battery_task14 as task14


OPS = Path(__file__).resolve().parent
FULL_SHA = "6432f647eb15bae46cfcc22b922f71958edb9d1b092e7bf3e63601df9084c47a"
SPLIT_SHA = {
    "FIT": "02caa03dc84c31afab2f4dd0d175a8d119ce7322e16e63427807bbe2a4df1d35",
    "SELECT": "4380825410da4e9b40bca432edf28687889080fae898a21945f0ad99d30c8d41",
    "TEST": "da98a4cd55fba3106f68c420168ebf6b7556b14594764da65bcca8ef2a787c54",
    "OOD": "6a30cd0e155c8211e8ad4b310a0b4cb39a7313893ba9cc717142eac049462cc6",
}

IDENTITY_FIELDS = (
    "schema", "task_id", "split", "seed", "group_number", "group_id", "transform_id",
    "head_pair", "attractor_pair", "second_head_pair", "second_attractor_pair",
    "surface_attractor_pair", "base_head_plural", "donor_head_plural",
    "base_attractor_plural", "donor_attractor_plural", "base_second_attractor_plural",
    "donor_second_attractor_plural", "base_template_id", "donor_template_id",
)


def authority():
    return task14.build_authority()


def _resign(row: dict[str, object]) -> None:
    row["row_id"] = task14._canonical_sha({key: row[key] for key in IDENTITY_FIELDS})


def test_exact_authority_shape_panels_and_digest() -> None:
    rows, digest = authority()
    assert task14.TASK_ID == "subject_verb.number_agreement"
    assert len(rows) == 512 and digest == FULL_SHA
    assert digest == contract.validate_rows(task14.TASK_SPEC, rows)
    panels = defaultdict(set)
    for row in rows:
        panels[(row["split"], row["group_id"])].add(row["transform_id"])
    assert len(panels) == 128
    assert all(value == set(contract.TRANSFORMS) for value in panels.values())


def test_deterministic_across_hash_seeds_and_seed_is_live() -> None:
    command = [sys.executable, "-c", (
        "import circuit_battery_task14 as t; print(t.build_authority()[1])"
    )]
    for hash_seed in ("0", "1", "999"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(OPS))
        assert subprocess.check_output(command, env=environment, text=True).strip() == FULL_SHA
    assert task14.build_authority(seed=71419)[1] != FULL_SHA


def test_split_hashes_phase_isolation_and_exact_templates() -> None:
    rows, _ = authority()
    prompts = defaultdict(set)
    nouns = defaultdict(set)
    templates = defaultdict(set)
    for row in rows:
        split = row["split"]
        prompts[split].update((row["base_text"], row["donor_text"]))
        templates[split].update((row["base_template_id"], row["donor_template_id"]))
        for role in (
            "head_pair", "attractor_pair", "second_head_pair", "second_attractor_pair",
            "surface_attractor_pair",
        ):
            nouns[split].update(row[role])
    for split in contract.PHASES:
        selected, digest = task14.split_rows(rows, split)
        assert len(selected) == 128 and digest == SPLIT_SHA[split]
        assert templates[split] == {
            template for pair in task14._PHASE_TEMPLATES[split].values() for template in pair
        }
    for index, split in enumerate(contract.PHASES):
        for other in contract.PHASES[index + 1:]:
            assert not prompts[split] & prompts[other]
            assert not nouns[split] & nouns[other]
            assert not templates[split] & templates[other]


def test_noun_number_answer_and_foil_roles_are_balanced() -> None:
    rows, _ = authority()
    panels = defaultdict(dict)
    for row in rows:
        panels[(row["split"], row["group_id"])][row["transform_id"]] = row
    for phase in contract.PHASES:
        phase_panels = [panel for (split, _), panel in panels.items() if split == phase]
        role_counts = {role: Counter() for role in (
            "head_pair", "attractor_pair", "second_head_pair", "second_attractor_pair",
            "surface_attractor_pair",
        )}
        number_cells = Counter()
        for panel in phase_panels:
            a1 = panel["A1"]
            for role in role_counts:
                role_counts[role][tuple(a1[role])] += 1
            number_cells[(a1["base_head_plural"], a1["base_attractor_plural"])] += 1
        assert all(len(counts) == 16 and set(counts.values()) == {2}
                   for counts in role_counts.values())
        assert len(number_cells) == 4 and set(number_cells.values()) == {8}
        for transform in ("A1", "A2", "P"):
            for side in ("base", "donor"):
                answers = Counter(panel[transform][f"{side}_answer"] for panel in phase_panels)
                foils = Counter(panel[transform][f"{side}_foil"] for panel in phase_panels)
                assert answers == {" is": 16, " are": 16}
                assert foils == {" is": 16, " are": 16}
        for side in ("base", "donor"):
            assert Counter(panel["C"][f"{side}_answer"] for panel in phase_panels) == {" are": 32}
            assert Counter(panel["C"][f"{side}_foil"] for panel in phase_panels) == {" is": 32}


def test_joint_tokenization_equal_positions_and_exact_semantic_edits() -> None:
    rows, _ = authority()
    for row in rows:
        assert len(row["base_ids"]) == len(row["donor_ids"])
        assert row["base_prediction_position"] == row["donor_prediction_position"]
        assert row["base_head_positions"] == row["donor_head_positions"]
        assert row["base_attractor_positions"] == row["donor_attractor_positions"]
        assert len(row["intervention_token_positions"]) == 1
        for side in ("base", "donor"):
            prompt = row[f"{side}_text"]
            answer = row[f"{side}_answer"]
            prompt_ids = task14.ENCODING.encode(prompt)
            complete_ids = task14.ENCODING.encode(prompt + answer)
            assert prompt_ids == row[f"{side}_ids"]
            assert complete_ids == prompt_ids + [row[f"{side}_answer_id"]]
            assert answer != row[f"{side}_foil"]
        changed = row["intervention_token_positions"]
        if row["transform_id"] in ("A1", "A2"):
            assert changed == [row["base_head_positions"][0]]
            assert row["base_head_plural"] is not row["donor_head_plural"]
            assert row["base_attractor_plural"] == row["donor_attractor_plural"]
            assert row["base_answer"] != row["donor_answer"]
        elif row["transform_id"] == "P":
            assert changed == [row["base_attractor_positions"][-1]]
            assert row["base_answer"] == row["donor_answer"]
            assert row["base_head_plural"] == row["donor_head_plural"]
            assert row["base_attractor_plural"] == row["donor_attractor_plural"]
            assert row["base_second_attractor_plural"] == row["donor_second_attractor_plural"]
            assert row["surface_attractor_pair"] not in (
                row["head_pair"], row["attractor_pair"], row["second_attractor_pair"],
            )
            changed_position = changed[0]
            expected_donor_word = task14._form(
                tuple(row["surface_attractor_pair"]),
                row[
                    "donor_second_attractor_plural"
                    if row["split"] == "OOD" else "donor_attractor_plural"
                ],
            )
            expected_donor_id = task14.ENCODING.encode(" " + expected_donor_word)
            assert expected_donor_id == [row["donor_ids"][changed_position]]
        else:
            assert changed == [row["base_attractor_positions"][-1]]
            assert len(row["base_head_positions"]) == 2
            assert row["base_answer"] == row["donor_answer"] == " are"
            assert row["control_relation"] == "two_singular_conjuncts_require_plural_agreement"
            assert row["base_head_plural"] is row["donor_head_plural"] is False
            if row["split"] == "OOD":
                assert row["base_attractor_plural"] == row["donor_attractor_plural"]
                assert row["base_second_attractor_plural"] is not row["donor_second_attractor_plural"]
            else:
                assert row["base_attractor_plural"] is not row["donor_attractor_plural"]
                assert row["base_second_attractor_plural"] == row["donor_second_attractor_plural"]


def test_ood_moves_the_head_and_adds_a_second_attractor() -> None:
    rows, _ = authority()
    a1 = [row for row in rows if row["split"] == "OOD" and row["transform_id"] == "A1"]
    a2 = [row for row in rows if row["split"] == "OOD" and row["transform_id"] == "A2"]
    assert all(len(row["base_attractor_positions"]) == 2 for row in a1 + a2)
    assert all(row["base_head_positions"][0] > max(row["base_attractor_positions"]) for row in a1)
    assert all(row["base_head_positions"][0] < min(row["base_attractor_positions"]) for row in a2)


@pytest.mark.parametrize(
    "attack", ("schema", "coherent_text_ids", "effect", "position", "c_answer", "ood_template"),
)
def test_semantic_validator_rejects_coherent_or_resigned_mutations(attack: str) -> None:
    rows, _ = authority()
    if attack == "schema":
        for row in rows:
            row["schema"] = "wrong_schema"
            _resign(row)
    elif attack == "ood_template":
        row = next(item for item in rows if item["split"] == "OOD" and item["transform_id"] == "A1")
        row["base_template_id"] = "fit_pp_near"
        row["donor_template_id"] = "fit_pp_near"
        _resign(row)
    else:
        row = next(item for item in rows if item["split"] == "FIT" and item["transform_id"] == "C")
        if attack == "coherent_text_ids":
            row["base_text"] = row["donor_text"]
            row["base_ids"] = list(row["donor_ids"])
        elif attack == "effect":
            row["expected_effect"] = "invariant"
        elif attack == "position":
            row["intervention_token_positions"] = [0]
        else:
            row["base_answer"] = " is"
            _, row["base_answer_id"] = task14._joint_encoding(row["base_text"], " is")
            row["base_foil"] = " are"
    with pytest.raises(contract.BatteryContractError):
        task14.validate_authority(rows)


def test_missing_row_duplicate_panel_and_invalid_group_count_fail() -> None:
    rows, _ = authority()
    with pytest.raises(contract.BatteryContractError):
        task14.validate_authority(rows[:-1])
    rows, _ = authority()
    rows[-1] = dict(rows[0])
    with pytest.raises(contract.BatteryContractError):
        task14.validate_authority(rows)
    with pytest.raises(ValueError, match="exactly 32"):
        task14.build_authority(groups_per_phase=31)


def test_generator_is_cpu_data_only() -> None:
    source = Path(task14.__file__).read_text()
    for forbidden in (
        "import torch", "torch.load", "cuda(", "queue.txt", "enqueue.sh", "_results.json",
        "_evidence/", "run_science(",
    ):
        assert forbidden not in source.lower()
