#!/usr/bin/env python3
"""CPU-only semantic tests for the positional-list battery adapter."""

from collections import defaultdict

import pytest

import circuit_battery_integration_contract as contract
import circuit_battery_task17 as task17


def authority(groups: int = 16):
    return task17.build_authority(groups_per_phase=groups, seed=1701)[0]


def test_complete_split_disjoint_panels_and_shared_contract() -> None:
    rows, digest = task17.build_authority(groups_per_phase=24)
    assert len(rows) == 4 * 24 * 4
    assert digest == contract.validate_rows(task17.TASK_SPEC, rows)
    panels = defaultdict(list)
    for row in rows:
        panels[(row["split"], row["group_id"])].append(row["transform_id"])
    assert all(sorted(transforms) == sorted(contract.TRANSFORMS) for transforms in panels.values())


def test_deterministic_content_hash() -> None:
    first_rows, first_hash = task17.build_authority(groups_per_phase=32, seed=99)
    second_rows, second_hash = task17.build_authority(groups_per_phase=32, seed=99)
    assert first_rows == second_rows
    assert first_hash == second_hash
    assert task17.build_authority(groups_per_phase=32, seed=100)[1] != first_hash


def test_joint_tokenization_and_position_alignment_are_exhaustive() -> None:
    for row in authority(32):
        for side in ("base", "donor"):
            prompt = row[f"{side}_text"]
            answer = row[f"{side}_answer"]
            prompt_ids = task17.ENCODING.encode(prompt)
            complete = task17.ENCODING.encode(prompt + answer)
            assert complete[:len(prompt_ids)] == prompt_ids
            assert len(complete[len(prompt_ids):]) == 1
            assert complete[-1] == row[f"{side}_answer_id"]
        assert len(row["base_ids"]) == len(row["donor_ids"])


def test_each_transform_changes_only_its_registered_variable() -> None:
    for row in authority(32):
        base, donor = row["base_values"], row["donor_values"]
        q, dq = row["base_query"], row["donor_query"]
        changed_positions = [i for i, pair in enumerate(zip(base, donor)) if pair[0] != pair[1]]
        if row["transform_id"] == "A1":
            assert base == donor and q != dq
            assert row["base_answer"] != row["donor_answer"]
        elif row["transform_id"] == "A2":
            assert q == dq and len(changed_positions) == 2
            assert donor[q] in base and row["base_answer"] != row["donor_answer"]
        elif row["transform_id"] == "P":
            assert q == dq and len(changed_positions) == 1 and changed_positions[0] != q
            assert row["base_answer"] == row["donor_answer"]
        else:
            assert base == donor and q != dq and base[q] == base[dq]
            assert len(set(base)) >= 2
            assert row["base_answer"] == row["donor_answer"]


def test_phase_payloads_and_prompts_are_disjoint_and_ood_is_longer() -> None:
    rows = authority(24)
    phase_values = defaultdict(set)
    phase_prompts = defaultdict(set)
    phase_lengths = defaultdict(set)
    for row in rows:
        phase_values[row["split"]].update(row["base_values"] + row["donor_values"])
        phase_prompts[row["split"]].update((row["base_text"], row["donor_text"]))
        phase_lengths[row["split"]].add(row["list_length"])
    for i, phase in enumerate(contract.PHASES):
        for other in contract.PHASES[i + 1:]:
            assert not phase_values[phase] & phase_values[other]
            assert not phase_prompts[phase] & phase_prompts[other]
    assert phase_lengths == {"FIT": {4}, "SELECT": {4}, "TEST": {4}, "OOD": {6}}


def test_foils_are_nondegenerate() -> None:
    for row in authority(32):
        assert row["foil_answers"]
        assert row["base_answer"] not in row["foil_answers"]
        assert all(len(task17.ENCODING.encode(foil)) == 1 for foil in row["foil_answers"])


def test_static_gate_rejects_cross_phase_payload_leakage() -> None:
    rows = authority(2)
    fit = next(row for row in rows if row["split"] == "FIT")
    select = next(row for row in rows if row["split"] == "SELECT")
    select["base_values"][0] = fit["base_values"][0]
    with pytest.raises(contract.BatteryContractError, match="payload vocabulary leakage"):
        task17.validate_authority(rows)


def test_invalid_group_count_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        task17.build_authority(groups_per_phase=0)
