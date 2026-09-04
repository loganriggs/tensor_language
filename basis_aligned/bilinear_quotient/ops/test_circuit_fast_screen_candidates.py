#!/usr/bin/env python3
"""CPU tests for the small reusable circuit-screen candidate bank."""

from __future__ import annotations

from collections import Counter, defaultdict
import os
from pathlib import Path
import subprocess
import sys

import pytest

import circuit_fast_screen_candidates as bank


def rows() -> list[dict]:
    return bank.build_rows(bank.TASK_ID)


def panels(materialized: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in materialized:
        result[row["group_id"]].append(row)
    return result


def test_registry_and_exact_battery_surface() -> None:
    assert tuple(bank.CANDIDATES) == (bank.TASK_ID,)
    spec = bank.CANDIDATES[bank.TASK_ID]
    assert spec.task is bank.TASK_SPEC
    assert spec.answer_vocabulary == (".", "?")
    required = {
        "schema", "task_id", "split", "group_id", "row_id", "transform_id",
        "family_id", "family", "role", "answer_changes", "expected_effect",
        "changed_variable", "capability_cell_id", "construction_id", "direction_id",
        "base_text", "donor_text", "base_ids", "donor_ids", "base_answer",
        "donor_answer", "base_foil", "donor_foil", "base_answer_id",
        "donor_answer_id", "base_foil_id", "donor_foil_id",
        "base_prediction_position", "donor_prediction_position",
        "base_semantic_position", "donor_semantic_position",
        "semantic_details", "construction_checks",
    }
    for row in rows():
        assert required <= set(row)
        assert row["split"] == "FIT"
        assert row["family"] == row["transform_id"]


def test_complete_linked_groups_and_unique_ids() -> None:
    materialized = rows()
    assert len(materialized) == 32 * 4
    grouped = panels(materialized)
    assert len(grouped) == 32
    for panel in grouped.values():
        assert [row["transform_id"] for row in panel] == list(bank.TRANSFORMS)
        assert len({row["seed"] for row in panel}) == 1
        assert len({(row["adjective"], row["object_name"]) for row in panel}) == 1
    assert len({row["row_id"] for row in materialized}) == len(materialized)
    assert bank.validate_rows(materialized) == bank.authority_sha256()


def test_answer_semantics_foils_and_capability_cells() -> None:
    materialized = rows()
    assert all(row["capability_cell_id"] ==
               f"{row['transform_id']}/{row['construction_id']}/{row['direction_id']}"
               for row in materialized)
    for row in materialized:
        assert {row["base_answer"], row["base_foil"]} == set(bank.PUNCTUATION)
        assert {row["donor_answer"], row["donor_foil"]} == set(bank.PUNCTUATION)
        assert {row["base_answer_id"], row["base_foil_id"]} == {
            row["donor_answer_id"], row["donor_foil_id"]
        }
        assert row["answer_changes"] is (row["transform_id"] != "P")
    # Every family+construction+direction cell is explicit rather than pooled.
    observed = Counter(row["capability_cell_id"] for row in materialized)
    assert all(count > 0 for count in observed.values())
    assert len(observed) == 8  # 2 A1 + 2 A2 + 2 P + 2 C


def test_ordered_directions_are_exactly_balanced() -> None:
    materialized = rows()
    directions = Counter(
        (row["transform_id"], row["direction_id"]) for row in materialized
    )
    for transform in ("A1", "A2"):
        assert directions[(transform, "declarative_to_interrogative")] == 16
        assert directions[(transform, "interrogative_to_declarative")] == 16
    assert directions[("C", "period_to_question")] == 16
    assert directions[("C", "question_to_period")] == 16
    assert directions[("P", "primary_to_alternative")] == 16
    assert directions[("P", "alternative_to_primary")] == 16


def test_matched_suffix_and_endpoint_hold_for_target_and_copy_control() -> None:
    for panel in panels(rows()).values():
        suffixes = {row["semantic_details"]["matched_final_suffix"] for row in panel}
        assert len(suffixes) == 1
        suffix = next(iter(suffixes))
        endpoint_ids = set()
        endpoint_words = set()
        for row in panel:
            assert row["base_text"].endswith(suffix)
            assert row["donor_text"].endswith(suffix)
            assert row["base_ids"][-1] == row["donor_ids"][-1]
            endpoint_ids.update((row["base_ids"][-1], row["donor_ids"][-1]))
            endpoint_words.update((row["base_text"].split()[-1], row["donor_text"].split()[-1]))
        assert len(endpoint_ids) == 1
        assert len(endpoint_words) == 1


def test_no_prompt_opener_is_a_universal_terminal_cue() -> None:
    materialized = rows()
    # Reporting-frame questions and declarations share the exact first two words.
    reporting = [row for row in materialized if row["transform_id"] == "A1"]
    for row in reporting:
        assert row["base_text"].split()[:2] == row["donor_text"].split()[:2]
    # Across A1 and A2, interrogatives occur with both `The` and `Does`, and
    # prompts beginning `The` have both legal terminals.  Thus no one opener is
    # a universal answer label across the target families.
    target_sides = []
    for row in materialized:
        if row["transform_id"] in ("A1", "A2"):
            target_sides.extend(((row["base_text"], row["base_answer"]),
                                 (row["donor_text"], row["donor_answer"])))
    question_openers = {text.split()[0] for text, answer in target_sides if answer == "?"}
    the_answers = {answer for text, answer in target_sides if text.split()[0] == "The"}
    assert question_openers == {"The", "Does"}
    assert the_answers == {".", "?"}


def test_copy_control_is_explicit_and_has_the_same_endpoint() -> None:
    for row in rows():
        if row["transform_id"] != "C":
            continue
        assert f"mark {row['base_answer']} after" in row["base_text"]
        assert f"mark {row['donor_answer']} after" in row["donor_text"]
        assert row["semantic_details"]["copy_control"] is True


def test_joint_tokenization_roundtrip_and_final_semantic_position() -> None:
    for row in rows():
        assert all(row["construction_checks"].values())
        for side in ("base", "donor"):
            text = row[f"{side}_text"]
            ids = row[f"{side}_ids"]
            assert bank.ENCODING.decode(ids) == text
            assert text[-1] not in " \t\r\n\"'"
            assert row[f"{side}_prediction_position"] == len(ids) - 1
            assert row[f"{side}_semantic_position"] == len(ids) - 1
            for role in ("answer", "foil"):
                punctuation = row[f"{side}_{role}"]
                token_id = row[f"{side}_{role}_id"]
                assert bank.ENCODING.encode(punctuation) == [token_id]
                assert bank.ENCODING.encode(text + punctuation) == ids + [token_id]


def test_default_authority_hash_is_frozen_and_hashseed_independent() -> None:
    expected = "d0da3cda58fa77e93f982932f9a890af8b77d9e0162f5144c2cb9288004a81ab"
    assert bank.authority_sha256() == expected
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import circuit_fast_screen_candidates as b; print(b.authority_sha256())"
        % str(Path(bank.__file__).parent)
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": str(value)},
        ).stdout.strip()
        for value in (0, 1, 999)
    }
    assert outputs == {expected}


def test_validator_rejects_coherent_semantic_and_identity_mutations() -> None:
    materialized = rows()
    changed = [dict(row) for row in materialized]
    changed[0] = dict(changed[0], base_answer="?", base_foil=".")
    # Even a coherently rehashed row must differ from the deterministic authority.
    identity_keys = {
        key for key in changed[0]
        if key not in {
            "row_id", "family_id", "family", "role", "answer_changes",
            "expected_effect", "changed_variable", "base_ids", "donor_ids",
            "base_answer_id", "donor_answer_id", "base_foil_id", "donor_foil_id",
            "base_semantic_position", "donor_semantic_position", "semantic_details",
            "construction_checks",
        }
    }
    changed[0]["row_id"] = bank.canonical_sha256(
        {key: changed[0][key] for key in identity_keys}
    )
    with pytest.raises(bank.CandidateBankError, match="deterministic semantic authority"):
        bank.validate_rows(changed)


@pytest.mark.parametrize("groups", [True, 1, 3, 34])
def test_invalid_group_requests_fail_closed(groups: object) -> None:
    with pytest.raises(bank.CandidateBankError):
        bank.build_rows(bank.TASK_ID, groups=groups)  # type: ignore[arg-type]


def test_unknown_candidate_and_bad_seed_fail_closed() -> None:
    with pytest.raises(KeyError):
        bank.build_rows("unknown.task")
    with pytest.raises(bank.CandidateBankError):
        bank.build_rows(bank.TASK_ID, seed=True)  # type: ignore[arg-type]
