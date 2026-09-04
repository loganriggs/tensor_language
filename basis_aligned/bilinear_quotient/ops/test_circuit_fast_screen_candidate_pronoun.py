#!/usr/bin/env python3
"""CPU-only semantic and authority tests for the pronoun screen candidate."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import circuit_fast_screen_candidate_pronoun as bank
import circuit_prior_art


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "circuits" / "fast_screen_pronoun_prior_art.json"


def rows() -> list[dict]:
    return bank.build_rows(bank.TASK_ID)


def panels(materialized: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in materialized:
        grouped[row["group_id"]].append(row)
    return grouped


def test_exact_surface_and_complete_linked_panels() -> None:
    materialized = rows()
    assert tuple(bank.CANDIDATES) == (bank.TASK_ID,)
    assert bank.CANDIDATES[bank.TASK_ID].answer_vocabulary == (" he", " she")
    assert len(materialized) == 128
    grouped = panels(materialized)
    assert len(grouped) == 32
    for panel in grouped.values():
        assert [row["transform_id"] for row in panel] == list(bank.TRANSFORMS)
        assert len({row["seed"] for row in panel}) == 1
        assert len({(row["woman_label"], row["man_label"], row["object_name"])
                    for row in panel}) == 1
        target_suffixes = {
            row["semantic_details"]["matched_final_suffix"]
            for row in panel if row["transform_id"] != "C"
        }
        assert target_suffixes == {"Answer with he or she. The answer is"}
        assert panel[3]["semantic_details"]["matched_final_suffix"] \
            == "The pronoun shown on the card is"
    assert len({row["row_id"] for row in materialized}) == len(materialized)
    assert bank.validate_rows(materialized) == bank.authority_sha256()


def test_a1_a2_are_answer_changing_antecedent_bindings() -> None:
    for panel in panels(rows()).values():
        for row in panel[:2]:
            assert row["transform_id"] in ("A1", "A2")
            assert row["answer_changes"] is True
            assert row["base_antecedent"] != row["donor_antecedent"]
            assert {row["base_antecedent"], row["donor_antecedent"]} == {
                row["woman_label"], row["man_label"]
            }
            assert row["base_answer"] != row["donor_answer"]
            for side in ("base", "donor"):
                actor = row[f"{side}_antecedent"]
                expected = " she" if actor == row["woman_label"] else " he"
                assert row[f"{side}_answer"] == expected
                assert actor in row[f"{side}_text"]
        # The independently worded constructions encode the same actor switch.
        assert panel[0]["base_antecedent"] == panel[1]["base_antecedent"]
        assert panel[0]["donor_antecedent"] == panel[1]["donor_antecedent"]


def test_p_changes_only_location_and_preserves_actor_and_answer() -> None:
    for row in rows():
        if row["transform_id"] != "P":
            continue
        assert row["answer_changes"] is False
        assert row["base_answer"] == row["donor_answer"]
        assert row["base_antecedent"] == row["donor_antecedent"]
        assert row["primary_location"] != row["alternate_location"]
        # Replacing the two registered location words makes the prompts identical.
        canonical_base = row["base_text"].replace(row["primary_location"], "LOCATION") \
            .replace(row["alternate_location"], "LOCATION")
        canonical_donor = row["donor_text"].replace(row["primary_location"], "LOCATION") \
            .replace(row["alternate_location"], "LOCATION")
        assert canonical_base == canonical_donor
        assert row["semantic_details"]["location_invariance_control"] is True


def test_c_is_unrelated_answer_changing_endpoint_control() -> None:
    for row in rows():
        if row["transform_id"] != "C":
            continue
        assert row["answer_changes"] is True
        assert row["base_antecedent"] is None and row["donor_antecedent"] is None
        assert row["semantic_details"]["antecedent_binding_task"] is False
        assert row["semantic_details"]["copy_control"] is True
        assert row["base_text"].endswith("The pronoun shown on the card is")
        assert row["donor_text"].endswith("The pronoun shown on the card is")
        assert f"pronoun {row['base_answer'].strip()}" in row["base_text"]
        assert f"pronoun {row['donor_answer'].strip()}" in row["donor_text"]


def test_capability_cells_and_ordered_directions_are_balanced() -> None:
    materialized = rows()
    observed = Counter(row["capability_cell_id"] for row in materialized)
    assert len(observed) == 8
    assert set(observed.values()) == {16}
    directions = Counter((row["transform_id"], row["direction_id"])
                         for row in materialized)
    for transform in ("A1", "A2"):
        assert directions[(transform, "female_to_male")] == 16
        assert directions[(transform, "male_to_female")] == 16
    assert directions[("P", "female_location_primary_to_alternate")] == 16
    assert directions[("P", "male_location_alternate_to_primary")] == 16
    assert directions[("C", "she_to_he")] == 16
    assert directions[("C", "he_to_she")] == 16


def test_neutral_label_gender_assignment_is_balanced_within_target_cells() -> None:
    # Neither Person A nor Person B is a stable proxy for woman/man, even after
    # conditioning on construction and base-to-donor answer direction.
    assignments = Counter(
        (row["transform_id"], row["direction_id"], row["woman_label"])
        for row in rows() if row["transform_id"] in ("A1", "A2")
    )
    for transform in ("A1", "A2"):
        for direction in ("female_to_male", "male_to_female"):
            assert assignments[(transform, direction, "Person A")] == 8
            assert assignments[(transform, direction, "Person B")] == 8


def test_exact_joint_tokenization_and_final_position() -> None:
    answer_ids = {bank.ENCODING.encode(item)[0] for item in bank.PRONOUNS}
    assert len(answer_ids) == 2
    for row in rows():
        assert all(row["construction_checks"].values())
        assert {row["base_answer_id"], row["base_foil_id"]} == answer_ids
        assert {row["donor_answer_id"], row["donor_foil_id"]} == answer_ids
        for side in ("base", "donor"):
            text = row[f"{side}_text"]
            ids = row[f"{side}_ids"]
            assert bank.ENCODING.decode(ids) == text
            assert text.endswith(row["semantic_details"]["matched_final_suffix"])
            assert text[-1] not in " \t\r\n\"'"
            assert row[f"{side}_prediction_position"] == len(ids) - 1
            assert row[f"{side}_semantic_position"] == len(ids) - 1
            for role in ("answer", "foil"):
                continuation = row[f"{side}_{role}"]
                token_id = row[f"{side}_{role}_id"]
                assert bank.ENCODING.encode(continuation) == [token_id]
                assert bank.ENCODING.encode(text + continuation) == ids + [token_id]


def test_same_long_suffix_and_final_token_across_each_linked_panel() -> None:
    for panel in panels(rows()).values():
        final_tokens = set()
        for row in panel:
            suffix = row["semantic_details"]["matched_final_suffix"]
            assert row["base_text"].endswith(suffix)
            assert row["donor_text"].endswith(suffix)
            final_tokens.update((row["base_ids"][-1], row["donor_ids"][-1]))
        assert len(final_tokens) == 1


def test_authority_hash_is_frozen_and_hashseed_independent() -> None:
    expected = "a4acf288af74f6e6787f01e06818a55d03174370323d07b6940cf85df964ab5b"
    assert bank.authority_sha256() == expected
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import circuit_fast_screen_candidate_pronoun as b; print(b.authority_sha256())"
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


def test_validator_rejects_semantic_mutation_even_if_row_is_rehashed() -> None:
    materialized = rows()
    changed = [dict(row) for row in materialized]
    changed[0] = dict(changed[0], base_answer=" he", base_foil=" she")
    changed[0]["row_id"] = bank.canonical_sha256({"coherent_but_wrong": changed[0]})
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


def test_prior_art_receipt_is_current_and_classifies_extension() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["candidate_id"] == bank.TASK_ID
    assert receipt["relation"] == "extension"
    digest = circuit_prior_art.validate_source_files(receipt, ROOT)
    assert len(digest) == 64
