#!/usr/bin/env python3
"""CPU tests for the pending-quote-state fast-screen candidate."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import circuit_fast_screen_candidate_quote_parity as bank
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_spec as screen
import circuit_prior_art


ROOT = Path(__file__).resolve().parent.parent


def rows() -> list[dict]:
    return bank.build_rows(bank.TASK_ID)


def panels(materialized: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in materialized:
        result[row["group_id"]].append(row)
    return result


def test_registry_and_exact_battery_surface() -> None:
    assert tuple(bank.CANDIDATES) == (bank.TASK_ID,)
    assert bank.CANDIDATES[bank.TASK_ID].task is bank.TASK_SPEC
    assert bank.CANDIDATES[bank.TASK_ID].answer_vocabulary == (".", '"')
    required = {
        "schema", "task_id", "split", "group_id", "row_id", "transform_id",
        "family_id", "family", "role", "answer_changes", "expected_effect",
        "changed_variable", "capability_cell_id", "construction_id", "direction_id",
        "base_text", "donor_text", "base_ids", "donor_ids", "base_answer",
        "donor_answer", "base_foil", "donor_foil", "base_answer_id",
        "donor_answer_id", "base_foil_id", "donor_foil_id",
        "base_prediction_position", "donor_prediction_position",
        "base_semantic_position", "donor_semantic_position", "semantic_details",
        "construction_checks",
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


def test_target_pairs_change_only_one_quote_and_parity_forces_answer() -> None:
    for row in rows():
        if row["transform_id"] not in ("A1", "A2"):
            continue
        assert row["base_text"].replace('"', "") == row["donor_text"].replace('"', "")
        assert abs(row["base_text"].count('"') - row["donor_text"].count('"')) == 1
        for side in ("base", "donor"):
            text = row[f"{side}_text"]
            answer = row[f"{side}_answer"]
            assert (text.count('"') % 2 == 1) is (answer == '"')
            assert (text + answer).count('"') % 2 == 0
            # The quoted material is an attributive noun phrase, not a complete
            # sentence whose period would conventionally precede the quote.
            assert "classifies this record as" in text
            assert row["semantic_details"]["matched_final_suffix"].startswith("a ")


def test_second_construction_controls_quote_presence_and_tests_parity() -> None:
    a1 = [row for row in rows() if row["transform_id"] == "A1"]
    a2 = [row for row in rows() if row["transform_id"] == "A2"]
    assert {row["construction_id"] for row in a1} == {"single_span"}
    assert {row["construction_id"] for row in a2} == {"balanced_prefix"}
    for row in a2:
        assert 'files "sample", then' in row["base_text"]
        assert 'files "sample", then' in row["donor_text"]
        assert {row["base_text"].count('"'), row["donor_text"].count('"')} == {2, 3}


def test_same_answer_invariance_changes_writer_only() -> None:
    for row in rows():
        if row["transform_id"] != "P":
            continue
        assert row["base_answer"] == row["donor_answer"]
        assert row["base_text"].replace(row["base_text"].split()[1], "WRITER", 1) \
            == row["donor_text"].replace(row["donor_text"].split()[1], "WRITER", 1)
        assert row["base_text"].count('"') == row["donor_text"].count('"')


def test_unrelated_control_is_natural_inch_mark_completion_without_quote_parity() -> None:
    for row in rows():
        if row["transform_id"] != "C":
            continue
        assert '"' not in row["base_text"]
        assert '"' not in row["donor_text"]
        for side in ("base", "donor"):
            text = row[f"{side}_text"]
            answer = row[f"{side}_answer"]
            if answer == ".":
                assert text.startswith("The inventory has ")
                assert "items; its final count is exactly" in text
            else:
                assert answer == '"'
                assert text.startswith("The board is ")
                assert "inches long; in customary notation it measures" in text
            assert text.split()[-1].isdigit()


def test_answers_foils_and_capability_cells() -> None:
    materialized = rows()
    observed = Counter(row["capability_cell_id"] for row in materialized)
    assert len(observed) == 8  # A1x2, A2x2, Px2, Cx2
    assert set(observed.values()) == {16}
    for row in materialized:
        assert row["capability_cell_id"] == (
            f"{row['transform_id']}/{row['construction_id']}/{row['direction_id']}"
        )
        assert {row["base_answer"], row["base_foil"]} == set(bank.ANSWER_VOCABULARY)
        assert {row["donor_answer"], row["donor_foil"]} == set(bank.ANSWER_VOCABULARY)
        assert {row["base_answer_id"], row["base_foil_id"]} == {
            row["donor_answer_id"], row["donor_foil_id"]
        }
        assert row["answer_changes"] is (row["transform_id"] != "P")


def test_ordered_directions_are_exactly_balanced() -> None:
    directions = Counter((row["transform_id"], row["direction_id"]) for row in rows())
    for transform in ("A1", "A2"):
        assert directions[(transform, "outside_to_pending")] == 16
        assert directions[(transform, "pending_to_outside")] == 16
    assert directions[("C", "period_to_quote")] == 16
    assert directions[("C", "quote_to_period")] == 16
    assert directions[("P", "primary_to_alternative")] == 16
    assert directions[("P", "alternative_to_primary")] == 16


def test_matched_suffix_joint_tokenization_and_final_position() -> None:
    for panel in panels(rows()).values():
        target_and_invariance = [
            row for row in panel if row["transform_id"] in ("A1", "A2", "P")
        ]
        suffixes = {
            row["semantic_details"]["matched_final_suffix"]
            for row in target_and_invariance
        }
        assert len(suffixes) == 1
        suffix = next(iter(suffixes))
        target_final_ids = set()
        for row in target_and_invariance:
            assert row["base_text"].endswith(suffix)
            assert row["donor_text"].endswith(suffix)
            assert row["base_ids"][-1] == row["donor_ids"][-1]
            target_final_ids.update((row["base_ids"][-1], row["donor_ids"][-1]))
        assert len(target_final_ids) == 1
        for row in panel:
            assert all(row["construction_checks"].values())
            for side in ("base", "donor"):
                text = row[f"{side}_text"]
                ids = row[f"{side}_ids"]
                assert bank.ENCODING.decode(ids) == text
                assert text[-1] not in " \t\r\n\"'"
                assert row[f"{side}_prediction_position"] == len(ids) - 1
                assert row[f"{side}_semantic_position"] == len(ids) - 1
                for role in ("answer", "foil"):
                    answer = row[f"{side}_{role}"]
                    token_id = row[f"{side}_{role}_id"]
                    assert bank.ENCODING.encode(answer) == [token_id]
                    assert bank.ENCODING.encode(text + answer) == ids + [token_id]


def test_default_authority_hash_is_frozen_and_hashseed_independent() -> None:
    expected = "f3e904042b2a7edfa5dbc0c1bd6cd7484b3c98e9cd5eaf1091e17e1cb2a7ccd1"
    assert bank.authority_sha256() == expected
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import circuit_fast_screen_candidate_quote_parity as b; "
        "print(b.authority_sha256())" % str(Path(bank.__file__).parent)
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": str(value)},
        ).stdout.strip()
        for value in (0, 1, 999)
    }
    assert outputs == {expected}


def test_candidate_compiles_through_reusable_screen_without_special_adapter() -> None:
    materialized = rows()
    spec = screen.CircuitFastScreenSpec(
        experiment_id="fast-screen-quote-parity-pending-close-v1",
        hypothesis=screen.CandidateHypothesis(
            behavior=bank.TASK_ID,
            answer_score=screen.ANSWER_SCORE,
            information_read="whether one ASCII quote remains pending",
            proposed_operation="carry quote-count parity across surface constructions",
            proposed_write="evidence for a closing quote rather than a period",
            candidate_sites=screen.CEILING_SITE_IDS,
            alternative_explanation=(
                "the known closer output or a generic period-versus-quote endpoint direction"
            ),
            circuit_prediction=(
                "one site transfers both pending-quote constructions and spares both controls"
            ),
            opposing_null_prediction=(
                "native capability fails or no common selective site transfers quote state"
            ),
        ),
        task=bank.TASK_SPEC,
        authority_sha256=bank.authority_sha256(),
        expected_fit_rows=len(materialized),
        batch_size=32,
        semantic_position=screen.SemanticPositionSpec(
            role="final input token before period or closing quote",
            recipient_field="base_semantic_position",
            donor_field="donor_semantic_position",
        ),
        fields=screen.AuthorityFieldSpec(),
        bars=kernel.FIXED_BARS,
        declared_max_price=screen.battery.ExactPhasePrice(
            phase="FIT", forward_calls=264, example_evaluations=8448,
            backward_calls=0, model_updates=0, evidence_bytes=67584,
        ),
    )
    receipt = screen.compile_dryrun(spec, materialized)
    assert receipt["authority_sha256"] == bank.authority_sha256()
    assert receipt["active_price"]["forward_calls"] == 228
    assert receipt["max_price"]["forward_calls"] == 264
    assert receipt["model_loaded"] is False
    assert receipt["gpu_accessed"] is False


def test_validator_rejects_semantic_mutation() -> None:
    materialized = rows()
    changed = [dict(row) for row in materialized]
    changed[0] = dict(changed[0], base_answer='"', base_foil=".")
    with pytest.raises(bank.QuoteParityCandidateError, match="deterministic"):
        bank.validate_rows(changed)


@pytest.mark.parametrize("groups", [True, 1, 3, 34])
def test_invalid_group_requests_fail_closed(groups: object) -> None:
    with pytest.raises(bank.QuoteParityCandidateError):
        bank.build_rows(bank.TASK_ID, groups=groups)  # type: ignore[arg-type]


def test_unknown_candidate_and_bad_seed_fail_closed() -> None:
    with pytest.raises(KeyError):
        bank.build_rows("unknown.task")
    with pytest.raises(bank.QuoteParityCandidateError):
        bank.build_rows(bank.TASK_ID, seed=True)  # type: ignore[arg-type]


def test_prior_art_receipt_is_current_and_classified_as_extension() -> None:
    receipt_path = ROOT / "circuits" / "fast_screen_quote_parity_prior_art.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["candidate_id"] == bank.TASK_ID
    assert receipt["relation"] == "extension"
    assert circuit_prior_art.validate_source_files(receipt, ROOT)
