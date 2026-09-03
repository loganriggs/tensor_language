"""Focused CPU tests for the fresh rung-578 induction counterfactual rows."""

from __future__ import annotations

import ast
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


OPS = Path(__file__).parent
PATH = OPS / "induction_selector_payload_three_source_rows_rung578.py"
SPEC = importlib.util.spec_from_file_location("induction_rows_r578", PATH)
ROWS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ROWS
SPEC.loader.exec_module(ROWS)
OLD_R552 = OPS.parent / "induction_selector_payload_factorial_rows_rung552.json"


def _saved():
    return json.loads(ROWS.OUT.read_text()), json.loads(ROWS.RECEIPT.read_text())


def _diff_positions(left, right):
    if len(left) != len(right):
        return ()
    return tuple(
        index for index, (a, b) in enumerate(zip(left, right, strict=True)) if a != b
    )


def test_saved_rows_receipt_and_preregistration_hashes_are_exact():
    payload, receipt = _saved()
    encoded = ROWS.OUT.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == receipt["rows_sha256"]
    assert hashlib.sha256(ROWS.PREREG.read_bytes()).hexdigest() == receipt[
        "preregistration_sha256"
    ]
    assert receipt["prior_r552_rows_sha256"] == (
        "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460"
    )
    assert payload["model_loaded"] is False
    assert payload["model_forwards"] == payload["model_backwards"] == 0
    assert payload["outcomes_opened"] == []


def test_r552_brittleness_diagnosis_is_exact_for_every_old_control_row():
    old = json.loads(OLD_R552.read_text())
    controls = [
        row for row in old["rows"] if row["family_id"] == "irrelevant_source_edit"
    ]
    assert len(controls) == 180
    by_cell = collections.Counter(
        (row["split"], row["family_variant"]) for row in controls
    )
    assert by_cell[("SELECT", "s1p0")] == 9
    for row in controls:
        selector = row["base_selector"]
        structure = row["base_structure"]
        contrast_slot = 1 - selector
        changed_source = structure["source_positions"][contrast_slot]
        contrast_payload = structure["payload_positions"][contrast_slot]
        assert _diff_positions(row["base_ids"], row["donor_ids"]) == (changed_source,)
        assert contrast_payload == changed_source + 1
        assert row["base_ids"][contrast_payload] == structure["payload_ids"][contrast_slot]
        assert structure["payload_ids"][contrast_slot] != row["base_answer_id"]


def test_builder_is_deterministic_and_independent_validator_accepts_saved_rows():
    saved, receipt = _saved()
    rebuilt = ROWS.build_dataset()
    assert rebuilt == saved
    validation = ROWS.validate_dataset(saved)
    for key, value in validation.items():
        assert receipt[key] == value


def test_exact_census_and_complete_family_coverage_per_group():
    payload, receipt = _saved()
    assert receipt["group_count"] == 180
    assert receipt["row_count"] == 5400
    assert receipt["factorial_condition_count"] == 720
    assert receipt["unique_prompt_sequence_count"] == 5040
    assert receipt["split_group_counts"] == {
        "FIT": 72,
        "SELECT": 36,
        "FINAL_TEST": 36,
        "OOD": 36,
    }
    assert receipt["family_row_counts"] == {
        "two_valid_sources_selector_swap": 360,
        "payload_swap_match_preserved": 360,
        "selector_payload_joint_answer_preserved": 360,
        "match_break_payload_preserved": 720,
        "irrelevant_source_edit": 720,
        "irrelevant_payload_edit": 720,
        "contrast_target_source_edit": 720,
        "copy_relation_preserved_nuisance_change": 1440,
    }
    by_group = collections.defaultdict(collections.Counter)
    for row in payload["rows"]:
        by_group[row["group_id"]][row["family_id"]] += 1
    assert all(counts == ROWS.EXPECTED_PER_GROUP for counts in by_group.values())


def test_factorial_has_exact_selector_payload_interaction_and_pair_relations():
    payload, _ = _saved()
    for group in payload["groups"]:
        variables = group["variable_token_ids"]
        cells = group["factorial_conditions"]
        assert [cells[name]["answer_id"] for name in ("s0p0", "s1p0", "s0p1", "s1p1")] == [
            variables["B"], variables["D"], variables["D"], variables["B"]
        ]
        # The ideal B-minus-D signed coordinate is +,-,-,+, whose
        # selector-by-payload interaction contrast is exactly +1.
        z00, z10, z01, z11 = 1.0, -1.0, -1.0, 1.0
        assert (z00 - z10 - z01 + z11) / 4 == 1.0
        for cell in cells.values():
            q = cell["query_position"]
            earlier_matches = [
                index for index, token in enumerate(cell["ids"][:q])
                if token == cell["query_id"]
            ]
            assert earlier_matches == [cell["source_positions"][cell["selector"]]]

    families = collections.defaultdict(list)
    for row in payload["rows"]:
        families[row["family_id"]].append(row)
    assert all(row["answer_changes"] for row in families["two_valid_sources_selector_swap"])
    assert all(row["answer_changes"] for row in families["payload_swap_match_preserved"])
    assert all(
        not row["answer_changes"]
        and row["base_selector"] != row["donor_selector"]
        and row["base_payload_assignment"] != row["donor_payload_assignment"]
        for row in families["selector_payload_joint_answer_preserved"]
    )


def test_neutral_source_control_is_outside_both_endpoint_payloads():
    payload, _ = _saved()
    rows = [row for row in payload["rows"] if row["family_id"] == "irrelevant_source_edit"]
    assert len(rows) == 720
    for row in rows:
        changed = _diff_positions(row["base_ids"], row["donor_ids"])
        assert changed == (row["base_structure"]["N_source_position"],)
        position = changed[0]
        assert position not in row["base_structure"]["source_positions"]
        assert row["base_ids"][position + 1] == row["base_structure"]["neutral_payload_id"]
        assert row["base_structure"]["neutral_payload_id"] not in {
            row["base_answer_id"], row["base_other_answer_id"]
        }
        assert row["base_answer_id"] == row["donor_answer_id"]
        assert row["base_other_answer_id"] == row["donor_other_answer_id"]


def test_old_brittle_edit_is_retained_under_honest_contrast_scope():
    payload, _ = _saved()
    rows = [
        row for row in payload["rows"]
        if row["family_id"] == "contrast_target_source_edit"
    ]
    assert len(rows) == 720
    for row in rows:
        changed = _diff_positions(row["base_ids"], row["donor_ids"])
        selector = row["base_selector"]
        contrast_position = row["base_structure"]["source_positions"][1 - selector]
        assert changed == (contrast_position,)
        # This is why it cannot be called irrelevant: the changed source is
        # immediately before the competing payload used by the endpoint.
        assert row["base_ids"][contrast_position + 1] == row["base_other_answer_id"]
        assert row["role"] == "competition_diagnostic"


def test_edits_are_live_and_source_roles_are_position_balanced():
    payload, _ = _saved()
    expected_roles = {
        "match_break_payload_preserved": "selected_target_source",
        "irrelevant_source_edit": "neutral_source",
        "irrelevant_payload_edit": "neutral_payload",
        "contrast_target_source_edit": "contrast_target_source",
    }
    for row in payload["rows"]:
        if row["family_id"] in expected_roles:
            assert _diff_positions(row["base_ids"], row["donor_ids"]) == (
                row["edit_position"],
            )
            assert row["edit_role"] == expected_roles[row["family_id"]]
            assert not row["answer_changes"]
    by_split = collections.defaultdict(collections.Counter)
    for group in payload["groups"]:
        by_split[group["split"]][tuple(group["pair_order"])] += 1
    for split, counts in by_split.items():
        assert len(counts) == 3
        assert max(counts.values()) == min(counts.values()), split


def test_groups_tokens_sequences_and_prompt_answers_do_not_cross_splits():
    payload, receipt = _saved()
    group_split = {group["group_id"]: group["split"] for group in payload["groups"]}
    sampled_owner = {}
    for group in payload["groups"]:
        for token in group["sampled_token_ids"]:
            assert token not in sampled_owner
            sampled_owner[token] = group["group_id"]
    sequence_owner = collections.defaultdict(set)
    answer_owner = collections.defaultdict(set)
    for row in payload["rows"]:
        assert row["split"] == group_split[row["group_id"]]
        for side in ("base", "donor"):
            ids = tuple(row[f"{side}_ids"])
            sequence_owner[ids].add(row["group_id"])
            answer_owner[(ids, row[f"{side}_answer_id"])].add(row["group_id"])
    assert all(len(owner) == 1 for owner in sequence_owner.values())
    assert all(len(owner) == 1 for owner in answer_owner.values())
    assert receipt["sampled_tokens_never_cross_groups"]
    assert receipt["prompt_sequences_never_cross_groups"]
    assert receipt["prompt_answer_pairs_never_cross_groups"]


def test_builder_has_no_model_torch_or_cuda_dependency():
    tree = ast.parse(PATH.read_text())
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported_roots <= {
        "__future__", "collections", "hashlib", "json", "random", "re",
        "pathlib", "typing", "tiktoken",
    }
    source = PATH.read_text().lower()
    assert "cuda" not in source
    assert "load_model" not in source
