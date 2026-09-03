"""CPU-only tests for the R580 native-capability instrument."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


OPS = Path(__file__).parent
PATH = OPS / "induction_selector_payload_native_capability_rung580.py"
SPEC = importlib.util.spec_from_file_location("induction_capability_r580", PATH)
R580 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R580
SPEC.loader.exec_module(R580)


@pytest.fixture(scope="module")
def authority():
    groups, rows, by_group = R580.load_authority()
    specs = R580.collect_sequence_specs(groups, rows)
    return groups, rows, by_group, specs


def _fast_score(authority, monkeypatch):
    groups, rows, _by_group, specs = authority
    monkeypatch.setattr(R580, "BOOTSTRAPS", 40)
    measurements = R580.planted_sequence_measurements(specs, rows)
    raw = R580.build_raw_evidence(groups, rows, measurements)
    return raw, R580.score_raw_evidence(raw), measurements


def test_script_parses_and_binds_all_reviewed_r578_authority():
    tree = ast.parse(PATH.read_text())
    assert tree is not None
    for expected in R580.HASHES.values():
        assert len(expected) == 64
        int(expected, 16)
    assert R580.HASHES[R580.ROWS] == (
        "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6"
    )
    assert R580.HASHES[R580.ROWS_RECEIPT] == (
        "9e4e63ebd98503d6aa5daa27617a20fea595829c5a372f27b1ce4371d7c05b45"
    )
    assert R580.HASHES[R580.PREREG] == (
        "8f80926d0a90360a66ebce605732d32ff3e283a3428eb7245f4813a521d12580"
    )


def test_authority_opens_only_fit_select_and_has_exact_literal_price(authority):
    groups, rows, by_group, specs = authority
    assert len(groups) == len(by_group) == R580.EXPECTED_GROUPS == 108
    assert len(rows) == R580.EXPECTED_ROWS == 3240
    assert len(specs) == R580.EXPECTED_SEQUENCES == 3024
    assert R580.EXPECTED_FORWARDS == 95
    assert {group["split"] for group in groups} == {"FIT", "SELECT"}
    assert not {row["split"] for row in rows} & set(R580.FORBIDDEN_SPLITS)
    assert len({spec["sequence_id"] for spec in specs}) == len(specs)
    assert all(spec["final_position"] == spec["length"] - 1 for spec in specs)


def test_raw_evidence_is_complete_and_preserves_stable_ids(authority, monkeypatch):
    groups, rows, _by_group, specs = authority
    monkeypatch.setattr(R580, "BOOTSTRAPS", 20)
    measurements = R580.planted_sequence_measurements(specs, rows)
    raw = R580.build_raw_evidence(groups, rows, measurements)
    assert len(raw["sequence_measurements"]) == 3024
    assert len(raw["row_measurements"]) == 3240
    assert len(raw["group_factorial_measurements"]) == 108
    assert len(raw["group_condition_effect_measurements"]) == 432
    assert len({row["row_id"] for row in raw["row_measurements"]}) == 3240
    assert len({row["sequence_id"] for row in raw["sequence_measurements"]}) == 3024
    first = raw["row_measurements"][0]
    assert {
        "row_id", "group_id", "split", "family_id", "condition",
        "base_sequence_id", "donor_sequence_id", "base_margin", "donor_margin",
        "base_ce", "donor_ce", "donor_minus_base_margin",
    } <= first.keys()


def test_exact_sha_bootstrap_is_reproducible_and_clustered_by_group(monkeypatch):
    monkeypatch.setattr(R580, "BOOTSTRAPS", 50)
    values = {"g2": [4.0, 6.0], "g0": [0.0, 2.0], "g1": [2.0, 4.0]}
    left = R580.bootstrap_summary(values, "test:clustered", two_sided=True)
    right = R580.bootstrap_summary(values, "test:clustered", two_sided=True)
    assert left == right
    assert left["ordered_group_ids"] == ["g0", "g1", "g2"]
    assert left["observation_count"] == 6
    assert left["point_mean"] == pytest.approx(3.0)
    # Independently recompute every bootstrap mean from the frozen hash rule.
    ordered = left["ordered_group_ids"]
    means = []
    for replicate in range(50):
        chosen = []
        for draw in range(3):
            payload = (
                f"{R580.BOOTSTRAP_NAMESPACE}:test:clustered:{replicate}:{draw}"
            ).encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3
            chosen.extend(values[ordered[index]])
        means.append(sum(chosen) / len(chosen))
    assert left["lower95"] == float(np.quantile(means, 0.025, method="lower"))
    assert left["upper95"] == float(np.quantile(means, 0.975, method="higher"))


def test_planted_fixture_passes_every_frozen_gate(authority, monkeypatch):
    raw, score, _measurements = _fast_score(authority, monkeypatch)
    assert score["verdict"] == "held_capability_screen"
    assert score["all_scientific_gates_pass"]
    assert score["failed_scientific_clauses"] == []
    assert score["pred_a_native_factorial_and_controls"]
    assert score["pred_b_selector_payload_interaction"]
    assert score["pred_c_selected_match_necessity_and_neutral_selectivity"]
    for split in R580.SPLITS:
        assert all(cell["passes"] for cell in score["factorial_cells"][split].values())
        assert score["selector_payload_interaction"][split]["passes"]
        assert score["selected_match_necessity_and_neutral_selectivity"][split]["passes"]
        assert set(score["relation_preserving_controls"][split]) == {
            "neutral_source", "neutral_payload", "filler", "lag"
        }
    assert len(raw["group_condition_effect_measurements"]) == 4 * 108


def test_all_control_cells_include_both_endpoints_and_all_four_factorial_cells(
    authority, monkeypatch
):
    _raw, score, _measurements = _fast_score(authority, monkeypatch)
    for split in R580.SPLITS:
        controls = score["relation_preserving_controls"][split]
        for label in R580.CONTROL_FAMILIES:
            assert set(controls[label]) == set(R580.CONDITIONS)
            for condition in R580.CONDITIONS:
                assert set(controls[label][condition]) == {"base", "donor"}
                expected = 72 if split == "FIT" else 36
                assert controls[label][condition]["base"]["group_count"] == expected
                assert controls[label][condition]["donor"]["group_count"] == expected


def test_scientific_failure_returns_null_with_raw_rows_instead_of_raising(
    authority, monkeypatch
):
    groups, rows, _by_group, specs = authority
    monkeypatch.setattr(R580, "BOOTSTRAPS", 40)
    passing = R580.planted_sequence_measurements(specs, rows)
    failed = R580.make_planted_scientific_null(passing, groups)
    raw = R580.build_raw_evidence(groups, rows, failed)
    score = R580.score_raw_evidence(raw)
    assert score["verdict"] == "scientific_null"
    assert not score["all_scientific_gates_pass"]
    assert "factorial:SELECT:s0p0" in score["failed_scientific_clauses"]
    assert not score["pred_a_native_factorial_and_controls"]
    assert score["pred_b_selector_payload_interaction"]
    assert not score["pred_c_selected_match_necessity_and_neutral_selectivity"]
    assert len(raw["sequence_measurements"]) == 3024
    assert len(raw["row_measurements"]) == 3240


def test_contrast_source_is_reported_but_cannot_change_gate_decision(
    authority, monkeypatch
):
    raw, baseline, _measurements = _fast_score(authority, monkeypatch)
    modified = json.loads(json.dumps(raw))
    for row in modified["group_condition_effect_measurements"]:
        row["contrast_source_signed_margin_change"] = 1_000_000.0
    changed = R580.score_raw_evidence(modified)
    assert changed["all_scientific_gates_pass"] == baseline["all_scientific_gates_pass"]
    assert changed["failed_scientific_clauses"] == baseline["failed_scientific_clauses"]
    assert changed["contrast_source_diagnostics_not_gated"] != baseline[
        "contrast_source_diagnostics_not_gated"
    ]


def test_selected_gap_uses_worse_of_both_neutral_controls(authority, monkeypatch):
    raw, _baseline, _measurements = _fast_score(authority, monkeypatch)
    item = raw["group_condition_effect_measurements"][0]
    expected = item["selected_match_drop"] - max(
        item["neutral_source_absolute_effect"],
        item["neutral_payload_absolute_effect"],
    )
    assert item["selected_vs_neutral_gap"] == pytest.approx(expected)


def test_dryrun_contract_is_model_free_and_static():
    text = PATH.read_text()
    assert 'if os.environ.get("BQLIB_DRYRUN") == "1"' in text
    assert '"literal_expected_forwards": EXPECTED_FORWARDS' in text
    assert '"model_loaded": False' in text
    assert '"forbidden_splits_opened": []' in text
    assert "import bilin18_observed_model_facade as facade" in text
    # Model facade is a lazy scientific-path import, not an import-time dependency.
    tree = ast.parse(text)
    top_level_imports = {
        (node.module or "")
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "bilin18_observed_model_facade" not in top_level_imports
