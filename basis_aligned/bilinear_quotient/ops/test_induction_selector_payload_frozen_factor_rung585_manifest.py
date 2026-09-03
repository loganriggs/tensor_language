from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


OPS = Path(__file__).resolve().parent
MODULE_PATH = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"


def load_module():
    name = "r585_outcome_blind_manifest_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract():
    return load_module()


@pytest.fixture(scope="module")
def authority(contract):
    return contract.build_authority_manifest()


@pytest.fixture(scope="module")
def manifests(contract, authority):
    return contract.build_cell_manifests(authority)


def test_is_import_safe_and_pins_only_model_free_authorities(contract):
    source = MODULE_PATH.read_text()
    assert "import torch" not in source
    assert "transformer_lens" not in source
    assert contract.sha256_file(contract.ROWS_PATH) == contract.ROWS_SHA256
    assert contract.sha256_file(contract.AMENDMENT_PATH) == contract.AMENDMENT_SHA256
    assert "r586_results" not in source
    assert "r587_audit.json" not in source


def test_exact_rows_endpoints_and_directions(contract, authority):
    for split, expected in contract.EXPECTED_SPLIT_COUNTS.items():
        rows = [row for row in authority["rows"] if row["split"] == split]
        endpoints = [row for row in authority["endpoints"] if row["split"] == split]
        directions = [row for row in authority["directions"] if row["split"] == split]
        assert len(rows) == expected["rows"]
        assert len(endpoints) == expected["endpoints"]
        assert len(directions) == expected["directions"]
        assert len({row["directed_id"] for row in directions}) == len(directions)
        assert all(row["family"] != contract.F_CONTRAST for row in rows)
        assert {row["recipient_condition"] for row in directions} == set(contract.CONDITIONS)


def test_direction_to_recipient_condition_mapping(contract, authority):
    expected = {
        (contract.F_SELECTOR, "payload_assignment_0"): ("s0p0", "s1p0"),
        (contract.F_SELECTOR, "payload_assignment_1"): ("s0p1", "s1p1"),
        (contract.F_PAYLOAD, "selector_0"): ("s0p0", "s0p1"),
        (contract.F_PAYLOAD, "selector_1"): ("s1p0", "s1p1"),
        (contract.F_JOINT, "payload_B"): ("s0p0", "s1p1"),
        (contract.F_JOINT, "payload_D"): ("s1p0", "s0p1"),
    }
    for key, (base_condition, donor_condition) in expected.items():
        records = [
            row for row in authority["directions"]
            if (row["family"], row["variant"]) == key
        ]
        assert {
            (row["direction"], row["recipient_condition"])
            for row in records
        } == {
            ("base_to_donor", base_condition),
            ("donor_to_base", donor_condition),
        }

    match = [row for row in authority["directions"] if row["family"] == contract.F_MATCH]
    assert all(not row["answer_changes"] for row in match)
    assert all(row["recipient_is_coherent"] for row in match if row["direction"] == "base_to_donor")
    assert all(not row["donor_is_coherent"] for row in match if row["direction"] == "base_to_donor")
    assert all(not row["recipient_is_coherent"] for row in match if row["direction"] == "donor_to_base")
    assert all(row["donor_is_coherent"] for row in match if row["direction"] == "donor_to_base")
    assert {row["donor_coherence_sign"] for row in match if row["direction"] == "base_to_donor"} == {-1}
    assert {row["donor_coherence_sign"] for row in match if row["direction"] == "donor_to_base"} == {1}


def test_literal_cell_coverage_and_structural_manifests(contract, manifests):
    for split in contract.SPLITS:
        targets = [cell for cell in manifests["target_cells"] if cell["split"] == split]
        controls = [cell for cell in manifests["control_cells"] if cell["split"] == split]
        keys = [cell for cell in manifests["coverage_keys"] if cell["split"] == split]
        eligible = [cell for cell in manifests["eligible_control_arm_cells"] if cell["split"] == split]
        identities = [
            identity for identity in manifests["structural_identities"]
            if identity["cell_id"].startswith(split + "|")
        ]
        assert (len(targets), len(controls), len(keys), len(eligible), len(identities)) == (
            20, 32, 24, 88, 32
        )
        assert all(len(cell["group_ids"]) == contract.EXPECTED_SPLIT_COUNTS[split]["groups_per_cell"] for cell in targets + controls)
        assert len({tuple(sorted(key.items())) for key in keys}) == 24

        lag_payload = [
            cell for cell in eligible
            if cell["control_kind"] == "lag" and cell["arm"] == "payload"
        ]
        assert lag_payload == []

    relations = {
        (identity["left_arm"], identity["right_arm"])
        for identity in manifests["structural_identities"]
    }
    assert relations == {("payload", "replay"), ("joint", "score")}


def test_control_scale_lookup_is_unique_and_fit_only(contract, manifests):
    lookup = contract.build_control_scale_lookup(manifests)
    assert len(lookup) == 32 * 3 * 2
    assert len({(row["control_cell_id"], row["arm"]) for row in lookup}) == len(lookup)
    expected_family = {
        "score": contract.F_SELECTOR,
        "payload": contract.F_PAYLOAD,
        "joint": contract.F_SELECTOR,
    }
    for row in lookup:
        parts = row["fit_target_cell_id"].split("|")
        assert parts[0] == "FIT"
        assert parts[1] == expected_family[row["arm"]]
        assert parts[3] == row["recipient_condition"]
        assert row["fit_target_arm"] == row["arm"]


def test_bootstrap_cell_ids_are_exact_and_use_canonical_family_ids(contract, manifests):
    cells = contract.expected_bootstrap_cells(manifests)
    assert len(cells) == 248
    assert len({cell["cell_id"] for cell in cells}) == 248
    assert sum(cell["cell_id"].startswith("FIT|") for cell in cells) == 124
    assert sum(cell["cell_id"].startswith("SELECT|") for cell in cells) == 124
    assert all("selector swap" not in cell["cell_id"] for cell in cells)
    assert all(len(cell["cell_id"].split("|")) == 7 for cell in cells)
    assert {
        cell["cell_id"].split("|")[1] for cell in cells
    } == set(contract.TARGET_FAMILIES)


def test_sha_draw_algorithm_matches_independent_definition(contract):
    cell_id = "FIT|family|variant|s0p0|base_to_donor|score|numerator_mean"
    groups = ("g2", "g0", "g1")
    matrix = contract.bootstrap_draw_matrix(cell_id, groups, replicates=3)
    expected = []
    for b in range(3):
        row = []
        for k in range(3):
            payload = f"{contract.BOOTSTRAP_NAMESPACE}:{cell_id}:{b}:{k}".encode()
            row.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3)
        expected.append(tuple(row))
    assert matrix == tuple(expected)
    raw = b"".join(value.to_bytes(2, "big") for row in expected for value in row)
    assert contract.big_endian_uint16_matrix_sha256(matrix) == hashlib.sha256(raw).hexdigest()
    assert matrix == contract.bootstrap_draw_matrix(cell_id, reversed(groups), replicates=3)


def test_phase_accounting_is_exact(contract, authority):
    accounting = contract.phase_accounting(authority)
    assert accounting["phases"]["FIT"] == {
        "unique_endpoints": 1728,
        "directed_pairs": 3744,
        "capture_replay_forwards": 54,
        "intervention_forwards_per_arm": 117,
        "intervention_forwards": 351,
        "native_comparator_forwards": 54,
        "phase_max_forwards": 459,
    }
    assert accounting["phases"]["SELECT"]["phase_max_forwards"] == 231
    assert accounting["total_max_forwards"] == 690
    assert accounting["model_backwards"] == accounting["weight_updates"] == 0


def test_dependency_lock_accepts_planted_held_and_null_without_path_reads(contract):
    held, held_hashes = contract.build_planted_dependency_fixture(True)
    null, null_hashes = contract.build_planted_dependency_fixture(False)
    assert contract.validate_dependency_lock(held, held_hashes) == {
        "schema_valid": True,
        "hashes_valid": True,
        "runnable": True,
        "terminal": "dependency_held",
        "reasons": [],
    }
    null_result = contract.validate_dependency_lock(null, null_hashes)
    assert not null_result["runnable"]
    assert null_result["terminal"] == "not_executed_upstream_dependency"
    assert null_result["reasons"] == ["r586_not_held"]
    assert all(path.startswith("fixture://") for path in held_hashes)


@pytest.mark.parametrize("mutation", ["hash", "split", "extra", "verdict_type"])
def test_dependency_lock_fails_closed_on_malformed_fixture(contract, mutation):
    lock, hashes = contract.build_planted_dependency_fixture(True)
    lock = copy.deepcopy(lock)
    if mutation == "hash":
        lock["r586_result_sha256"] = "0" * 64
    elif mutation == "split":
        lock["evaluated_splits"] = ["FIT"]
    elif mutation == "extra":
        lock["unexpected"] = True
    elif mutation == "verdict_type":
        lock["r586_verdict"] = ["held_capability_screen"]
    with pytest.raises(ValueError):
        contract.validate_dependency_lock(lock, hashes)


def test_dryrun_is_deterministic_and_scientifically_closed(contract):
    first = contract.build_dryrun()
    second = contract.build_dryrun()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["model_loaded"] is False
    assert first["outcomes_opened"] == []
    assert first["phase_accounting"]["total_max_forwards"] == 690
    assert first["planted_dependency_checks"]["held"]["runnable"] is True
    assert first["planted_dependency_checks"]["null"]["runnable"] is False
