"""Focused, outcome-blind tests for the independent R588 CPU auditor."""

# BQLANE: cpu

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
AUDITOR_PATH = HERE / "audit_numbered_list_cached_value_downstream_use_rung588.py"
SPEC = importlib.util.spec_from_file_location("r588_auditor_under_test", AUDITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
r588 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r588)


@pytest.fixture(scope="module")
def authority():
    return r588.load_authority()


@pytest.fixture(scope="module")
def held_fixture():
    return r588.make_fixture(held=True, replicates=3)


@pytest.fixture(scope="module")
def null_fixture():
    return r588.make_fixture(held=False, replicates=3)


def test_pinned_authority_and_exact_census(authority):
    rows, helper = authority
    assert len(rows) == 1_440
    assert {split: sum(row["split"] == split for row in rows)
            for split in r588.SPLIT_ROWS} == r588.SPLIT_ROWS
    assert {split: len({row["group_id"] for row in rows if row["split"] == split})
            for split in r588.SPLIT_GROUPS} == r588.SPLIT_GROUPS
    assert len(r588.expected_dryrun_provenance(rows, helper)) == 19
    assert len(r588.expected_result_provenance(rows, helper)) == 21
    assert r588.verify_preoutcome_authority()[str(r588.R584_REVIEW)] == \
        "9294bdf8df18a56cdae8705b69e0129bfe2d6376d642d4c9dc86386c0d898310"


def test_auditor_is_cpu_only_and_does_not_import_r584_scoring():
    source = AUDITOR_PATH.read_text()
    assert "# BQLANE: cpu" in source
    assert "import torch" not in source
    assert "load_bilin18" not in source
    assert "from numbered_list_cached_value_downstream_use_rung584" not in source
    assert "import numbered_list_cached_value_downstream_use_rung584" not in source
    assert r588.BOOTSTRAPS == 2_000


def test_independent_bootstrap_draws_and_trace_hash_are_exact():
    cells = [
        {"group_id": "g2", "value": 2.0},
        {"group_id": "g0", "value": 0.0},
        {"group_id": "g1", "value": 1.0},
    ]
    bootstrapper = r588.Bootstrapper(5)
    observed = bootstrapper.lower(cells, "value", "unit:cell")
    groups = ("g0", "g1", "g2")
    values = {"g0": 0.0, "g1": 1.0, "g2": 2.0}
    draws = np.empty((5, 3), dtype=np.uint16)
    statistics = np.empty(5, dtype=np.float64)
    for replicate in range(5):
        sample = []
        for draw in range(3):
            payload = f"r582-group-bootstrap-v1:unit:cell:{replicate}:{draw}".encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3
            draws[replicate, draw] = index
            sample.append(values[groups[index]])
        statistics[replicate] = np.mean(sample)
    trace = bootstrapper.traces["unit:cell"]
    assert observed == float(np.quantile(statistics, .025))
    assert trace["draw_matrix_sha256"] == hashlib.sha256(
        draws.astype(">u2", copy=False).tobytes(order="C")
    ).hexdigest()
    assert trace["statistic_vector_sha256"] == hashlib.sha256(
        statistics.astype(">f8", copy=False).tobytes(order="C")
    ).hexdigest()


def test_planted_held_path_reconstructs_all_metrics(held_fixture):
    audit = r588.audit_payload(held_fixture, replicates=3)
    assert audit["audit_verdict"] == "held_independent_audit"
    assert audit["independently_recomputed_decision"] == "downstream_use_component_held"
    assert audit["independently_recomputed_opened_splits"] == ["FIT", "SELECT"]
    assert audit["independently_recomputed_model_forwards"] == 510
    assert audit["bootstrap_cell_count"] == 588
    report = held_fixture["fit_reports"][r588.SELECTION_NAMES[0]]
    assert len(report["targets"]) == len(report["copies"]) == \
        len(report["action_gaps"]) == 12
    assert len(report["active_relation_and_conflict_controls"]) == 12
    assert len(report["conflicts"]) == 6
    for null_report in held_fixture["fit_null_reports"].values():
        assert len(null_report["representation_cells"]) == 12
        assert len(null_report["source_surface_comparisons"]) == 4
        assert null_report["real_bounds_reused_without_redraw"] is True


def test_planted_complete_scientific_null_is_not_an_audit_failure(null_fixture):
    audit = r588.audit_payload(null_fixture, replicates=3)
    assert audit["audit_verdict"] == "held_independent_audit"
    assert audit["independently_recomputed_decision"] == \
        "downstream_use_decomposition_null"
    assert audit["independently_recomputed_opened_splits"] == ["FIT"]
    assert audit["independently_recomputed_model_forwards"] == 379
    assert audit["bootstrap_cell_count"] == 432


def test_planted_fit_null_failure_reconstructs_419_path():
    fixture = r588.make_fit_null_failure_fixture(replicates=3)
    audit = r588.audit_payload(fixture, replicates=3)
    assert audit["audit_verdict"] == "held_independent_audit"
    assert audit["independently_recomputed_provisional"] == r588.SELECTION_NAMES[0]
    assert audit["independently_recomputed_selected"] is None
    assert audit["independently_recomputed_model_forwards"] == 419
    assert audit["bootstrap_cell_count"] == 456


@pytest.mark.parametrize("case", [
    "missing_arm", "missing_row", "nonfinite", "wrong_donor", "wrong_price",
    "wrong_interaction", "list_next_step", "stale_execution_plan",
    "stale_provenance", "bad_endpoint_identity", "bad_rms_identity",
])
def test_malformed_evidence_fails_closed(held_fixture, case):
    value = copy.deepcopy(held_fixture)
    name = r588.SELECTION_NAMES[0]
    null_key = f"{name}:null:{r588.NULLS[0]}"
    if case == "missing_arm":
        value["fit_raw"].pop(r588.SELECTION_NAMES[-1])
    elif case == "missing_row":
        value["fit_raw"][name].pop()
    elif case == "nonfinite":
        value["fit_raw"][name][0]["margin_damage"] = float("nan")
    elif case == "wrong_donor":
        row = value["fit_null_raw"][null_key][0]
        row["null_donor_row_id"] = row["row_id"]
    elif case == "wrong_price":
        value["model_forwards"] = 509
    elif case == "wrong_interaction":
        value["component_interactions"]["fit"][0]["cross_x_self"] += .1
    elif case == "list_next_step":
        value["next_step"] = [value["next_step"]]
    elif case == "stale_execution_plan":
        value["execution_plan"]["literal_executable_maximum_forwards"] = 509
    elif case == "stale_provenance":
        value["input_sha256"][str(r588.R584_DRYRUN)] = "0" * 64
    elif case == "bad_endpoint_identity":
        value["fit_raw"][name][0]["native"]["margin"] += .1
    elif case == "bad_rms_identity":
        value["fit_raw"][name][0]["full_vocabulary_logit_rms"] += .1
    audit = r588.audit_payload(value, replicates=3)
    assert audit["audit_verdict"] == "failed_independent_audit"
    assert audit["audit_failures"]
    assert set(audit) == {
        "audit_verdict", "audit_failures", "independently_recomputed_decision",
        "independently_recomputed_predicates", "independently_recomputed_provisional",
        "independently_recomputed_selected", "independently_recomputed_opened_splits",
        "independently_recomputed_model_forwards", "raw_counts",
        "bootstrap_replicates_per_cell", "bootstrap_cell_count",
        "bootstrap_trace_sha256", "bootstrap_traces",
    }


def test_null_gate_uses_cross_representation_min_max_and_active_norms(held_fixture):
    name = r588.SELECTION_NAMES[0]
    report = next(iter(held_fixture["fit_null_reports"].values()))
    for source in (0, 1):
        for surface in ("factorial", "surface"):
            cells = [report["representation_cells"][f"{rep}:source{source}:{surface}"]
                     for rep in ("list", "digit", "word")]
            comparison = report["source_surface_comparisons"][f"source{source}:{surface}"]
            assert comparison["minimum_real_gap_lower95_across_representations"] == \
                min(cell["real_gap_lower95_reused"] for cell in cells)
            assert comparison["maximum_null_gap_lower95_across_representations"] == \
                max(cell["null_gap_lower95"] for cell in cells)
            assert all(.8 <= cell["median_null_norm_over_median_real_norm"] <= 1.25
                       for cell in cells)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        r588.strict_loads(b'{"a":1,"a":2}', "duplicate")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        r588.strict_loads(b'{"a":NaN}', "nan")


def test_dryrun_never_touches_source_namespace(monkeypatch, tmp_path):
    class Bomb:
        def __getattribute__(self, name):
            raise AssertionError(f"source result was touched through {name}")

    monkeypatch.setattr(r588, "SOURCE_RESULT", Bomb())
    monkeypatch.setattr(r588, "DRYRUN", tmp_path / "dryrun.json")
    receipt = r588.run_dryrun()
    assert receipt["status"] == "dryrun_passed"
    assert receipt["source_result_touched"] is False
    assert receipt["model_forwards"] == 0


def test_stable_reader_binds_exact_bytes_on_a_synthetic_path(tmp_path):
    path = tmp_path / "synthetic_result.json"
    payload = {"message": "not the R584 namespace", "finite": 1.0}
    raw = (json.dumps(payload, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    observed = r588.read_stable_source(path)
    assert observed == raw
    assert hashlib.sha256(observed).hexdigest() == hashlib.sha256(raw).hexdigest()
