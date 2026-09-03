"""CPU-only owner tests for the prospective R590 contract replication."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("numbered_list_cached_value_downstream_use_rung590.py")
SPEC = importlib.util.spec_from_file_location("r590_owner_target", PATH)
r590 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(r590)


@pytest.fixture(scope="module")
def rows():
    return r590.r584.load_authority()


@pytest.fixture(scope="module")
def fit_null_evidence():
    fixture = r590.r588.make_fixture(held=False, replicates=8)
    return r590.evidence_from_legacy_payload(fixture)


@pytest.fixture(scope="module")
def fit_null_result(fit_null_evidence):
    digest = r590.canonical_sha256(fit_null_evidence)
    return r590.build_result(
        fit_null_evidence,
        evidence_sha256=digest,
        checkpoint_sha256=r590.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        replicates=8,
    )


@pytest.fixture(scope="module")
def active_null_evidence():
    fixture = r590.r588.make_fit_null_failure_fixture(replicates=8)
    return r590.evidence_from_legacy_payload(fixture)


@pytest.fixture(scope="module")
def held_evidence():
    fixture = r590.r588.make_fixture(held=True, replicates=8)
    return r590.evidence_from_legacy_payload(fixture)


def test_exact_authorities_and_historical_result_is_not_an_input():
    observed = r590.validate_authorities()
    assert observed[str(r590.R584_RUNNER)] == r590.AUTHORITY_HASHES[r590.R584_RUNNER]
    assert observed[str(r590.HANDOFF_V4)] == r590.AUTHORITY_HASHES[r590.HANDOFF_V4]
    assert observed[str(r590.HANDOFF_V5)] == r590.AUTHORITY_HASHES[r590.HANDOFF_V5]
    assert r590.r584.OUT not in r590.AUTHORITY_HASHES
    assert r590.OUT != r590.r584.OUT
    assert not r590.OUT.exists() and not r590.RECEIPT.exists() and not r590.EVIDENCE_DIR.exists()


def test_phase_specific_support_census_is_exact_and_order_hash_is_frozen(rows):
    census = r590.frozen_phase_support_census(rows, r590.AUTHORIZED_SPLITS)
    assert census["cell_fields"] == ["condition", "representation", "source_level"]
    for split, expected_rows, expected_per_cell in (
        ("FIT", 576, 16),
        ("SELECT", 288, 8),
        ("FINAL_TEST", 288, 8),
        ("OOD", 288, 8),
    ):
        report = census["splits"][split]
        assert report["row_count"] == expected_rows
        assert report["cell_count"] == 36
        assert all(cell["row_count"] == expected_per_cell for cell in report["cells"])
        assert report["ordered_row_ids_sha256"] == r590.canonical_sha256(
            report["ordered_row_ids"]
        )


def test_phase_panel_rejects_cross_split_replacement_duplicate_and_shrink(rows):
    fit = [row for row in rows if row["split"] == "FIT"]
    select = [row for row in rows if row["split"] == "SELECT"]
    attacks = (
        fit[:-1] + [select[0]],
        fit[:-1] + [fit[0]],
        fit[:-1],
        fit[1:] + fit[:1],
    )
    for planted in attacks:
        with pytest.raises(RuntimeError, match="borrowed|replacement|shrank"):
            r590.validate_full_phase_panel(planted, rows, "FIT")


def test_global_support_cannot_fill_a_missing_fit_cell(rows):
    fit = [row for row in rows if row["split"] == "FIT"]
    select = [row for row in rows if row["split"] == "SELECT"]
    missing_cell = tuple(fit[0][field] for field in r590.SUPPORT_CELL_FIELDS)
    depleted = [
        row for row in fit
        if tuple(row[field] for field in r590.SUPPORT_CELL_FIELDS) != missing_cell
    ]
    assert any(
        tuple(row[field] for field in r590.SUPPORT_CELL_FIELDS) == missing_cell
        for row in select
    )
    with pytest.raises(RuntimeError, match="shrank"):
        r590.validate_full_phase_panel(depleted, rows, "FIT")


def test_forward_call_manifest_is_literal_complete_and_variable_shape(rows):
    manifest = r590.build_forward_call_manifest(rows)
    report = r590.validate_forward_call_manifest(manifest, rows)
    assert report["call_count"] == 510
    assert report["call_kind_counts"] == {
        "component_suffix": 366,
        "native_logits_smoke": 2,
        "null_component_suffix": 60,
        "trajectory": 82,
    }
    assert report["source_callsite_census"] == {
        "capture_split.trajectory": 2,
        "capture_split.native_logits": 1,
        "evaluate_component.component_forward": 1,
    }
    assert report["wrapper_callsite_census"] == {
        "run_science.facade_load_bilin18": 1,
        "run_science.capture_split": 2,
        "run_science.evaluate_component": 4,
    }
    assert all(item["logical_batch_size"] <= 24 for item in manifest)
    assert all(item["shape_validation_mode"] == r590.SHAPE_MODE for item in manifest)


def test_fixed_4x256_or_hidden_shape_call_fails_before_enqueue(rows):
    manifest = r590.build_forward_call_manifest(rows)
    changed = copy.deepcopy(manifest)
    changed[0]["shape_validation_mode"] = "fixed_production_4x256"
    changed[0]["logical_batch_size"] = 4
    changed[0]["padded_sequence_length"] = 256
    with pytest.raises(RuntimeError, match="manifest differs|shape"):
        r590.validate_forward_call_manifest(changed, rows)
    with pytest.raises(RuntimeError, match="manifest differs|510"):
        r590.validate_forward_call_manifest(manifest[:-1], rows)


def test_conditional_call_paths_are_exact(rows):
    manifest = r590.build_forward_call_manifest(rows)
    assert len(r590.expected_executed_call_ids(manifest, None, None)) == 379
    assert len(r590.expected_executed_call_ids(
        manifest, "mlp8_background_cross", None
    )) == 419
    assert len(r590.expected_executed_call_ids(
        manifest, "mlp8_background_cross", "mlp8_background_cross"
    )) == 510
    with pytest.raises(RuntimeError, match="differs"):
        r590.expected_executed_call_ids(
            manifest, "mlp8_background_cross", "mlp10_background_cross"
        )


def test_fit_null_terminal_is_derived_from_complete_primitive_evidence(
    fit_null_evidence, fit_null_result,
):
    derived = r590.validate_result_against_evidence(
        fit_null_result, fit_null_evidence, replicates=8
    )
    assert derived["decision"] == "downstream_use_decomposition_null"
    assert derived["model_forwards"] == 379
    assert derived["bootstrap_cell_count"] == 432
    assert derived["evaluated_splits"] == ["FIT"]


def test_active_fit_null_and_fit_select_hold_are_both_phase_complete(
    active_null_evidence, held_evidence,
):
    active_null = r590.derive_scientific_summary(active_null_evidence, replicates=8)
    assert active_null["provisional_fit_selection"] is not None
    assert active_null["selected_component"] is None
    assert active_null["model_forwards"] == 419
    assert active_null["evaluated_splits"] == ["FIT"]

    held = r590.derive_scientific_summary(held_evidence, replicates=8)
    assert held["selected_component"] == held["provisional_fit_selection"]
    assert held["model_forwards"] == 510
    assert held["evaluated_splits"] == ["FIT", "SELECT"]
    assert held["decision"] == "downstream_use_component_held"


def test_active_null_donor_mutation_is_rejected(active_null_evidence):
    changed = copy.deepcopy(active_null_evidence)
    key = next(iter(changed["fit_null_raw"]))
    changed["fit_null_raw"][key][0]["null_donor_row_id"] = \
        changed["fit_null_raw"][key][0]["row_id"]
    with pytest.raises(RuntimeError, match="donor"):
        r590.derive_scientific_summary(changed, replicates=8)


@pytest.mark.parametrize("field,replacement", [
    ("fit_reports", {}),
    ("decision", "downstream_use_component_held"),
    ("next_step", "invented_action"),
    ("model_forwards", 510),
])
def test_report_and_terminal_mutations_are_rejected(
    fit_null_evidence, fit_null_result, field, replacement,
):
    changed = copy.deepcopy(fit_null_result)
    changed[field] = replacement
    with pytest.raises(RuntimeError):
        r590.validate_result_against_evidence(changed, fit_null_evidence, replicates=8)


def test_extra_result_field_is_rejected(fit_null_evidence, fit_null_result):
    changed = dict(fit_null_result, invented_summary=True)
    with pytest.raises(RuntimeError, match="field set"):
        r590.validate_result_against_evidence(changed, fit_null_evidence, replicates=8)


def test_missing_or_semantically_changed_primitive_row_is_rejected(fit_null_evidence):
    missing = copy.deepcopy(fit_null_evidence)
    arm = next(iter(missing["fit_raw"]))
    missing["fit_raw"][arm].pop()
    with pytest.raises(RuntimeError, match="membership|row"):
        r590.derive_scientific_summary(missing, replicates=8)

    changed = copy.deepcopy(fit_null_evidence)
    changed["fit_capture_raw"][0]["source_position"] += 1
    with pytest.raises(RuntimeError, match="source_position"):
        r590.derive_scientific_summary(changed, replicates=8)


def test_selffangled_arm_endpoint_cannot_disagree_with_native_capture(fit_null_evidence):
    changed = copy.deepcopy(fit_null_evidence)
    arm = next(iter(changed["fit_raw"]))
    record = next(row for row in changed["fit_raw"][arm] if row["condition"] != "step_two")
    record["native"]["answer_logit"] += 1.0
    record["native"]["margin"] += 1.0
    record["native"]["ce"] -= 1.0
    record["margin_damage"] += 1.0
    record["ce_increase"] += 1.0
    with pytest.raises(RuntimeError, match="native"):
        r590.derive_scientific_summary(changed, replicates=8)


def test_replay_and_exactness_failure_hard_aborts_before_a_terminal(fit_null_evidence):
    changed = copy.deepcopy(fit_null_evidence)
    changed["fit_capture_raw"][0]["native_replay_relative_squared_error_by_row"] = {
        "source_present": 1.0,
        "source_deleted": 1.0,
        "maximum": 1.0,
    }
    changed["fit_exactness"] = {
        key: 1.0 for key in changed["fit_exactness"]
    }
    with pytest.raises(r590.UnretainedInstrumentError, match="before publishable evidence"):
        r590.derive_scientific_summary(changed, replicates=8)


def test_evidence_call_ids_must_equal_derived_conditional_path(fit_null_evidence):
    changed = copy.deepcopy(fit_null_evidence)
    changed["executed_forward_call_ids"] = ["invented-call"]
    with pytest.raises(RuntimeError, match="executed call IDs"):
        r590.derive_scientific_summary(changed, replicates=8)


def test_evidence_phase_support_cannot_be_shrunk_or_rehashed(fit_null_evidence):
    changed = copy.deepcopy(fit_null_evidence)
    changed["phase_support_census"]["splits"]["FIT"]["cells"][0][
        "ordered_row_ids"
    ].pop()
    changed["phase_support_census_sha256"] = r590.canonical_sha256(
        changed["phase_support_census"]
    )
    with pytest.raises(RuntimeError, match="phase_support"):
        r590.derive_scientific_summary(changed, replicates=8)


def test_staged_package_is_finite_mutually_bound_and_atomically_publishable(
    tmp_path, fit_null_evidence,
):
    stage = r590.create_stage_root(tmp_path)
    stage, result, receipt = r590.stage_package(
        fit_null_evidence,
        checkpoint_sha256=r590.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        stage_root=stage,
        replicates=8,
    )
    out = tmp_path / "result.json"
    receipt_path = tmp_path / "receipt.json"
    evidence_dir = tmp_path / "evidence"
    r590.publish_staged_package(
        stage, out=out, receipt_path=receipt_path, evidence_dir=evidence_dir,
        replicates=8,
    )
    observed = r590.validate_complete_package(
        out=out, receipt_path=receipt_path, evidence_dir=evidence_dir, replicates=8
    )
    assert observed["decision"] == result["decision"]
    assert receipt["result_sha256"] == r590.sha256(out)
    assert receipt["evidence_sha256"] == r590.sha256(
        evidence_dir / r590.EVIDENCE_FILE.name
    )


@pytest.mark.parametrize("crash_label", [
    "published_evidence", "published_result", "published_receipt",
])
def test_publication_crash_rolls_final_paths_back_into_recognizable_stage(
    tmp_path, fit_null_evidence, crash_label,
):
    stage = r590.create_stage_root(tmp_path)
    stage, _, _ = r590.stage_package(
        fit_null_evidence,
        checkpoint_sha256=r590.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        stage_root=stage,
        replicates=8,
    )
    out = tmp_path / "result.json"
    receipt_path = tmp_path / "receipt.json"
    evidence_dir = tmp_path / "evidence-final"

    def crash(label):
        if label == crash_label:
            raise KeyboardInterrupt(label)

    with pytest.raises(KeyboardInterrupt):
        r590.publish_staged_package(
            stage, out=out, receipt_path=receipt_path,
            evidence_dir=evidence_dir, replicates=8, crash_injector=crash,
        )
    assert not out.exists() and not receipt_path.exists() and not evidence_dir.exists()
    assert stage.exists()


def test_receipt_or_evidence_mutation_is_rejected(tmp_path, fit_null_evidence):
    stage = r590.create_stage_root(tmp_path)
    stage, _, _ = r590.stage_package(
        fit_null_evidence,
        checkpoint_sha256=r590.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        stage_root=stage,
        replicates=8,
    )
    result = r590.strict_load_json(stage / "result.json")
    receipt = r590.strict_load_json(stage / "receipt.json")
    evidence_bytes = (stage / "evidence" / r590.EVIDENCE_FILE.name).read_bytes()
    result_bytes = (stage / "result.json").read_bytes()
    receipt["result_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="receipt"):
        r590.validate_receipt(receipt, result_bytes, evidence_bytes, result)


def test_staged_byte_mutation_is_rejected_before_any_final_rename(
    tmp_path, fit_null_evidence,
):
    stage = r590.create_stage_root(tmp_path)
    stage, _, _ = r590.stage_package(
        fit_null_evidence,
        checkpoint_sha256=r590.CHECKPOINT_SHA256,
        elapsed_seconds=1.0,
        stage_root=stage,
        replicates=8,
    )
    with (stage / "result.json").open("ab") as handle:
        handle.write(b"\n")
    out = tmp_path / "result-final.json"
    receipt = tmp_path / "receipt-final.json"
    evidence = tmp_path / "evidence-final"
    with pytest.raises(RuntimeError, match="receipt"):
        r590.publish_staged_package(
            stage, out=out, receipt_path=receipt, evidence_dir=evidence,
            replicates=8,
        )
    assert not out.exists() and not receipt.exists() and not evidence.exists()


def test_dryrun_is_model_free_closed_and_contains_shape_hash(monkeypatch):
    # The owner source exists by the time this test imports R590.
    plan = r590.run_dryrun()
    assert plan["model_loaded"] is False and plan["cuda_opened"] is False
    assert plan["opened_splits"] == [] and plan["forbidden_splits_opened"] == []
    assert plan["model_forwards"] == 0 and plan["model_backwards"] == 0
    assert plan["forward_call_shape_contract"]["call_count"] == 510
    assert len(plan["forward_call_shape_contract"]["manifest_sha256"]) == 64
    assert plan["phase_support_census"]["splits"]["FIT"]["row_count"] == 576
    assert plan["phase_support_census_sha256"] == r590.canonical_sha256(
        plan["phase_support_census"]
    )
