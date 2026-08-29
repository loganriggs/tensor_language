import json
from pathlib import Path

import pytest

import recover_block3_consequence_family_f_v2 as recovery


def _frozen():
    body = {
        "schema": "block3_consequence_family_f_v2_recovery_authority",
        "status": "frozen_before_v1_outcome_parent_row_or_checkpoint_tensor_load",
        "source_closure": {"commit": "a" * 40, "paths": {}, "sha256": "b" * 64},
        "v1_binding": {"sha256": "c" * 64},
        "prior_artifact_binding": {"sha256": "d" * 64},
        "row_binding": {"sha256": "e" * 64},
        "checkpoint": {},
        "protocol": {},
        "output_paths": {},
    }
    return {**body, "authority_sha256": recovery.logical_sha256(body)}


def _valid_result():
    arms = list(recovery.v1.call_contract.REPORT_STUDENT_ARMS)
    programs = [arm for arm in arms if arm != "continuous_teacher_F1"]
    expected_model = {
        str(site): recovery.N_BATCHES if site <= recovery.v1.LAYER
        else 19 * recovery.N_BATCHES
        for site in range(18)
    }
    return {
        "schema": "block3_consequence_family_f_v2_recovery_results",
        "status": "fit_reporting_recovered_no_validation_or_final_opened",
        "authority_sha256": _frozen()["authority_sha256"],
        "v1_programs_file_sha256": recovery.V1_PINS[
            str(recovery.v1.PROGRAMS.relative_to(recovery.ROOT))
        ],
        "postfit_report": {
            arm: {
                "document_balanced_teacher_kl": 0.1,
                "row_mean_teacher_kl": 0.1,
                "summed_write_nrmse": 0.2,
            }
            for arm in arms
        },
        "program_prices": {arm: {"total_bytes": 1} for arm in programs},
        "polarization_replay_by_device": {
            arm: {
                "cpu": {"max_absolute": 3e-4, "max_relative": 1e-6},
                "cuda": {"max_absolute": 1e-4, "max_relative": 2e-6},
            }
            for arm in programs
        },
        "report_call_ledger": {
            "schema": "block3_consequence_family_f_v2_recovery_calls",
            "prefixes": recovery.N_BATCHES,
            "teacher_suffixes": recovery.N_BATCHES,
            "student_suffixes": {arm: recovery.N_BATCHES for arm in arms},
            "optimizer_steps": 0,
            "program_refits": 0,
        },
        "model_call_ledger": {"attention": expected_model, "mlp": expected_model},
        "v1_fit_call_ledger_status": "complete_exact_from_preserved_failure",
        "v1_optimizer_traces": "unavailable_from_spent_v1_nonpromotive",
        "model_state_before_sha256": "f" * 64,
        "model_state_after_sha256": "f" * 64,
        "fit_rows_loaded": 480,
        "validation_rows_loaded": 0,
        "final_rows_loaded": 0,
        "ground_truth_target_tokens_used": 0,
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
        "elapsed_seconds": 10.0,
        "maximum_allocated_cuda_bytes": 1,
    }


def test_v2_namespace_and_source_closure_are_distinct_from_spent_v1():
    assert not set(recovery.output_namespace()) & {
        recovery.v1.AUTHORITY, recovery.v1.PROGRAMS, recovery.v1.RESULTS,
        recovery.v1.RECEIPT, recovery.v1.FAILURE,
    }
    assert str(recovery.PREREG.relative_to(recovery.ROOT)) in recovery.SOURCE_PATHS
    assert str(recovery.RUNNER.relative_to(recovery.ROOT)) in recovery.SOURCE_PATHS
    assert str(recovery.TEST.relative_to(recovery.ROOT)) in recovery.SOURCE_PATHS


def test_recovery_call_ledger_accepts_only_exact_frozen_census():
    calls = recovery.RecoveryCalls()
    for _ in range(recovery.N_BATCHES):
        calls.record_prefix(
            "postfit_report", recovery.v1.call_contract.REPORT_SHARED_ARM
        )
        calls.record_teacher_suffix(
            "postfit_report", recovery.v1.call_contract.REPORT_SHARED_ARM
        )
        for arm in recovery.v1.call_contract.REPORT_STUDENT_ARMS:
            calls.record_student_suffix("postfit_report", arm)
    assert calls.validate_exact()["optimizer_steps"] == 0
    calls.students[next(iter(calls.students))] -= 1
    with pytest.raises(RuntimeError, match="call census"):
        calls.validate_exact()


def test_recovery_call_ledger_rejects_wrong_phase_arm_and_donor():
    calls = recovery.RecoveryCalls()
    with pytest.raises(RuntimeError, match="prefix"):
        calls.record_prefix("score_fit", "wrong")
    with pytest.raises(RuntimeError, match="prefix"):
        calls.record_prefix(
            "postfit_report", recovery.v1.call_contract.REPORT_SHARED_ARM, donor=True
        )
    with pytest.raises(RuntimeError, match="student"):
        calls.record_student_suffix("postfit_report", "invented")


def test_semantic_result_accepts_backend_local_maxima_and_rejects_bad_relative_gate():
    frozen = _frozen()
    result = _valid_result()
    recovery.semantic_validate_result(result, frozen)
    first = next(iter(result["polarization_replay_by_device"]))
    result["polarization_replay_by_device"][first]["cuda"]["max_relative"] = 3e-5
    with pytest.raises(RuntimeError, match="polarization"):
        recovery.semantic_validate_result(result, frozen)


def test_semantic_result_rejects_permissions_calls_and_resource_drift():
    frozen = _frozen()
    result = _valid_result()
    result["authorized_for_validation"] = True
    with pytest.raises(RuntimeError, match="lineage or permissions"):
        recovery.semantic_validate_result(result, frozen)
    result = _valid_result()
    result["report_call_ledger"]["prefixes"] -= 1
    with pytest.raises(RuntimeError, match="call census"):
        recovery.semantic_validate_result(result, frozen)
    result = _valid_result()
    result["report_call_ledger"]["optimizer_steps"] = 1
    with pytest.raises(RuntimeError, match="call census"):
        recovery.semantic_validate_result(result, frozen)
    result = _valid_result()
    result["model_call_ledger"]["invented"] = {}
    with pytest.raises(RuntimeError, match="family registry"):
        recovery.semantic_validate_result(result, frozen)
    result = _valid_result()
    result["elapsed_seconds"] = recovery.MAX_WALL_SECONDS + 1
    with pytest.raises(RuntimeError, match="resource"):
        recovery.semantic_validate_result(result, frozen)


def test_receipt_is_an_exact_join():
    frozen = _frozen()
    receipt = {
        "schema": "block3_consequence_family_f_v2_recovery_receipt",
        "status": "fit_reporting_recovery_complete_receipt_last",
        "authority_sha256": frozen["authority_sha256"],
        "authority_file_sha256": "1" * 64,
        "results_file_sha256": "2" * 64,
        "v1_programs_file_sha256": recovery.V1_PINS[
            str(recovery.v1.PROGRAMS.relative_to(recovery.ROOT))
        ],
        "source_closure_sha256": frozen["source_closure"]["sha256"],
        "validation_rows_loaded": 0,
        "final_rows_loaded": 0,
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
    }
    recovery.semantic_validate_receipt(
        receipt, frozen=frozen, authority_hash="1" * 64, result_hash="2" * 64,
    )
    receipt["authorized_for_final"] = True
    with pytest.raises(RuntimeError, match="receipt"):
        recovery.semantic_validate_receipt(
            receipt, frozen=frozen, authority_hash="1" * 64,
            result_hash="2" * 64,
        )


def test_v1_binding_refuses_retrospective_result_or_receipt(monkeypatch, tmp_path):
    fake_result = tmp_path / "result.json"
    fake_result.write_text(json.dumps({"forbidden": True}))
    monkeypatch.setattr(recovery.v1, "RESULTS", fake_result)
    monkeypatch.setattr(recovery.v1, "RECEIPT", tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="unexpectedly has"):
        recovery.v1_file_binding()


def test_pristine_namespace_refuses_any_spent_path(monkeypatch, tmp_path):
    spent = tmp_path / "authority.json"
    spent.write_text("{}")
    monkeypatch.setattr(recovery, "output_namespace", lambda: (spent,))
    with pytest.raises(RuntimeError, match="namespace is spent"):
        recovery.require_pristine_namespace()


@pytest.mark.parametrize("loader", ["v1", "rows", "parents"])
def test_every_tensor_loader_rejects_fake_or_drifted_authority(
    monkeypatch, tmp_path, loader,
):
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps({"fake": True}))
    monkeypatch.setattr(recovery, "AUTHORITY", authority)
    frozen = _frozen()
    with pytest.raises(RuntimeError, match="authority capability"):
        if loader == "v1":
            recovery.load_v1_after_authority(frozen)
        elif loader == "rows":
            recovery.load_rows_after_authority(frozen, {})
        else:
            recovery.load_parents_after_authority(frozen)


def test_receipt_publication_refuses_replaced_lock(monkeypatch):
    events = []

    class ReplacedClaim:
        def verify(self):
            events.append("verify")
            raise RuntimeError("claim replaced")

    monkeypatch.setattr(recovery, "require_resource_ceiling", lambda _started: (1.0, 0))
    monkeypatch.setattr(
        recovery.collector, "create_json",
        lambda *_args, **_kwargs: events.append("receipt_written"),
    )
    with pytest.raises(RuntimeError, match="claim replaced"):
        recovery.publish_receipt_last({}, claim=ReplacedClaim(), started=0.0)
    assert events == ["verify"]


def test_receipt_publication_checks_resources_before_lock_and_write(monkeypatch):
    events = []

    class Claim:
        def verify(self):
            events.append("verify")

    def reject(_started):
        events.append("resources")
        raise RuntimeError("resource ceiling")

    monkeypatch.setattr(recovery, "require_resource_ceiling", reject)
    monkeypatch.setattr(
        recovery.collector, "create_json",
        lambda *_args, **_kwargs: events.append("receipt_written"),
    )
    with pytest.raises(RuntimeError, match="resource ceiling"):
        recovery.publish_receipt_last({}, claim=Claim(), started=0.0)
    assert events == ["resources"]
