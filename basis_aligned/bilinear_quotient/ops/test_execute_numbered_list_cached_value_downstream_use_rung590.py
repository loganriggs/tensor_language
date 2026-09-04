"""CPU-only tests for the hash-pinned R590 managed adapter."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest


PATH = Path(__file__).with_name("execute_numbered_list_cached_value_downstream_use_rung590.py")
SPEC = importlib.util.spec_from_file_location("r590_adapter_target", PATH)
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter)


def test_frozen_bytes_and_complete_executable_closure():
    observed = adapter.verify_frozen_bytes()
    assert observed[str(adapter.PRODUCER)] == adapter.FROZEN_HASHES[adapter.PRODUCER]
    assert observed[str(adapter.HANDOFF_V6)] == adapter.FROZEN_HASHES[adapter.HANDOFF_V6]
    required = {
        adapter.R584_RUNNER, adapter.R588_AUDITOR, adapter.RESULT_CONTRACT,
        adapter.FACADE, adapter.R576_RUNNER, adapter.R573_RUNNER,
        adapter.R582_HELPER, adapter.JACCLUST_PACKAGE, adapter.TT_MODEL,
    }
    assert required <= set(adapter.FROZEN_HASHES)
    assert required == {path for _, path, _ in adapter.EXECUTABLE_LOAD_ORDER}


def test_recursive_helper_and_provenance_use_verified_snapshot(monkeypatch, tmp_path):
    snapshot = adapter.capture_frozen_bytes()
    producer = adapter.load_frozen_producer(snapshot)
    assert producer.r588.load_r582_helper() is producer.r584.r582
    assert producer._role_sha256("implementation") == adapter.FROZEN_HASHES[adapter.PRODUCER]

    changed = tmp_path / "changed_producer.py"
    changed.write_bytes(b"raise RuntimeError('must never execute or attest this')\n")
    monkeypatch.setattr(producer, "SCRIPT", changed)
    hashes = producer.source_hashes()
    assert hashes[str(adapter.PRODUCER)] == adapter.FROZEN_HASHES[adapter.PRODUCER]
    assert str(changed) not in hashes


def test_model_free_plan_never_hashes_prior_outcome_artifacts(monkeypatch):
    producer = adapter.load_frozen_producer()
    auditor = producer.r588
    forbidden_paths = {
        producer.r584.r582.R576_RESULT.resolve(),
        producer.r584.r582.R579_AUDIT.resolve(),
    }
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry run reached a broad prior-outcome authority loader")

    def guarded_read_bytes(path):
        if path.resolve() in forbidden_paths:
            raise AssertionError(f"dry run read prior outcome bytes: {path}")
        return original_read_bytes(path)

    def guarded_read_text(path, *args, **kwargs):
        if path.resolve() in forbidden_paths:
            raise AssertionError(f"dry run read prior outcome text: {path}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(auditor, "verify_preoutcome_authority", forbidden)
    monkeypatch.setattr(auditor, "load_authority", forbidden)
    monkeypatch.setattr(producer.r584, "load_authority", forbidden)
    monkeypatch.setattr(producer.r584.r582, "validate_authorities", forbidden)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    plan = producer.run_dryrun()
    assert plan["literal_executable_maximum_forwards"] == 510
    assert not any(
        "rung576_results" in path or "rung579_audit" in path
        for path in plan["input_sha256"]
    )


def test_subprocess_dryrun_is_model_free_shape_checked_and_outcome_closed():
    environment = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        [sys.executable, str(PATH)], env=environment,
        check=True, capture_output=True, text=True,
    )
    report = json.loads(completed.stdout)
    assert report["mode"] == "model_free_dryrun"
    assert report["dryrun_status"] == "deterministic_cpu_dryrun_passed"
    assert report["forward_call_shape_contract"]["call_count"] == 510
    assert report["forward_call_shape_contract"]["validation_mode"] == \
        "dynamic_batched_token_matrix_exact_common_length_v1"
    assert report["model_forwards"] == 0 and report["model_backwards"] == 0
    assert report["model_weights_updated"] is False
    assert report["next_step"] == "independent_preexecution_review_required"


def test_real_branch_uses_same_verified_module_without_reopening_path():
    calls = []

    def fake_science(producer):
        calls.append(producer)
        raise SystemExit(73)

    with pytest.raises(SystemExit, match="73"):
        adapter.dispatch(
            {}, science_function=fake_science, recovery_function=lambda: None,
            namespace_paths=(),
        )
    assert len(calls) == 1
    assert calls[0].__name__ == "r590_managed_producer"


def test_verified_source_bytes_survive_a_path_swap(tmp_path):
    path = tmp_path / "planted_module.py"
    original = b"VALUE = 'verified'\n"
    path.write_bytes(original)
    captured = path.read_bytes()
    path.write_bytes(b"VALUE = 'swapped'\n")
    name = "r590_planted_" + uuid.uuid4().hex
    module = adapter._module_from_verified_bytes(name, path, captured)
    assert module.VALUE == "verified"


def test_tampered_frozen_source_fails_closed(tmp_path):
    planted = tmp_path / "producer.py"
    planted.write_text("changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        adapter.verify_frozen_bytes({planted: "0" * 64})


def test_incomplete_verified_snapshot_is_rejected_before_import():
    snapshot = adapter.capture_frozen_bytes()
    snapshot.pop(adapter.R588_AUDITOR)
    with pytest.raises(RuntimeError, match="snapshot is incomplete"):
        adapter.load_frozen_producer(snapshot)


def test_invalid_mode_and_arguments_fail_closed():
    with pytest.raises(RuntimeError, match="exactly '1'"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "0"}, recovery_function=lambda: None,
            namespace_paths=(),
        )
    with pytest.raises(SystemExit, match="no command-line arguments"):
        adapter.main(["--execute-science"])


def test_hard_crash_stage_reaches_quarantine_through_managed_preflight(tmp_path):
    producer = adapter.load_frozen_producer()
    stage = producer.create_stage_root(tmp_path)
    (stage / "evidence").mkdir()
    (stage / "evidence" / producer.EVIDENCE_FILE.name).write_bytes(b'{"partial":')
    (stage / "result.json").write_bytes(b'{"partial":')
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"

    def recovery():
        producer.recover_stale_publication(
            root=tmp_path, out=out, receipt_path=receipt, evidence_dir=evidence
        )

    with pytest.raises(RuntimeError, match="recovered incomplete"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"}, recovery_function=recovery,
            namespace_paths=(out, receipt, evidence),
            dry_validator=lambda: {"status": "deterministic_cpu_dryrun_passed"},
        )
    assert not stage.exists()
    recoveries = list(tmp_path.glob(producer.RECOVERY_PREFIX + "*"))
    assert len(recoveries) == 1
    report = adapter.dispatch(
        {"BQLIB_DRYRUN": "1"}, recovery_function=recovery,
        namespace_paths=(out, receipt, evidence),
        dry_validator=lambda: {"status": "deterministic_cpu_dryrun_passed"},
    )
    assert report["mode"] == "model_free_dryrun"


def test_complete_existing_package_is_refused_untouched(tmp_path):
    producer = adapter.load_frozen_producer()
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"
    out.write_bytes(b"result")
    receipt.write_bytes(b"receipt")
    evidence.mkdir()
    before = (out.read_bytes(), receipt.read_bytes(), sorted(evidence.iterdir()))

    with pytest.raises(RuntimeError, match="complete output namespace"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"},
            recovery_function=lambda: producer.recover_stale_publication(
                root=tmp_path, out=out, receipt_path=receipt, evidence_dir=evidence
            ),
            namespace_paths=(out, receipt, evidence),
        )
    assert (out.read_bytes(), receipt.read_bytes(), sorted(evidence.iterdir())) == before


def test_arbitrary_partial_bytes_are_refused_not_quarantined(tmp_path):
    producer = adapter.load_frozen_producer()
    producer.create_stage_root(tmp_path)
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"
    payload = b"arbitrary unrelated bytes"
    out.write_bytes(payload)

    with pytest.raises(RuntimeError, match="unrecognized"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"},
            recovery_function=lambda: producer.recover_stale_publication(
                root=tmp_path, out=out, receipt_path=receipt, evidence_dir=evidence
            ),
            namespace_paths=(out, receipt, evidence),
        )
    assert out.read_bytes() == payload
    assert list(tmp_path.glob(producer.STAGE_PREFIX + "*"))
    assert not list(tmp_path.glob(producer.RECOVERY_PREFIX + "*"))
