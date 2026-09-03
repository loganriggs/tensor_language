"""CPU-only tests for the hash-pinned R590 managed adapter."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).with_name("execute_numbered_list_cached_value_downstream_use_rung590.py")
SPEC = importlib.util.spec_from_file_location("r590_adapter_target", PATH)
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter)


def test_frozen_bytes_and_exact_scientific_command():
    observed = adapter.verify_frozen_bytes()
    assert observed[str(adapter.PRODUCER)] == adapter.FROZEN_HASHES[adapter.PRODUCER]
    assert observed[str(adapter.HANDOFF_V5)] == adapter.FROZEN_HASHES[adapter.HANDOFF_V5]
    executable, argv = adapter.scientific_command()
    assert executable == sys.executable
    assert argv == [sys.executable, str(adapter.PRODUCER), "--execute-science"]


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


def test_real_branch_uses_exact_exec_without_falling_into_dryrun():
    calls = []

    def fake_exec(executable, argv):
        calls.append((executable, argv))
        raise SystemExit(73)

    with pytest.raises(SystemExit, match="73"):
        adapter.dispatch(
            {}, exec_function=fake_exec, recovery_function=lambda: None,
            namespace_paths=(),
        )
    assert calls == [adapter.scientific_command()]


def test_tampered_frozen_source_fails_closed(tmp_path):
    planted = tmp_path / "producer.py"
    planted.write_text("changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        adapter.verify_frozen_bytes({planted: "0" * 64})


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
