"""CPU-only owner tests for the hash-pinned R585 managed adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).with_name(
    "execute_induction_selector_payload_frozen_factor_rung585.py"
)


def load_adapter():
    name = "r585_managed_adapter_owner_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def adapter():
    return load_adapter()


def test_exact_commit_and_execution_authorities_are_pinned(adapter):
    assert adapter.REPAIR_BASE_COMMIT == "a4e7c46c6339c75fc7f89c1e35339e15e3b74fd9"
    assert adapter.REPAIR_BASE_COMMIT_SHORT == "a4e7c46c6"
    assert adapter.FROZEN_HASHES[adapter.PRODUCER] == (
        "33b5cbbc26e5ba62bb60a5bf62d69a1ef7ea51d1bf64e51fd3b95049e55f4327"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_TEST] == (
        "1a3419e3aa19abc2b03424d02ff5c474296472811e780dd10bbde4cc34f410d7"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_DRYRUN] == (
        "ac02054a22452911150e173792f28902351fdf1b04d04b87007a570837cf026d"
    )
    assert adapter.FROZEN_HASHES[adapter.IMPLEMENTATION_REVIEW] == (
        "9bf8ae3c89d7c504bfdd42694771ef44bb87883429060d16335f0a1266d75a30"
    )
    assert adapter.FROZEN_HASHES[adapter.SECOND_IMPLEMENTATION_REVIEW] == (
        "02b513f9eca7d2582d462de95ca7423cb8150962d9a4bc3e5e40deb36762ca19"
    )
    assert adapter.FROZEN_HASHES[adapter.SECOND_IMPLEMENTATION_ADVERSARIAL_TEST] == (
        "5d5fa1e2628f3dcee41e330c4e6dba5f0a7d6cc0248e1694cedbf39bd7422c47"
    )
    assert adapter.FROZEN_HASHES[adapter.FINAL_IMPLEMENTATION_REVIEW] == (
        "8ddbcf3037b890a3fd1ae6933a526a29c1bd767a22d7fa3af8044d7d660d9238"
    )
    assert adapter.FROZEN_HASHES[adapter.FINAL_IMPLEMENTATION_ADVERSARIAL_TEST] == (
        "693b70f70b72334affd2c8da7e5e02e8b5a41125b29e1df7f943a1856a345277"
    )
    assert adapter.FROZEN_HASHES[adapter.ITERATION4_IMPLEMENTATION_REVIEW] == (
        "302e9ba506931e8513c5f069a332cc1445342ab282344269ee00b866a9e6a9fc"
    )
    assert adapter.FROZEN_HASHES[adapter.ITERATION4_IMPLEMENTATION_ADVERSARIAL_TEST] == (
        "29d8023ddbc56c70df7097394717d70fe7a2b6289fae0bce197f0d0e8f9eafd3"
    )
    assert adapter.verify_frozen_bytes() == {
        str(path): digest for path, digest in adapter.FROZEN_HASHES.items()
    }


def test_preflight_reports_four_adapter_claims_and_no_model_work(adapter):
    plan = adapter.preflight()
    assert plan["pred_a_repair_base_and_bytes_match"] is True
    assert plan["pred_b_outcome_namespaces_are_unused"] is True
    assert plan["pred_c_dryrun_is_model_free"] is True
    assert plan["pred_d_science_command_is_explicit"] is True
    assert plan["model_forwards"] == plan["model_backwards"] == 0
    assert plan["model_weights_updated"] is False
    json.dumps(plan, allow_nan=False)


def test_dryrun_branch_calls_only_model_free_validator(adapter):
    calls = []

    def planted_validator():
        calls.append("dry")
        return {"status": "deterministic_cpu_dryrun_passed"}

    def forbidden_exec(executable, argv):
        pytest.fail(f"dry-run attempted exec: {executable} {argv}")

    report = adapter.dispatch(
        {"BQLIB_DRYRUN": "1"},
        dry_validator=planted_validator,
        exec_function=forbidden_exec,
    )
    assert calls == ["dry"]
    assert report["mode"] == "model_free_dryrun"
    assert report["dryrun_status"] == "deterministic_cpu_dryrun_passed"


def test_real_branch_uses_only_exact_execute_science_command(adapter):
    calls = []

    class ExecObserved(Exception):
        pass

    def observed_exec(executable, argv):
        calls.append((executable, argv))
        raise ExecObserved

    with pytest.raises(ExecObserved):
        adapter.dispatch({}, exec_function=observed_exec)
    assert calls == [
        (
            sys.executable,
            [sys.executable, str(adapter.PRODUCER), "--execute-science"],
        )
    ]
    assert "--dry-run" not in calls[0][1]


def test_tampered_byte_fails_before_dispatch(adapter, tmp_path):
    planted = tmp_path / "authority.txt"
    planted.write_bytes(b"expected")
    expected = adapter.sha256(planted)
    planted.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="frozen R585 file changed"):
        adapter.verify_frozen_bytes({planted: expected})


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_occupied_namespace_fails_closed(adapter, tmp_path, kind):
    occupied = tmp_path / "outcome"
    if kind == "file":
        occupied.write_text("planted\n")
    else:
        occupied.mkdir()
    with pytest.raises(RuntimeError, match="outcome namespace already exists"):
        adapter.require_unused_namespaces((occupied,))


def test_managed_retry_reaches_recognized_stale_stage_quarantine(adapter, tmp_path):
    producer = adapter.load_frozen_producer()
    out = tmp_path / "result.json"
    receipt = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence"
    stage = producer.create_stage_root(tmp_path)

    def recover():
        producer.recover_stale_publication(
            root=tmp_path, out=out, receipt=receipt, evidence=evidence
        )

    with pytest.raises(RuntimeError, match="recovered incomplete"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "1"}, recovery_function=recover,
            namespace_paths=(out, receipt, evidence),
            dry_validator=lambda: pytest.fail("dry run reached before quarantine"),
        )
    assert not stage.exists()
    recovered = list(tmp_path.glob(producer.RECOVERY_PREFIX + "*"))
    assert len(recovered) == 1


def test_managed_recovery_refuses_complete_and_arbitrary_occupied_paths(adapter, tmp_path):
    producer = adapter.load_frozen_producer()
    for suffix, complete in (("complete", True), ("arbitrary", False)):
        root = tmp_path / suffix
        root.mkdir()
        out = root / "result.json"
        receipt = root / "receipt.json"
        evidence = root / "evidence"
        if complete:
            out.write_text("complete-result")
            receipt.write_text("complete-receipt")
            evidence.mkdir()
        else:
            out.write_text("arbitrary bytes")
            stage = producer.create_stage_root(root)

        def recover(root=root, out=out, receipt=receipt, evidence=evidence):
            producer.recover_stale_publication(
                root=root, out=out, receipt=receipt, evidence=evidence
            )

        expected = "complete output namespace" if complete else "unrecognized"
        with pytest.raises(RuntimeError, match=expected):
            adapter.dispatch(
                {"BQLIB_DRYRUN": "1"}, recovery_function=recover,
                namespace_paths=(out, receipt, evidence),
                dry_validator=lambda: pytest.fail("unsafe recovery continued"),
            )
        assert out.read_text() == ("complete-result" if complete else "arbitrary bytes")
        if complete:
            assert receipt.read_text() == "complete-receipt" and evidence.is_dir()
        else:
            assert stage.exists()
        assert not list(root.glob(producer.RECOVERY_PREFIX + "*"))


def test_ambiguous_environment_and_arguments_fail_closed(adapter):
    with pytest.raises(RuntimeError, match="absent or exactly '1'"):
        adapter.dispatch({"BQLIB_DRYRUN": "true"})
    with pytest.raises(SystemExit, match="accepts no command-line arguments"):
        adapter.main(["--execute-science"])


def test_exact_model_free_validation_preserves_committed_dryrun(adapter):
    before = adapter.sha256(adapter.PRODUCER_DRYRUN)
    payload = adapter.run_model_free_validation()
    after = adapter.sha256(adapter.PRODUCER_DRYRUN)
    assert before == after == adapter.FROZEN_HASHES[adapter.PRODUCER_DRYRUN]
    assert payload["status"] == "deterministic_cpu_dryrun_passed"
    assert payload["model_loaded"] is False
    assert payload["cuda_opened"] is False
    assert payload["price"] == {
        "FIT": 459, "SELECT": 231, "maximum": 690, "backwards": 0, "updates": 0,
    }
