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
        "dcdb6470e481dcbc58e86997f4a4d0e3203607ae29a0b74b0e58f59abf62db58"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_TEST] == (
        "cf4326ba6500814767e4b5ee17952753cbda39d6368e91c45a47a1ddce10cc63"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_DRYRUN] == (
        "a30d8206b11beb691e2b9dd2ce33a3a3c2df6752388643f13f0fc81442c69118"
    )
    assert adapter.FROZEN_HASHES[adapter.IMPLEMENTATION_REVIEW] == (
        "9bf8ae3c89d7c504bfdd42694771ef44bb87883429060d16335f0a1266d75a30"
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
