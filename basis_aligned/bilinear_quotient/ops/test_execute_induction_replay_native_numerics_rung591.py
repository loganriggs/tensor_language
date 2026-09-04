"""CPU-only tests for the hash-pinned R591 managed adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).with_name("execute_induction_replay_native_numerics_rung591.py")


def load_adapter():
    name = "r591_managed_adapter_owner_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def adapter():
    return load_adapter()


def test_all_candidate_and_method_dependency_bytes_are_pinned(adapter):
    assert adapter.FROZEN_HASHES[adapter.PRODUCER] == (
        "fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_TEST] == (
        "8a24a9903d10ada8a4048c7adcb33cb4ef3e8aeef11d6f9718f8e50e57b6212c"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_DRYRUN] == (
        "8a6331fb1a4d3800abff5ab6b7e291105872b06b41a43b003436312b6e50dc5d"
    )
    assert adapter.FROZEN_HASHES[adapter.PREREGISTRATION] == (
        "2dd8f918f767a6e5d91af357cfaa14770b79334ebac837d1bf52e8046ce190a5"
    )
    assert adapter.FROZEN_HASHES[adapter.BUILDER_HANDOFF] == (
        "202f1268e583a82f6cca385f4223b6edf4e8f8bbaee2c1cc975b09e51cd95f12"
    )
    assert adapter.FROZEN_HASHES[adapter.R585] == (
        "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b"
    )
    assert adapter.FROZEN_HASHES[adapter.FACADE] == (
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c"
    )
    assert adapter.FROZEN_HASHES[adapter.INDUCTION] == (
        "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a"
    )
    assert adapter.FROZEN_HASHES[adapter.METHOD_HANDOFF_V5] == (
        "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80"
    )
    assert adapter.FROZEN_HASHES[adapter.METHOD_HANDOFF_V6] == (
        "d1fdedd90ffff29e6790042b9c9a6ad84278849c3f66707cb586317832fdad1c"
    )
    assert adapter.FROZEN_HASHES[adapter.R578_ROWS] == (
        "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6"
    )
    assert adapter.FROZEN_HASHES[adapter.TT_MODEL] == (
        "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2"
    )
    assert adapter.verify_frozen_bytes() == {
        str(path): digest for path, digest in adapter.FROZEN_HASHES.items()
    }


def test_preflight_has_no_model_work_and_keeps_234_price(adapter):
    plan = adapter.preflight()
    assert all(plan[key] is True for key in adapter.REGISTERED_PREDICATES)
    assert plan["registered_diagnostic_forwards"] == 234
    assert plan["model_forwards"] == plan["model_backwards"] == 0
    assert plan["model_weights_updated"] is False
    json.dumps(plan, allow_nan=False)


def test_dryrun_calls_only_model_free_validator(adapter):
    calls = []

    def planted_validator():
        calls.append("dry")
        return {"schema": "planted"}

    def forbidden_exec(executable, argv):
        pytest.fail(f"dryrun attempted exec: {executable} {argv}")

    report = adapter.dispatch(
        {"BQLIB_DRYRUN": "1"}, dry_validator=planted_validator,
        exec_function=forbidden_exec,
    )
    assert calls == ["dry"]
    assert report["mode"] == "model_free_dryrun"
    assert report["next_step"] == "different_agent_review_required"


def test_real_branch_execs_only_exact_diagnostic_command(adapter):
    calls = []

    class ExecObserved(Exception):
        pass

    def observed_exec(executable, argv):
        calls.append((executable, argv))
        raise ExecObserved

    with pytest.raises(ExecObserved):
        adapter.dispatch({}, exec_function=observed_exec)
    assert calls == [adapter.diagnostic_command()]
    executable, argv = calls[0]
    assert executable == sys.executable
    assert argv[:4] == [sys.executable, "-I", "-c", argv[3]]
    assert str(adapter.PRODUCER) in argv[3]


def test_immutable_command_executes_checked_bytes_after_path_swap(adapter, tmp_path):
    planted = tmp_path / "producer.py"
    planted.write_text("print('original checked bytes')\n")
    expected = adapter.sha256(planted)
    _, argv = adapter.diagnostic_command(planted, expected)
    planted.write_text("print('swapped unchecked bytes')\n")
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "original checked bytes"
    assert "swapped" not in completed.stdout


def test_tampered_byte_fails_before_dispatch(adapter, tmp_path):
    planted = tmp_path / "authority.txt"
    planted.write_bytes(b"expected")
    expected = adapter.sha256(planted)
    planted.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="frozen R591 file changed"):
        adapter.verify_frozen_bytes({planted: expected})


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_any_occupied_scientific_namespace_fails_closed(adapter, tmp_path, kind):
    occupied = tmp_path / "scientific-outcome"
    if kind == "file":
        occupied.write_text("planted\n")
    else:
        occupied.mkdir()
    with pytest.raises(RuntimeError, match="scientific namespace already exists"):
        adapter.require_unused_namespaces((occupied,))


def test_exact_model_free_validation_matches_committed_dryrun(adapter):
    payload = adapter.run_model_free_validation()
    committed = json.loads(adapter.PRODUCER_DRYRUN.read_text(encoding="utf-8"))
    assert payload == committed
    assert payload["model_forwards"] == payload["model_backwards"] == 0
    assert payload["model_weights_updated"] is False
    assert payload["scientific_status"] == "diagnostic_only_no_scientific_terminal"


def test_managed_subprocess_dryrun_is_cpu_only(adapter):
    environment = {"BQLIB_DRYRUN": "1", "BQLIB_NO_MODEL": "1", "CUDA_VISIBLE_DEVICES": ""}
    process = subprocess.run(
        [sys.executable, str(SCRIPT)], check=True, capture_output=True, text=True,
        env=environment,
    )
    report = json.loads(process.stdout)
    assert process.stderr == ""
    assert report["mode"] == "model_free_dryrun"
    assert report["model_forwards"] == report["model_backwards"] == 0


def test_ambiguous_environment_and_arguments_fail_closed(adapter):
    with pytest.raises(RuntimeError, match="absent or exactly '1'"):
        adapter.dispatch({"BQLIB_DRYRUN": "true"})
    with pytest.raises(SystemExit, match="accepts no command-line arguments"):
        adapter.main(["--execute-diagnostic"])
