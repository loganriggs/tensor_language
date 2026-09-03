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
        "b2b266529f0f842211fea46856064133df5e3f4a8a7758c9095e7d29a94b6c49"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_TEST] == (
        "e756ba3d17d3ebee2f81e97e573dd216090555de1fd3f1cfc926268f902d9ce7"
    )
    assert adapter.FROZEN_HASHES[adapter.PRODUCER_DRYRUN] == (
        "161193de5d90da69aafcd681e375993fa91d32e99100f0ed02fb586d5a629d8b"
    )
    assert adapter.FROZEN_HASHES[adapter.PREREGISTRATION] == (
        "e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593"
    )
    assert adapter.FROZEN_HASHES[adapter.BUILDER_HANDOFF] == (
        "61f8fb407dc026a7a2b126f2dce02b60266d040ffcce7159c5dc6a0d2517cc4f"
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
    assert calls == [
        (sys.executable, [sys.executable, str(adapter.PRODUCER)])
    ]


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
