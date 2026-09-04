#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial tests for the blocked task-21 managed producer adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

import execute_circuit_battery_task21_capability_fit as adapter


REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
PRODUCER = OPS / "circuit_battery_task21_capability_fit_producer.py"
NOTE = (
    REPO_ROOT / "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK21_CAPABILITY_FIT_PRODUCER_IMPLEMENTATION_PREREGISTRATION_2026-09-04.md"
)
REVIEW = (
    REPO_ROOT / "basis_aligned/polynomial_causal/"
    "TASK21_VERBATIM_COPY_AUTHORITY_COMPILER_REPRODUCIBILITY_REVIEW_2026-09-04.md"
)
DRYRUN = (
    REPO_ROOT / "basis_aligned/bilinear_quotient/"
    "circuit_battery_task21_capability_fit_producer_v1_dryrun.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dryrun_compiles_exact_plan_and_excludes_runtime_sources() -> None:
    report = adapter.dispatch({"BQLIB_DRYRUN": "1"})
    assert report["execution_authorized"] is False
    assert report["status"] == "model_free_plan_validated_execution_unauthorized"
    assert report["compiled_contract_sha256"] \
        == "5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2"
    assert report["call_manifest_sha256"] \
        == "ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e"
    assert report["completed_calls"] == 8
    assert report["example_evaluations"] == 168
    assert report["raw_numeric_evidence_bytes"] == 1344
    assert report["evidence_file_count"] == 24
    assert report["passing_fixture_terminal"] == "ok"
    assert report["failing_fixture_terminal"] == "hard_abort"
    assert report["failing_fixture_projection_all_null"] is True
    assert report["model_loaded"] is False and report["model_forwards"] == 0
    assert report["queue_touched"] is False
    assert report["runtime_only_roles_excluded"] == sorted((
        "canary1_source", "canary2_source", "jacclust_package",
        "model_source", "observed_model_facade",
    ))
    assert not set(report["runtime_only_roles_excluded"]) & set(report["captured_roles"])
    assert "compiler_review" in report["captured_roles"]
    assert "producer_implementation_preregistration" in report["captured_roles"]


def test_checked_in_producer_dryrun_is_exact() -> None:
    assert sha256(DRYRUN) == "58c3821a8812062fd8fd5b0cd4dcb8aff7166dfbbe76ba10d85138e1dfa96bd6"
    assert json.loads(DRYRUN.read_text()) == adapter.dispatch({"BQLIB_DRYRUN": "1"})


def test_subprocess_dryrun_never_imports_torch_or_opens_model() -> None:
    code = (
        "import runpy,sys; p=sys.argv[1]; sys.argv=[p]; "
        "runpy.run_path(p,run_name='__main__'); "
        "raise SystemExit(97 if 'torch' in sys.modules else 0)"
    )
    environment = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(adapter.ADAPTER)], cwd=REPO_ROOT,
        env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"model_loaded": false' in completed.stdout
    assert '"execution_authorized": false' in completed.stdout


def test_real_branch_is_blocked_before_bootstrap_capture_or_safe_read(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("blocked real branch touched the managed closure")
    monkeypatch.setattr(adapter, "bootstrap", forbidden)
    monkeypatch.setattr(adapter, "capture", forbidden)
    monkeypatch.setattr(adapter, "safe_read", forbidden)
    with pytest.raises(adapter.AdapterError, match="not authorized"):
        adapter.dispatch({})
    with pytest.raises(adapter.AdapterError, match="absent or exactly"):
        adapter.dispatch({"BQLIB_DRYRUN": "true"})
    with pytest.raises(SystemExit, match="accepts no arguments"):
        adapter.main(["unmanaged-argument"])


def test_import_cache_substitution_is_replaced_by_captured_modules(monkeypatch) -> None:
    names = (
        "circuit_experiment_spec", "circuit_artifact_package",
        "circuit_battery_task21", "circuit_battery_task21_capability_fit",
        "circuit_battery_task21_capability_fit_producer",
    )
    planted = {}
    for name in names:
        module = ModuleType(name); module.planted_cache_attack = True
        planted[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    report = adapter.dispatch({"BQLIB_DRYRUN": "1"})
    assert report["completed_calls"] == 8
    for name in names:
        assert sys.modules[name] is not planted[name]
        assert not hasattr(sys.modules[name], "planted_cache_attack")
    producer = sys.modules["circuit_battery_task21_capability_fit_producer"]
    assert producer.framework is sys.modules["circuit_experiment_spec"]
    assert producer.package is sys.modules["circuit_artifact_package"]
    assert producer.capability is sys.modules["circuit_battery_task21_capability_fit"]


def test_disk_import_substitution_cannot_replace_verified_modules(tmp_path: Path, monkeypatch) -> None:
    names = (
        "circuit_experiment_spec", "circuit_artifact_package",
        "circuit_battery_task21", "circuit_battery_task21_capability_fit",
        "circuit_battery_task21_capability_fit_producer",
    )
    for name in names:
        (tmp_path / f"{name}.py").write_text(
            "raise RuntimeError('planted disk import substitution executed')\n"
        )
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert adapter.dispatch({"BQLIB_DRYRUN": "1"})["completed_calls"] == 8
    assert not any(str(tmp_path) in str(sys.modules[name].__file__) for name in names)


def test_changed_captured_compiler_producer_or_authority_bytes_are_rejected() -> None:
    _, managed, captured = adapter.capture("1")
    for role in ("capability_compiler", "producer", "fit_authority"):
        attacked = dict(captured)
        attacked[role] += b"\n# planted mutation\n"
        with pytest.raises(adapter.AdapterError, match="captured frozen bytes changed"):
            adapter.load_verified_closure(managed, attacked, real=False)


def test_safe_read_rejects_changed_bytes_symlink_and_nonregular(tmp_path: Path) -> None:
    payload = tmp_path / "payload.py"; payload.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    assert adapter.safe_read(payload, expected) == b"trusted"
    with pytest.raises(adapter.AdapterError, match="changed"):
        adapter.safe_read(payload, "0" * 64)
    link = tmp_path / "link.py"; link.symlink_to(payload)
    with pytest.raises(adapter.AdapterError, match="safely open"):
        adapter.safe_read(link, expected)
    directory = tmp_path / "directory"; directory.mkdir()
    with pytest.raises(adapter.AdapterError):
        adapter.safe_read(directory, expected)


def test_every_frozen_file_matches_and_review_chain_is_exact() -> None:
    assert len({item.role for item in adapter.FILES}) == len(adapter.FILES)
    assert len({item.relative_path for item in adapter.FILES}) == len(adapter.FILES)
    for item in adapter.FILES:
        assert sha256(REPO_ROOT / item.relative_path) == item.sha256
    assert adapter.COMPILER_COMMIT == "9ebab94615eade27b1eb63e4f2c6239337b71dc9"
    assert adapter.COMPILER_REVIEW_COMMIT == "ca088ce0906160958a2586cff50b707699b7eb88"
    assert adapter.file_by_role("compiler_review").sha256 == sha256(REVIEW) \
        == "3f66075ab775ce27084203999859ea6941efec6d2154a6987994b48e011c7c50"
    assert adapter.file_by_role("producer").sha256 == sha256(PRODUCER) \
        == "395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf"
    assert adapter.EXECUTION_AUTHORIZED is False


def test_implementation_note_binds_producer_review_model_and_scope() -> None:
    note = NOTE.read_text()
    for value in (
        sha256(PRODUCER), sha256(REVIEW), adapter.COMPILER_COMMIT,
        "191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b",
        "5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2",
        "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    ):
        assert value in note
    assert "does not authorize model/checkpoint access" in note
    assert "168 explicit row-side evaluations" in note
    assert "1,344 raw numeric bytes" in note
    assert "every scientific projection field null" in note


def test_adapter_source_has_no_queue_outcome_or_future_authority_path() -> None:
    source = adapter.ADAPTER.read_text()
    assert "queue.txt" not in source and "enqueue.sh" not in source
    assert "_results.json" not in source and "_evidence" not in source
    assert "build_authority(" not in source
    assert "select_authority" not in source.lower()
    assert "test_authority" not in source.lower()
    assert "ood_authority" not in source.lower()
    assert "EXECUTION_AUTHORIZED = False" in source
    assert "EXECUTION_AUTHORIZED = True" not in source
    assert not any(item.kind == "outcome" for item in adapter.FILES)


def test_only_fit_authority_is_present_and_real_modules_are_runtime_only() -> None:
    authorities = [item for item in adapter.FILES if item.kind == "authority"]
    assert [(item.role, item.relative_path) for item in authorities] == [
        ("fit_authority", "basis_aligned/bilinear_quotient/ops/circuit_battery_task21_copy_fit_authority.json")
    ]
    runtime = {item.role for item in adapter.FILES if not item.dryrun_access}
    assert runtime == {
        "jacclust_package", "model_source", "observed_model_facade",
        "canary1_source", "canary2_source",
    }
