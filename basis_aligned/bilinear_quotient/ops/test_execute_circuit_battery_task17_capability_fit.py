#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial tests for the review-blocked task-17 managed adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

import execute_circuit_battery_task17_capability_fit as adapter


REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
PRODUCER = OPS / "circuit_battery_task17_capability_fit_producer.py"
DRYRUN = (
    REPO_ROOT / "basis_aligned/bilinear_quotient/"
    "circuit_battery_task17_capability_fit_v1_dryrun.json"
)
AMENDMENT = (
    REPO_ROOT / "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK17_CAPABILITY_FIT_EXECUTION_AMENDMENT.md"
)
REPAIR_AMENDMENT = (
    REPO_ROOT / "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK17_CAPABILITY_FIT_PUBLICATION_REPAIR_AMENDMENT.md"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dryrun_compiles_exact_plan_and_excludes_runtime_sources() -> None:
    report = adapter.dispatch({"BQLIB_DRYRUN": "1"})
    assert report == json.loads(DRYRUN.read_text())
    assert report["execution_authorized"] is False
    assert report["compiled_contract_sha256"] == (
        "526f292338abb5583942f95241be6aa2485db8421270e395bb9fa64bb34751c9"
    )
    assert report["call_manifest_sha256"] == (
        "0edd2541dcddb0d3442b05e6df3f65971a9d973281a676fc9117338435567bdf"
    )
    assert report["completed_calls"] == 8
    assert report["example_evaluations"] == 192
    assert report["raw_numeric_evidence_bytes"] == 1536
    assert report["model_loaded"] is False
    assert report["model_forwards"] == 0
    assert report["runtime_only_roles_excluded"] == sorted((
        "canary1_source", "canary2_source", "jacclust_package",
        "model_source", "observed_model_facade",
    ))
    assert not set(report["runtime_only_roles_excluded"]) & set(report["captured_roles"])
    assert "compiler_review" in report["captured_roles"]


def test_subprocess_dryrun_never_imports_torch_or_opens_model() -> None:
    code = (
        "import json,runpy,sys; p=sys.argv[1]; sys.argv=[p]; "
        "runpy.run_path(p,run_name='__main__'); "
        "raise SystemExit(97 if 'torch' in sys.modules else 0)"
    )
    environment = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(adapter.ADAPTER)],
        cwd=REPO_ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"model_loaded": false' in completed.stdout
    assert '"execution_authorized": false' in completed.stdout


def test_import_cache_substitution_is_replaced_by_captured_verified_modules(monkeypatch) -> None:
    names = (
        "circuit_experiment_spec", "circuit_artifact_package",
        "circuit_battery_task17_capability_fit",
        "circuit_battery_task17_capability_fit_producer",
    )
    planted = {}
    for name in names:
        module = ModuleType(name)
        module.planted_disk_cache_attack = True
        planted[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    report = adapter.dispatch({"BQLIB_DRYRUN": "1"})
    assert report["completed_calls"] == 8
    for name in names:
        assert sys.modules[name] is not planted[name]
        assert not hasattr(sys.modules[name], "planted_disk_cache_attack")
    producer = sys.modules["circuit_battery_task17_capability_fit_producer"]
    assert producer.framework is sys.modules["circuit_experiment_spec"]
    assert producer.package is sys.modules["circuit_artifact_package"]
    assert producer.capability is sys.modules["circuit_battery_task17_capability_fit"]


def test_disk_import_substitution_cannot_replace_verified_modules(tmp_path: Path, monkeypatch) -> None:
    for name in (
        "circuit_experiment_spec", "circuit_artifact_package",
        "circuit_battery_task17_capability_fit",
        "circuit_battery_task17_capability_fit_producer",
    ):
        (tmp_path / f"{name}.py").write_text(
            "raise RuntimeError('planted disk import substitution executed')\n"
        )
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    report = adapter.dispatch({"BQLIB_DRYRUN": "1"})
    assert report["completed_calls"] == 8
    assert not any(str(tmp_path) in str(sys.modules[name].__file__) for name in (
        "circuit_experiment_spec", "circuit_artifact_package",
        "circuit_battery_task17_capability_fit",
        "circuit_battery_task17_capability_fit_producer",
    ))


def test_real_branch_is_blocked_before_capture_model_or_gpu(monkeypatch) -> None:
    def forbidden(_mode):
        raise AssertionError("real blocked branch reached immutable/runtime capture")

    monkeypatch.setattr(adapter, "capture", forbidden)
    monkeypatch.setattr(adapter, "safe_read", lambda *_args: (_ for _ in ()).throw(
        AssertionError("blocked real branch read a frozen file")
    ))
    with pytest.raises(adapter.AdapterError, match="not authorized"):
        adapter.dispatch({})
    with pytest.raises(adapter.AdapterError, match="absent or exactly"):
        adapter.dispatch({"BQLIB_DRYRUN": "true"})


def test_changed_captured_compiler_or_producer_bytes_are_rejected() -> None:
    _, managed, captured = adapter.capture("1")
    for role in ("capability_compiler", "producer"):
        attacked = dict(captured)
        attacked[role] = attacked[role] + b"\n# planted mutation\n"
        with pytest.raises(adapter.AdapterError, match="captured frozen bytes changed"):
            adapter.load_verified_closure(managed, attacked, real=False)


def test_safe_read_rejects_changed_bytes_symlink_and_midread_identity(tmp_path: Path) -> None:
    payload = tmp_path / "payload.py"
    payload.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    assert adapter.safe_read(payload, expected) == b"trusted"
    with pytest.raises(adapter.AdapterError, match="changed"):
        adapter.safe_read(payload, "0" * 64)
    link = tmp_path / "link.py"
    link.symlink_to(payload)
    with pytest.raises(adapter.AdapterError, match="safely open"):
        adapter.safe_read(link, expected)


def test_every_frozen_file_matches_adapter_digest_and_review_is_bound() -> None:
    assert len({item.role for item in adapter.FILES}) == len(adapter.FILES)
    assert len({item.relative_path for item in adapter.FILES}) == len(adapter.FILES)
    for item in adapter.FILES:
        assert sha256(REPO_ROOT / item.relative_path) == item.sha256
    review = adapter.file_by_role("compiler_review")
    assert review.sha256 == "0494f037748a5e781d038c9960875fbb1e1ee219711c78649246d402e8e6b5c4"
    assert adapter.COMPILER_COMMIT == "5da7c8cea"
    assert adapter.EXECUTION_AUTHORIZED is False


def test_versioned_publication_repair_binds_new_producer_and_preserves_old_amendment() -> None:
    producer_digest = sha256(PRODUCER)
    amendment = AMENDMENT.read_text()
    repair = REPAIR_AMENDMENT.read_text()
    assert producer_digest == adapter.file_by_role("producer").sha256
    assert "`a46b64410d0090d2034523be5b1eee58250c876131d78f97b3262c25ca637750`" in amendment
    assert f"`{producer_digest}`" in repair
    assert adapter.file_by_role("execution_amendment").sha256 == sha256(AMENDMENT)
    assert adapter.file_by_role("publication_repair_amendment").sha256 == sha256(
        REPAIR_AMENDMENT
    )
    assert "not authorized for model" in amendment
    assert "RENAME_NOREPLACE" in repair
    assert "0 model updates" in amendment


def test_adapter_source_has_no_enqueue_or_outcome_read_path() -> None:
    source = adapter.ADAPTER.read_text()
    assert "circuit_battery.py" not in source
    assert "circuit_battery_tasks" not in source
    assert "queue.txt" not in source
    assert "enqueue.sh" not in source
    assert "build_authority(" not in source
    assert "_results.json" not in source
    assert "EXECUTION_AUTHORIZED = False" in source
