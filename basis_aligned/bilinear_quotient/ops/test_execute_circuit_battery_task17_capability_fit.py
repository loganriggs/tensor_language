#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial tests for the authorization-candidate task-17 managed adapter."""

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
REPAIR_REVIEW = (
    REPO_ROOT / "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK17_CAPABILITY_FIT_PUBLICATION_REPAIR_REVIEW_2026-09-04.md"
)
AUTHORIZATION_AMENDMENT = (
    REPO_ROOT / "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK17_CAPABILITY_FIT_AUTHORIZATION_AMENDMENT.md"
)
PROVENANCE_CORRECTION = (
    REPO_ROOT / "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK17_CAPABILITY_FIT_PUBLICATION_REPAIR_PROVENANCE_CORRECTION.md"
)
REPAIR_COMMIT = "538cef96451b3e8f07758f20cca2be1b7bfdf561"
REPAIR_COMMIT_TIME = "2026-09-04T05:13:56+00:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dryrun_compiles_exact_plan_and_excludes_runtime_sources() -> None:
    report = adapter.dispatch({"BQLIB_DRYRUN": "1"})
    assert report == json.loads(DRYRUN.read_text())
    assert report["execution_authorized"] is True
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
    assert '"execution_authorized": true' in completed.stdout


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


def test_real_branch_accepts_only_exact_managed_mode_and_delegates_once(monkeypatch) -> None:
    calls = []
    producer = ModuleType("fake_authorized_producer")
    producer.run_science = lambda captured: calls.append(captured) or {"terminal": "fixture"}
    captured = {"frozen": b"bytes"}
    monkeypatch.setattr(adapter, "capture", lambda mode: (None, "managed", captured))
    monkeypatch.setattr(
        adapter, "load_verified_closure",
        lambda managed, observed, *, real: (
            {"producer": producer}
            if managed == "managed" and observed is captured and real is True
            else (_ for _ in ()).throw(AssertionError("wrong managed real dispatch"))
        ),
    )
    assert adapter.dispatch({}) == {"terminal": "fixture"}
    assert calls == [captured]
    with pytest.raises(adapter.AdapterError, match="absent or exactly"):
        adapter.dispatch({"BQLIB_DRYRUN": "true"})
    assert calls == [captured]
    with pytest.raises(SystemExit, match="accepts no arguments"):
        adapter.main(["unmanaged-argument"])


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
    assert adapter.EXECUTION_AUTHORIZED is True


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


def test_provenance_correction_preserves_original_and_binds_git_freeze_event() -> None:
    repair_bytes = REPAIR_AMENDMENT.read_bytes()
    repair = repair_bytes.decode()
    correction = PROVENANCE_CORRECTION.read_text()
    original_digest = hashlib.sha256(repair_bytes).hexdigest()

    assert original_digest == (
        "0c4a20b751cc05c5373b3a1d0eab95164ffc70e5dbe685cc12a9dbb341ff8301"
    )
    assert "Frozen prospectively:** 2026-09-04 05:22 UTC" in repair
    assert original_digest in correction
    assert REPAIR_COMMIT in correction
    assert REPAIR_COMMIT_TIME in correction
    assert "future-time transcription error" in correction
    assert "Git commit object and its timestamp supersede only" in correction
    assert "did not access a model checkpoint or GPU" in correction
    assert "real adapter branch remains blocked" in correction
    assert adapter.file_by_role("publication_repair_amendment").sha256 == original_digest
    assert adapter.file_by_role("publication_repair_provenance_correction").sha256 == sha256(
        PROVENANCE_CORRECTION
    )
    assert adapter.EXECUTION_AUTHORIZED is True

    committed_repair = subprocess.run(
        ["git", "show", f"{REPAIR_COMMIT}:{REPAIR_AMENDMENT.relative_to(REPO_ROOT)}"],
        cwd=REPO_ROOT, check=True, capture_output=True,
    ).stdout
    metadata = subprocess.run(
        ["git", "show", "-s", "--format=%H%n%aI%n%cI", REPAIR_COMMIT],
        cwd=REPO_ROOT, check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    assert committed_repair == repair_bytes
    assert metadata == [REPAIR_COMMIT, REPAIR_COMMIT_TIME, REPAIR_COMMIT_TIME]


def test_adapter_source_has_no_enqueue_or_outcome_read_path() -> None:
    source = adapter.ADAPTER.read_text()
    assert "circuit_battery.py" not in source
    assert "circuit_battery_tasks" not in source
    assert "queue.txt" not in source
    assert "enqueue.sh" not in source
    assert "build_authority(" not in source
    assert "_results.json" not in source
    assert "EXECUTION_AUTHORIZED = True" in source


def test_authorization_binds_complete_review_chain_and_exact_single_run_scope() -> None:
    authorization = AUTHORIZATION_AMENDMENT.read_text()
    expected = {
        "capability_preregistration": "0fea3731f59c8b9f9b1d1e898f2b4dbca65f706406b69f1b3e429e85bc621a63",
        "compiler_review": "0494f037748a5e781d038c9960875fbb1e1ee219711c78649246d402e8e6b5c4",
        "execution_amendment": "f90b0b91ee5256ed6d5962300cf8a82666efc304edbc5d273d043b623388e7e4",
        "publication_repair_amendment": "0c4a20b751cc05c5373b3a1d0eab95164ffc70e5dbe685cc12a9dbb341ff8301",
        "publication_repair_provenance_correction": "14a982abbc79de99e970dea2d352952e22e70717e7e9f677ace23370f3e7685b",
        "publication_repair_review": "6b4c526ec69342f33d731eadc34d50b78014dedc39cac9d1a2b89df02b8077b4",
        "producer": "3dcf04c0f776c056f3701967a666025ed8b63cab4d7e60a868fd766b00ac98ea",
    }
    for role, digest in expected.items():
        assert adapter.file_by_role(role).sha256 == digest
        assert digest in authorization
    assert "e722e50717962c3da0b63cf875a0ceda1872ed844bfdfaac23426c719fe77348" in authorization
    assert "15d60e1760581228b69d214ffcebebf5231a15cd5a09d018bda4bd98bae69ca5" in authorization
    assert "exactly one managed invocation" in authorization
    assert "exactly 8 native forward calls and 192 explicit row-side evaluations" in authorization
    assert "exactly 1,536 bytes" in authorization
    assert "no generated or read SELECT, TEST, or OOD rows" in authorization
    assert "all scientific projection fields remain null" in authorization.replace("\n", " ")
    assert "a tab, and its absolute path" in authorization
    assert "compile and execute those captured bytes without reopening" in authorization
    assert adapter.file_by_role("authorization_amendment").sha256 == sha256(
        AUTHORIZATION_AMENDMENT
    )
    assert sha256(REPAIR_REVIEW) == adapter.file_by_role("publication_repair_review").sha256
