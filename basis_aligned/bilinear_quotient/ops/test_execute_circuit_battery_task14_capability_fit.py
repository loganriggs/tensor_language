#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial tests for the prospectively authorized task14 adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

import execute_circuit_battery_task14_capability_fit as adapter


ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
DRYRUN = ROOT / "basis_aligned/bilinear_quotient/circuit_battery_task14_capability_fit_producer_v1_dryrun.json"
AUTHORIZATION = ROOT / "basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK14_CAPABILITY_FIT_AUTHORIZATION_AMENDMENT_2026-09-04.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dryrun_exact_and_runtime_sources_excluded():
    report = adapter.dispatch({"BQLIB_DRYRUN": "1"})
    assert report["execution_authorized"] is True
    assert report["compiled_contract_sha256"] == "84f8e1cf85323dba94d13c7c716afef448b8621bff6b534c2025715420e86a82"
    assert report["call_manifest_sha256"] == "4b4da44c5090914f87d52e018bc9a8d18b74a202bdb82667283a9f1564682e0e"
    assert report["metric_manifest_sha256"] == "5da9f66829156e352afe087c75f92a7a6a37f06fe1ec5177efeffd9442609dcc"
    assert (report["completed_calls"], report["example_evaluations"]) == (8, 256)
    assert report["raw_numeric_evidence_bytes"] == 2048
    assert report["model_loaded"] is report["gpu_accessed"] is False
    assert report["model_forwards"] == report["model_backwards"] == 0
    assert report["queue_touched"] is report["publication_attempted"] is False
    assert set(report["runtime_only_roles_excluded"]) == {
        "receipt_source", "jacclust_package", "model_source", "observed_model_facade",
        "fastload_dependency", "fastload_source", "canary1_source", "canary2_source",
    }
    assert not set(report["captured_roles"]) & set(report["runtime_only_roles_excluded"])
    assert "compiler_review" in report["captured_roles"]
    assert "authorization_amendment" in report["captured_roles"]
    assert "producer_review" in report["captured_roles"]


def test_checked_in_dryrun_matches_exact_adapter_output():
    assert json.loads(DRYRUN.read_text()) == adapter.dispatch({"BQLIB_DRYRUN": "1"})


def test_subprocess_dryrun_imports_no_torch_and_has_no_cuda(monkeypatch):
    code = (
        "import runpy,sys;p=sys.argv[1];sys.argv=[p];"
        "runpy.run_path(p,run_name='__main__');"
        "raise SystemExit(97 if 'torch' in sys.modules else 0)"
    )
    environment = dict(os.environ, BQLIB_DRYRUN="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", code, str(adapter.ADAPTER)], cwd=ROOT,
        env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"execution_authorized": true' in result.stdout
    assert '"gpu_accessed": false' in result.stdout


def test_real_dispatch_delegates_once_through_verified_real_closure(monkeypatch):
    events = []
    captured = {"authorization_amendment": b"a", "producer_review": b"r"}
    producer = ModuleType("fake_producer")
    producer.run_science = lambda value: events.append(("science", value)) or {"terminal": "fake"}
    def fake_capture(mode):
        events.append(("capture", mode)); return ModuleType("compiler"), ModuleType("managed"), captured
    def fake_load(_managed, value, *, real):
        events.append(("load", real, value)); return {"producer": producer}
    monkeypatch.setattr(adapter, "capture", fake_capture)
    monkeypatch.setattr(adapter, "load_verified_closure", fake_load)
    assert adapter.dispatch({}) == {"terminal": "fake"}
    assert events == [
        ("capture", None), ("load", True, captured), ("science", captured),
    ]


def test_real_capture_includes_exact_runtime_roles_only():
    _, _, captured = adapter.capture(None)
    assert set(captured) == {item.role for item in adapter.FILES}
    assert set(adapter.REAL_LOAD_ORDER).issubset(captured)
    assert not any(item.kind == "outcome" for item in adapter.FILES)


def _fake_real_modules():
    modules = {role: ModuleType(f"fake_{role}") for role in adapter.BASE_LOAD_ORDER + adapter.REAL_LOAD_ORDER}
    model = modules["model_source"]
    modules["observed_model_facade"].TT = model
    modules["jacclust_package"].tt_model = model
    modules["producer"].capability = modules["capability_compiler"]
    modules["producer"].package = modules["artifact_package"]
    modules["producer"].framework = modules["experiment_spec"]
    exec("def load_model_fast(): return None", modules["fastload_source"].__dict__)
    exec("def load_model(): return None", modules["fastload_dependency"].__dict__)
    return modules


def test_dynamic_real_module_identity_graph(monkeypatch):
    modules = _fake_real_modules()
    names = {
        "jacclust": "jacclust_package", "jacclust.tt_model": "model_source",
        "bilin18_observed_model_facade": "observed_model_facade",
        "mlp_in_situ_usage_rank_map_probe": "fastload_dependency",
        "fastload": "fastload_source",
    }
    for name, role in names.items():
        monkeypatch.setitem(sys.modules, name, modules[role])
    adapter.validate_real_module_identities(modules)


def test_real_loader_uses_exact_base_then_runtime_order_without_disk(monkeypatch):
    modules = _fake_real_modules()
    calls = []

    class FakeManaged:
        @staticmethod
        def module_from_verified_bytes(name, _path, _payload, *, is_package=False):
            role = next(item.role for item in adapter.FILES if item.module_name == name)
            module = modules[role]
            module.__name__ = name
            module.__package__ = name if is_package else name.rpartition(".")[0]
            sys.modules[name] = module
            parent_name, _, child = name.rpartition(".")
            if parent_name:
                setattr(sys.modules[parent_name], child, module)
            calls.append(role)
            return module

    monkeypatch.setattr(adapter, "validate_captured_bytes", lambda _captured: None)
    captured = {item.role: b"captured" for item in adapter.FILES}
    loaded = adapter.load_verified_closure(FakeManaged(), captured, real=True)
    assert calls == list(adapter.BASE_LOAD_ORDER + adapter.REAL_LOAD_ORDER)
    assert loaded == modules


@pytest.mark.parametrize("name,role", (
    ("jacclust.tt_model", "model_source"),
    ("bilin18_observed_model_facade", "observed_model_facade"),
    ("mlp_in_situ_usage_rank_map_probe", "fastload_dependency"),
    ("fastload", "fastload_source"),
))
def test_dynamic_real_module_substitution_rejected(monkeypatch, name, role):
    modules = _fake_real_modules()
    bindings = {
        "jacclust": "jacclust_package", "jacclust.tt_model": "model_source",
        "bilin18_observed_model_facade": "observed_model_facade",
        "mlp_in_situ_usage_rank_map_probe": "fastload_dependency",
        "fastload": "fastload_source",
    }
    for module_name, bound_role in bindings.items():
        monkeypatch.setitem(sys.modules, module_name, modules[bound_role])
    monkeypatch.setitem(sys.modules, name, ModuleType("planted"))
    with pytest.raises(adapter.AdapterError):
        adapter.validate_real_module_identities(modules)


def test_only_exact_mode_and_zero_arguments():
    with pytest.raises(adapter.AdapterError, match="absent or exactly"):
        adapter.dispatch({"BQLIB_DRYRUN": "true"})
    with pytest.raises(SystemExit, match="accepts no arguments"):
        adapter.main(["planted"])


def test_import_cache_poisoning_replaced_by_verified_bytes(monkeypatch):
    names = (
        "circuit_experiment_spec", "circuit_artifact_package",
        "circuit_battery_integration_contract", "circuit_managed_entry",
        "circuit_battery_task14", "circuit_battery_task14_capability_fit",
        "circuit_battery_task14_capability_fit_producer",
    )
    planted = {}
    for name in names:
        module = ModuleType(name); module.planted = True
        planted[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    assert adapter.dispatch({"BQLIB_DRYRUN": "1"})["completed_calls"] == 8
    for name in names:
        assert sys.modules[name] is not planted[name]
        assert not hasattr(sys.modules[name], "planted")
    producer = sys.modules["circuit_battery_task14_capability_fit_producer"]
    assert producer.capability is sys.modules["circuit_battery_task14_capability_fit"]
    assert producer.package is sys.modules["circuit_artifact_package"]
    assert producer.framework is sys.modules["circuit_experiment_spec"]


def test_disk_module_poisoning_cannot_execute(tmp_path, monkeypatch):
    names = (
        "circuit_experiment_spec", "circuit_artifact_package",
        "circuit_battery_integration_contract", "circuit_managed_entry",
        "circuit_battery_task14", "circuit_battery_task14_capability_fit",
        "circuit_battery_task14_capability_fit_producer",
    )
    for name in names:
        (tmp_path / f"{name}.py").write_text("raise RuntimeError('disk poison')\n")
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert adapter.dispatch({"BQLIB_DRYRUN": "1"})["completed_calls"] == 8
    assert not any(str(tmp_path) in str(sys.modules[name].__file__) for name in names)


@pytest.mark.parametrize("role", (
    "capability_compiler", "task14_generator", "producer", "fit_authority",
    "compiler_review", "capability_preregistration",
    "producer_implementation_preregistration", "producer_review",
    "authorization_amendment",
))
def test_captured_source_authority_and_review_mutation_reject(role):
    _, managed, captured = adapter.capture("1")
    attack = dict(captured); attack[role] += b"\nplanted"
    with pytest.raises(adapter.AdapterError, match="captured frozen bytes changed"):
        adapter.load_verified_closure(managed, attack, real=False)


@pytest.mark.parametrize("role", ("authorization_amendment", "producer_review"))
def test_missing_authorization_closure_rejected(role):
    _, _, captured = adapter.capture("1")
    attack = dict(captured); attack.pop(role)
    with pytest.raises(adapter.AdapterError, match="review and authorization"):
        adapter.validate_captured_bytes(attack)


def test_safe_read_rejects_hash_symlink_and_directory(tmp_path):
    source = tmp_path / "source"; source.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    assert adapter.safe_read(source, expected) == b"trusted"
    with pytest.raises(adapter.AdapterError, match="changed"):
        adapter.safe_read(source, "0" * 64)
    link = tmp_path / "link"; link.symlink_to(source)
    with pytest.raises(adapter.AdapterError, match="safely open"):
        adapter.safe_read(link, expected)
    directory = tmp_path / "directory"; directory.mkdir()
    with pytest.raises(adapter.AdapterError):
        adapter.safe_read(directory, expected)


def test_every_frozen_file_and_exact_review_chain():
    assert len({x.role for x in adapter.FILES}) == len(adapter.FILES)
    assert len({x.relative_path for x in adapter.FILES}) == len(adapter.FILES)
    for item in adapter.FILES:
        assert sha(ROOT / item.relative_path) == item.sha256
    assert adapter.COMPILER_COMMIT == "fc586c1158ddeee7df8f4b502deec54189609c4c"
    assert adapter.COMPILER_REVIEW_COMMIT == "10afc5d6005d169879b07e92cb5fcb4e3a65f312"
    assert adapter.PRODUCER_BUILD_COMMIT == "26d45e89797515240eec368bc313728925d5f48a"
    assert adapter.PRODUCER_REVIEW_COMMIT == "753afa27e05b594acc39b0c1d84d72272c26e640"
    assert adapter.file_by_role("compiler_review").sha256 == \
        "a1707dd88949a9b5beb439b275e665cda1a7a62a6d5eedf076d20d192c852e59"
    assert adapter.file_by_role("producer").sha256 == \
        "9ba9448fcebcd764aa2b91e91333b3bbb2549a899b1f8304f2ce3f83bf741e3e"
    assert adapter.file_by_role("fastload_source").sha256 == \
        "5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90"
    assert adapter.file_by_role("fastload_dependency").sha256 == \
        "c701af71491d29f33f5ad691f89380a9fa7c2d86514a61fd7423ad8a78fd4d16"
    assert adapter.file_by_role("producer_review").sha256 == \
        "fddb2bac0595f733b765669cb41de1d21ad81a17205df4156505f332c0ea1ccc"
    assert adapter.file_by_role("authorization_amendment").sha256 == sha(AUTHORIZATION) == \
        "e20878d9dcbcf1c2ce0de289a6aed390b44167297a26fe89966c423a010bbee8"
    assert adapter.EXECUTION_AUTHORIZED is True


def test_authorization_binds_exact_scope_chain_and_final_review_dependency():
    note = AUTHORIZATION.read_text()
    for value in (
        adapter.COMPILER_COMMIT, adapter.COMPILER_REVIEW_COMMIT,
        adapter.PRODUCER_BUILD_COMMIT, adapter.PRODUCER_REVIEW_COMMIT,
        "9ba9448fcebcd764aa2b91e91333b3bbb2549a899b1f8304f2ce3f83bf741e3e",
        "7c0ef18db572dede3a65a355860efbc8d15787e7486c10f48e2643c0aa6f4f38",
        "fddb2bac0595f733b765669cb41de1d21ad81a17205df4156505f332c0ea1ccc",
        "84f8e1cf85323dba94d13c7c716afef448b8621bff6b534c2025715420e86a82",
        "4b4da44c5090914f87d52e018bc9a8d18b74a202bdb82667283a9f1564682e0e",
        "5da9f66829156e352afe087c75f92a7a6a37f06fe1ec5177efeffd9442609dcc",
    ):
        assert value in note
    assert "exactly one managed invocation" in note
    assert "exactly 256 unique row-side evaluations" in note
    assert "exactly 2,048 raw numeric evidence bytes" in note
    assert "no automatic retry" in note
    assert "final independent approval" in note


def test_only_fit_authority_and_runtime_fastload_closure():
    authorities = [item for item in adapter.FILES if item.kind == "authority"]
    assert [(x.role, x.relative_path) for x in authorities] == [(
        "fit_authority",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14_agreement_fit_authority.json",
    )]
    runtime = {x.role for x in adapter.FILES if not x.dryrun_access}
    assert runtime == {
        "receipt_source", "jacclust_package", "model_source", "observed_model_facade",
        "fastload_dependency", "fastload_source", "canary1_source", "canary2_source",
    }
    assert adapter.REAL_LOAD_ORDER == (
        "receipt_source", "jacclust_package", "model_source", "observed_model_facade",
        "fastload_dependency", "fastload_source",
    )


def test_source_has_no_queue_results_or_future_authority():
    source = adapter.ADAPTER.read_text()
    assert "queue.txt" not in source and "enqueue.sh" not in source
    assert "_results.json" not in source and "_evidence" not in source
    assert "build_authority(" not in source
    assert "EXECUTION_AUTHORIZED = True" in source
    assert "EXECUTION_AUTHORIZED = False" not in source
    assert not any(item.kind == "outcome" for item in adapter.FILES)
