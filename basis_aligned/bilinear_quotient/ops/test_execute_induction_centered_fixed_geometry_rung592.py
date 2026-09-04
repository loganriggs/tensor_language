#!/usr/bin/env python3
# BQLANE: cpu
"""Managed-boundary tests for the prospective R592 preflight adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import pytest


MODULE = Path(__file__).with_name("execute_induction_centered_fixed_geometry_rung592.py")
SPEC = importlib.util.spec_from_file_location("r592_adapter", MODULE)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def test_frozen_bytes_and_model_free_preflight(tmp_path: Path) -> None:
    assert adapter.verify_frozen_bytes() == {
        str(path): digest for path, digest in adapter.FROZEN_HASHES.items()
    }
    report = adapter.preflight(namespace_paths=(tmp_path / "unused",))
    assert report["registered_fit_forwards"] == 639
    assert report["registered_select_forwards"] == 322
    assert report["registered_max_forwards"] == 961
    assert report["capacity_preflight"]["required_free_bytes"] == 9_000_000_000
    assert report["model_forwards"] == report["model_backwards"] == 0
    assert report["model_weights_updated"] is False
    assert report["select_opened"] is report["final_opened"] is report["ood_opened"] is False


def test_occupied_any_normal_or_invalid_namespace_blocks(tmp_path: Path) -> None:
    for name in ("result.json", "invalid_receipt.json", "evidence"):
        path = tmp_path / name
        path.mkdir() if name == "evidence" else path.write_text("occupied")
        with pytest.raises(RuntimeError, match="already exists"):
            adapter.require_unused_namespaces((path,))


def test_hash_mutation_blocks_before_import(tmp_path: Path) -> None:
    path = tmp_path / "candidate.py"
    path.write_text("pass\n")
    with pytest.raises(RuntimeError, match="changed or missing"):
        adapter.verify_frozen_bytes({path: "0" * 64})


def test_real_mode_dispatches_only_embedded_verified_producer(tmp_path: Path) -> None:
    called = {}
    def fake_exec(executable, argv):
        called["executable"] = executable; called["argv"] = argv
        return None
    with pytest.raises(RuntimeError, match="unexpectedly returned"):
        adapter.dispatch({}, exec_function=fake_exec, namespace_paths=(tmp_path / "unused",))
    assert called["executable"] == called["argv"][0]
    assert called["argv"][1:3] == ["-I", "-c"]
    assert "base64.b64decode" in called["argv"][3]
    assert "read_bytes" not in called["argv"][3]


def test_invalid_dryrun_environment_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="must be absent"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "true"},
            exec_function=lambda *_: None,
            namespace_paths=(tmp_path / "unused",),
        )


def test_dryrun_matches_committed_bytes() -> None:
    observed = adapter.run_model_free_validation()
    assert observed["status"] == "prospective_model_free_only"
    assert observed["scientific_terminal"] is None


def test_insufficient_capacity_blocks_before_dispatch(tmp_path: Path) -> None:
    low = lambda _path: types.SimpleNamespace(f_bavail=8_999_999_999, f_frsize=1)
    with pytest.raises(RuntimeError, match="insufficient free space"):
        adapter.preflight(
            namespace_paths=(tmp_path / "unused",), capacity_path=tmp_path,
            statvfs_function=low,
        )
