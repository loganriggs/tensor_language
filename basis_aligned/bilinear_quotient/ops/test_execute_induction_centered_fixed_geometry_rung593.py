#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free managed-adapter tests for prospective R593."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path
import types

import pytest


MODULE = Path(__file__).with_name("execute_induction_centered_fixed_geometry_rung593.py")
SPEC = importlib.util.spec_from_file_location("execute_r593_test", MODULE)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def stat(available: int):
    return lambda _path: types.SimpleNamespace(f_bavail=available, f_frsize=1)


def test_exact_frozen_bytes_and_model_free_preflight() -> None:
    assert all("TO_FREEZE" not in digest for digest in adapter.FROZEN_HASHES.values())
    observed = adapter.verify_frozen_bytes()
    assert observed == {str(path): digest for path, digest in adapter.FROZEN_HASHES.items()}
    report = adapter.preflight(
        namespace_paths=(), capacity_path=Path("."),
        statvfs_function=stat(adapter.MINIMUM_FREE_BYTES),
    )
    assert report["registered_fit_forwards"] == 639
    assert report["registered_select_forwards"] == 322
    assert report["registered_max_forwards"] == 961
    assert report["capacity_thresholds"] == {
        "before_model": 9_455_639_040,
        "before_select_after_fit": 3_954_175_488,
        "fit_canonical_data_bytes": 5_501_463_552,
        "remaining_select_plus_chunk_bytes": 2_794_172_416,
        "safety_margin_bytes": 1_160_003_072,
    }
    assert report["model_forwards"] == report["model_backwards"] == 0
    assert report["model_weights_updated"] is False
    assert not any(report[key] for key in ("select_opened", "final_opened", "ood_opened"))


def test_all_six_r593_namespaces_block_dispatch(tmp_path: Path) -> None:
    for occupied_index in range(len(adapter.OUTCOME_NAMESPACES)):
        paths = [tmp_path / f"namespace-{index}" for index in range(len(adapter.OUTCOME_NAMESPACES))]
        paths[occupied_index].touch()
        with pytest.raises(RuntimeError, match="namespace already exists"):
            adapter.preflight(
                namespace_paths=paths, capacity_path=tmp_path,
                statvfs_function=stat(adapter.MINIMUM_FREE_BYTES),
            )


def test_one_byte_below_capacity_fails_before_dispatch() -> None:
    called = []
    with pytest.raises(RuntimeError, match="insufficient free space before model boundary"):
        adapter.dispatch(
            {}, exec_function=lambda *_args: called.append(True), namespace_paths=(),
            capacity_path=Path("."), statvfs_function=stat(adapter.MINIMUM_FREE_BYTES - 1),
        )
    assert called == []


def test_dryrun_and_invalid_environment_are_model_free() -> None:
    report = adapter.dispatch(
        {"BQLIB_DRYRUN": "1"}, namespace_paths=(), capacity_path=Path("."),
        statvfs_function=stat(adapter.MINIMUM_FREE_BYTES),
    )
    assert report["mode"] == "model_free_dryrun"
    with pytest.raises(RuntimeError, match="must be absent or exactly '1'"):
        adapter.dispatch(
            {"BQLIB_DRYRUN": "0"}, namespace_paths=(), capacity_path=Path("."),
            statvfs_function=stat(adapter.MINIMUM_FREE_BYTES),
        )


def test_real_dispatch_embeds_only_verified_producer_and_both_provenance_hashes() -> None:
    captured = []
    with pytest.raises(RuntimeError, match="unexpectedly returned"):
        adapter.dispatch(
            {}, exec_function=lambda executable, argv: captured.append((executable, argv)),
            namespace_paths=(), capacity_path=Path("."),
            statvfs_function=stat(adapter.MINIMUM_FREE_BYTES),
        )
    assert len(captured) == 1
    assert captured[0][0] == captured[0][1][0]


def test_scientific_command_contains_exact_bytes_and_provenance() -> None:
    executable, argv = adapter.scientific_command()
    assert argv[:3] == [executable, "-I", "-c"]
    launcher = argv[3]
    source = adapter.PRODUCER.read_bytes()
    encoded = base64.b64encode(source).decode("ascii")
    assert encoded in launcher
    assert adapter.FROZEN_HASHES[adapter.PRODUCER] in launcher
    assert hashlib.sha256(MODULE.read_bytes()).hexdigest() in launcher
    assert "__r593_immutable_sha256__" in launcher
    assert "__r593_adapter_sha256__" in launcher


def test_hash_mutation_fails_before_import(tmp_path: Path) -> None:
    altered = tmp_path / "producer.py"
    altered.write_bytes(adapter.PRODUCER.read_bytes() + b"\n# mutation\n")
    with pytest.raises(RuntimeError, match="changed or missing"):
        adapter.verify_frozen_bytes({altered: adapter.FROZEN_HASHES[adapter.PRODUCER]})
