#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free managed-adapter tests for prospective R593."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
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
    captured: list[tuple[str, list[str], int]] = []
    def returning(executable, argv):
        captured.append((executable, argv, int(argv[-1])))
    with pytest.raises(RuntimeError, match="unexpectedly returned"):
        adapter.dispatch(
            {}, exec_function=returning,
            namespace_paths=(), capacity_path=Path("."),
            statvfs_function=stat(adapter.MINIMUM_FREE_BYTES),
        )
    assert len(captured) == 1
    assert captured[0][0] == captured[0][1][0]
    with pytest.raises(OSError):
        os.fstat(captured[0][2])


def test_scientific_command_uses_exact_fully_sealed_memfd_and_small_argv() -> None:
    executable, argv, descriptor = adapter.scientific_command()
    try:
        assert argv[:3] == [executable, "-I", "-c"]
        assert all(len(argument.encode("utf-8")) < 4096 for argument in argv)
        assert max(map(len, argv)) < 4096
        source = adapter.PRODUCER.read_bytes()
        assert os.pread(descriptor, len(source) + 1, 0) == source
        assert fcntl.fcntl(descriptor, adapter.F_GET_SEALS) == adapter.FULL_SEAL_MASK
        assert os.get_inheritable(descriptor)
        with pytest.raises(OSError):
            os.write(descriptor, b"x")
        launcher = argv[3]
        assert adapter.FROZEN_HASHES[adapter.PRODUCER] in launcher
        assert hashlib.sha256(MODULE.read_bytes()).hexdigest() in launcher
        assert "__r593_immutable_sha256__" in launcher
        assert "__r593_adapter_sha256__" in launcher
        assert "base64" not in launcher
    finally:
        os.close(descriptor)


def test_canonical_constants_and_forced_glibc_fallback() -> None:
    assert adapter.MFD_ALLOW_SEALING == 0x0002
    assert adapter.F_ADD_SEALS == 1033 and adapter.F_GET_SEALS == 1034
    assert adapter.FULL_SEAL_MASK == 0x0001 | 0x0002 | 0x0004 | 0x0008
    descriptor = adapter.linux_memfd_create("r593-fallback-test", os_function=None)
    try:
        assert os.fstat(descriptor).st_size == 0
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("mutation", ("truncated", "appended", "wrong_digest"))
def test_harmless_child_rejects_length_or_digest_corruption(mutation: str) -> None:
    registered = b"print('FIXTURE_EXECUTED')\n"
    payload = {
        "truncated": registered[:-1],
        "appended": registered + b"#extra\n",
        "wrong_digest": registered,
    }[mutation]
    digest = hashlib.sha256(registered).hexdigest()
    if mutation == "wrong_digest":
        digest = "0" * 64
    executable, argv, descriptor = adapter.sealed_python_command(
        payload, logical_path="r593_harmless_fixture.py",
        expected_length=len(registered), expected_sha256=digest,
        adapter_sha256="a" * 64,
    )
    try:
        completed = subprocess.run(
            argv, pass_fds=(descriptor,), capture_output=True, text=True, check=False,
        )
    finally:
        os.close(descriptor)
    assert executable == argv[0]
    assert completed.returncode != 0
    assert "FIXTURE_EXECUTED" not in completed.stdout
    assert "sealed producer length/hash mismatch" in completed.stderr


def test_harmless_child_executes_exact_fixture_and_closes_child_fd() -> None:
    source = b"""import os
links=[]
for name in os.listdir('/proc/self/fd'):
 try: links.append(os.readlink('/proc/self/fd/'+name))
 except FileNotFoundError: pass
assert not any('memfd:r593-immutable-producer' in link for link in links)
print('FIXTURE_EXECUTED', end='')
"""
    digest = hashlib.sha256(source).hexdigest()
    executable, argv, descriptor = adapter.sealed_python_command(
        source, logical_path="r593_harmless_fixture.py",
        expected_length=len(source), expected_sha256=digest,
        adapter_sha256="0" * 64,
    )
    try:
        completed = subprocess.run(
            argv, pass_fds=(descriptor,), capture_output=True, text=True, check=False,
        )
    finally:
        os.close(descriptor)
    assert executable == argv[0]
    assert completed.returncode == 0
    assert completed.stdout == "FIXTURE_EXECUTED"


def test_failing_exec_function_closes_descriptor() -> None:
    observed = []
    def failing(_executable, argv):
        observed.append(int(argv[-1]))
        raise OSError("planted exec failure")
    with pytest.raises(OSError, match="planted exec failure"):
        adapter.dispatch(
            {}, exec_function=failing, namespace_paths=(), capacity_path=Path("."),
            statvfs_function=stat(adapter.MINIMUM_FREE_BYTES),
        )
    with pytest.raises(OSError):
        os.fstat(observed[0])


def test_hash_mutation_fails_before_import(tmp_path: Path) -> None:
    altered = tmp_path / "producer.py"
    altered.write_bytes(adapter.PRODUCER.read_bytes() + b"\n# mutation\n")
    with pytest.raises(RuntimeError, match="changed or missing"):
        adapter.verify_frozen_bytes({altered: adapter.FROZEN_HASHES[adapter.PRODUCER]})
