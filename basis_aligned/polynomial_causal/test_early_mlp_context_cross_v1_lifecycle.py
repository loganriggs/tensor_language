from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import early_mlp_context_cross_v1_lifecycle as lifecycle
import compilation_mask_cut_rank_v1_gpu_adapter as inherited


def test_canonical_roles_load_with_exact_hashes_and_are_disjoint() -> None:
    roles = lifecycle.load_two_roles()
    assert roles.skip7000.wave.row_count == 192
    assert roles.skip7000.wave.document_count == 79
    assert roles.skip11000.wave.row_count == 192
    assert roles.skip11000.wave.document_count == 105
    assert len(roles.disjointness_sha256) == 64


def test_namespace_is_create_only_and_lock_owned(tmp_path) -> None:
    paths = lifecycle.output_paths(tmp_path, "safe_cross_test")
    paths.require_pristine()
    lock = lifecycle.RunLock(paths.lock)
    lock.acquire()
    try:
        lifecycle.publish_json_create_only(paths.authority, {"status": "frozen"}, lock)
        assert json.loads(paths.authority.read_text()) == {"status": "frozen"}
        with pytest.raises(Exception):
            lifecycle.publish_json_create_only(paths.authority, {"status": "changed"}, lock)
    finally:
        lock.release()
    with pytest.raises(RuntimeError, match="spent"):
        paths.require_pristine()


def test_lock_theft_fails_publication(tmp_path) -> None:
    paths = lifecycle.output_paths(tmp_path, "stolen_cross_test")
    lock = lifecycle.RunLock(paths.lock)
    lock.acquire()
    paths.lock.write_text("stolen\n")
    with pytest.raises(RuntimeError, match="ownership"):
        lifecycle.publish_json_create_only(paths.authority, {"x": 1}, lock)
    # The owner deliberately cannot release a stolen lock.
    paths.lock.unlink()


def test_source_closure_requires_launch_commit_on_origin_main(monkeypatch) -> None:
    source = inherited.SourceClosure(
        source_commit="a" * 40, path_sha256s=(("source.py", "b" * 64),),
    )
    monkeypatch.setattr(
        inherited, "committed_source_closure", lambda repo, paths: source,
    )
    observed = []

    def completed(arguments, **kwargs):
        observed.append(arguments)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(lifecycle.subprocess, "run", completed)
    assert lifecycle.committed_source_closure() is source
    assert observed[0][-2:] == (source.source_commit, "origin/main")

    monkeypatch.setattr(
        lifecycle.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"not pushed",
        ),
    )
    with pytest.raises(RuntimeError, match="origin/main"):
        lifecycle.committed_source_closure()


def test_source_closure_rehash_detects_pre_authority_drift(tmp_path, monkeypatch) -> None:
    path = tmp_path / "source.py"
    path.write_text("frozen\n")
    source = inherited.SourceClosure(
        source_commit="c" * 40,
        path_sha256s=(("source.py", lifecycle.file_sha256(path)),),
    )
    monkeypatch.setattr(lifecycle, "REPO", tmp_path)
    lifecycle.verify_source_closure(source)
    path.write_text("changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        lifecycle.verify_source_closure(source)
