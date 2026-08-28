from __future__ import annotations

import json

import pytest

import early_mlp_context_cross_v1_lifecycle as lifecycle


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
