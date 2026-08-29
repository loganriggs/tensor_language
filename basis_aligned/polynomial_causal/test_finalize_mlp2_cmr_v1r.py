from __future__ import annotations

import hashlib
import copy
from pathlib import Path

import pytest
import torch

import finalize_mlp2_cmr_v1r as finalizer


HASH = "a" * 64


def test_exact_role_rows_hash_metadata_is_allowed() -> None:
    finalizer.reject_forbidden_payloads({
        "role_summary": {"tensor_hashes": {"rows": HASH}}
    })


@pytest.mark.parametrize("bad", [[1], 1, "g" * 64, "a" * 63, True])
def test_allowed_path_rejects_non_sha256_values(bad: object) -> None:
    with pytest.raises(RuntimeError, match="forbidden raw payload"):
        finalizer.reject_forbidden_payloads({
            "role_summary": {"tensor_hashes": {"rows": bad}}
        })


@pytest.mark.parametrize("value", [
    {"role_summary": {"rows": HASH}}, {"score": {"rows": HASH}},
    {"rows": HASH}, {"tokens": [1]}, {"targets": [2]},
    {"raw_logits": [0.0]}, {"responses": [0.0]},
])
def test_forbidden_payloads_remain_forbidden_everywhere_else(value: object) -> None:
    with pytest.raises(RuntimeError, match="forbidden raw payload"):
        finalizer.reject_forbidden_payloads(value)


@pytest.mark.parametrize("value", [torch.tensor([1]), b"bytes", bytearray(b"x")])
def test_binary_and_tensor_payloads_are_forbidden(value: object) -> None:
    with pytest.raises(RuntimeError, match="binary/tensor"):
        finalizer.reject_forbidden_payloads({"safe": value})


def test_v1_replay_refuses_to_parse_before_v1r_authority(monkeypatch, tmp_path) -> None:
    with pytest.raises(RuntimeError, match="sealed capability"):
        finalizer.replay_v1(object())


def test_capability_cannot_be_constructed_directly() -> None:
    assert not hasattr(finalizer, "ReplayCapability")
    assert not hasattr(finalizer, "_CAPABILITY_KEY")


def test_frozen_v1_snapshot_and_receipt_absence_match() -> None:
    assert finalizer.current_protected() == (
        finalizer.EXPECTED_V1, finalizer.EXPECTED_PARENTS, False,
    )


def test_source_declares_distinct_create_only_namespace_and_no_model_path() -> None:
    source = finalizer.Path(finalizer.__file__).read_text()
    assert "mlp2_cmr_v1r_finalization_authority.json" in source
    assert "model_access_authorized\": False" in source
    assert "row_deserialization_authorized\": False" in source
    assert "replication_access_authorized\": False" in source
    assert "write_create_only_guarded(RECEIPT" in source
    assert "RECEIPT.exists() or FAILURE.exists()" in source


def _fake_main_environment(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    v1_paths = {}
    expected_v1 = {}
    for name in finalizer.EXPECTED_V1:
        path = tmp_path / f"v1_{name}"
        data = f"fixed-{name}".encode()
        path.write_bytes(data)
        v1_paths[name] = path
        expected_v1[name] = hashlib.sha256(data).hexdigest()
    parent_paths = {}
    expected_parents = {}
    for name in finalizer.EXPECTED_PARENTS:
        path = tmp_path / f"parent_{name}"
        data = f"fixed-parent-{name}".encode()
        path.write_bytes(data)
        parent_paths[name] = path
        expected_parents[name] = hashlib.sha256(data).hexdigest()
    outputs = {
        "AUTHORITY": tmp_path / "v1r_authority.json",
        "RESULT": tmp_path / "v1r_result.json",
        "RECEIPT": tmp_path / "v1r_receipt.json",
        "FAILURE": tmp_path / "v1r_failure.json",
        "LOCK": tmp_path / ".v1r.lock",
        "V1_RECEIPT": tmp_path / "absent_v1_receipt.json",
    }
    monkeypatch.setattr(finalizer, "V1_PATHS", v1_paths)
    monkeypatch.setattr(finalizer, "PARENT_PATHS", parent_paths)
    monkeypatch.setattr(finalizer, "EXPECTED_V1", expected_v1)
    monkeypatch.setattr(finalizer, "EXPECTED_PARENTS", expected_parents)
    for name, path in outputs.items():
        monkeypatch.setattr(finalizer, name, path)
    monkeypatch.setattr(finalizer, "committed_source", lambda: ("c" * 40, {}))
    score = {"validation_passed": False, "replication_authorized": False}
    def replay(capability):
        finalizer.consume_replay_capability(capability)
        return {"score": score}, score
    monkeypatch.setattr(finalizer, "replay_v1", replay)
    return outputs


def test_mocked_main_success_is_receipt_last(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    finalizer.main()
    assert out["AUTHORITY"].exists() and out["RESULT"].exists()
    assert out["RECEIPT"].exists() and not out["FAILURE"].exists()


def test_replay_capability_is_one_use(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    nonce = "n"
    finalizer.write_create_only_guarded(
        out["LOCK"], finalizer.canonical_json_bytes({
            "experiment_id": "bilin18_mlp2_cmr_v1r_finalization", "nonce": nonce,
        }), before_link=lambda: None,
    )
    stat = out["LOCK"].stat(follow_symlinks=False)
    inode = (stat.st_dev, stat.st_ino)
    source_commit = "c" * 40
    authority = finalizer.canonical_authority(source_commit, {})
    finalizer.write_create_only_guarded(
        out["AUTHORITY"], finalizer.canonical_json_bytes(authority),
        before_link=lambda: None,
    )
    authority_hash = finalizer.file_sha256(out["AUTHORITY"])
    capability = finalizer.mint_replay_capability(
        authority, authority_hash, source_commit, {}, nonce, inode,
    )
    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.copy(capability)
    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.deepcopy(capability)
    finalizer.consume_replay_capability(capability)
    with pytest.raises(RuntimeError, match="already consumed"):
        finalizer.consume_replay_capability(capability)


def test_mint_rejects_every_noncanonical_authority(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    nonce = "n"
    finalizer.write_create_only_guarded(
        out["LOCK"], finalizer.canonical_json_bytes({
            "experiment_id": "bilin18_mlp2_cmr_v1r_finalization", "nonce": nonce,
        }), before_link=lambda: None,
    )
    stat = out["LOCK"].stat(follow_symlinks=False)
    inode = (stat.st_dev, stat.st_ino)
    commit = "c" * 40
    canonical = finalizer.canonical_authority(commit, {})
    variants = [
        {"exact": True},
        {**canonical, "source_commit": "d" * 40},
        {**canonical, "v1_artifacts": {}},
        {**canonical, "v1_parents": {}},
        {**canonical, "model_access_authorized": True},
        {**canonical, "authorized_outputs": ["wrong.json"]},
    ]
    for authority in variants:
        out["AUTHORITY"].unlink(missing_ok=True)
        finalizer.write_create_only_guarded(
            out["AUTHORITY"], finalizer.canonical_json_bytes(authority),
            before_link=lambda: None,
        )
        with pytest.raises(RuntimeError, match="exact canonical authority"):
            finalizer.mint_replay_capability(
                authority, finalizer.file_sha256(out["AUTHORITY"]), commit, {}, nonce, inode,
            )


def test_mint_rejects_self_consistent_caller_chosen_source(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    nonce = "n"
    finalizer.write_create_only_guarded(
        out["LOCK"], finalizer.canonical_json_bytes({
            "experiment_id": "bilin18_mlp2_cmr_v1r_finalization", "nonce": nonce,
        }), before_link=lambda: None,
    )
    stat = out["LOCK"].stat(follow_symlinks=False)
    inode = (stat.st_dev, stat.st_ino)
    caller_commit = "c" * 40
    authority = finalizer.canonical_authority(caller_commit, {})
    finalizer.write_create_only_guarded(
        out["AUTHORITY"], finalizer.canonical_json_bytes(authority),
        before_link=lambda: None,
    )
    monkeypatch.setattr(finalizer, "committed_source", lambda: ("d" * 40, {"real": HASH}))
    with pytest.raises(RuntimeError, match="independently committed source identity"):
        finalizer.mint_replay_capability(
            authority, finalizer.file_sha256(out["AUTHORITY"]), caller_commit, {}, nonce, inode,
        )


def test_mocked_main_failure_is_failure_only(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    def fail(capability):
        finalizer.consume_replay_capability(capability)
        raise RuntimeError("planned replay failure")
    monkeypatch.setattr(finalizer, "replay_v1", fail)
    with pytest.raises(RuntimeError, match="planned replay failure"):
        finalizer.main()
    assert out["FAILURE"].exists() and not out["RECEIPT"].exists()


def test_receipt_loses_to_late_failure_rival(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    original = finalizer.write_create_only_guarded
    def rival(path, data, *, before_link):
        if path == out["RECEIPT"]:
            out["FAILURE"].write_text("rival")
        return original(path, data, before_link=before_link)
    monkeypatch.setattr(finalizer, "write_create_only_guarded", rival)
    with pytest.raises(RuntimeError, match="terminal namespace"):
        finalizer.main()
    assert out["FAILURE"].exists() and not out["RECEIPT"].exists()


def test_failure_loses_to_late_receipt_rival(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    def fail(capability):
        finalizer.consume_replay_capability(capability)
        raise RuntimeError("planned replay failure")
    monkeypatch.setattr(finalizer, "replay_v1", fail)
    original = finalizer.write_create_only_guarded
    def rival(path, data, *, before_link):
        if path == out["FAILURE"]:
            out["RECEIPT"].write_text("rival")
        return original(path, data, before_link=before_link)
    monkeypatch.setattr(finalizer, "write_create_only_guarded", rival)
    with pytest.raises(RuntimeError, match="planned replay failure"):
        finalizer.main()
    assert out["RECEIPT"].exists() and not out["FAILURE"].exists()


def test_late_v1_drift_blocks_both_terminals(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    original = finalizer.write_create_only_guarded
    def drift(path, data, *, before_link):
        if path == out["RECEIPT"]:
            finalizer.V1_PATHS["result"].write_text("drift")
        return original(path, data, before_link=before_link)
    monkeypatch.setattr(finalizer, "write_create_only_guarded", drift)
    with pytest.raises(RuntimeError, match="protected v1 snapshot"):
        finalizer.main()
    assert not out["RECEIPT"].exists() and not out["FAILURE"].exists()


def test_late_lock_drift_blocks_both_terminals(monkeypatch, tmp_path: Path) -> None:
    out = _fake_main_environment(monkeypatch, tmp_path)
    original = finalizer.write_create_only_guarded
    def drift(path, data, *, before_link):
        if path == out["RECEIPT"]:
            out["LOCK"].write_text("drift")
        return original(path, data, before_link=before_link)
    monkeypatch.setattr(finalizer, "write_create_only_guarded", drift)
    with pytest.raises(Exception):
        finalizer.main()
    assert not out["RECEIPT"].exists() and not out["FAILURE"].exists()
