from __future__ import annotations

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
    monkeypatch.setattr(finalizer, "AUTHORITY", tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="authority must exist"):
        finalizer.replay_v1()


def test_frozen_v1_snapshot_and_receipt_absence_match() -> None:
    assert finalizer.current_protected() == (
        finalizer.EXPECTED_V1, finalizer.EXPECTED_PARENTS, False,
    )


def test_source_declares_distinct_create_only_namespace_and_no_model_path() -> None:
    source = finalizer.Path(finalizer.__file__).read_text()
    assert "mlp2_cmr_v1r_finalization_authority.json" in source
    assert "model_access_authorized\": False" in source
    assert "row_access_authorized\": False" in source
    assert "replication_access_authorized\": False" in source
    assert "write_create_only_guarded(RECEIPT" in source
    assert "RECEIPT.exists() or FAILURE.exists()" in source

