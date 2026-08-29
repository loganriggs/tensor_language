from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import freeze_gauge_transport_triangle_unique_rows_v1 as v1
import freeze_gauge_transport_triangle_unique_rows_v2 as v2


def test_raw_and_composite_hash_known_answer_are_distinct():
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    assert v2.raw_tensor_sha256(tensor) == (
        "73e200e2b048c86d4e8c86b86bf62bbda84c7384e34e250b01aa30ab29d234a4"
    )
    assert v1.tensor_sha256(tensor) == (
        "b6b2a4005eaebba5b51bd452db94fc271ec88ee459b92588903de5bf4e9de67e"
    )
    assert v2.raw_tensor_sha256(tensor) != v1.tensor_sha256(tensor)


def test_source_cache_loader_uses_parent_raw_hash_not_output_composite_hash(tmp_path: Path):
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    cache = tmp_path / "cache.pt"
    torch.save(tensor, cache)
    binding = {
        "path": str(cache),
        "shape": [2, 2],
        "dtype": "torch.int64",
        "file_sha256": v1.file_sha256(cache),
        "tensor_raw_sha256": v2.raw_tensor_sha256(tensor),
    }
    loaded = v2.load_cache_tensors({"cache_bindings": {"known": binding}})
    assert torch.equal(loaded["known"], tensor)

    wrong_currency = copy.deepcopy(binding)
    wrong_currency["tensor_raw_sha256"] = v1.tensor_sha256(tensor)
    with pytest.raises(RuntimeError, match="source-cache replay changed"):
        v2.load_cache_tensors({"cache_bindings": {"known": wrong_currency}})


def test_v2_metadata_rebuild_binds_exact_spent_v1_plan_without_tensor_load(monkeypatch):
    def forbidden_tensor_load(*_args, **_kwargs):
        raise AssertionError("metadata recovery must not load a row tensor")

    monkeypatch.setattr(torch, "load", forbidden_tensor_load)
    v1_authority, failure = v2.load_v1_parents()
    parent = v2.load_parent_metadata()
    source = {"commit": "synthetic", "paths": {}, "sha256": "s" * 64}
    authority = v2.build_authority(source, parent, v1_authority)
    assert failure["status"] == "terminal_failure_no_receipt"
    assert authority["selection_plan"] == v1_authority["selection_plan"]
    assert authority["selection_plan"]["selection_plan_sha256"] == v2.SELECTION_PLAN_SHA256
    assert authority["hash_protocols"] == {
        "source_cache_tensor": "sha256(contiguous_cpu_tensor_raw_bytes_only)",
        "output_role_tensor": (
            "sha256(dtype_utf8_then_shape_json_then_contiguous_cpu_raw_bytes)"
        ),
        "v2_delta_from_v1": "source_cache_validation_uses_parent_raw_byte_hash",
    }
    for key, binding in authority["cache_bindings"].items():
        assert binding["tensor_raw_sha256"] == parent["entries"][key]["tensor_raw_sha256"]
        assert "tensor_sha256" not in binding
    assert authority["outputs"]["authority"].endswith("unique_rows_v2_authority.json")


def test_v2_build_rejects_even_self_consistent_selection_change():
    v1_authority, _ = v2.load_v1_parents()
    changed_parent = copy.deepcopy(v2.load_parent_metadata())
    first = changed_parent["document_provenance"]["sets"]["n480_skip80"]
    first[0], first[2] = first[2], first[0]
    source = {"commit": "synthetic", "paths": {}, "sha256": "s" * 64}
    with pytest.raises(RuntimeError, match="not exactly the spent v1 selection"):
        v2.build_authority(source, changed_parent, v1_authority)


def test_v2_namespace_adapter_restores_every_v1_global_after_exception():
    names = (
        "PREREG", "RUNNER", "TEST", "SOURCE_FILES", "AUTHORITY", "ROWS",
        "MANIFEST", "RECEIPT", "FAILURE", "LOCK",
    )
    before = {name: getattr(v1, name) for name in names}
    with pytest.raises(RuntimeError, match="synthetic"):
        with v2._v2_base_namespace():
            assert v1.AUTHORITY == v2.AUTHORITY
            assert v1.LOCK == v2.LOCK
            raise RuntimeError("synthetic")
    assert {name: getattr(v1, name) for name in names} == before


def test_v2_manifest_retains_composite_output_hash(monkeypatch, tmp_path: Path):
    rows_file = tmp_path / "rows.pt"
    rows_file.write_bytes(b"synthetic-v2-rows")
    monkeypatch.setattr(v2, "ROWS", rows_file)
    monkeypatch.setattr(v1, "ROLE_SIZES", {"basis": 1})
    monkeypatch.setattr(v1, "EXPECTED_CONTRIBUTIONS", {"basis": {"cache": 1}})
    tensor = torch.tensor([[1, 2]], dtype=torch.int64)
    authority = {
        "authority_sha256": "a" * 64,
        "hash_protocols": {
            "source_cache_tensor": "raw",
            "output_role_tensor": "composite",
            "v2_delta_from_v1": "raw source only",
        },
        "permissions": {"conditional_future_row_eligibility": "separate runner required"},
    }
    payload = {
        "roles": {"basis": tensor},
        "records": {"basis": [{"document_id": "one"}]},
    }
    manifest = v2.build_manifest(payload, authority)
    assert manifest["role_tensor_composite_sha256s"]["basis"] == v1.tensor_sha256(tensor)
    assert manifest["role_tensor_composite_sha256s"]["basis"] != v2.raw_tensor_sha256(tensor)


def test_v2_parent_files_are_exact_and_v1_terminal_outputs_remain_absent():
    authority, failure = v2.load_v1_parents()
    assert v1.file_sha256(v2.V1_AUTHORITY) == v2.V1_AUTHORITY_FILE_SHA256
    assert v1.file_sha256(v2.V1_FAILURE) == v2.V1_FAILURE_FILE_SHA256
    assert authority["authority_sha256"] == v2.V1_AUTHORITY_SHA256
    assert failure["receipt_exists"] is False
    assert all(not Path(authority["outputs"][key]).exists() for key in (
        "rows", "manifest", "receipt",
    ))
