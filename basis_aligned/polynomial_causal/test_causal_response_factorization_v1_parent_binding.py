import json
import os
import inspect

import pytest

import causal_response_factorization_v1_parent_binding as parent


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _protocol():
    return {
        "role": "FIT",
        "model_rows_sha256": "1" * 64,
        "fit_role_sha256": "2" * 64,
        "fit_document_ids_sha256": "3" * 64,
        "spec_order_sha256": "4" * 64,
        "support_hashes_sha256": "5" * 64,
        "rows": 496,
        "source_documents": 343,
        "positions": 256,
        "sources": 49,
        "targets": 49,
        "phases": ["full", "residual"],
        "batch_size": 4,
        "batches": 124,
        "outer_forwards": 12_400,
        "projection_event_shape": [2, 49, 124],
        "capture_event_shape": [6, 124],
        "model_dtype": "torch.float32",
        "fit_arithmetic": "CPU torch.float64 then one deploy cast",
        "artifact_order": ["authority", "bundle", "manifest", "receipt"],
        "authorized_for_eval": False,
        "authorized_for_factor_selection": False,
    }


def _fixture(tmp_path):
    paths = parent.FitParentPaths(
        authority=tmp_path / "authority.json",
        bundle=tmp_path / "bundle.pt",
        manifest=tmp_path / "manifest.json",
        receipt=tmp_path / "receipt.json",
        failure=tmp_path / "failure.json",
        terminal=tmp_path / "terminal.json",
    )
    protocol = _protocol()
    output_paths = {
        "authority": str(paths.authority),
        "bundle": str(paths.bundle),
        "manifest": str(paths.manifest),
        "receipt": str(paths.receipt),
        "failure": str(paths.failure),
        "terminal": str(paths.terminal),
        "lock": str(tmp_path / "lock"),
    }
    authority_body = {
        "schema": "causal_response_tensor_v1_fit_authority",
        "status": "frozen_before_any_parent_tensor_or_bilin18_model_load",
        "source_closure": {
            "commit": "a" * 40, "paths": {}, "sha256": "b" * 64,
        },
        "independent_audit": {
            "path": str(tmp_path / "audit.json"),
            "sha256": "c" * 64,
            "reviewer": "fixture",
        },
        "parents": {},
        "protocol": protocol,
        "output_paths": output_paths,
        "outcome_access_before_authority": {
            "parent_tensors_loaded": False,
            "model_loaded": False,
            "model_forward_calls": 0,
            "scientific_outcomes_read": False,
        },
        "authorized_for_fit_execution": True,
        "authorized_for_eval": False,
    }
    authority = {
        **authority_body,
        "authority_sha256": parent._logical_sha256(authority_body),
    }
    _write_json(paths.authority, authority)
    paths.bundle.write_bytes(b"opaque tensor bytes never deserialized by parent binding")
    authority_record, _ = parent._stable_record(paths.authority)
    bundle_record, _ = parent._stable_record(paths.bundle)
    manifest_body = {
        "schema": "causal_response_tensor_v1_fit_manifest",
        "status": "complete_fit_bundle_semantically_replayed",
        "authority_artifact_sha256": authority_record["sha256"],
        "authority_logical_sha256": authority["authority_sha256"],
        "bundle": {
            "path": str(paths.bundle),
            "sha256": bundle_record["sha256"],
            "bytes": bundle_record["bytes"],
        },
        "bundle_summary": {},
        "protocol": protocol,
        "authorized_for_eval": False,
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": parent._logical_sha256(manifest_body),
    }
    _write_json(paths.manifest, manifest)
    manifest_record, _ = parent._stable_record(paths.manifest)
    terminal = {
        "schema": "causal_response_tensor_v1_fit_terminal",
        "kind": "receipt",
        "authority_artifact_sha256": authority_record["sha256"],
        "authority_logical_sha256": authority["authority_sha256"],
        "aggregate": {
            "authority": authority_record,
            "bundle": bundle_record,
            "manifest": manifest_record,
        },
        "payload": {
            "status": "complete",
            "authorized_for_eval": False,
            "checkpoint": {},
            "model_state_sha256_before": "d" * 64,
            "model_state_sha256_after": "d" * 64,
            "outer_forwards": 12_400,
            "projection_event_shape": [2, 49, 124],
            "capture_event_shape": [6, 124],
        },
    }
    _write_json(paths.terminal, terminal)
    os.link(paths.terminal, paths.receipt)
    return paths, terminal


def test_parent_binding_replays_success_without_tensor_deserialization(tmp_path):
    paths, _ = _fixture(tmp_path)
    value = parent.fit_parent_binding_without_tensor_load(paths)
    assert value["tensor_values_deserialized"] is False
    assert value["authorized_for_eval"] is False
    assert value["receipt_sha256"] == value["terminal_sha256"]
    assert value["binding_sha256"] == parent._logical_sha256({
        key: item for key, item in value.items() if key != "binding_sha256"
    })


def test_parent_binding_module_has_no_torch_or_tensor_loader_surface():
    assert "torch" not in parent.__dict__
    source = inspect.getsource(parent)
    assert "torch.load" not in source
    assert "import torch" not in source


def test_parent_binding_rejects_failure_terminal(tmp_path):
    paths, _ = _fixture(tmp_path)
    paths.failure.write_text("failed\n")
    with pytest.raises(RuntimeError, match="ended in failure"):
        parent.fit_parent_binding_without_tensor_load(paths)


def test_parent_binding_requires_terminal_and_receipt_same_inode(tmp_path):
    paths, terminal = _fixture(tmp_path)
    paths.receipt.unlink()
    _write_json(paths.receipt, terminal)
    with pytest.raises(RuntimeError, match="same published inode"):
        parent.fit_parent_binding_without_tensor_load(paths)


def test_parent_binding_rejects_bundle_mutation(tmp_path):
    paths, _ = _fixture(tmp_path)
    paths.bundle.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bundle artifact changed"):
        parent.fit_parent_binding_without_tensor_load(paths)


def test_parent_binding_rejects_authority_semantic_mutation(tmp_path):
    paths, terminal = _fixture(tmp_path)
    authority = json.loads(paths.authority.read_text())
    authority["authorized_for_eval"] = True
    _write_json(paths.authority, authority)
    terminal["aggregate"]["authority"], _ = parent._stable_record(paths.authority)
    terminal["authority_artifact_sha256"] = terminal["aggregate"]["authority"]["sha256"]
    paths.receipt.unlink()
    _write_json(paths.terminal, terminal)
    os.link(paths.terminal, paths.receipt)
    with pytest.raises(RuntimeError, match="authority semantics"):
        parent.fit_parent_binding_without_tensor_load(paths)


def test_parent_binding_rejects_manifest_bundle_substitution(tmp_path):
    paths, terminal = _fixture(tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    manifest["bundle"]["sha256"] = "e" * 64
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = parent._logical_sha256(body)
    _write_json(paths.manifest, manifest)
    terminal["aggregate"]["manifest"], _ = parent._stable_record(paths.manifest)
    paths.receipt.unlink()
    _write_json(paths.terminal, terminal)
    os.link(paths.terminal, paths.receipt)
    with pytest.raises(RuntimeError, match="manifest identity"):
        parent.fit_parent_binding_without_tensor_load(paths)


def test_parent_binding_rejects_model_state_drift(tmp_path):
    paths, terminal = _fixture(tmp_path)
    terminal["payload"]["model_state_sha256_after"] = "f" * 64
    paths.receipt.unlink()
    _write_json(paths.terminal, terminal)
    os.link(paths.terminal, paths.receipt)
    with pytest.raises(RuntimeError, match="success receipt payload"):
        parent.fit_parent_binding_without_tensor_load(paths)


def test_parent_binding_rejects_parent_self_authorizing_selection(tmp_path):
    paths, terminal = _fixture(tmp_path)
    authority = json.loads(paths.authority.read_text())
    authority["protocol"]["authorized_for_factor_selection"] = True
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    authority["authority_sha256"] = parent._logical_sha256(body)
    _write_json(paths.authority, authority)
    terminal["aggregate"]["authority"], _ = parent._stable_record(paths.authority)
    terminal["authority_artifact_sha256"] = terminal["aggregate"]["authority"]["sha256"]
    terminal["authority_logical_sha256"] = authority["authority_sha256"]
    paths.receipt.unlink()
    _write_json(paths.terminal, terminal)
    os.link(paths.terminal, paths.receipt)
    with pytest.raises(RuntimeError, match="protocol"):
        parent.fit_parent_binding_without_tensor_load(paths)
