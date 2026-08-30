import json
from pathlib import Path

import pytest
import torch

import causal_response_factorization_v1_parent_binding as parent
import causal_response_factorization_v1_training_loader as loader
import causal_response_tensor_v1_fit_bundle as fit_bundle
from test_causal_response_tensor_v1_fit_bundle import _payload


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _fixture(tmp_path, monkeypatch):
    paths = parent.FitParentPaths(
        authority=tmp_path / "fit_authority.json",
        bundle=tmp_path / "fit_bundle.pt",
        manifest=tmp_path / "fit_manifest.json",
        receipt=tmp_path / "fit_receipt.json",
        failure=tmp_path / "fit_failure.json",
        terminal=tmp_path / "fit_terminal.json",
        lock=tmp_path / "fit.lock",
    )
    payload = _payload()
    torch.save(payload, paths.bundle)
    bundle_record, _ = parent._stable_record(paths.bundle)
    authority_logical = payload["binding"]["authority_sha256"]
    summary = fit_bundle.fit_bundle_manifest_summary(
        paths.bundle,
        expected_authority_sha256=authority_logical,
        expected_artifact_sha256=bundle_record["sha256"],
        require_production=False,
    )
    manifest = {"bundle_summary": summary}
    _write_json(paths.manifest, manifest)
    manifest_record, _ = parent._stable_record(paths.manifest)
    receipt = {
        "payload": {
            "status": "complete",
            "authorized_for_eval": False,
            "model_state_sha256_before": payload["binding"]["model_state_sha256_before"],
            "model_state_sha256_after": payload["binding"]["model_state_sha256_after"],
            "outer_forwards": payload["call_ledger"]["outer_forwards"],
            "projection_event_shape": [2, 49, 124],
            "capture_event_shape": [6, 124],
            "checkpoint": {
                "config_sha256": payload["binding"]["config_sha256"],
                "weights_sha256": payload["binding"]["weights_sha256"],
            },
        }
    }
    _write_json(paths.receipt, receipt)
    receipt_record, _ = parent._stable_record(paths.receipt)
    parent_body = {
        "schema": "causal_response_factorization_v1_fit_parent_binding",
        "receipt_sha256": receipt_record["sha256"],
        "terminal_sha256": receipt_record["sha256"],
        "authority_artifact_sha256": "2" * 64,
        "authority_logical_sha256": authority_logical,
        "bundle_sha256": bundle_record["sha256"],
        "bundle_bytes": bundle_record["bytes"],
        "manifest_artifact_sha256": manifest_record["sha256"],
        "manifest_logical_sha256": "5" * 64,
        "source_closure_sha256": payload["binding"]["source_closure_sha256"],
        "fit_protocol": {},
        "tensor_values_deserialized": False,
        "authorized_for_eval": False,
    }
    parent_binding = {
        **parent_body, "binding_sha256": parent._logical_sha256(parent_body)
    }
    authority_body = {
        "schema": "causal_response_factorization_v1_training_authority",
        "status": "frozen_before_fit_bundle_tensor_deserialization",
        "source_closure": {},
        "independent_audit": {},
        "parent_binding_sha256": parent_binding["binding_sha256"],
        "protocol": {
            "role": "FIT_TRAINING",
            "training_documents": 229,
            "validation_documents_exposed": 0,
            "eval_documents_exposed": 0,
        },
        "output_paths": {},
        "outcome_access_before_authority": {
            "fit_bundle_deserialized": False,
            "fit_response_values_read": False,
            "validation_values_read": False,
            "eval_values_read": False,
        },
        "authorized_for_training_input": True,
        "authorized_for_validation": False,
        "authorized_for_eval": False,
    }
    authority = {
        **authority_body, "authority_sha256": parent._logical_sha256(authority_body)
    }
    monkeypatch.setattr(
        parent, "fit_parent_binding_without_tensor_load", lambda _paths: parent_binding
    )
    return paths, parent_binding, authority, payload


def test_one_use_loader_exposes_only_training_role_and_exact_artifacts(
    tmp_path, monkeypatch,
):
    paths, binding, authority, _ = _fixture(tmp_path, monkeypatch)
    capability = loader.OneUseFitTrainingLoader(
        paths, require_production=False, train_documents=3
    )
    result = capability.load_once(
        parent_binding=binding, analysis_authority=authority
    )
    assert capability.spent is True
    assert result.response.shape[-1] == 3
    assert result.document_ids.numel() == 3
    assert result.artifacts.bundle_sha256 == binding["bundle_sha256"]
    assert not hasattr(result, "validation_response")
    with pytest.raises(RuntimeError, match="already spent"):
        capability.load_once(parent_binding=binding, analysis_authority=authority)


def test_loader_poisoned_even_when_first_attempt_fails(tmp_path, monkeypatch):
    paths, binding, authority, _ = _fixture(tmp_path, monkeypatch)
    capability = loader.OneUseFitTrainingLoader(
        paths, require_production=False, train_documents=3
    )
    bad = dict(authority)
    bad["authority_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="logical identity"):
        capability.load_once(parent_binding=binding, analysis_authority=bad)
    assert capability.spent is True
    with pytest.raises(RuntimeError, match="already spent"):
        capability.load_once(parent_binding=binding, analysis_authority=authority)


def test_loader_rejects_bundle_changed_after_parent_binding(tmp_path, monkeypatch):
    paths, binding, authority, _ = _fixture(tmp_path, monkeypatch)
    paths.bundle.write_bytes(b"changed after authority")
    capability = loader.OneUseFitTrainingLoader(
        paths, require_production=False, train_documents=3
    )
    with pytest.raises(RuntimeError, match="bundle bytes differ"):
        capability.load_once(parent_binding=binding, analysis_authority=authority)


def test_loader_rejects_manifest_summary_substitution(tmp_path, monkeypatch):
    paths, binding, authority, _ = _fixture(tmp_path, monkeypatch)
    manifest = json.loads(paths.manifest.read_text())
    manifest["bundle_summary"]["axes"]["model_width"] += 1
    _write_json(paths.manifest, manifest)
    binding = dict(binding)
    binding["manifest_artifact_sha256"] = parent._stable_record(paths.manifest)[0]["sha256"]
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding["binding_sha256"] = parent._logical_sha256(body)
    authority = dict(authority)
    authority["parent_binding_sha256"] = binding["binding_sha256"]
    authority_body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    authority["authority_sha256"] = parent._logical_sha256(authority_body)
    monkeypatch.setattr(
        parent, "fit_parent_binding_without_tensor_load", lambda _paths: binding
    )
    capability = loader.OneUseFitTrainingLoader(
        paths, require_production=False, train_documents=3
    )
    with pytest.raises(RuntimeError, match="manifest summary"):
        capability.load_once(parent_binding=binding, analysis_authority=authority)


def test_loader_rejects_receipt_model_state_or_checkpoint_mismatch(
    tmp_path, monkeypatch,
):
    paths, binding, authority, _ = _fixture(tmp_path, monkeypatch)
    receipt = json.loads(paths.receipt.read_text())
    receipt["payload"]["model_state_sha256_after"] = "f" * 64
    _write_json(paths.receipt, receipt)
    binding = dict(binding)
    binding["receipt_sha256"] = parent._stable_record(paths.receipt)[0]["sha256"]
    binding["terminal_sha256"] = binding["receipt_sha256"]
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding["binding_sha256"] = parent._logical_sha256(body)
    authority = dict(authority)
    authority["parent_binding_sha256"] = binding["binding_sha256"]
    authority_body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    authority["authority_sha256"] = parent._logical_sha256(authority_body)
    monkeypatch.setattr(
        parent, "fit_parent_binding_without_tensor_load", lambda _paths: binding
    )
    capability = loader.OneUseFitTrainingLoader(
        paths, require_production=False, train_documents=3
    )
    with pytest.raises(RuntimeError, match="receipt does not join"):
        capability.load_once(parent_binding=binding, analysis_authority=authority)


def test_loader_has_no_validation_eval_model_or_corpus_surface():
    public = [name.lower() for name in dir(loader) if not name.startswith("_")]
    assert not any("validation" in name for name in public)
    assert not any("eval" in name for name in public)
    assert not any("model" in name or "corpus" in name for name in public)


def test_synthetic_loader_cannot_target_exact_production_paths():
    with pytest.raises(RuntimeError, match="production FIT paths"):
        loader.OneUseFitTrainingLoader(
            parent.PRODUCTION_PATHS, require_production=False, train_documents=3
        )


def test_synthetic_loader_rejects_dotdot_aliases_to_production_paths():
    aliases = parent.FitParentPaths(*(
        path.parent / "synthetic-looking-subdir" / ".." / path.name
        for path in (
            getattr(parent.PRODUCTION_PATHS, field)
            for field in loader.FIT_PARENT_PATH_FIELDS
        )
    ))
    with pytest.raises(RuntimeError, match="production FIT paths"):
        loader.OneUseFitTrainingLoader(
            aliases, require_production=False, train_documents=3
        )


def test_synthetic_loader_rejects_even_one_production_artifact_alias(tmp_path):
    values = [tmp_path / f"synthetic-{index}" for index in range(7)]
    values[1] = (
        parent.PRODUCTION_PATHS.bundle.parent / "alias" / ".."
        / parent.PRODUCTION_PATHS.bundle.name
    )
    mixed = parent.FitParentPaths(*values)
    with pytest.raises(RuntimeError, match="production FIT paths"):
        loader.OneUseFitTrainingLoader(
            mixed, require_production=False, train_documents=3
        )


def test_synthetic_loader_rejects_hardlink_alias_to_production_bundle(tmp_path):
    alias = tmp_path / "innocent-looking-bundle.pt"
    alias.hardlink_to(parent.PRODUCTION_PATHS.bundle)
    values = [tmp_path / f"synthetic-{index}" for index in range(7)]
    values[1] = alias
    with pytest.raises(RuntimeError, match="production FIT paths"):
        loader.OneUseFitTrainingLoader(
            parent.FitParentPaths(*values), require_production=False, train_documents=3
        )


def test_synthetic_loader_rejects_cross_role_production_path(tmp_path):
    values = [tmp_path / f"synthetic-{index}" for index in range(7)]
    values[0] = parent.PRODUCTION_PATHS.bundle
    with pytest.raises(RuntimeError, match="production FIT paths"):
        loader.OneUseFitTrainingLoader(
            parent.FitParentPaths(*values), require_production=False, train_documents=3
        )


def test_synthetic_loader_rechecks_aliases_at_load_boundary(tmp_path, monkeypatch):
    paths, binding, authority, _ = _fixture(tmp_path, monkeypatch)
    capability = loader.OneUseFitTrainingLoader(
        paths, require_production=False, train_documents=3
    )
    paths.bundle.unlink()
    paths.bundle.hardlink_to(parent.PRODUCTION_PATHS.bundle)
    with pytest.raises(RuntimeError, match="became production aliases"):
        capability.load_once(parent_binding=binding, analysis_authority=authority)
    assert capability.spent is True


def test_synthetic_loader_rejects_production_inode_opened_after_final_lookup(
    tmp_path, monkeypatch,
):
    paths, binding, authority, _ = _fixture(tmp_path, monkeypatch)
    calls = 0

    def switch_after_final_lookup(candidate_paths):
        nonlocal calls
        calls += 1
        if calls == 3:
            candidate_paths.bundle.unlink()
            candidate_paths.bundle.hardlink_to(parent.PRODUCTION_PATHS.bundle)
        return False

    monkeypatch.setattr(loader, "_touches_production_parent", switch_after_final_lookup)
    capability = loader.OneUseFitTrainingLoader(
        paths, require_production=False, train_documents=3
    )
    with pytest.raises(RuntimeError, match="opened a production bundle inode"):
        capability.load_once(parent_binding=binding, analysis_authority=authority)
    assert calls == 3
    assert capability.spent is True
