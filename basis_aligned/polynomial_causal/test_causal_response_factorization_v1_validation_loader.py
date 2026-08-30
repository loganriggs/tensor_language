import hashlib
import json

import pytest

import causal_response_factorization_v1_parent_binding as parent
import causal_response_factorization_v1_parent_rebinding as rebinding
import causal_response_factorization_v1_validation_loader as loader
from test_causal_response_factorization_v1_training_loader import _fixture as _training_fixture
from test_causal_response_factorization_v1_validation_input import _freeze_inputs


def _authority(parent_binding, freeze_sha):
    body = {
        "schema": loader.AUTHORITY_SCHEMA,
        "status": loader.AUTHORITY_STATUS,
        "source_closure": {},
        "self_review": {},
        "parent_binding_sha256": parent_binding["binding_sha256"],
        "candidate_freeze": {"artifact_sha256": freeze_sha},
        "grid_terminal": {},
        "protocol": {
            "role": "FIT_INTERNAL_VALIDATION",
            "validation_documents": 114,
            "training_response_values_exposed": 0,
            "eval_documents_exposed": 0,
            "candidate_programs": 27,
            "candidates_dropped_after_scoring": 0,
            "winner_selected_inside_scorer": False,
        },
        "output_paths": {},
        "outcome_access_before_authority": dict(loader.OUTCOME_BOUNDARY),
        "authorized_for_validation_scoring": True,
        "authorized_for_candidate_selection": False,
        "authorized_for_eval": False,
    }
    return {**body, "authority_sha256": parent._logical_sha256(body)}


def _fixture(tmp_path, monkeypatch):
    paths, binding, _training_authority, payload = _training_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rebinding, "fit_parent_binding_by_content_identity", lambda _paths=None: binding
    )
    freeze, audit, freeze_sha, audit_sha = _freeze_inputs()
    authority = _authority(binding, freeze_sha)
    return paths, binding, authority, (freeze, audit, freeze_sha, audit_sha)


def _load(capability, binding, authority, freeze_inputs):
    freeze, audit, freeze_sha, audit_sha = freeze_inputs
    return capability.load_once(
        parent_binding=binding, candidate_freeze=freeze, candidate_freeze_audit=audit,
        candidate_freeze_artifact_sha256=freeze_sha,
        candidate_freeze_audit_artifact_sha256=audit_sha,
        validation_authority=authority,
    )


def test_one_use_loader_exposes_only_validation_role(tmp_path, monkeypatch):
    paths, binding, authority, freeze_inputs = _fixture(tmp_path, monkeypatch)
    capability = loader.OneUseFitValidationLoader(
        paths, require_production=False, train_documents=3
    )
    result = _load(capability, binding, authority, freeze_inputs)
    assert capability.spent is True
    assert result.response.shape == (2, 4, 4, 1)
    assert result.candidate_freeze.artifact_sha256 == freeze_inputs[2]
    assert not hasattr(result, "training_response")
    with pytest.raises(RuntimeError, match="already spent"):
        _load(capability, binding, authority, freeze_inputs)


def test_loader_poisoned_when_authority_is_forged(tmp_path, monkeypatch):
    paths, binding, authority, freeze_inputs = _fixture(tmp_path, monkeypatch)
    capability = loader.OneUseFitValidationLoader(
        paths, require_production=False, train_documents=3
    )
    forged = dict(authority)
    forged["authorized_for_candidate_selection"] = True
    with pytest.raises(RuntimeError, match="schema or role"):
        _load(capability, binding, forged, freeze_inputs)
    assert capability.spent is True
    with pytest.raises(RuntimeError, match="already spent"):
        _load(capability, binding, authority, freeze_inputs)


def test_loader_rejects_authority_bound_to_other_freeze_or_parent(tmp_path, monkeypatch):
    paths, binding, authority, freeze_inputs = _fixture(tmp_path, monkeypatch)
    other = _authority(binding, "9" * 64)
    with pytest.raises(RuntimeError, match="different candidate freeze"):
        _load(
            loader.OneUseFitValidationLoader(paths, require_production=False, train_documents=3),
            binding, other, freeze_inputs,
        )
    body = {k: v for k, v in binding.items() if k != "binding_sha256"}
    body["bundle_bytes"] = 1
    other_parent = {**body, "binding_sha256": parent._logical_sha256(body)}
    with pytest.raises(RuntimeError, match="FIT parent changed"):
        _load(
            loader.OneUseFitValidationLoader(paths, require_production=False, train_documents=3),
            other_parent, _authority(other_parent, freeze_inputs[2]), freeze_inputs,
        )


def test_production_loader_refuses_synthetic_paths_and_split():
    with pytest.raises(RuntimeError, match="229/114"):
        loader.OneUseFitValidationLoader(train_documents=3)


def test_synthetic_loader_refuses_production_authority_surface(tmp_path, monkeypatch):
    paths, binding, authority, freeze_inputs = _fixture(tmp_path, monkeypatch)
    capability = loader.OneUseFitValidationLoader(
        paths, require_production=False, train_documents=3
    )
    freeze, audit, freeze_sha, audit_sha = freeze_inputs
    with pytest.raises(RuntimeError, match="authority surface"):
        capability.load_once(
            parent_binding=binding, candidate_freeze=freeze, candidate_freeze_audit=audit,
            candidate_freeze_artifact_sha256=freeze_sha,
            candidate_freeze_audit_artifact_sha256=audit_sha,
            expected_validation_authority_artifact_sha256="a" * 64,
        )


def test_authority_validator_requires_exact_protocol_and_boundary():
    binding = {"binding_sha256": "1" * 64}
    authority = _authority(binding, "2" * 64)
    loader.validate_validation_authority(
        authority, binding, candidate_freeze_artifact_sha256="2" * 64,
    )
    forged = json.loads(json.dumps(authority))
    forged["protocol"]["winner_selected_inside_scorer"] = True
    body = {k: v for k, v in forged.items() if k != "authority_sha256"}
    forged["authority_sha256"] = parent._logical_sha256(body)
    with pytest.raises(RuntimeError, match="protocol changed"):
        loader.validate_validation_authority(
            forged, binding, candidate_freeze_artifact_sha256="2" * 64,
        )
    forged = json.loads(json.dumps(authority))
    forged["outcome_access_before_authority"]["validation_values_read"] = True
    body = {k: v for k, v in forged.items() if k != "authority_sha256"}
    forged["authority_sha256"] = parent._logical_sha256(body)
    with pytest.raises(RuntimeError, match="outcome boundary"):
        loader.validate_validation_authority(
            forged, binding, candidate_freeze_artifact_sha256="2" * 64,
        )
