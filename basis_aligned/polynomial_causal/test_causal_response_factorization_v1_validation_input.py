from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

import causal_response_factorization_v1_validation_input as validation
import causal_response_tensor_v1_fit_bundle as fit_bundle
from causal_response_factorization_v1 import prospective_document_split, signed_response_from_sums
from test_causal_response_factorization_v1_fit_adapter import _parent_binding
from test_causal_response_tensor_v1_fit_bundle import _payload


HERE = Path(__file__).resolve().parent
FREEZE_PATH = HERE / "causal_response_factorization_v1_candidate_freeze_v2.json"
AUDIT_PATH = HERE / "causal_response_factorization_v1_candidate_freeze_v2_independent_audit.json"


def _freeze_inputs():
    freeze_raw = FREEZE_PATH.read_bytes()
    audit_raw = AUDIT_PATH.read_bytes()
    return (
        json.loads(freeze_raw), json.loads(audit_raw),
        hashlib.sha256(freeze_raw).hexdigest(), hashlib.sha256(audit_raw).hexdigest(),
    )


def _input():
    payload = _payload()
    freeze, audit, freeze_sha, audit_sha = _freeze_inputs()
    value = validation.validation_input_from_fit_payload(
        payload, parent_binding=_parent_binding(payload), candidate_freeze=freeze,
        candidate_freeze_audit=audit, candidate_freeze_artifact_sha256=freeze_sha,
        candidate_freeze_audit_artifact_sha256=audit_sha,
        require_production=False, train_documents=3,
    )
    return payload, value


def test_validation_adapter_exposes_only_hashed_validation_documents():
    payload, value = _input()
    raw = payload["fit_response"]
    response, valid = signed_response_from_sums(
        raw["statistics"], raw["member_count"], raw["off_count"],
    )
    training_indices, validation_indices = prospective_document_split(
        raw["document_ids"], train_documents=3,
    )
    assert torch.equal(value.response, response[..., validation_indices])
    assert torch.equal(value.valid, valid[..., validation_indices])
    assert torch.equal(value.original_document_indices, validation_indices)
    assert set(value.original_document_indices.tolist()).isdisjoint(training_indices.tolist())
    assert value.response.shape == (2, 4, 4, 1)
    assert value.candidate_freeze.candidate_programs == 27


def test_validation_output_does_not_alias_private_fit_payload():
    payload, value = _input()
    response = value.response.clone()
    documents = value.document_ids.clone()
    payload["fit_response"]["statistics"]["member_signed_sum"].zero_()
    payload["fit_response"]["document_ids"].zero_()
    assert torch.equal(value.response, response)
    assert torch.equal(value.document_ids, documents)


def test_freeze_requires_exact_all_seed_census_and_no_score_fields():
    payload = _payload()
    freeze, audit, freeze_sha, audit_sha = _freeze_inputs()
    freeze["candidate_programs"][0]["validation_mse"] = 0.0
    body = {key: item for key, item in freeze.items() if key != "manifest_sha256"}
    freeze["manifest_sha256"] = validation._logical_sha256(body)
    audit["candidate_freeze_manifest_sha256"] = freeze["manifest_sha256"]
    with pytest.raises(RuntimeError, match="contains a score"):
        validation.validation_input_from_fit_payload(
            payload, parent_binding=_parent_binding(payload), candidate_freeze=freeze,
            candidate_freeze_audit=audit, candidate_freeze_artifact_sha256=freeze_sha,
            candidate_freeze_audit_artifact_sha256=audit_sha,
            require_production=False, train_documents=3,
        )


def test_validation_adapter_rejects_non_go_or_mismatched_freeze_audit():
    payload = _payload()
    freeze, audit, freeze_sha, audit_sha = _freeze_inputs()
    audit["status"] = "NO-GO"
    with pytest.raises(RuntimeError, match="audit"):
        validation.validation_input_from_fit_payload(
            payload, parent_binding=_parent_binding(payload), candidate_freeze=freeze,
            candidate_freeze_audit=audit, candidate_freeze_artifact_sha256=freeze_sha,
            candidate_freeze_audit_artifact_sha256=audit_sha,
            require_production=False, train_documents=3,
        )


def test_validation_adapter_replays_bundle_before_exposure():
    payload = _payload()
    freeze, audit, freeze_sha, audit_sha = _freeze_inputs()
    payload["fit_response"]["statistics"]["member_abs_sum"][0, 0, 0, 0] = -1
    payload["tensor_hashes"] = fit_bundle._tensor_hash_map({
        key: value for key, value in payload.items() if key != "tensor_hashes"
    })
    with pytest.raises(ValueError, match="nonnegative"):
        validation.validation_input_from_fit_payload(
            payload, parent_binding=_parent_binding(payload), candidate_freeze=freeze,
            candidate_freeze_audit=audit, candidate_freeze_artifact_sha256=freeze_sha,
            candidate_freeze_audit_artifact_sha256=audit_sha,
            require_production=False, train_documents=3,
        )


def test_validation_input_rejects_forged_owner_topology():
    _, value = _input()
    with pytest.raises(ValueError, match="owner topology"):
        replace(value, source_groups=torch.zeros_like(value.source_groups))


def test_validation_adapter_has_no_filesystem_model_training_or_eval_surface():
    assert not hasattr(validation, "Path")
    assert not hasattr(validation, "open")
    assert not hasattr(validation, "load")
    assert "training_response" not in validation.FitValidationInput.__dataclass_fields__
    assert not any("eval" in name.lower() for name in dir(validation))
