from dataclasses import replace

import pytest
import torch

import causal_response_factorization_v1_fit_adapter as adapter
import causal_response_tensor_v1_fit_bundle as fit_bundle
from causal_response_factorization_v1 import (
    prospective_document_split,
    signed_response_from_sums,
)
from test_causal_response_tensor_v1_fit_bundle import _payload


def _artifacts(payload):
    return adapter.FitArtifactBinding(
        parent_binding_sha256="0" * 64,
        receipt_sha256="1" * 64,
        terminal_sha256="1" * 64,
        authority_artifact_sha256="2" * 64,
        authority_logical_sha256=payload["binding"]["authority_sha256"],
        bundle_sha256="3" * 64,
        manifest_artifact_sha256="4" * 64,
        manifest_logical_sha256="5" * 64,
        source_closure_sha256=payload["binding"]["source_closure_sha256"],
    )


def _parent_binding(payload):
    artifacts = _artifacts(payload)
    body = {
        "schema": "causal_response_factorization_v1_fit_parent_binding",
        "receipt_sha256": artifacts.receipt_sha256,
        "terminal_sha256": artifacts.terminal_sha256,
        "authority_artifact_sha256": artifacts.authority_artifact_sha256,
        "authority_logical_sha256": artifacts.authority_logical_sha256,
        "bundle_sha256": artifacts.bundle_sha256,
        "bundle_bytes": 123,
        "manifest_artifact_sha256": artifacts.manifest_artifact_sha256,
        "manifest_logical_sha256": artifacts.manifest_logical_sha256,
        "source_closure_sha256": artifacts.source_closure_sha256,
        "fit_protocol": {},
        "tensor_values_deserialized": False,
        "authorized_for_eval": False,
    }
    return {**body, "binding_sha256": adapter._logical_sha256(body)}


def _replace_binding(value, **updates):
    body = {key: item for key, item in value.items() if key != "binding_sha256"}
    body.update(updates)
    return {**body, "binding_sha256": adapter._logical_sha256(body)}


def _analysis_input():
    payload = _payload()
    return payload, adapter.training_input_from_fit_payload(
        payload,
        parent_binding=_parent_binding(payload),
        require_production=False,
        train_documents=3,
    )


def test_adapter_derives_train_only_signed_response_and_owner_topology():
    payload, result = _analysis_input()
    raw = payload["fit_response"]
    full_response, full_valid = signed_response_from_sums(
        raw["statistics"], raw["member_count"], raw["off_count"]
    )
    train, validation = prospective_document_split(raw["document_ids"], train_documents=3)
    assert torch.equal(result.response, full_response[..., train])
    assert torch.equal(result.valid, full_valid[..., train])
    assert torch.equal(result.original_document_indices, train)
    assert set(result.original_document_indices.tolist()).isdisjoint(validation.tolist())
    assert result.document_ids.numel() == 3
    assert result.owner_components == ("a1", "m2")
    assert torch.equal(result.source_groups, torch.tensor([0, 0, 1, 1]))
    assert result.response.shape == (2, 4, 4, 3)


def test_adapter_outputs_do_not_alias_the_validated_payload():
    payload, result = _analysis_input()
    saved_response = result.response.clone()
    saved_documents = result.document_ids.clone()
    payload["fit_response"]["statistics"]["member_signed_sum"].zero_()
    payload["fit_response"]["document_ids"].zero_()
    assert torch.equal(result.response, saved_response)
    assert torch.equal(result.document_ids, saved_documents)


def test_adapter_replays_semantics_before_exposing_response():
    payload = _payload()
    parent_binding = _parent_binding(payload)
    payload["fit_response"]["statistics"]["member_abs_sum"][0, 0, 0, 0] = -1
    payload["tensor_hashes"] = fit_bundle._tensor_hash_map({
        key: value for key, value in payload.items() if key != "tensor_hashes"
    })
    with pytest.raises(ValueError, match="nonnegative"):
        adapter.training_input_from_fit_payload(
            payload, parent_binding=parent_binding,
            require_production=False, train_documents=3
        )


def test_adapter_rejects_wrong_authority_source_closure_and_production_split():
    payload = _payload()
    parent_binding = _parent_binding(payload)
    with pytest.raises(RuntimeError, match="authority"):
        adapter.training_input_from_fit_payload(
            payload,
            parent_binding=_replace_binding(
                parent_binding, authority_logical_sha256="9" * 64
            ),
            require_production=False,
            train_documents=3,
        )
    with pytest.raises(RuntimeError, match="source closure"):
        adapter.training_input_from_fit_payload(
            payload,
            parent_binding=_replace_binding(
                parent_binding, source_closure_sha256="8" * 64
            ),
            require_production=False,
            train_documents=3,
        )
    with pytest.raises(RuntimeError, match="preregistration"):
        adapter.training_input_from_fit_payload(
            payload, parent_binding=parent_binding,
            require_production=True, train_documents=228
        )


def test_training_input_rejects_forged_owner_topology():
    _, result = _analysis_input()
    with pytest.raises(ValueError, match="do not match"):
        replace(result, source_groups=torch.zeros_like(result.source_groups))
    with pytest.raises(ValueError, match="canonical"):
        replace(result, owner_components=("m2", "a1"))


def test_adapter_carries_all_parent_artifact_identities():
    payload, result = _analysis_input()
    expected = replace(
        _artifacts(payload),
        parent_binding_sha256=_parent_binding(payload)["binding_sha256"],
    )
    assert result.artifacts == expected
    assert result.artifacts.receipt_sha256 == result.artifacts.terminal_sha256
    with pytest.raises(ValueError, match="terminal"):
        replace(result.artifacts, terminal_sha256="f" * 64)


def test_adapter_rejects_parent_binding_with_unreplayed_logical_hash():
    payload = _payload()
    parent_binding = _parent_binding(payload)
    parent_binding["bundle_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="logical identity"):
        adapter.training_input_from_fit_payload(
            payload, parent_binding=parent_binding,
            require_production=False, train_documents=3,
        )


def test_adapter_has_no_file_validation_or_eval_capability_surface():
    assert not hasattr(adapter, "Path")
    assert not hasattr(adapter, "open")
    assert not hasattr(adapter, "load")
    assert not any("validation" in name.lower() for name in dir(adapter))
    assert not any("eval" in name.lower() for name in dir(adapter))
    assert "validation" not in adapter.FitTrainingInput.__dataclass_fields__
