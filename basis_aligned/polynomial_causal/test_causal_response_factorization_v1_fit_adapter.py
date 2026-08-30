import pytest
import torch

import causal_response_factorization_v1_fit_adapter as adapter
import causal_response_tensor_v1_fit_bundle as fit_bundle
from test_causal_response_tensor_v1_fit_bundle import _payload


def _analysis_input():
    payload = _payload()
    authority = payload["binding"]["authority_sha256"]
    return payload, adapter.factorization_input_from_fit_payload(
        payload,
        expected_authority_sha256=authority,
        require_production=False,
        train_documents=3,
    )


def test_adapter_derives_signed_response_split_and_owner_topology():
    payload, result = _analysis_input()
    raw = payload["fit_response"]
    member_count = raw["member_count"][None, None].expand_as(result.response)
    off_count = raw["off_count"][None, None].expand_as(result.response)
    expected = torch.zeros_like(result.response)
    expected[result.valid] = (
        raw["statistics"]["member_signed_sum"][result.valid]
        / member_count[result.valid]
        - raw["statistics"]["off_signed_sum"][result.valid]
        / off_count[result.valid]
    )
    assert torch.equal(result.response, expected)
    assert result.train_document_indices.numel() == 3
    assert result.validation_document_indices.numel() == 1
    assert result.owner_components == ("a1", "m2")
    assert torch.equal(result.source_groups, torch.tensor([0, 0, 1, 1]))
    assert result.response.shape == (2, 4, 4, 4)


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
    authority = payload["binding"]["authority_sha256"]
    payload["fit_response"]["statistics"]["member_abs_sum"][0, 0, 0, 0] = -1
    payload["tensor_hashes"] = fit_bundle._tensor_hash_map({
        key: value for key, value in payload.items() if key != "tensor_hashes"
    })
    with pytest.raises(ValueError, match="nonnegative"):
        adapter.factorization_input_from_fit_payload(
            payload,
            expected_authority_sha256=authority,
            require_production=False,
            train_documents=3,
        )


def test_adapter_rejects_wrong_authority_and_production_split_change():
    payload = _payload()
    with pytest.raises(RuntimeError, match="authority"):
        adapter.factorization_input_from_fit_payload(
            payload,
            expected_authority_sha256="9" * 64,
            require_production=False,
            train_documents=3,
        )
    with pytest.raises(RuntimeError, match="preregistration"):
        adapter.factorization_input_from_fit_payload(
            payload,
            expected_authority_sha256=payload["binding"]["authority_sha256"],
            require_production=True,
            train_documents=228,
        )


def test_adapter_has_no_file_or_eval_capability_surface():
    assert not hasattr(adapter, "Path")
    assert not hasattr(adapter, "open")
    assert not hasattr(adapter, "load")
    assert not any("eval" in name.lower() for name in dir(adapter))
