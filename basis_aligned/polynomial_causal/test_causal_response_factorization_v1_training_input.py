import pytest
import torch

import causal_response_factorization_v1_training_input as artifact
from test_causal_response_factorization_v1_fit_adapter import _analysis_input


def _payload():
    _, value = _analysis_input()
    return artifact.build_training_input_payload(
        value, analysis_authority_sha256="a" * 64
    )


def test_training_artifact_round_trip_is_train_only_and_nonaliasing(tmp_path):
    payload = _payload()
    path = tmp_path / "training.pt"
    digest = artifact.publish_training_input(
        path, payload, expected_analysis_authority_sha256="a" * 64,
        require_production=False,
    )
    result, replay_digest = artifact.replay_training_input(
        path, expected_analysis_authority_sha256="a" * 64,
        expected_artifact_sha256=digest, require_production=False,
    )
    assert replay_digest == digest
    assert result.response.shape[-1] == 3
    assert not any("validation" in key or "eval" in key for key in payload)
    saved = result.response.clone()
    payload["response"].zero_()
    assert torch.equal(result.response, saved)


def test_training_artifact_rejects_tensor_tampering_even_if_shape_survives():
    payload = _payload()
    payload["response"][0, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="tensor hashes"):
        artifact.validate_training_input_payload(
            payload, expected_analysis_authority_sha256="a" * 64,
            require_production=False,
        )


def test_training_artifact_rejects_validation_field_or_authority_substitution():
    payload = _payload()
    payload["validation_response"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="schema"):
        artifact.validate_training_input_payload(
            payload, expected_analysis_authority_sha256="a" * 64,
            require_production=False,
        )
    payload.pop("validation_response")
    with pytest.raises(RuntimeError, match="authority"):
        artifact.validate_training_input_payload(
            payload, expected_analysis_authority_sha256="b" * 64,
            require_production=False,
        )


def test_training_artifact_rejects_forged_owner_topology_after_rehash():
    payload = _payload()
    payload["source_groups"].zero_()
    payload["tensor_hashes"] = artifact._tensor_hashes(payload)
    with pytest.raises(ValueError, match="do not match"):
        artifact.validate_training_input_payload(
            payload, expected_analysis_authority_sha256="a" * 64,
            require_production=False,
        )


def test_training_artifact_publication_is_create_only(tmp_path):
    payload = _payload()
    path = tmp_path / "training.pt"
    artifact.publish_training_input(
        path, payload, expected_analysis_authority_sha256="a" * 64,
        require_production=False,
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        artifact.publish_training_input(
            path, payload, expected_analysis_authority_sha256="a" * 64,
            require_production=False,
        )
    assert path.read_bytes() == original
