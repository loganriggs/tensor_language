"""Semantic, create-only artifact for receipt-bound FIT training responses.

The artifact contains exactly the 229 training documents emitted by the one-use
loader. It contains no full FIT payload, validation response, EVAL value, model,
tokens, logits, or activations.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from causal_response_factorization_v1_fit_adapter import (
    FitArtifactBinding,
    FitTrainingInput,
)
from causal_response_tensor_v1_backend import tensor_sha256


SCHEMA = "causal_response_factorization_v1_training_input"
CLAIM_BOUNDARY = (
    "FIT training role only: 229 documents; no validation response, EVAL value, "
    "model, token, logit, activation, or original full-role payload."
)
PAYLOAD_KEYS = {
    "schema", "claim_boundary", "analysis_authority_sha256", "response", "valid",
    "document_ids", "original_document_indices", "source_groups", "phases",
    "source_tags", "target_tags", "source_components", "owner_components",
    "artifacts", "tensor_hashes", "forbidden_payload_contract",
}
FORBIDDEN_PAYLOAD_CONTRACT = {
    "validation_response": False,
    "validation_document_ids": False,
    "eval_response": False,
    "full_fit_payload": False,
    "raw_tokens": False,
    "targets": False,
    "activations": False,
    "logits": False,
    "model": False,
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: tensor_sha256(value)
        for key, value in payload.items()
        if type(value) is torch.Tensor
    }


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.clone().contiguous()


def build_training_input_payload(
    value: FitTrainingInput, *, analysis_authority_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, FitTrainingInput):
        raise TypeError("training input builder requires a FitTrainingInput")
    if not _is_sha256(analysis_authority_sha256):
        raise ValueError("training analysis authority hash is malformed")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "analysis_authority_sha256": analysis_authority_sha256,
        "response": _clone(value.response),
        "valid": _clone(value.valid),
        "document_ids": _clone(value.document_ids),
        "original_document_indices": _clone(value.original_document_indices),
        "source_groups": _clone(value.source_groups),
        "phases": list(value.phases),
        "source_tags": list(value.source_tags),
        "target_tags": list(value.target_tags),
        "source_components": list(value.source_components),
        "owner_components": list(value.owner_components),
        "artifacts": asdict(value.artifacts),
        "forbidden_payload_contract": dict(FORBIDDEN_PAYLOAD_CONTRACT),
    }
    payload["tensor_hashes"] = _tensor_hashes(payload)
    validate_training_input_payload(
        payload,
        expected_analysis_authority_sha256=analysis_authority_sha256,
        require_production=False,
    )
    return payload


def validate_training_input_payload(
    payload: Mapping[str, Any],
    *,
    expected_analysis_authority_sha256: str,
    require_production: bool = True,
) -> FitTrainingInput:
    if type(payload) is not dict or set(payload) != PAYLOAD_KEYS or (
        payload.get("schema") != SCHEMA or payload.get("claim_boundary") != CLAIM_BOUNDARY
        or payload.get("forbidden_payload_contract") != FORBIDDEN_PAYLOAD_CONTRACT
    ):
        raise RuntimeError("training input artifact schema or claim boundary changed")
    if not _is_sha256(expected_analysis_authority_sha256) or (
        payload["analysis_authority_sha256"] != expected_analysis_authority_sha256
    ):
        raise RuntimeError("training input artifact authority binding changed")
    replay_hashes = _tensor_hashes(payload)
    if payload["tensor_hashes"] != replay_hashes:
        raise RuntimeError("training input tensor hashes do not replay")
    artifact_raw = payload["artifacts"]
    if type(artifact_raw) is not dict or set(artifact_raw) != set(
        FitArtifactBinding.__dataclass_fields__
    ):
        raise RuntimeError("training input FIT artifact binding schema changed")
    artifacts = FitArtifactBinding(**artifact_raw)
    tensors = {
        name: payload[name]
        for name in (
            "response", "valid", "document_ids", "original_document_indices",
            "source_groups",
        )
    }
    expected_dtypes = {
        "response": torch.float64,
        "valid": torch.bool,
        "document_ids": torch.int64,
        "original_document_indices": torch.int64,
        "source_groups": torch.int64,
    }
    for name, value in tensors.items():
        if type(value) is not torch.Tensor or value.dtype != expected_dtypes[name] or (
            value.device.type != "cpu" or not value.is_contiguous()
            or (value.dtype.is_floating_point and not bool(torch.isfinite(value).all()))
        ):
            raise RuntimeError(f"training input {name} tensor is malformed")
    result = FitTrainingInput(
        response=_clone(payload["response"]),
        valid=_clone(payload["valid"]),
        document_ids=_clone(payload["document_ids"]),
        original_document_indices=_clone(payload["original_document_indices"]),
        source_groups=_clone(payload["source_groups"]),
        phases=tuple(payload["phases"]),
        source_tags=tuple(payload["source_tags"]),
        target_tags=tuple(payload["target_tags"]),
        source_components=tuple(payload["source_components"]),
        owner_components=tuple(payload["owner_components"]),
        artifacts=artifacts,
    )
    if require_production and (
        result.response.shape != (2, 49, 49, 229)
        or result.document_ids.numel() != 229
        or len(result.owner_components) != 6
    ):
        raise RuntimeError("training input artifact differs from production dimensions")
    return result


def publish_training_input(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_analysis_authority_sha256: str,
    require_production: bool = True,
) -> str:
    path = path.resolve()
    validate_training_input_payload(
        payload,
        expected_analysis_authority_sha256=expected_analysis_authority_sha256,
        require_production=require_production,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            descriptor = -1
            torch.save(payload, sink)
            sink.flush(); os.fsync(sink.fileno())
        digest = replay_training_input(
            temporary,
            expected_analysis_authority_sha256=expected_analysis_authority_sha256,
            require_production=require_production,
        )[1]
        os.link(temporary, path)
        return replay_training_input(
            path,
            expected_analysis_authority_sha256=expected_analysis_authority_sha256,
            expected_artifact_sha256=digest,
            require_production=require_production,
        )[1]
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def replay_training_input(
    path: Path,
    *,
    expected_analysis_authority_sha256: str,
    expected_artifact_sha256: str | None = None,
    require_production: bool = True,
) -> tuple[FitTrainingInput, str]:
    before = _file_sha256(path)
    raw = path.read_bytes()
    after = _file_sha256(path)
    digest = hashlib.sha256(raw).hexdigest()
    if before != after or digest != before or (
        expected_artifact_sha256 is not None and digest != expected_artifact_sha256
    ):
        raise RuntimeError("training input artifact changed during stable replay")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    result = validate_training_input_payload(
        payload,
        expected_analysis_authority_sha256=expected_analysis_authority_sha256,
        require_production=require_production,
    )
    return result, digest
