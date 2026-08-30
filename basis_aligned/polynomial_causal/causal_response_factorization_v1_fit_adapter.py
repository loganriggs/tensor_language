"""Pure train-role boundary from a receipt-bound FIT payload.

This module deliberately has no filesystem, model, corpus, validation-role, or EVAL
capability. A source-closed lifecycle must bind and load the exact FIT artifact, then
pass both its in-memory payload and immutable artifact identities here. The factory
replays bundle semantics and returns only the frozen training documents. Validation
values require a different, later source closure after candidates are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

import causal_response_tensor_v1_fit_bundle as fit_bundle
from causal_response_factorization_v1 import (
    FIT_TRAIN_DOCUMENTS,
    prospective_document_split,
    signed_response_from_sums,
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _logical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class FitArtifactBinding:
    """Exact identities joining training tensors to one completed FIT transaction."""

    parent_binding_sha256: str
    receipt_sha256: str
    terminal_sha256: str
    authority_artifact_sha256: str
    authority_logical_sha256: str
    bundle_sha256: str
    manifest_artifact_sha256: str
    manifest_logical_sha256: str
    source_closure_sha256: str

    def __post_init__(self) -> None:
        if not all(_is_sha256(value) for value in self.__dict__.values()):
            raise ValueError("every FIT artifact identity must be a lowercase SHA-256")
        if self.receipt_sha256 != self.terminal_sha256:
            raise ValueError("FIT receipt and shared terminal identities differ")

    @classmethod
    def from_parent_binding(cls, value: Mapping[str, Any]) -> "FitArtifactBinding":
        """Reduce the outcome-blind parent's exact binding without retaining aliases."""

        if type(value) is not dict or set(value) != {
            "schema", "receipt_sha256", "terminal_sha256",
            "authority_artifact_sha256", "authority_logical_sha256",
            "bundle_sha256", "bundle_bytes", "manifest_artifact_sha256",
            "manifest_logical_sha256", "source_closure_sha256", "fit_protocol",
            "tensor_values_deserialized", "authorized_for_eval", "binding_sha256",
        } or value.get("schema") != (
            "causal_response_factorization_v1_fit_parent_binding"
        ) or value.get("tensor_values_deserialized") is not False or (
            value.get("authorized_for_eval") is not False
        ):
            raise RuntimeError("FIT parent binding schema or role changed")
        body = {key: item for key, item in value.items() if key != "binding_sha256"}
        if value["binding_sha256"] != _logical_sha256(body):
            raise RuntimeError("FIT parent binding logical identity does not replay")
        return cls(
            parent_binding_sha256=value["binding_sha256"],
            receipt_sha256=value["receipt_sha256"],
            terminal_sha256=value["terminal_sha256"],
            authority_artifact_sha256=value["authority_artifact_sha256"],
            authority_logical_sha256=value["authority_logical_sha256"],
            bundle_sha256=value["bundle_sha256"],
            manifest_artifact_sha256=value["manifest_artifact_sha256"],
            manifest_logical_sha256=value["manifest_logical_sha256"],
            source_closure_sha256=value["source_closure_sha256"],
        )


@dataclass(frozen=True)
class FitTrainingInput:
    """Receipt-bound training-role tensors; no validation document is present.

    ``response`` and ``valid`` use axes ``[phase, source, target, train_document]``.
    ``source_groups`` is exactly derived from ``source_components`` and the ordered
    first occurrence of each owner in ``owner_components``.
    """

    response: torch.Tensor
    valid: torch.Tensor
    document_ids: torch.Tensor
    original_document_indices: torch.Tensor
    source_groups: torch.Tensor
    phases: tuple[str, ...]
    source_tags: tuple[str, ...]
    target_tags: tuple[str, ...]
    source_components: tuple[str, ...]
    owner_components: tuple[str, ...]
    artifacts: FitArtifactBinding

    def __post_init__(self) -> None:
        p = len(self.phases)
        s = len(self.source_tags)
        t = len(self.target_tags)
        d = self.document_ids.numel()
        if self.response.dtype != torch.float64 or self.response.shape != (p, s, t, d):
            raise ValueError("training response does not match the declared FIT axes")
        if self.valid.dtype != torch.bool or self.valid.shape != self.response.shape:
            raise ValueError("training valid mask does not align with the response")
        if self.document_ids.dtype != torch.int64 or self.document_ids.shape != (d,):
            raise ValueError("training document IDs are malformed")
        if self.original_document_indices.dtype != torch.int64 or (
            self.original_document_indices.shape != (d,)
            or torch.unique(self.original_document_indices).numel() != d
            or bool((self.original_document_indices < 0).any())
        ):
            raise ValueError("training source-document indices are malformed")
        if self.source_groups.dtype != torch.int64 or self.source_groups.shape != (s,):
            raise ValueError("source owner assignments are malformed")
        tensors = (
            self.response,
            self.valid,
            self.document_ids,
            self.original_document_indices,
            self.source_groups,
        )
        if any(value.device.type != "cpu" or not value.is_contiguous() for value in tensors):
            raise ValueError("factorization training inputs must be contiguous CPU tensors")
        expected_owners = tuple(dict.fromkeys(self.source_components))
        if not self.owner_components or self.owner_components != expected_owners or (
            len(set(self.owner_components)) != len(self.owner_components)
        ):
            raise ValueError("owner component ordering is not canonical")
        owner_index = {owner: index for index, owner in enumerate(self.owner_components)}
        expected_groups = torch.tensor(
            [owner_index[owner] for owner in self.source_components], dtype=torch.int64
        )
        if not torch.equal(self.source_groups, expected_groups):
            raise ValueError("source owner assignments do not match component topology")
        if self.source_tags != self.target_tags or len(self.source_components) != s:
            raise ValueError("FIT source/target or component topology changed")
        if not isinstance(self.artifacts, FitArtifactBinding):
            raise TypeError("training input lacks an exact FIT artifact binding")


def training_input_from_fit_payload(
    payload: Mapping[str, Any],
    *,
    parent_binding: Mapping[str, Any],
    require_production: bool = True,
    train_documents: int = FIT_TRAIN_DOCUMENTS,
) -> FitTrainingInput:
    """Replay one receipt-loaded payload and expose the training role only."""

    artifacts = FitArtifactBinding.from_parent_binding(parent_binding)
    if require_production and train_documents != FIT_TRAIN_DOCUMENTS:
        raise RuntimeError("production document split differs from the preregistration")
    fit_bundle.validate_fit_bundle_payload(
        payload, require_production=require_production
    )
    if payload["binding"]["authority_sha256"] != artifacts.authority_logical_sha256:
        raise RuntimeError("FIT analysis authority differs from the receipt binding")
    if payload["binding"]["source_closure_sha256"] != artifacts.source_closure_sha256:
        raise RuntimeError("FIT analysis source closure differs from the receipt binding")

    fit_response = payload["fit_response"]
    response, valid = signed_response_from_sums(
        fit_response["statistics"],
        fit_response["member_count"],
        fit_response["off_count"],
    )
    all_document_ids = fit_response["document_ids"]
    train_indices, validation_indices = prospective_document_split(
        all_document_ids, train_documents=train_documents
    )
    del validation_indices
    source_components = tuple(payload["source_components"])
    owner_components = tuple(dict.fromkeys(source_components))
    owner_index = {owner: index for index, owner in enumerate(owner_components)}
    source_groups = torch.tensor(
        [owner_index[owner] for owner in source_components], dtype=torch.int64
    )

    return FitTrainingInput(
        response=response[..., train_indices].clone().contiguous(),
        valid=valid[..., train_indices].clone().contiguous(),
        document_ids=all_document_ids[train_indices].clone().contiguous(),
        original_document_indices=train_indices.clone().contiguous(),
        source_groups=source_groups.contiguous(),
        phases=tuple(payload["phases"]),
        source_tags=tuple(payload["source_tags"]),
        target_tags=tuple(payload["target_tags"]),
        source_components=source_components,
        owner_components=owner_components,
        artifacts=artifacts,
    )
