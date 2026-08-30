"""Pure boundary from a validated FIT payload to factor-analysis tensors.

This module deliberately has no filesystem, model, corpus, or EVAL capability.  A
separate receipt-bound lifecycle must load the exact FIT artifact and verify its
artifact digest.  Only then may it pass the in-memory payload here.  This adapter
replays the bundle's semantic validator, checks the authority binding, derives the
signed response, and returns cloned CPU tensors with the frozen document split and
owner topology.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FitFactorizationInput:
    """The complete FIT-only numerical input to factorization v1.

    ``response`` and ``valid`` use axes ``[phase, source, target, document]``.
    ``source_groups`` maps each source to the corresponding entry of
    ``owner_components``.  All tensors are independent contiguous CPU clones.
    """

    response: torch.Tensor
    valid: torch.Tensor
    document_ids: torch.Tensor
    train_document_indices: torch.Tensor
    validation_document_indices: torch.Tensor
    source_groups: torch.Tensor
    phases: tuple[str, ...]
    source_tags: tuple[str, ...]
    target_tags: tuple[str, ...]
    source_components: tuple[str, ...]
    owner_components: tuple[str, ...]
    authority_sha256: str

    def __post_init__(self) -> None:
        p = len(self.phases)
        s = len(self.source_tags)
        t = len(self.target_tags)
        d = self.document_ids.numel()
        if self.response.dtype != torch.float64 or self.response.shape != (p, s, t, d):
            raise ValueError("response does not match the declared FIT axes")
        if self.valid.dtype != torch.bool or self.valid.shape != self.response.shape:
            raise ValueError("valid mask does not align with the response")
        if self.document_ids.dtype != torch.int64 or self.document_ids.shape != (d,):
            raise ValueError("document IDs are malformed")
        if self.source_groups.dtype != torch.int64 or self.source_groups.shape != (s,):
            raise ValueError("source owner assignments are malformed")
        tensors = (
            self.response,
            self.valid,
            self.document_ids,
            self.train_document_indices,
            self.validation_document_indices,
            self.source_groups,
        )
        if any(value.device.type != "cpu" or not value.is_contiguous() for value in tensors):
            raise ValueError("factorization inputs must be contiguous CPU tensors")
        roles = torch.cat((self.train_document_indices, self.validation_document_indices))
        if roles.numel() != d or not torch.equal(
            torch.sort(roles).values, torch.arange(d, dtype=torch.int64)
        ):
            raise ValueError("document roles must partition the FIT documents")
        if len(self.owner_components) == 0 or set(self.source_components) != set(
            self.owner_components
        ):
            raise ValueError("owner component topology is incomplete")
        if not _is_sha256(self.authority_sha256):
            raise ValueError("authority binding is not a lowercase SHA-256")


def factorization_input_from_fit_payload(
    payload: Mapping[str, Any],
    *,
    expected_authority_sha256: str,
    require_production: bool = True,
    train_documents: int = FIT_TRAIN_DOCUMENTS,
) -> FitFactorizationInput:
    """Replay and reduce an already receipt-loaded FIT payload.

    This function cannot establish the artifact digest because it accepts no bytes or
    path.  Its caller must first prove that the payload came from the receipt-bound
    artifact.  It does establish the complete semantic bundle contract and authority
    identity before exposing any response value to the optimizer.
    """

    if not _is_sha256(expected_authority_sha256):
        raise ValueError("expected FIT authority hash is malformed")
    if require_production and train_documents != FIT_TRAIN_DOCUMENTS:
        raise RuntimeError("production document split differs from the preregistration")
    fit_bundle.validate_fit_bundle_payload(
        payload, require_production=require_production
    )
    if payload["binding"]["authority_sha256"] != expected_authority_sha256:
        raise RuntimeError("FIT analysis authority differs from the receipt binding")

    fit_response = payload["fit_response"]
    response, valid = signed_response_from_sums(
        fit_response["statistics"],
        fit_response["member_count"],
        fit_response["off_count"],
    )
    document_ids = fit_response["document_ids"].clone().contiguous()
    train_indices, validation_indices = prospective_document_split(
        document_ids, train_documents=train_documents
    )
    source_components = tuple(payload["source_components"])
    owner_components = tuple(dict.fromkeys(source_components))
    owner_index = {owner: index for index, owner in enumerate(owner_components)}
    source_groups = torch.tensor(
        [owner_index[owner] for owner in source_components], dtype=torch.int64
    )

    return FitFactorizationInput(
        response=response.clone().contiguous(),
        valid=valid.clone().contiguous(),
        document_ids=document_ids,
        train_document_indices=train_indices.clone().contiguous(),
        validation_document_indices=validation_indices.clone().contiguous(),
        source_groups=source_groups.contiguous(),
        phases=tuple(payload["phases"]),
        source_tags=tuple(payload["source_tags"]),
        target_tags=tuple(payload["target_tags"]),
        source_components=source_components,
        owner_components=owner_components,
        authority_sha256=expected_authority_sha256,
    )
