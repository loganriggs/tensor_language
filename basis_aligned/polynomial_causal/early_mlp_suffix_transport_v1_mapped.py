"""Document-block mappings for suffix-transport shuffled and A-null controls.

This module constructs and validates the only legal relation between a student fit
batch and a false-pairing teacher batch.  It does not execute a model or expose a
teacher capability.  A later mapped broker must consume :class:`MappedRunContext`
instead of accepting a control name or arbitrary target tensor from a runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_rows as row_contract


DOCUMENT_SHUFFLE_SEED = 2026083050
A_NULL_SEED_START = 2026083100
PROVENANCE_KEYS = {
    "document_id", "dataset_document_index", "chunk_id", "token_start",
}


def control_seed(control: str) -> int:
    if control == "document_shuffle":
        return DOCUMENT_SHUFFLE_SEED
    if isinstance(control, str) and control.startswith("A_null_"):
        suffix = control.removeprefix("A_null_")
        if len(suffix) == 2 and suffix.isdigit() and int(suffix) in range(20):
            return A_NULL_SEED_START + int(suffix)
    raise ValueError("mapped control must be document_shuffle or A_null_00..19")


@dataclass(frozen=True)
class StratumRotation:
    rows_per_document: int
    documents: tuple[str, ...]
    offset: int


@dataclass(frozen=True)
class DocumentBlockPlan:
    """Immutable bijection from source rows to false-pairing target rows."""

    control: str
    seed: int
    row_count: int
    row_targets: tuple[int, ...]
    source_documents: tuple[str, ...]
    target_documents: tuple[str, ...]
    strata: tuple[StratumRotation, ...]
    sha256: str

    def target_indices(self, source_indices: Sequence[int]) -> tuple[int, ...]:
        indices = tuple(source_indices)
        if not indices or any(
            type(index) is not int or not 0 <= index < self.row_count for index in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError("mapped source indices are malformed or duplicated")
        targets = tuple(self.row_targets[index] for index in indices)
        if any(
            self.source_documents[source] == self.target_documents[source]
            for source in indices
        ):
            raise RuntimeError("mapped batch contains a fixed document")
        return targets


def _validate_records(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not records:
        raise ValueError("document-block mapping requires provenance records")
    documents = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != PROVENANCE_KEYS:
            raise ValueError(f"mapped provenance schema changed at row {index}")
        document = record["document_id"]
        dataset_index = record["dataset_document_index"]
        chunk_id = record["chunk_id"]
        token_start = record["token_start"]
        if not isinstance(document, str) or not document or type(dataset_index) is not int \
                or dataset_index < 0 or type(chunk_id) is not int or chunk_id < 0 \
                or type(token_start) is not int or token_start < 0:
            raise ValueError(f"mapped provenance value changed at row {index}")
        documents.append(document)
    return tuple(documents)


def build_document_block_plan(
    records: Sequence[Mapping[str, Any]], *, control: str,
) -> DocumentBlockPlan:
    """Rotate whole documents within equal-row-count strata using the frozen seed."""

    seed = control_seed(control)
    source_documents = _validate_records(records)
    rows_by_document: dict[str, list[int]] = {}
    document_order: list[str] = []
    for row_index, document in enumerate(source_documents):
        if document not in rows_by_document:
            rows_by_document[document] = []
            document_order.append(document)
        rows_by_document[document].append(row_index)
    # A whole-document block must really be a block in canonical source order.  If
    # documents interleave, a cyclic document permutation is no longer a block map.
    for document, indices in rows_by_document.items():
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise RuntimeError(f"document rows are not contiguous: {document}")

    documents_by_count: dict[int, list[str]] = {}
    for document in document_order:
        documents_by_count.setdefault(len(rows_by_document[document]), []).append(document)
    if any(len(documents) < 2 for documents in documents_by_count.values()):
        raise RuntimeError("every equal-row-count stratum needs at least two documents")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    row_targets = [-1] * len(records)
    target_documents = [""] * len(records)
    rotations = []
    for row_count in sorted(documents_by_count):
        documents = documents_by_count[row_count]
        offset = int(torch.randint(
            1, len(documents), (1,), generator=generator, dtype=torch.int64,
        ))
        rotations.append(StratumRotation(
            rows_per_document=row_count, documents=tuple(documents), offset=offset,
        ))
        for source_ordinal, source_document in enumerate(documents):
            target_document = documents[(source_ordinal + offset) % len(documents)]
            for within_document, source_row in enumerate(rows_by_document[source_document]):
                target_row = rows_by_document[target_document][within_document]
                row_targets[source_row] = target_row
                target_documents[source_row] = target_document

    if sorted(row_targets) != list(range(len(records))):
        raise RuntimeError("document-block map is not a row permutation")
    if any(
        source == target for source, target in zip(source_documents, target_documents, strict=True)
    ):
        raise RuntimeError("document-block map has a fixed document")
    payload = {
        "schema_version": 1,
        "control": control,
        "seed": seed,
        "row_count": len(records),
        "source_documents": list(source_documents),
        "target_documents": target_documents,
        "row_targets": row_targets,
        "strata": [
            {
                "rows_per_document": item.rows_per_document,
                "documents": list(item.documents),
                "offset": item.offset,
            }
            for item in rotations
        ],
    }
    return DocumentBlockPlan(
        control=control, seed=seed, row_count=len(records),
        row_targets=tuple(row_targets), source_documents=source_documents,
        target_documents=tuple(target_documents), strata=tuple(rotations),
        sha256=runtime.logical_identity_sha256(payload),
    )


@dataclass(frozen=True)
class MappedRunContext:
    """Bind one control plan to the sealed fit role and exact target token batches."""

    base: capabilities.RunContext
    plan: DocumentBlockPlan

    def __post_init__(self) -> None:
        if not isinstance(self.base, capabilities.RunContext) or not isinstance(
            self.plan, DocumentBlockPlan
        ) or self.plan.row_count != self.base.fit_row_count:
            raise ValueError("mapped run context differs from the sealed fit role")

    def require_identity(
        self, identity: runtime.TraceIdentity, *, fit_rows: torch.Tensor,
        student_inputs: torch.Tensor, student_indices: Sequence[int],
        teacher_inputs: torch.Tensor, teacher_indices: Sequence[int],
    ) -> None:
        self.base.require_common_identity(identity, student_inputs, student_indices)
        if identity.phase != "fit" or identity.control != self.plan.control or (
            identity.teacher_mapping_sha256 != self.plan.sha256
        ):
            raise RuntimeError("trace differs from the sealed mapped control")
        if self.plan.control == "document_shuffle":
            if identity.route not in {"L", "R", "S0", "S1"}:
                raise RuntimeError("document shuffle is not licensed for this route")
        elif identity.route != "T":
            raise RuntimeError("A-null mapping is licensed only for T")
        if not torch.is_tensor(fit_rows) or fit_rows.device.type != "cpu" or (
            fit_rows.dtype != torch.long
        ) or tuple(fit_rows.shape) != (
            self.base.fit_row_count, row_contract.TOKEN_LENGTH,
        ) or runtime.tensor_identity_sha256(fit_rows) != self.base.fit_role_tensor_sha256:
            raise RuntimeError("mapped context fit tensor differs from the sealed role")
        expected_indices = self.plan.target_indices(student_indices)
        if tuple(teacher_indices) != expected_indices:
            raise RuntimeError("teacher indices differ from the document-block map")
        if not torch.is_tensor(teacher_inputs) or teacher_inputs.dtype != torch.long or tuple(
            teacher_inputs.shape
        ) != (runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH):
            raise RuntimeError("teacher tokens differ from the mapped fit rows")
        expected_inputs = fit_rows[
            torch.tensor(expected_indices, dtype=torch.long), :runtime.SEQUENCE_LENGTH
        ].to(device=teacher_inputs.device)
        if not torch.equal(teacher_inputs, expected_inputs):
            raise RuntimeError("teacher tokens differ from the mapped fit rows")
