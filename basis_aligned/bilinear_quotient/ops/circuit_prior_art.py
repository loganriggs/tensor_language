#!/usr/bin/env python3
# BQLANE: cpu
"""Pure validation for circuit prior-art and novelty review receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "circuit_prior_art_and_novelty_receipt_v1"
RELATIONS = frozenset({"replication", "extension", "contradiction_test", "new_question"})
RECEIPT_FIELDS = frozenset({
    "schema", "candidate_id", "canonical_objects", "aliases_searched",
    "method_families", "matched_prior_claims", "relation", "novelty_delta",
    "decision_changed", "reviewer", "reviewed_sources",
})
SOURCE_FIELDS = frozenset({"source", "sha256", "searched_terms"})


class PriorArtError(ValueError):
    """The proposed receipt is incomplete, inconsistent, or non-canonical."""


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PriorArtError("receipt is not canonical finite JSON") from error


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise PriorArtError(f"{label} must be nonempty trimmed text")
    return value


def _text_list(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if type(value) is not list or (not value and not allow_empty):
        raise PriorArtError(f"{label} must be a{' possibly empty' if allow_empty else ' nonempty'} list")
    output = [_text(item, f"{label} item") for item in value]
    if len(output) != len(set(output)):
        raise PriorArtError(f"{label} contains duplicates")
    return output


def validate_receipt(receipt: Mapping[str, Any]) -> str:
    """Validate one receipt and return its canonical SHA-256 identity."""
    if type(receipt) is not dict or set(receipt) != RECEIPT_FIELDS:
        raise PriorArtError("receipt fields differ from the exact v1 schema")
    if receipt["schema"] != SCHEMA:
        raise PriorArtError("receipt schema changed")
    _text(receipt["candidate_id"], "candidate_id")
    _text_list(receipt["canonical_objects"], "canonical_objects")
    _text_list(receipt["aliases_searched"], "aliases_searched")
    _text_list(receipt["method_families"], "method_families")
    matches = _text_list(
        receipt["matched_prior_claims"], "matched_prior_claims", allow_empty=True,
    )
    relation = receipt["relation"]
    if type(relation) is not str or relation not in RELATIONS:
        raise PriorArtError("relation is not registered")
    if relation == "new_question" and matches:
        raise PriorArtError("new_question cannot have matched prior claims")
    if relation != "new_question" and not matches:
        raise PriorArtError("replication/extension/contradiction_test requires a prior match")
    _text(receipt["novelty_delta"], "novelty_delta")
    _text(receipt["decision_changed"], "decision_changed")
    _text(receipt["reviewer"], "reviewer")

    sources = receipt["reviewed_sources"]
    if type(sources) is not list or not sources:
        raise PriorArtError("reviewed_sources must contain evidence")
    source_names: list[str] = []
    searched_evidence = 0
    for index, source in enumerate(sources):
        if type(source) is not dict or set(source) != SOURCE_FIELDS:
            raise PriorArtError(f"reviewed_sources[{index}] fields changed")
        source_names.append(_text(source["source"], f"reviewed_sources[{index}].source"))
        digest = source["sha256"]
        if type(digest) is not str or len(digest) != 64 or digest != digest.lower():
            raise PriorArtError(f"reviewed_sources[{index}].sha256 is not exact lowercase SHA-256")
        try:
            bytes.fromhex(digest)
        except ValueError as error:
            raise PriorArtError(f"reviewed_sources[{index}].sha256 is not hexadecimal") from error
        searched_evidence += len(_text_list(
            source["searched_terms"], f"reviewed_sources[{index}].searched_terms",
        ))
    if len(source_names) != len(set(source_names)):
        raise PriorArtError("reviewed_sources contains duplicate source identities")
    if searched_evidence == 0:
        raise PriorArtError("receipt contains no searched-term evidence")
    return canonical_hash(receipt)


def validate_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate an ordered receipt collection, rejecting candidate reuse."""
    if type(receipts) is not list or not receipts:
        raise PriorArtError("receipt collection must be a nonempty list")
    candidate_ids: set[str] = set()
    hashes: list[str] = []
    for receipt in receipts:
        digest = validate_receipt(receipt)
        candidate_id = receipt["candidate_id"]
        if candidate_id in candidate_ids:
            raise PriorArtError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        hashes.append(digest)
    return hashes


def validate_source_files(receipt: Mapping[str, Any], base_dir: Path) -> str:
    """Validate the receipt and prove that every reviewed source hash is current."""
    digest = validate_receipt(receipt)
    root = base_dir.resolve()
    for source in receipt["reviewed_sources"]:
        relative = Path(source["source"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PriorArtError(f"reviewed source path is not contained: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise PriorArtError(f"reviewed source is missing or unsafe: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != source["sha256"]:
            raise PriorArtError(
                f"reviewed source changed: {relative}; "
                f"expected={source['sha256']}, observed={observed}"
            )
    return digest
