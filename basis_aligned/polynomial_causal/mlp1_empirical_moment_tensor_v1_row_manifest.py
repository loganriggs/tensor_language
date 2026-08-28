"""Pure document-role manifest for the MLP1 empirical-moment experiment.

This module imports no data, tokenizer, tensor, or model library.  It converts a
frozen parquet row count and a set of excluded document indices into the exact
SHA-ordered FIT/VALIDATION/REPLICATION assignment registered by the execution
addendum.  The returned object is not an activation or scoring authority.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


EXPERIMENT_ID = "bilin18_mlp1_empirical_moment_v1"
ROLES = ("FIT", "VALIDATION", "REPLICATION")
DOCUMENTS_PER_ROLE = 2_084
TOTAL_DOCUMENTS = len(ROLES) * DOCUMENTS_PER_ROLE
WINDOW_TOKENS = 256
POSITION_START = 64
POSITION_STOP = 256
POSITIONS_PER_FULL_WINDOW = POSITION_STOP - POSITION_START
ROWS_PER_ROLE = 400_000
ROLE_FULL_WINDOWS = 2_083
ROLE_FINAL_POSITION_STOP = 128

FIT_PREFIXES = {
    "FIT100": {
        "rows": 100_000,
        "full_windows": 520,
        "partial_document_ordinal": 520,
        "partial_position_start": 64,
        "partial_position_stop": 224,
    },
    "FIT200": {
        "rows": 200_000,
        "full_windows": 1_041,
        "partial_document_ordinal": 1_041,
        "partial_position_start": 64,
        "partial_position_stop": 192,
    },
    "FIT400": {
        "rows": 400_000,
        "full_windows": 2_083,
        "partial_document_ordinal": 2_083,
        "partial_position_start": 64,
        "partial_position_stop": 128,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def document_position_ledger_sha256(
    indices: Iterable[int], *, full_windows: int, final_stop: int,
) -> str:
    """Hash an ordered document/position ledger without materializing its rows."""
    digest = hashlib.sha256(b"mlp1-empirical-moment-document-position-ledger-v1\0")
    values = tuple(indices)
    if len(values) != full_windows + 1:
        raise RuntimeError("document-position ledger window count changed")
    for ordinal, document_index in enumerate(values):
        checked = _literal_index(document_index, "ledger document index")
        stop = POSITION_STOP if ordinal < full_windows else final_stop
        for position in range(POSITION_START, stop):
            digest.update(checked.to_bytes(8, "little", signed=False))
            digest.update(position.to_bytes(2, "little", signed=False))
    return digest.hexdigest()


def _literal_index(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context} must be a literal nonnegative integer")
    return value


def ordering_digest(document_index: int) -> bytes:
    index = _literal_index(document_index, "document index")
    return hashlib.sha256(
        EXPERIMENT_ID.encode("utf-8") + b"\0" + str(index).encode("ascii")
    ).digest()


def order_surviving_indices(
    parquet_rows: int, excluded_indices: Iterable[int], *, limit: int = TOTAL_DOCUMENTS,
) -> tuple[int, ...]:
    """Return the exact prospective prefix of the SHA-ordered survivor population."""
    count = _literal_index(parquet_rows, "parquet_rows")
    wanted = _literal_index(limit, "limit")
    excluded = {
        _literal_index(value, "excluded document index") for value in excluded_indices
    }
    outside = sorted(value for value in excluded if value >= count)
    if outside:
        raise RuntimeError(
            f"registry exclusion is outside pinned parquet range: {outside[:3]}"
        )
    survivors = count - len(excluded)
    if survivors < wanted:
        raise RuntimeError(
            f"pinned parquet has only {survivors} unexcluded documents; need {wanted}"
        )
    return tuple(sorted(
        (index for index in range(count) if index not in excluded),
        key=lambda index: (ordering_digest(index), index),
    )[:wanted])


def _position_mask(stop: int) -> dict[str, int]:
    if not POSITION_START < stop <= POSITION_STOP:
        raise RuntimeError("invalid registered position-mask stop")
    return {
        "position_start_inclusive": POSITION_START,
        "position_stop_exclusive": stop,
        "position_count": stop - POSITION_START,
    }


def _role_record(role: str, indices: tuple[int, ...]) -> dict[str, Any]:
    if role not in ROLES or len(indices) != DOCUMENTS_PER_ROLE:
        raise RuntimeError("role assignment does not have the registered shape")
    if len(set(indices)) != DOCUMENTS_PER_ROLE:
        raise RuntimeError("role assignment repeats a document")
    full_rows = ROLE_FULL_WINDOWS * POSITIONS_PER_FULL_WINDOW
    final_rows = ROLE_FINAL_POSITION_STOP - POSITION_START
    if full_rows + final_rows != ROWS_PER_ROLE:
        raise AssertionError("registered 400k role arithmetic changed")
    return {
        "ordered_document_indices": list(indices),
        "ordered_document_indices_sha256": sha256_json(list(indices)),
        "document_count": DOCUMENTS_PER_ROLE,
        "eligible_row_count": ROWS_PER_ROLE,
        "complete_window_count": ROLE_FULL_WINDOWS,
        "complete_window_position_mask": _position_mask(POSITION_STOP),
        "partial_window": {
            "document_ordinal_zero_indexed": ROLE_FULL_WINDOWS,
            **_position_mask(ROLE_FINAL_POSITION_STOP),
        },
        "ordered_document_position_ledger_sha256": document_position_ledger_sha256(
            indices, full_windows=ROLE_FULL_WINDOWS,
            final_stop=ROLE_FINAL_POSITION_STOP,
        ),
    }


def _fit_prefix_record(
    fit_indices: tuple[int, ...], label: str, specification: Mapping[str, int],
) -> dict[str, Any]:
    full = specification["full_windows"]
    ordinal = specification["partial_document_ordinal"]
    stop = specification["partial_position_stop"]
    rows = specification["rows"]
    if ordinal != full or ordinal >= len(fit_indices):
        raise AssertionError(f"{label} document ordinal changed")
    if full * POSITIONS_PER_FULL_WINDOW + stop - POSITION_START != rows:
        raise AssertionError(f"{label} row arithmetic changed")
    selected = fit_indices[:ordinal + 1]
    return {
        "row_count": rows,
        "complete_window_count": full,
        "document_count": len(selected),
        "ordered_document_indices_sha256": sha256_json(list(selected)),
        "partial_window": {
            "document_ordinal_zero_indexed": ordinal,
            **_position_mask(stop),
        },
        "ordered_document_position_ledger_sha256": document_position_ledger_sha256(
            selected, full_windows=full, final_stop=stop,
        ),
    }


def build_role_manifest(
    *, parquet_rows: int, excluded_indices: Iterable[int], registry_census: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = order_surviving_indices(parquet_rows, excluded_indices)
    role_indices = {
        role: ordered[offset:offset + DOCUMENTS_PER_ROLE]
        for role, offset in zip(ROLES, range(0, TOTAL_DOCUMENTS, DOCUMENTS_PER_ROLE), strict=True)
    }
    if len(set(ordered)) != TOTAL_DOCUMENTS:
        raise RuntimeError("cross-role document collision")
    excluded = sorted({_literal_index(value, "excluded document index")
                       for value in excluded_indices})
    roles = {role: _role_record(role, role_indices[role]) for role in ROLES}
    fit_prefixes = {
        label: _fit_prefix_record(role_indices["FIT"], label, specification)
        for label, specification in FIT_PREFIXES.items()
    }
    manifest = {
        "schema_version": 1,
        "manifest_kind": "mlp1_empirical_moment_tensor_v1_row_roles",
        "status": "non_authorizing_role_manifest_pending_last_written_receipt",
        "authority": "none",
        "authorized_for_tokenization": False,
        "authorized_for_activation_capture": False,
        "authorized_for_model_forward": False,
        "authorized_for_scientific_outcomes": False,
        "experiment_id": EXPERIMENT_ID,
        "selection_rule": (
            "sha256(experiment_id+NUL+decimal_document_index),numeric_index;"
            "first_2084_FIT_next_2084_VALIDATION_next_2084_REPLICATION"
        ),
        "parquet_row_count": parquet_rows,
        "excluded_document_index_count": len(excluded),
        "excluded_document_indices_sha256": sha256_json(excluded),
        "registry_census": dict(registry_census),
        "roles": roles,
        "fit_nested_prefixes": fit_prefixes,
        "cross_role_document_disjoint": all(
            set(role_indices[left]).isdisjoint(role_indices[right])
            for left, right in (("FIT", "VALIDATION"), ("FIT", "REPLICATION"),
                                ("VALIDATION", "REPLICATION"))
        ),
        "ordered_all_roles_document_indices_sha256": sha256_json(list(ordered)),
        "token_identity_state": "not_loaded_not_hashed_not_authorized",
    }
    validate_role_manifest(manifest)
    return manifest


def validate_role_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("authority") != "none":
        raise RuntimeError("row-role manifest header changed")
    if any(manifest.get(key) is not False for key in (
        "authorized_for_tokenization", "authorized_for_activation_capture",
        "authorized_for_model_forward", "authorized_for_scientific_outcomes",
    )):
        raise RuntimeError("row-role manifest leaked downstream authority")
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or tuple(roles) != ROLES:
        raise RuntimeError("row-role manifest role order changed")
    seen: set[int] = set()
    for role in ROLES:
        record = roles[role]
        indices = record.get("ordered_document_indices")
        if not isinstance(indices, list) or len(indices) != DOCUMENTS_PER_ROLE:
            raise RuntimeError(f"{role} document assignment changed")
        checked = tuple(_literal_index(value, f"{role} document index") for value in indices)
        if seen.intersection(checked) or len(set(checked)) != DOCUMENTS_PER_ROLE:
            raise RuntimeError("row roles are not document-disjoint")
        seen.update(checked)
        rebuilt = _role_record(role, checked)
        if record != rebuilt:
            raise RuntimeError(f"{role} mask or hash is not canonical")
    fit = tuple(roles["FIT"]["ordered_document_indices"])
    expected_prefixes = {
        label: _fit_prefix_record(fit, label, specification)
        for label, specification in FIT_PREFIXES.items()
    }
    if manifest.get("fit_nested_prefixes") != expected_prefixes:
        raise RuntimeError("FIT nested-prefix masks or hashes changed")
    if manifest.get("cross_role_document_disjoint") is not True:
        raise RuntimeError("cross-role disjointness gate is not literal true")
    if manifest.get("ordered_all_roles_document_indices_sha256") != sha256_json([
        index for role in ROLES for index in roles[role]["ordered_document_indices"]
    ]):
        raise RuntimeError("all-role ordered document hash changed")
