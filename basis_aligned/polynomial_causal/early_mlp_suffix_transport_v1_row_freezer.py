"""CPU-only harvester and create-only row transaction for suffix transport v1.

This module cannot import the model.  It traverses the pinned first FineWeb parquet
in canonical order, constructs one registered candidate triple at a time, delegates
all collision decisions to :mod:`early_mlp_suffix_transport_v1_rows`, deletes every
rejected staging directory, and publishes the first collision-free receipt last.

Importing this module performs no I/O.  Numerical execution remains forbidden until
these bytes and their tests are committed, pushed, and independently audited.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import shutil
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

import early_mlp_suffix_transport_v1_lifecycle as life
import early_mlp_suffix_transport_v1_rows as identity


PINNED_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
PINNED_RELATIVE_PATH = "data/CC-MAIN-2013-20/000_00000.parquet"
PINNED_SIZE = 2_147_531_358
PINNED_SHA256 = "c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930"
ORDERED_MANIFEST_SHA256 = "ba5e92b0d157f47cc6f8656eb1c37e46b7aac6957be8be68c1596736b98e6f90"
GPT2_ENCODING_SHA256 = "0be287937901b1baae837369293dd6f63da1bece9609006e6485b57a3de37335"
DATASETS_VERSION = "5.0.1"
CANONICAL_FINEWEB_RECEIPT = life.BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
DEDUP_REFERENCE = life.BQ / "bilin18_eval_tokens_large.pt"
ROLE_TO_LICENSE_NAME = dict(zip(identity.ROLES, life.ROLE_NAMES, strict=True))


def encoding_fingerprint(encoding: Any) -> str:
    digest = __import__("hashlib").sha256()
    for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda item: item[1]):
        digest.update(len(token).to_bytes(8, "little"))
        digest.update(token)
        digest.update(int(rank).to_bytes(8, "little"))
    for token, rank in sorted(encoding._special_tokens.items()):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(rank).to_bytes(8, "little"))
    return digest.hexdigest()


def validate_ordered_source() -> tuple[dict[str, Any], Path]:
    if not CANONICAL_FINEWEB_RECEIPT.is_file() or life.file_sha256(
        CANONICAL_FINEWEB_RECEIPT
    ) != identity.CANONICAL_REGISTRY_FILES[CANONICAL_FINEWEB_RECEIPT]:
        raise RuntimeError("canonical FineWeb receipt identity changed")
    receipt = json.loads(CANONICAL_FINEWEB_RECEIPT.read_text())
    gate = receipt.get("ordered_manifest_local_parquet_identity_gate", {})
    required = {
        "passed": True,
        "revision": PINNED_REVISION,
        "config": "default",
        "ordered_file_count": 27468,
        "ordered_manifest_sha256": ORDERED_MANIFEST_SHA256,
        "first_relative_path": PINNED_RELATIVE_PATH,
        "source_size": PINNED_SIZE,
        "source_sha256": PINNED_SHA256,
        "datasets_version": DATASETS_VERSION,
    }
    if receipt.get("authorized_for_scored_experiments") is not True or any(
        gate.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError("ordered FineWeb authority changed")
    source = Path(gate.get("source_local_path", ""))
    if not source.is_file() or source.stat().st_size != PINNED_SIZE \
            or life.file_sha256(source) != PINNED_SHA256:
        raise RuntimeError("pinned FineWeb parquet identity changed")
    return dict(gate), source


def parquet_texts(source: Path) -> Iterator[tuple[str, str]]:
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(source)
    row_index = 0
    for batch in parquet_file.iter_batches(columns=["text"], batch_size=256):
        for text in batch.column(0).to_pylist():
            yield (
                f"{PINNED_REVISION}:{PINNED_RELATIVE_PATH}:{row_index}", text,
            )
            row_index += 1


def harvest_candidate(
    texts: Iterable[tuple[str, str]],
    encode: Callable[[str], list[int]],
    *,
    candidate_index: int,
    seen_prefixes: set[tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, Any]]]]:
    """Reproduce canonical skip/chunk semantics for one registered triple."""
    triple = life.candidate_triple(candidate_index)
    specs = {
        "fit": (triple.fit_n, triple.fit_skip),
        "validation": (triple.validation_n, triple.validation_skip),
        "final": (triple.final_n, triple.final_skip),
    }
    selected: dict[str, list[list[int]]] = {role: [] for role in identity.ROLES}
    records: dict[str, list[dict[str, Any]]] = {role: [] for role in identity.ROLES}
    for document_index, item in enumerate(texts):
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError("ordered source must yield (document_id,text) tuples")
        document_id, text = item
        if not isinstance(document_id, str) or not document_id \
                or not isinstance(text, str):
            raise RuntimeError("ordered source yielded malformed document")
        if not any(
            document_index >= skip and len(selected[role]) < n
            for role, (n, skip) in specs.items()
        ):
            continue
        tokens = encode(text)
        if not isinstance(tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int) for token in tokens
        ):
            raise RuntimeError("tokenizer returned malformed token sequence")
        for role, (n, skip) in specs.items():
            if document_index < skip or len(selected[role]) >= n:
                continue
            for start in range(
                0, len(tokens) - identity.TOKEN_LENGTH, identity.TOKEN_LENGTH,
            ):
                row = tokens[start:start + identity.TOKEN_LENGTH]
                if tuple(row[:32]) in seen_prefixes:
                    continue
                selected[role].append(row)
                records[role].append({
                    "document_id": document_id,
                    "dataset_document_index": document_index,
                    "chunk_id": start // identity.TOKEN_LENGTH,
                    "token_start": start,
                })
                if len(selected[role]) >= n:
                    break
        if all(len(selected[role]) == n for role, (n, _) in specs.items()):
            break
    tensors: dict[str, torch.Tensor] = {}
    for role, (n, _) in specs.items():
        value = torch.tensor(selected[role], dtype=torch.long)
        if tuple(value.shape) != (n, identity.TOKEN_LENGTH) \
                or len(records[role]) != n:
            raise RuntimeError(
                f"pinned source ended before candidate {candidate_index} {role}: "
                f"rows={tuple(value.shape)} records={len(records[role])}"
            )
        tensors[role] = value
    return tensors, records


def load_dedup_prefixes() -> set[tuple[int, ...]]:
    expected_file, expected_tensor, payload_key = identity.CANONICAL_ROW_TENSORS[
        DEDUP_REFERENCE
    ]
    if payload_key is not None or life.file_sha256(DEDUP_REFERENCE) != expected_file:
        raise RuntimeError("canonical dedup reference file changed")
    value = torch.load(DEDUP_REFERENCE, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long \
            or value.ndim != 2 or identity.tensor_raw_sha256(value) != expected_tensor:
        raise RuntimeError("canonical dedup reference tensor changed")
    return {tuple(int(token) for token in row[:32].tolist()) for row in value}


def _stage_candidate(
    staging: Path, rows_by_role: Mapping[str, torch.Tensor], candidate_index: int,
) -> dict[str, dict[str, Any]]:
    if set(rows_by_role) != set(identity.ROLES):
        raise RuntimeError("staging requires exactly fit, validation, and final rows")
    staging.mkdir(parents=False, exist_ok=False)
    triple = life.candidate_triple(candidate_index)
    skips = {
        "fit": triple.fit_skip,
        "validation": triple.validation_skip,
        "final": triple.final_skip,
    }
    entries: dict[str, dict[str, Any]] = {}
    for role in identity.ROLES:
        filename = f"{role}_n{len(rows_by_role[role])}_skip{skips[role]}.pt"
        path = staging / filename
        with path.open("xb") as handle:
            torch.save(rows_by_role[role], handle)
            handle.flush()
            os.fsync(handle.fileno())
        entries[role] = {
            "staged_path": str(path.resolve()),
            "filename": filename,
            "cache_file_sha256": life.file_sha256(path),
            "shape_full": list(rows_by_role[role].shape),
            "tensor_full_raw_sha256": life.tensor_sha256(rows_by_role[role]),
            "tensor_bytes_raw_sha256": identity.tensor_raw_sha256(rows_by_role[role]),
        }
    return entries


def expected_filename(role: str, candidate_index: int) -> str:
    if role not in identity.ROLES:
        raise ValueError(f"unknown row role {role!r}")
    triple = life.candidate_triple(candidate_index)
    counts = {"fit": triple.fit_n, "validation": triple.validation_n, "final": triple.final_n}
    skips = {
        "fit": triple.fit_skip, "validation": triple.validation_skip,
        "final": triple.final_skip,
    }
    return f"{role}_n{counts[role]}_skip{skips[role]}.pt"


def install_cache_create_only(
    staging: Path, entries: Mapping[str, Mapping[str, Any]], cache: Path,
) -> dict[str, dict[str, Any]]:
    """Install staged files by hard link without replacing any destination."""
    if set(entries) != set(identity.ROLES):
        raise RuntimeError("cache installation requires exactly three role entries")
    cache.mkdir(parents=False, exist_ok=False)
    installed: dict[str, dict[str, Any]] = {}
    for role in identity.ROLES:
        source = Path(entries[role]["staged_path"])
        destination = cache / entries[role]["filename"]
        os.link(source, destination)
        if life.file_sha256(destination) != entries[role]["cache_file_sha256"]:
            raise RuntimeError(f"installed cache file differs for {role}")
        installed[ROLE_TO_LICENSE_NAME[role]] = {
            "cache_path": str(destination.resolve()),
            "cache_file_sha256": entries[role]["cache_file_sha256"],
            "shape_full": list(entries[role]["shape_full"]),
            "tensor_full_raw_sha256": entries[role]["tensor_full_raw_sha256"],
            "tensor_bytes_raw_sha256": entries[role]["tensor_bytes_raw_sha256"],
        }
    directory = os.open(cache, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return installed


def validate_staged_payloads(
    *, staged_entries: Mapping[str, Mapping[str, Any]],
    rows_by_role: Mapping[str, torch.Tensor], candidate_index: int,
) -> None:
    """Prove serialized staging bytes contain the adjudicated in-memory tensors."""
    if set(staged_entries) != set(identity.ROLES) or set(rows_by_role) != set(
        identity.ROLES
    ):
        raise RuntimeError("staged validation requires exactly three roles")
    for role in identity.ROLES:
        entry = staged_entries[role]
        source = rows_by_role[role]
        path = Path(entry["staged_path"])
        if entry.get("filename") != expected_filename(role, candidate_index) \
                or path.name != entry["filename"] or not path.is_file():
            raise RuntimeError(f"staged row path changed for {role}")
        before = life.file_sha256(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        after = life.file_sha256(path)
        if after != before:
            raise RuntimeError(f"staged row file changed while loading: {role}")
        if not isinstance(value, torch.Tensor) or value.dtype != torch.long \
                or not torch.equal(value, source) or list(value.shape) != entry["shape_full"] \
                or before != entry["cache_file_sha256"] \
                or life.tensor_sha256(value) != entry["tensor_full_raw_sha256"] \
                or identity.tensor_raw_sha256(value) != entry["tensor_bytes_raw_sha256"]:
            raise RuntimeError(f"serialized staging differs from adjudicated rows: {role}")


def validate_publication_bindings(
    *, paths: life.ArtifactPaths, installed: Mapping[str, Mapping[str, Any]],
    rows_manifest_binding: Mapping[str, Any], collision_binding: Mapping[str, Any],
) -> None:
    if life.artifact_binding(paths.rows_manifest) != dict(rows_manifest_binding):
        raise RuntimeError("rows manifest changed before receipt publication")
    if collision_binding.get("absent") is True:
        if paths.collision_manifest.exists():
            raise RuntimeError("unexpected collision manifest appeared before publication")
    elif life.artifact_binding(paths.collision_manifest) != dict(collision_binding):
        raise RuntimeError("collision manifest changed before receipt publication")
    for role, entry in installed.items():
        path = Path(entry["cache_path"])
        if not path.is_file() or life.file_sha256(path) != entry["cache_file_sha256"]:
            raise RuntimeError(f"installed cache changed before receipt publication: {role}")


def canonical_snapshot() -> dict[str, Any]:
    """Recompute every canonical CPU identity used by the row transaction."""
    source_closure = life.freeze_source_closure(require_origin=True)
    gate, source = validate_ordered_source()
    prior, registry_census = identity.load_canonical_prior()
    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    return {
        "source_closure": source_closure,
        "gate": gate,
        "source": source,
        "source_identity": _source_identity(gate, source, encoding),
        "prior": prior,
        "registry_census": registry_census,
    }


def replay_candidate_history(
    *, selection_snapshot: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
    chosen_rows: Mapping[str, torch.Tensor],
    chosen_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Re-harvest canonical source and prove every recorded candidate decision."""
    if not history:
        raise RuntimeError("cannot replay an empty candidate history")
    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    prefixes = load_dedup_prefixes()
    source = selection_snapshot["source"]
    prior = selection_snapshot["prior"]
    for candidate_index, expected_report in enumerate(history):
        rows_by_role, records_by_role = harvest_candidate(
            parquet_texts(source), encoding.encode_ordinary,
            candidate_index=candidate_index, seen_prefixes=prefixes,
        )
        report = identity.adjudicate_candidate(
            candidate_index=candidate_index,
            rows_by_role=rows_by_role,
            records_by_role=records_by_role,
            prior=prior,
        )
        if report != expected_report:
            raise RuntimeError(f"canonical candidate replay changed at {candidate_index}")
        if candidate_index == len(history) - 1:
            if any(not torch.equal(rows_by_role[role], chosen_rows[role])
                   for role in identity.ROLES) or {
                       role: records_by_role[role] for role in identity.ROLES
                   } != {
                       role: list(chosen_records[role]) for role in identity.ROLES
                   }:
                raise RuntimeError("chosen staged rows/provenance differ from canonical harvest")


def publish_frozen_candidate_locked(
    *,
    lock_nonce: str,
    paths: life.ArtifactPaths,
    staging: Path,
    staged_entries: Mapping[str, Mapping[str, Any]],
    rows_by_role: Mapping[str, torch.Tensor],
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    history: Sequence[Mapping[str, Any]],
    selection_snapshot: Mapping[str, Any],
    lock_path: Path = life.RUN_LOCK,
) -> dict[str, Any]:
    """Install cache, write optional collisions and manifest, then receipt last."""
    life.require_run_claim(lock_nonce, lock_path)
    current = canonical_snapshot()
    for key in ("source_closure", "gate", "source", "source_identity", "prior",
                "registry_census"):
        if current[key] != selection_snapshot[key]:
            raise RuntimeError(f"canonical selection snapshot changed: {key}")
    prior = current["prior"]
    registry_census = current["registry_census"]
    source_closure = current["source_closure"]
    source_identity = current["source_identity"]
    chosen_index = len(history) - 1
    if set(rows_by_role) != set(identity.ROLES) or set(records_by_role) != set(
        identity.ROLES
    ) or set(staged_entries) != set(identity.ROLES):
        raise RuntimeError("publication requires exactly three candidate roles")
    identity.validate_collision_history(history, chosen_index)
    replay_candidate_history(
        selection_snapshot=current,
        history=history,
        chosen_rows=rows_by_role,
        chosen_records=records_by_role,
    )
    recomputed = identity.adjudicate_candidate(
        candidate_index=chosen_index,
        rows_by_role=rows_by_role,
        records_by_role=records_by_role,
        prior=prior,
    )
    if recomputed != history[-1]:
        raise RuntimeError("chosen collision decision does not bind staged rows/provenance")
    validate_staged_payloads(
        staged_entries=staged_entries,
        rows_by_role=rows_by_role,
        candidate_index=chosen_index,
    )
    if paths.cache.exists() or any(path.exists() for path in paths.output_files()):
        raise RuntimeError("suffix-transport row namespace is already spent")
    installed = install_cache_create_only(staging, staged_entries, paths.cache)
    rejected = [dict(report) for report in history[:-1]]
    collision_binding: dict[str, Any]
    if rejected:
        collision_payload = {
            "schema_version": 1,
            "status": "rejected_candidates_hash_only",
            "authority": "none",
            "reports": rejected,
            "reports_sha256": life.logical_json_sha256(rejected),
        }
        life.atomic_create_json(collision_payload, paths.collision_manifest)
        collision_binding = life.artifact_binding(paths.collision_manifest)
    else:
        collision_binding = {
            "path": str(paths.collision_manifest.resolve()), "absent": True,
        }

    named_records = {
        ROLE_TO_LICENSE_NAME[role]: [dict(record) for record in records_by_role[role]]
        for role in identity.ROLES
    }
    manifest = {
        "schema_version": 1,
        "status": "rows_frozen_before_any_model_forward",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "chosen_candidate_index": chosen_index,
        "chosen_decision": dict(history[-1]),
        "collision_history_sha256": identity.collision_history_hash(history),
        "collision_manifest": collision_binding,
        "entries": installed,
        "role_record_counts": {role: len(values) for role, values in named_records.items()},
        "role_record_hashes": {
            role: life.logical_json_sha256(values) for role, values in named_records.items()
        },
        "role_licenses": life.ROLE_LICENSES,
        "registry_census": dict(registry_census),
        "source_closure": dict(source_closure),
        "source_identity": dict(source_identity),
    }
    life.atomic_create_json(manifest, paths.rows_manifest)
    rows_manifest_binding = life.artifact_binding(paths.rows_manifest)
    receipt = {
        **manifest,
        "status": "row_roles_frozen_before_any_model_forward",
        "receipt_kind": "early_mlp_suffix_transport_v1_rows",
        "rows_manifest": rows_manifest_binding,
        "document_provenance": {"schema_version": 1, "sets": named_records},
    }
    life.require_run_claim(lock_nonce, lock_path)
    final_snapshot = canonical_snapshot()
    for key in ("source_closure", "gate", "source", "source_identity", "prior",
                "registry_census"):
        if final_snapshot[key] != selection_snapshot[key]:
            raise RuntimeError(f"canonical snapshot drifted before receipt: {key}")
    validate_publication_bindings(
        paths=paths, installed=installed,
        rows_manifest_binding=rows_manifest_binding,
        collision_binding=collision_binding,
    )
    # The canonical replay and artifact rehashes above are deliberately heavyweight.
    # Re-establish exclusive ownership at the actual authority boundary rather than
    # relying on the claim that preceded them.
    life.require_run_claim(lock_nonce, lock_path)
    life.atomic_create_json(receipt, paths.rows_receipt)
    reloaded = json.loads(paths.rows_receipt.read_text())
    if reloaded != receipt:
        raise RuntimeError("last-written row receipt did not reload exactly")
    return receipt


def _source_identity(gate: Mapping[str, Any], source: Path, encoding: Any) -> dict[str, Any]:
    import datasets
    import pyarrow
    import tiktoken

    fingerprint = encoding_fingerprint(encoding)
    if datasets.__version__ != DATASETS_VERSION or fingerprint != GPT2_ENCODING_SHA256:
        raise RuntimeError("dataset/tokenizer environment identity changed")
    return {
        "ordered_manifest_gate": dict(gate),
        "source_path": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "source_sha256": life.file_sha256(source),
        "datasets_version": datasets.__version__,
        "pyarrow_version": pyarrow.__version__,
        "tiktoken_version": tiktoken.__version__,
        "gpt2_encoding_sha256": fingerprint,
        "loader_semantics": (
            "pinned first parquet in certified order; gpt2 encode_ordinary; "
            "canonical eval prefix32 dedup; range(0,len(tokens)-513,513)"
        ),
    }


def freeze_rows_locked(
    *, lock_nonce: str, paths: life.ArtifactPaths = life.PATHS,
    lock_path: Path = life.RUN_LOCK,
) -> dict[str, Any]:
    """Execute the complete CPU row transaction under an owned experiment lock."""
    life.require_run_claim(lock_nonce, lock_path)
    paths.assert_stage_preconditions("rows")
    selection_snapshot = canonical_snapshot()
    source = selection_snapshot["source"]
    prior = selection_snapshot["prior"]
    prefixes = load_dedup_prefixes()
    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    history: list[dict[str, Any]] = []
    candidate_index = 0
    while True:
        life.require_run_claim(lock_nonce, lock_path)
        rows_by_role, records_by_role = harvest_candidate(
            parquet_texts(source), encoding.encode_ordinary,
            candidate_index=candidate_index, seen_prefixes=prefixes,
        )
        staging = paths.cache.with_name(
            f".{paths.cache.name}.candidate{candidate_index}.tmp."
            f"{os.getpid()}.{secrets.token_hex(8)}"
        )
        accepted = False
        try:
            staged_entries = _stage_candidate(staging, rows_by_role, candidate_index)
            report = identity.adjudicate_candidate(
                candidate_index=candidate_index,
                rows_by_role=rows_by_role,
                records_by_role=records_by_role,
                prior=prior,
            )
            history.append(report)
            accepted = report["accepted"]
            if accepted:
                break
        finally:
            if not accepted and staging.exists():
                shutil.rmtree(staging)
                if staging.exists():
                    raise RuntimeError("rejected candidate staging was not deleted")
        candidate_index += 1

    try:
        # No authority can be published against drifted source, code, registries, or ship.
        return publish_frozen_candidate_locked(
            lock_nonce=lock_nonce,
            paths=paths,
            staging=staging,
            staged_entries=staged_entries,
            rows_by_role=rows_by_role,
            records_by_role=records_by_role,
            history=history,
            selection_snapshot=selection_snapshot,
            lock_path=lock_path,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def freeze_rows(paths: life.ArtifactPaths = life.PATHS) -> dict[str, Any]:
    with life.exclusive_run_claim() as nonce:
        return freeze_rows_locked(lock_nonce=nonce, paths=paths)
