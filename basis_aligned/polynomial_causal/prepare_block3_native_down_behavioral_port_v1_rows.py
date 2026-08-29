#!/usr/bin/env python3
"""Outcome-blind fresh-row freezer for the Block-3 native-Down port.

This source is deliberately incapable of loading a checkpoint or running a model.  It
selects one 257-token row from each of 192 new FineWeb source documents, after a
recursive census of every prior row-bearing registry JSON under ``basis_aligned``.
Publication is create-only: cache first, receipt last, with a complete re-harvest and
registry/source/lock revalidation immediately before the receipt write.

The canonical transaction must not be executed until these exact sources have been
committed, pushed, and independently audited.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import copy
from typing import Any, Callable, Iterable, Mapping, NamedTuple

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASIS = ROOT / "basis_aligned"
BQ = BASIS / "bilinear_quotient"
BASE_PATH = HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py"
REGISTRY_PATH = HERE / "prepare_mlp0_c512_mlp2_compensation_v1_rows.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = _load_module("block3_native_down_row_base", BASE_PATH)
REGISTRY = _load_module("block3_native_down_row_registry", REGISTRY_PATH)

START_DOCUMENT_INDEX = 60_000
N_SOURCE_DOCUMENTS = 192
TOKEN_LENGTH = 257
PREFIX_LENGTH = 32
MAX_CHUNKS_PER_DOCUMENT = 1

CACHE = BQ / ".rowcache_block3_native_down_behavioral_port_v1"
RECEIPT = BQ / "block3_native_down_behavioral_port_v1_rows_receipt.json"
LOCK = Path("/workspace/runs/.block3_native_down_behavioral_port_v1_rows.lock")
ADDENDUM = HERE / "BLOCK3_NATIVE_DOWN_BEHAVIORAL_PORT_V1_EXECUTION_ADDENDUM.md"
PREREGISTRATION = HERE / "BLOCK3_NATIVE_DOWN_BEHAVIORAL_PORT_V1_PREREGISTRATION.md"
FREEZER = Path(__file__).resolve()
TEST = HERE / "test_prepare_block3_native_down_behavioral_port_v1_rows.py"
RUNNER = HERE / "run_block3_native_down_behavioral_port_v1.py"
RUNNER_TEST = HERE / "test_run_block3_native_down_behavioral_port_v1.py"
LOCAL_HARVESTER = HERE / "local_fineweb_harvest.py"
REFERENCE_ROWS = BQ / "bilin18_eval_tokens_large.pt"
AUDIT = HERE / "block3_native_down_behavioral_port_v1_rows_independent_audit.json"

FAILED_ROW = (HERE / "gauge_transport_triangle_unique_rows_v1_rows.pt").resolve()
FAILED_ROW_AUTHORITY = (
    HERE / "gauge_transport_triangle_unique_rows_v1_authority.json"
).resolve()
FAILED_ROW_FAILURE = (
    HERE / "gauge_transport_triangle_unique_rows_v1_failure.json"
).resolve()
TERMINAL_COPY_V2_RECEIPT = (
    BQ / "terminal_copy_induction_v2_rows_receipt.json"
).resolve()
TERMINAL_COPY_FIT_FREQUENCIES = (
    BQ / ".rowcache_terminal_copy_induction_v2/fit_token_frequencies.pt"
).resolve()
FAILED_ROW_AUTHORITY_SHA256 = (
    "5f7435150561ef385c9a4ee51e2040c4a029e98faefbfe1bc0f92612d820498e"
)
FAILED_ROW_FAILURE_SHA256 = (
    "91859b52b55b8be8ac05dc61f26b95fd43cdb92db7b8c39dfa72d226df41eb58"
)
TERMINAL_COPY_V2_RECEIPT_SHA256 = (
    "aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8"
)
TERMINAL_COPY_FIT_FREQUENCIES_SHA256 = (
    "7ba995e6bcfa2704cc4c2220dfdc8bd5caea53b976359c6b86cb5e14ba7e4c9a"
)

SOURCE_PATHS = (
    FREEZER, TEST, PREREGISTRATION, ADDENDUM, RUNNER, RUNNER_TEST,
    REGISTRY_PATH, BASE_PATH,
    LOCAL_HARVESTER,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _committed_blob(path: Path, commit: str) -> bytes:
    relative = str(path.relative_to(ROOT))
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def source_closure(commit: str) -> dict[str, str]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = _committed_blob(path, commit)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"row-freezer source differs from {commit}: {relative}")
        hashes[relative] = digest
    return hashes


class RunClaim(NamedTuple):
    descriptor: int
    inode: int
    nonce: str


def acquire_claim(path: Path | None = None) -> RunClaim:
    path = LOCK if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"fresh-row namespace is locked: {path}") from exc
    try:
        os.write(descriptor, (nonce + "\n").encode())
        os.fsync(descriptor)
        inode = os.fstat(descriptor).st_ino
        return RunClaim(descriptor, inode, nonce)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def require_claim(claim: RunClaim, path: Path | None = None) -> None:
    path = LOCK if path is None else path
    if not path.is_file() or path.stat().st_ino != claim.inode:
        raise RuntimeError("fresh-row lock was replaced")
    if path.read_text() != claim.nonce + "\n":
        raise RuntimeError("fresh-row lock content changed")


def release_claim(claim: RunClaim, path: Path | None = None) -> None:
    path = LOCK if path is None else path
    try:
        if path.exists() and path.stat().st_ino == claim.inode:
            path.unlink()
    finally:
        os.close(claim.descriptor)


def write_json_create_only(
    payload: Mapping[str, Any], path: Path,
    *, pre_link_check: Callable[[], None] | None = None,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as sink:
            descriptor = None
            sink.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        if pre_link_check is not None:
            pre_link_check()
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def discover_registry_files() -> tuple[Path, ...]:
    """Census all prior row-capable registry JSONs, including sibling directories."""
    paths: set[Path] = {BASE.CANONICAL_RECEIPT}
    for pattern in ("*receipt*.json", "*manifest*.json", "*authority*.json"):
        paths.update(BASIS.rglob(pattern))
    paths.discard(RECEIPT)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise RuntimeError(f"registered prior identity file is missing: {missing}")
    return tuple(sorted(path.resolve() for path in paths))


def _resolve_registry_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _looks_row_like(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith((".pt", ".pth")):
        return False
    path = Path(value)
    name = path.name.lower()
    return any(part.startswith(".rowcache") for part in path.parts) or any(
        term in name for term in (
            "row", "fineweb", "eval_token", "oracle_corpus", "source_document",
        )
    )


def _replace_failed_row_reference(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_failed_row_reference(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_failed_row_reference(child) for child in value]
    if _looks_row_like(value) and _resolve_registry_path(value) == FAILED_ROW:
        return None
    return value


def _missing_row_paths(payload: Any) -> set[Path]:
    missing: set[Path] = set()
    for value in REGISTRY.walk_json(payload):
        if _looks_row_like(value):
            path = _resolve_registry_path(value)
            if not path.is_file():
                missing.add(path)
    return missing


def _strip_embedded_waiver_proofs(payload: Any) -> Any:
    """Remove only exact previously verified waiver receipts from recursive census.

    Fresh-row receipts preserve the two canonical failed-row proof records. Those
    records describe an absent tensor; they are not themselves new row references.
    Every byte-level lineage field is revalidated before the proof metadata is
    excluded from the recursive path scanner.
    """
    sanitized = copy.deepcopy(payload)
    if not isinstance(sanitized, dict) or "waiver_proofs" not in sanitized:
        return sanitized
    proofs = sanitized.pop("waiver_proofs")
    if not isinstance(proofs, list) or len(proofs) != 2:
        raise RuntimeError("embedded failed-row waiver family changed")
    expected_paths = {str(FAILED_ROW_AUTHORITY), str(TERMINAL_COPY_V2_RECEIPT)}
    observed_paths = set()
    for proof in proofs:
        if not isinstance(proof, dict) or set(proof) != {
            "registry_json", "registry_json_sha256", "omitted_missing_row_path",
            "authority_sha256", "failure_sha256", "proof",
        } or proof.get("omitted_missing_row_path") != str(FAILED_ROW) \
                or proof.get("authority_sha256") != FAILED_ROW_AUTHORITY_SHA256 \
                or proof.get("failure_sha256") != FAILED_ROW_FAILURE_SHA256 \
                or proof.get("proof") != (
                    "exact_spent_authority_and_terminal_failure_no_materialized_outputs"
                ):
            raise RuntimeError("embedded failed-row waiver proof changed")
        path = _resolve_registry_path(proof["registry_json"])
        if str(path) not in expected_paths or not path.is_file() \
                or file_sha256(path) != proof.get("registry_json_sha256"):
            raise RuntimeError("embedded failed-row waiver registry binding changed")
        observed_paths.add(str(path))
    if observed_paths != expected_paths:
        raise RuntimeError("embedded failed-row waiver set changed")
    return sanitized


def _failed_row_waiver(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the one historical unmaterialized row reference exactly.

    Two registry JSONs name this same absent tensor: its spent authority and the later
    terminal-copy v2 receipt that already bound the failed-unmaterialized proof.  No
    other file or missing row receives a waiver.
    """

    path = path.resolve()
    if path not in (FAILED_ROW_AUTHORITY, TERMINAL_COPY_V2_RECEIPT):
        raise RuntimeError("registry has an unregistered missing row-like reference")
    if _missing_row_paths(payload) != {FAILED_ROW}:
        raise RuntimeError("registered failed-row JSON has unexpected missing paths")
    if FAILED_ROW.exists() or file_sha256(FAILED_ROW_AUTHORITY) != (
        FAILED_ROW_AUTHORITY_SHA256
    ) or file_sha256(FAILED_ROW_FAILURE) != FAILED_ROW_FAILURE_SHA256:
        raise RuntimeError("failed-unmaterialized row lineage changed")
    authority = json.loads(FAILED_ROW_AUTHORITY.read_bytes())
    outputs = authority.get("outputs", {})
    expected_outputs = {"authority", "failure", "lock", "manifest", "receipt", "rows"}
    if set(outputs) != expected_outputs or authority.get("authority_sha256") != (
        "8901a7446f70358e7e058013bb81c72f477c8636f5a1f76088307eda437025b5"
    ) or _resolve_registry_path(outputs["rows"]) != FAILED_ROW:
        raise RuntimeError("failed-unmaterialized authority semantics changed")
    resolved = {key: _resolve_registry_path(value) for key, value in outputs.items()}
    failure = json.loads(FAILED_ROW_FAILURE.read_bytes())
    if failure.get("status") != "terminal_failure_no_receipt" or any(
        failure.get(f"{kind}_exists") is not False
        for kind in ("rows", "manifest", "receipt")
    ) or any(resolved[kind].exists() for kind in ("rows", "manifest", "receipt", "lock")):
        raise RuntimeError("failed-unmaterialized transaction is no longer terminal/absent")
    staging = sorted({
        str(candidate.resolve())
        for kind in ("rows", "manifest", "receipt", "failure")
        for candidate in resolved[kind].parent.glob(f".{resolved[kind].name}.tmp-*")
    })
    if staging:
        raise RuntimeError("failed-unmaterialized transaction has staging artifacts")
    if path == TERMINAL_COPY_V2_RECEIPT:
        if file_sha256(path) != TERMINAL_COPY_V2_RECEIPT_SHA256:
            raise RuntimeError("terminal-copy v2 waiver receipt changed")
        waivers = payload.get("failed_unmaterialized_registry_exclusions")
        if not isinstance(waivers, list) or len(waivers) != 1 or (
            waivers[0].get("authority_sha256") != FAILED_ROW_AUTHORITY_SHA256
            or waivers[0].get("failure_sha256") != FAILED_ROW_FAILURE_SHA256
            or waivers[0].get("omitted_missing_row_path") != str(FAILED_ROW)
            or waivers[0].get("proof", {}).get("rows_exists") is not False
        ):
            raise RuntimeError("terminal-copy v2 failed-row proof changed")
    return {
        "registry_json": str(path),
        "registry_json_sha256": file_sha256(path),
        "omitted_missing_row_path": str(FAILED_ROW),
        "authority_sha256": FAILED_ROW_AUTHORITY_SHA256,
        "failure_sha256": FAILED_ROW_FAILURE_SHA256,
        "proof": "exact_spent_authority_and_terminal_failure_no_materialized_outputs",
    }


def _validate_exact_nonrow_frequency_artifact(path: Path) -> tuple[dict[str, Any], str]:
    """Bind the one frequency-vector artifact overmatched by the filename census."""

    if path.resolve() != TERMINAL_COPY_FIT_FREQUENCIES or not path.is_file():
        raise RuntimeError("unexpected non-row registry artifact")
    before = file_sha256(path)
    if before != TERMINAL_COPY_FIT_FREQUENCIES_SHA256:
        raise RuntimeError("terminal-copy fit-frequency artifact changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    after = file_sha256(path)
    if before != after or not isinstance(payload, dict) or set(payload) != {
        "query", "target"
    } or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.long
        or tuple(value.shape) != (50_257,)
        for value in payload.values()
    ):
        raise RuntimeError("terminal-copy fit-frequency artifact schema changed")
    return ({
        "kind": "exact_nonrow_frequency_vector",
        "path": str(path.resolve()),
        "file_sha256": before,
        "keys": ["query", "target"],
        "shape": [50_257],
        "reason": "contains no document rows; filename-only row heuristic overmatch",
    }, before)


def load_registry_exclusions(
    registry_files: tuple[Path, ...],
) -> tuple[
    tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    dict[str, str], dict[str, str], list[dict[str, Any]], list[dict[str, Any]],
]:
    """Parse the full registry with each JSON and tensor bound to exact bytes."""
    documents: set[str] = set()
    indices: set[int] = set()
    full_rows: set[tuple[int, ...]] = set()
    prefixes: set[tuple[int, ...]] = set()
    registry_hashes: dict[str, str] = {}
    tensor_specs: dict[Path, list[dict[str, str]]] = {REFERENCE_ROWS.resolve(): []}
    waiver_proofs: list[dict[str, Any]] = []
    nonrow_proofs: list[dict[str, Any]] = []

    for path in registry_files:
        before = file_sha256(path)
        raw = path.read_bytes()
        after = file_sha256(path)
        raw_hash = hashlib.sha256(raw).hexdigest()
        if before != raw_hash or after != before:
            raise RuntimeError(f"registry JSON changed while reading: {path}")
        registry_hashes[str(path.resolve())] = before
        payload = json.loads(raw)
        sanitized = _strip_embedded_waiver_proofs(payload)
        missing = _missing_row_paths(sanitized)
        if missing:
            waiver_proofs.append(_failed_row_waiver(path, sanitized))
            sanitized = _replace_failed_row_reference(sanitized)
        for tensor_path, specifications in REGISTRY.referenced_row_specs(sanitized).items():
            tensor_specs.setdefault(tensor_path, []).extend(specifications)
        for value in REGISTRY.walk_json(sanitized):
            if not isinstance(value, dict):
                continue
            document = value.get("document_id")
            index = value.get("dataset_document_index")
            if isinstance(document, str) and document:
                documents.add(document)
            if isinstance(index, int):
                indices.add(index)

    tensor_hashes: dict[str, str] = {}
    if TERMINAL_COPY_FIT_FREQUENCIES in tensor_specs:
        proof, digest = _validate_exact_nonrow_frequency_artifact(
            TERMINAL_COPY_FIT_FREQUENCIES
        )
        nonrow_proofs.append(proof)
        tensor_hashes[str(TERMINAL_COPY_FIT_FREQUENCIES)] = digest
        del tensor_specs[TERMINAL_COPY_FIT_FREQUENCIES]
    for path in sorted(tensor_specs):
        tensors, digest = REGISTRY.load_verified_row_tensor(path, tensor_specs[path])
        tensor_hashes[str(path)] = digest
        for tensor in tensors:
            for row in tensor:
                values = tuple(int(item) for item in row.tolist())
                full_rows.add(values)
                prefixes.add(values[:PREFIX_LENGTH])
    waiver_proofs.sort(key=lambda item: item["registry_json"])
    expected_waiver_files = {str(FAILED_ROW_AUTHORITY), str(TERMINAL_COPY_V2_RECEIPT)}
    if {item["registry_json"] for item in waiver_proofs} != expected_waiver_files:
        raise RuntimeError("registry did not reproduce both exact failed-row waiver records")
    return (
        (documents, indices, full_rows, prefixes), registry_hashes, tensor_hashes,
        waiver_proofs, nonrow_proofs,
    )


def harvest_fresh_documents(
    texts: Iterable[tuple[str, str]],
    encode: Callable[[str], list[int]],
    prior: tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    *,
    start_document_index: int = START_DOCUMENT_INDEX,
    n_source_documents: int = N_SOURCE_DOCUMENTS,
    token_length: int = TOKEN_LENGTH,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Select exactly one collision-free row from each of N ordered documents."""
    if type(start_document_index) is not int or start_document_index < 0:
        raise ValueError("start_document_index must be a nonnegative integer")
    if type(n_source_documents) is not int or n_source_documents <= 0:
        raise ValueError("n_source_documents must be positive")
    if type(token_length) is not int or token_length < PREFIX_LENGTH:
        raise ValueError("token_length is too short")
    prior_docs, prior_indices, prior_rows, prior_prefixes = prior
    prior_rows_at_length = {
        row[:token_length] for row in prior_rows if len(row) >= token_length
    }
    used_prefixes = set(prior_prefixes)
    rows: list[list[int]] = []
    records: list[dict[str, Any]] = []
    for document_index, item in enumerate(texts):
        if document_index < start_document_index:
            continue
        if len(rows) == n_source_documents:
            break
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("ordered corpus must yield (document_id,text)")
        document_id, text = item
        if not isinstance(document_id, str) or not document_id or not isinstance(text, str):
            raise ValueError("ordered corpus identity is malformed")
        if document_id in prior_docs or document_index in prior_indices:
            continue
        tokens = encode(text)
        chosen: tuple[list[int], int] | None = None
        for token_start in range(0, len(tokens) - token_length + 1, token_length):
            row = tokens[token_start:token_start + token_length]
            row_tuple = tuple(int(value) for value in row)
            prefix = row_tuple[:PREFIX_LENGTH]
            if row_tuple in prior_rows_at_length or prefix in used_prefixes:
                continue
            chosen = (row, token_start)
            break
        if chosen is None:
            continue
        row, token_start = chosen
        used_prefixes.add(tuple(int(value) for value in row[:PREFIX_LENGTH]))
        ordinal = len(rows)
        rows.append(row)
        records.append({
            "document_id": document_id,
            "dataset_document_index": document_index,
            "source_document_ordinal": ordinal,
            "row_index": ordinal,
            "chunk_id": token_start // token_length,
            "token_start": token_start,
        })
    if len(rows) != n_source_documents:
        raise RuntimeError(
            f"ordered source ended after {len(rows)}/{n_source_documents} eligible documents"
        )
    tensor = torch.tensor(rows, dtype=torch.long)
    if tensor.shape != (n_source_documents, token_length):
        raise RuntimeError(f"fresh row tensor has wrong shape: {tuple(tensor.shape)}")
    return tensor, records


def summarize(rows: torch.Tensor, records: list[dict[str, Any]]) -> dict[str, Any]:
    documents = [record.get("document_id") for record in records]
    indices = [record.get("dataset_document_index") for record in records]
    ordinals = [record.get("source_document_ordinal") for record in records]
    row_indices = [record.get("row_index") for record in records]
    if rows.shape != (N_SOURCE_DOCUMENTS, TOKEN_LENGTH) or rows.dtype != torch.long:
        raise RuntimeError("canonical fresh-row tensor schema failed")
    if len(records) != N_SOURCE_DOCUMENTS or len(set(documents)) != N_SOURCE_DOCUMENTS:
        raise RuntimeError("fresh role is not one row per unique source document")
    if len(set(indices)) != N_SOURCE_DOCUMENTS:
        raise RuntimeError("fresh role repeats a dataset document index")
    expected = list(range(N_SOURCE_DOCUMENTS))
    if ordinals != expected or row_indices != expected:
        raise RuntimeError("fresh row-to-document identity is not canonical")
    return {
        "n_source_documents": N_SOURCE_DOCUMENTS,
        "n_rows": N_SOURCE_DOCUMENTS,
        "token_length": TOKEN_LENGTH,
        "scored_positions_per_row": 192,
        "n_scored_positions": N_SOURCE_DOCUMENTS * 192,
        "row_to_document_identity": expected,
        "first_dataset_document_index": min(indices),
        "last_dataset_document_index": max(indices),
    }


def validate_disjointness(
    rows: torch.Tensor,
    records: list[dict[str, Any]],
    prior: tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
) -> dict[str, bool]:
    prior_docs, prior_indices, prior_rows, prior_prefixes = prior
    documents = {record["document_id"] for record in records}
    indices = {record["dataset_document_index"] for record in records}
    full = {tuple(int(value) for value in row.tolist()) for row in rows}
    prefixes = {row[:PREFIX_LENGTH] for row in full}
    prior_at_length = {row[:TOKEN_LENGTH] for row in prior_rows if len(row) >= TOKEN_LENGTH}
    gates = {
        "source_documents_disjoint_from_all_prior_roles": documents.isdisjoint(prior_docs),
        "dataset_indices_disjoint_from_all_prior_roles": indices.isdisjoint(prior_indices),
        "full257_rows_disjoint_from_all_prior_roles": full.isdisjoint(prior_at_length),
        "prefix32_disjoint_from_all_prior_roles": prefixes.isdisjoint(prior_prefixes),
        "one_row_per_source_document": len(documents) == len(records) == rows.shape[0],
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"fresh-role disjointness failed: {failed}")
    return gates


def verify_snapshot(
    *,
    commit: str,
    sources: Mapping[str, str],
    registry_files: tuple[Path, ...],
    registry_hashes: Mapping[str, str],
    tensor_hashes: Mapping[str, str],
    prior: tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    waiver_proofs: list[dict[str, Any]],
    nonrow_proofs: list[dict[str, Any]],
    parquet: Path,
) -> None:
    if source_closure(commit) != dict(sources):
        raise RuntimeError("fresh-row source closure changed")
    current_registry = discover_registry_files()
    if current_registry != registry_files:
        raise RuntimeError("fresh-row registry membership changed")
    (
        current_prior, current_registry_hashes, current_tensor_hashes,
        current_waiver_proofs, current_nonrow_proofs,
    ) = load_registry_exclusions(current_registry)
    if discover_registry_files() != current_registry:
        raise RuntimeError("fresh-row registry membership changed during replay")
    if current_registry_hashes != dict(registry_hashes):
        raise RuntimeError("fresh-row registry files changed")
    if current_tensor_hashes != dict(tensor_hashes) or current_prior != prior:
        raise RuntimeError("fresh-row exclusion tensors changed")
    if current_waiver_proofs != waiver_proofs:
        raise RuntimeError("failed-unmaterialized registry waiver changed")
    if current_nonrow_proofs != nonrow_proofs:
        raise RuntimeError("exact non-row registry classification changed")
    if parquet.stat().st_size != BASE.local.PINNED_SIZE or (
        file_sha256(parquet) != BASE.local.PINNED_SHA256
    ):
        raise RuntimeError("pinned ordered FineWeb parquet changed")


def validate_independent_audit(path: Path = AUDIT) -> tuple[dict[str, Any], str]:
    """Require an outcome-blind GO that binds every freezer source byte."""

    if not path.is_file():
        raise RuntimeError("independent fresh-row freezer audit is absent")
    before = file_sha256(path)
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError("independent fresh-row freezer audit changed while reading")
    audit = json.loads(raw)
    required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if set(audit) != required or audit.get("schema") != (
        "block3_native_down_behavioral_port_v1_rows_independent_audit"
    ) or audit.get("status") != "GO" or audit.get("outcome_access") is not False or (
        not isinstance(audit.get("tests_passed"), int) or audit["tests_passed"] < 1
    ) or not isinstance(audit.get("reviewer"), str) or not audit["reviewer"]:
        raise RuntimeError("independent fresh-row freezer audit is not an exact GO")
    commit = audit.get("audited_source_commit")
    hashes = audit.get("audited_source_hashes")
    if not isinstance(commit, str) or len(commit) != 40 or not isinstance(hashes, dict):
        raise RuntimeError("independent fresh-row freezer audit binding is malformed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    expected_paths = {str(source.relative_to(ROOT)) for source in SOURCE_PATHS}
    if set(hashes) != expected_paths:
        raise RuntimeError("independent audit source closure is incomplete")
    for source in SOURCE_PATHS:
        relative = str(source.relative_to(ROOT))
        committed = _committed_blob(source, commit)
        digest = hashlib.sha256(committed).hexdigest()
        if hashes[relative] != digest or file_sha256(source) != digest:
            raise RuntimeError("source changed after independent freezer audit")
    return audit, before


def verify_installed_cache(path: Path, entry: Mapping[str, Any]) -> torch.Tensor:
    before = file_sha256(path)
    if before != entry.get("file_sha256"):
        raise RuntimeError("installed fresh-row file hash changed")
    rows = torch.load(path, map_location="cpu", weights_only=True)
    after = file_sha256(path)
    if before != after or not isinstance(rows, torch.Tensor) or rows.dtype != torch.long or (
        tuple(rows.shape) != (N_SOURCE_DOCUMENTS, TOKEN_LENGTH)
    ) or tensor_sha256(rows) != entry.get("tensor_sha256"):
        raise RuntimeError("installed fresh-row tensor changed")
    return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_locked(claim: RunClaim) -> dict[str, Any]:
    require_claim(claim)
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite native-Down fresh-row namespace")
    audit, audit_sha256 = validate_independent_audit()
    commit = _git("rev-parse", "HEAD")
    sources = source_closure(commit)
    canonical, parquet = BASE.validate_ordered_source()
    registry_files = discover_registry_files()
    (
        prior, registry_hashes, prior_tensor_hashes, waiver_proofs, nonrow_proofs,
    ) = load_registry_exclusions(registry_files)

    import tiktoken
    encoding = tiktoken.get_encoding("gpt2")
    rows, records = harvest_fresh_documents(
        BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
    )
    summary = summarize(rows, records)
    disjointness = validate_disjointness(rows, records, prior)

    staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        staged = staging / "fresh_192_source_documents.pt"
        torch.save(rows, staged)
        entry = {
            "cache_path": str((CACHE / staged.name).resolve()),
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
            "file_sha256": file_sha256(staged),
            "tensor_sha256": tensor_sha256(rows),
        }
        verify_snapshot(
            commit=commit, sources=sources, registry_files=registry_files,
            registry_hashes=registry_hashes, tensor_hashes=prior_tensor_hashes,
            prior=prior, waiver_proofs=waiver_proofs,
            nonrow_proofs=nonrow_proofs, parquet=parquet,
        )
        require_claim(claim)
        CACHE.mkdir(parents=False, exist_ok=False)
        installed = CACHE / staged.name
        os.replace(staged, installed)
        _fsync_directory(CACHE)
        _fsync_directory(CACHE.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    installed_rows = verify_installed_cache(installed, entry)
    replay_rows, replay_records = harvest_fresh_documents(
        BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
    )
    if not torch.equal(replay_rows, installed_rows) or replay_records != records:
        raise RuntimeError("canonical fresh-row re-harvest changed")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "block3_native_down_behavioral_port_v1_rows",
        "status": "frozen_before_native_down_behavioral_port_model_forward",
        "authority": "pinned_ordered_fineweb_registry_wide_fresh_documents",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": False,
        "source_commit": commit,
        "source_hashes": sources,
        "independent_audit": {
            "path": str(AUDIT.resolve()),
            "file_sha256": audit_sha256,
            "audited_source_commit": audit["audited_source_commit"],
            "reviewer": audit["reviewer"],
            "tests_passed": audit["tests_passed"],
        },
        "selection": {
            "start_dataset_document_index": START_DOCUMENT_INDEX,
            "n_source_documents": N_SOURCE_DOCUMENTS,
            "rows_per_source_document": 1,
            "token_length": TOKEN_LENGTH,
            "scored_position_slice": [64, 256],
        },
        "sample_summary": summary,
        "entries": {"fresh": entry},
        "document_provenance": {"schema_version": 1, "sets": {"fresh": records}},
        "disjointness_gates": disjointness,
        "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
        "source_receipt_path": str(BASE.CANONICAL_RECEIPT.resolve()),
        "source_receipt_sha256": registry_hashes[str(BASE.CANONICAL_RECEIPT.resolve())],
        "prior_registry_files": registry_hashes,
        "prior_row_tensors": prior_tensor_hashes,
        "failed_unmaterialized_registry_waivers": waiver_proofs,
        "exact_nonrow_registry_artifacts": nonrow_proofs,
        "exclusion_counts": {
            "source_documents": len(prior[0]),
            "dataset_document_indices": len(prior[1]),
            "full_rows": len(prior[2]),
            "prefix32": len(prior[3]),
            "registry_files": len(registry_hashes),
            "row_tensor_files": len(prior_tensor_hashes),
        },
        "outcome_access": {
            "checkpoint_loaded": False,
            "model_imported": False,
            "model_forward_calls": 0,
            "scientific_outcomes_read": False,
        },
    }
    verify_snapshot(
        commit=commit, sources=sources, registry_files=registry_files,
        registry_hashes=registry_hashes, tensor_hashes=prior_tensor_hashes,
        prior=prior, waiver_proofs=waiver_proofs,
        nonrow_proofs=nonrow_proofs, parquet=parquet,
    )
    verify_installed_cache(installed, entry)
    require_claim(claim)
    if RECEIPT.exists():
        raise RuntimeError("fresh-row receipt appeared before publication")

    def final_publication_guard() -> None:
        require_claim(claim)
        if RECEIPT.exists():
            raise RuntimeError("fresh-row receipt appeared before final publication")
        validate_independent_audit()
        verify_snapshot(
            commit=commit, sources=sources, registry_files=registry_files,
            registry_hashes=registry_hashes, tensor_hashes=prior_tensor_hashes,
            prior=prior, waiver_proofs=waiver_proofs,
            nonrow_proofs=nonrow_proofs, parquet=parquet,
        )
        verify_installed_cache(installed, entry)

    write_json_create_only(receipt, RECEIPT, pre_link_check=final_publication_guard)
    return receipt


def freeze() -> dict[str, Any]:
    claim = acquire_claim()
    try:
        return freeze_locked(claim)
    finally:
        release_claim(claim)


def load_frozen_rows(path: Path = RECEIPT) -> tuple[dict[str, Any], torch.Tensor]:
    before = file_sha256(path)
    receipt = json.loads(path.read_bytes())
    if file_sha256(path) != before:
        raise RuntimeError("fresh-row receipt changed while loading")
    if receipt.get("receipt_kind") != "block3_native_down_behavioral_port_v1_rows" or (
        receipt.get("status") != "frozen_before_native_down_behavioral_port_model_forward"
    ) or receipt.get("authorized_for_scored_experiments") is not True or (
        receipt.get("authorized_for_training") is not False
    ) or not all(receipt.get("disjointness_gates", {}).values()):
        raise RuntimeError("fresh-row receipt is not authoritative")
    entry = receipt.get("entries", {}).get("fresh", {})
    rows = verify_installed_cache(Path(entry.get("cache_path", "")), entry)
    summarize(rows, receipt["document_provenance"]["sets"]["fresh"])
    return receipt, rows


if __name__ == "__main__":
    frozen = freeze()
    print(json.dumps({
        "receipt": str(RECEIPT),
        "receipt_sha256": file_sha256(RECEIPT),
        "sample_summary": frozen["sample_summary"],
        "disjointness_gates": frozen["disjointness_gates"],
        "exclusion_counts": frozen["exclusion_counts"],
    }, indent=2))
