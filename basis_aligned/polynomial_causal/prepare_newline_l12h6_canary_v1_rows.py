#!/usr/bin/env python3
"""Prospective create-only fresh-row freezer for the newline L12H6 canary.

The transaction is model-free and cannot mint its authority or independent audit.
Authority and audit must exist before parquet/code bytes are opened.  Import is I/O
free.  The canonical freezer has not been executed by this source change.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tokenize
from typing import Any, Iterable, Mapping, NamedTuple

import torch

from circuit_newline_fixed_crew_v1 import NewlineMaskSpec, build_newline_masks
import newline_l12h6_canary_rows_v1 as contract
import newline_l12h6_canary_v1_readiness as readiness
import newline_l12h6_token_registry_v1 as token_registry


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PARQUET = Path("/workspace/fineweb_pinned/data/CC-MAIN-2013-20/000_00000.parquet")
PARQUET_SIZE = 2_147_531_358
PARQUET_SHA256 = "c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930"
FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
FINEWEB_RELATIVE = "data/CC-MAIN-2013-20/000_00000.parquet"
CANONICAL_FINEWEB_RECEIPT = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
CANONICAL_FINEWEB_RECEIPT_SHA256 = (
    "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16"
)

AUTHORITY = HERE / "newline_l12h6_canary_v1_rows_authority.json"
AUDIT = HERE / "newline_l12h6_canary_v1_rows_independent_audit.json"
CACHE = BQ / ".rowcache_newline_l12h6_canary_v1"
MANIFEST = CACHE / "manifest.json"
RECEIPT = BQ / "newline_l12h6_canary_v1_rows_receipt.json"
FAILURE = BQ / "newline_l12h6_canary_v1_rows_failure.json"
LOCK = Path("/workspace/runs/.newline_l12h6_canary_v1_rows.lock")
ALLOCATION_SEED = "newline_l12h6_canary_v1_fresh_roles_20260830"
EXTRA_CANDIDATES_PER_CELL = 16

SOURCE_PATHS = tuple(dict.fromkeys((
    *readiness.SOURCE_PATHS,
    "basis_aligned/polynomial_causal/prepare_newline_l12h6_canary_v1_rows.py",
    "basis_aligned/polynomial_causal/test_prepare_newline_l12h6_canary_v1_rows.py",
    "basis_aligned/polynomial_causal/local_fineweb_harvest.py",
)))
OWN_JSON_PATHS = frozenset(path.resolve() for path in (AUTHORITY, AUDIT, MANIFEST, RECEIPT, FAILURE))
ROLE_FILES = {
    role: CACHE / f"{role.lower()}_rows.pt" for role in contract.ROLE_ORDER
}
ROLE_AUTHORIZATIONS = {
    "CANARY_SELECT": "l12h6_canary_selection_only",
    "FINAL": "sealed_pending_canary_pass_and_new_authority",
    "OOD": "sealed_pending_canary_pass_and_new_authority",
}
CODE_EXCLUDED_PARTS = frozenset({
    ".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__",
})
LIST_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d{1,4}[.)]\s+|\|.+\||[^\t]+\t[^\t]+)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _stable_bytes(path: Path) -> tuple[bytes, str]:
    before = file_sha256(path); raw = path.read_bytes(); after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"newline input changed during read: {path}")
    return raw, before


def _stable_json(path: Path) -> tuple[dict[str, Any], str]:
    raw, digest = _stable_bytes(path); value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"newline JSON root is not an object: {path}")
    return value, digest


def source_closure(commit: str) -> dict[str, str]:
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("newline source commit is malformed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    output = {}
    for relative in SOURCE_PATHS:
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        live = ROOT / relative
        if not live.is_file() or file_sha256(live) != digest:
            raise RuntimeError(f"newline source drift: {relative}")
        output[relative] = digest
    return output


def discover_registry_files() -> tuple[Path, ...]:
    return tuple(sorted(
        (path.resolve() for path in (ROOT / "basis_aligned").rglob("*.json")
         if path.resolve() not in OWN_JSON_PATHS),
        key=str,
    ))


def registry_snapshot() -> dict[str, str]:
    output = {}
    for path in discover_registry_files():
        raw, digest = _stable_bytes(path)
        value = json.loads(raw)
        if not isinstance(value, (dict, list)):
            raise RuntimeError(f"newline registry JSON is not a container: {path}")
        output[str(path)] = digest
    return output


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def historical_exclusions(snapshot: Mapping[str, str]) -> contract.HistoricalExclusions:
    sets = {name: set() for name in (
        "document", "source", "blob", "normalized", "row", "prefix",
    )}
    scalar_keys = {
        "document_id": "document", "source_file": "source", "path": "source",
        "source_path": "source", "file_path": "source",
        "source_blob_sha256": "blob", "blob_sha256": "blob",
        "normalized_python_sha256": "normalized", "row_sha256": "row",
        "rows_sha256": "row", "prefix32_sha256": "prefix",
    }
    list_keys = {
        "document_ids": "document", "source_files": "source",
        "row_sha256s": "row", "prefix32_sha256s": "prefix",
    }
    if dict(snapshot) != registry_snapshot():
        raise RuntimeError("newline registry snapshot changed before exclusion replay")
    for name, digest in snapshot.items():
        path = Path(name); raw, observed = _stable_bytes(path)
        if observed != digest:
            raise RuntimeError(f"newline registry changed: {path}")
        payload = json.loads(raw)
        for value in _walk(payload):
            if not isinstance(value, Mapping):
                continue
            for key, target in scalar_keys.items():
                item = value.get(key)
                if isinstance(item, str) and item:
                    sets[target].add(item)
            for key, target in list_keys.items():
                items = value.get(key)
                if isinstance(items, list):
                    sets[target].update(item for item in items if isinstance(item, str) and item)
    return contract.HistoricalExclusions(
        frozenset(sets["document"]), frozenset(sets["source"]),
        frozenset(sets["blob"]), frozenset(sets["normalized"]),
        frozenset(sets["row"]), frozenset(sets["prefix"]),
    )


def code_path_is_eligible(path: str) -> bool:
    parts = Path(path).parts; name = Path(path).name
    return bool(
        path.endswith(".py") and not any(part in CODE_EXCLUDED_PARTS for part in parts)
        and not name.startswith("test_") and not name.endswith("_test.py")
        and not path.startswith("basis_aligned/bilinear_quotient/runlogs/")
    )


def normalized_python_sha256(blob: bytes) -> str:
    normalized = []
    try:
        for token in tokenize.tokenize(io.BytesIO(blob).readline):
            if token.type in {
                tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL,
                tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
            }:
                continue
            normalized.append("STRING" if token.type == tokenize.STRING else (
                "NUMBER" if token.type == tokenize.NUMBER else token.string
            ))
    except (IndentationError, SyntaxError, tokenize.TokenError, UnicodeDecodeError):
        normalized = [" ".join(blob.decode("utf-8", errors="replace").split())]
    return hashlib.sha256("\0".join(normalized).encode()).hexdigest()


def code_tree_manifest(commit: str) -> dict[str, Any]:
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True,
    ).splitlines()
    entries = []
    for path in sorted(path for path in paths if code_path_is_eligible(path)):
        blob = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        entries.append({
            "path": path, "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "normalized_python_sha256": normalized_python_sha256(blob),
        })
    return {"eligible_file_count": len(entries), "entries_sha256": _logical_sha(entries)}


def _fineweb_identity() -> dict[str, Any]:
    receipt, receipt_sha = _stable_json(CANONICAL_FINEWEB_RECEIPT)
    gate = receipt.get("ordered_manifest_local_parquet_identity_gate")
    if receipt_sha != CANONICAL_FINEWEB_RECEIPT_SHA256 or not isinstance(gate, Mapping) or (
        gate.get("passed") is not True or gate.get("revision") != FINEWEB_REVISION
        or gate.get("first_relative_path") != FINEWEB_RELATIVE
        or gate.get("source_local_path") != str(PARQUET)
        or gate.get("source_size") != PARQUET_SIZE
        or gate.get("source_sha256") != PARQUET_SHA256
    ):
        raise RuntimeError("newline canonical FineWeb identity changed")
    if PARQUET.stat().st_size != PARQUET_SIZE or file_sha256(PARQUET) != PARQUET_SHA256:
        raise RuntimeError("newline pinned FineWeb parquet changed")
    return {
        "canonical_receipt_path": str(CANONICAL_FINEWEB_RECEIPT),
        "canonical_receipt_sha256": receipt_sha,
        "parquet_path": str(PARQUET), "parquet_size": PARQUET_SIZE,
        "parquet_sha256": PARQUET_SHA256, "revision": FINEWEB_REVISION,
        "relative_path": FINEWEB_RELATIVE,
    }


def source_identity(commit: str) -> dict[str, Any]:
    import tiktoken

    encoding = tiktoken.get_encoding(token_registry.ENCODING_NAME)
    registry = token_registry.build_registry(encoding)
    return {
        "fineweb": _fineweb_identity(), "code_tree": code_tree_manifest(commit),
        "tokenizer": {
            "name": token_registry.ENCODING_NAME,
            "encoding_sha256": token_registry.ENCODING_SHA256,
            "registry_sha256": token_registry.REGISTRY_SHA256,
            "class_counts": dict(token_registry.EXPECTED_COUNTS),
            "class_hashes": dict(token_registry.EXPECTED_ID_SHA256),
        },
        "enumerator": (
            "one nonoverlapping 257-token row per document/file; exact mask eligibility; "
            "role-license SHA split before allocation; no model or outcome"
        ),
        "registry_recomputed_sha256": _logical_sha({name: list(ids) for name, ids in registry.items()}),
    }


def _expected_outputs() -> dict[str, Any]:
    return {
        "cache": str(CACHE), "manifest": str(MANIFEST), "receipt": str(RECEIPT),
        "failure": str(FAILURE), "lock": str(LOCK),
        "role_files": {role: str(path) for role, path in ROLE_FILES.items()},
    }


def validate_authority(
    payload: Mapping[str, Any], *, replay_protected: bool = True,
) -> dict[str, Any]:
    if set(payload) != {
        "schema", "status", "outcome_access", "source_commit", "source_hashes",
        "audit_path", "registry_snapshot", "source_identity", "allocation_seed", "outputs",
    } or payload.get("schema") != "newline_l12h6_canary_v1_rows_authority" or payload.get(
        "status"
    ) != "frozen_before_candidate_enumeration_or_row_access" or payload.get(
        "outcome_access"
    ) is not False:
        raise RuntimeError("newline row authority schema/status changed")
    if not isinstance(payload.get("source_commit"), str) or len(payload["source_commit"]) != 40 \
            or not isinstance(payload.get("source_hashes"), Mapping) or not payload["source_hashes"] \
            or any(not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64
                   for path, digest in payload["source_hashes"].items()) or (
        payload.get("audit_path") != str(AUDIT) or payload.get("allocation_seed") != ALLOCATION_SEED
        or payload.get("outputs") != _expected_outputs()
    ):
        raise RuntimeError("newline row authority source/namespace changed")
    if not isinstance(payload.get("registry_snapshot"), Mapping) or not payload[
        "registry_snapshot"
    ] or not isinstance(payload.get("source_identity"), Mapping):
        raise RuntimeError("newline row authority protected identity is malformed")
    if replay_protected and (
        payload["source_hashes"] != source_closure(payload["source_commit"])
        or payload["registry_snapshot"] != registry_snapshot()
        or payload["source_identity"] != source_identity(payload["source_commit"])
    ):
        raise RuntimeError("newline row authority protected identity changed")
    return dict(payload)


def validate_audit(
    audit: Mapping[str, Any], *, authority_sha256: str, authority: Mapping[str, Any],
) -> None:
    if set(audit) != {
        "schema", "status", "outcome_access", "authority_sha256", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    } or audit.get("schema") != "newline_l12h6_canary_v1_rows_independent_audit" or (
        audit.get("status") != "GO" or audit.get("outcome_access") is not False
        or audit.get("authority_sha256") != authority_sha256
        or audit.get("audited_source_commit") != authority["source_commit"]
        or audit.get("audited_source_hashes") != authority["source_hashes"]
        or type(audit.get("tests_passed")) is not int or audit["tests_passed"] < 1
        or not isinstance(audit.get("reviewer"), str) or not audit["reviewer"]
    ):
        raise RuntimeError("newline independent row audit is not an exact outcome-blind GO")


def frozen_mask_spec(registry: Mapping[str, object]) -> NewlineMaskSpec:
    token_registry.validate_registry(registry)
    return NewlineMaskSpec(
        tuple(registry["newline"]), tuple(registry["punctuation"]),
        tuple(registry["capitalized"]), tuple(registry["quote_bracket"]),
        first_prediction=64,
        jitter_offsets=(2, -2, 3, -3, 4, -4, 8, -8, 16, -16, 32, -32),
        random_seed=2_026_083_000,
    )


def role_license(document_id: str, domain: contract.NewlineDomain) -> str:
    digest = hashlib.sha256(
        f"{ALLOCATION_SEED}\0{domain.value}\0{document_id}".encode()
    ).digest()[0] % 11
    return "CANARY_SELECT" if digest < 3 else ("FINAL" if digest < 7 else "OOD")


def is_list_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False
    structural = sum(bool(LIST_PATTERN.match(line)) for line in lines)
    return structural >= 3 and 3 * structural >= len(lines)


def _row_for_document(
    tokens: list[int], spec: NewlineMaskSpec, record_factory, exclusions,
) -> tuple[torch.Tensor, contract.CandidateRecord] | None:
    candidates = []
    for start in range(0, len(tokens) - contract.ROW_LENGTH + 1, contract.ROW_LENGTH):
        row = torch.tensor(tokens[start:start + contract.ROW_LENGTH], dtype=torch.long)
        record = record_factory(start)
        row_hash = contract.tensor_sha256(row.contiguous())
        prefix_hash = contract.tensor_sha256(row[:contract.PREFIX_LENGTH].contiguous())
        if row_hash in exclusions.row_sha256s or prefix_hash in exclusions.prefix_sha256s:
            continue
        try:
            masks = build_newline_masks(row.unsqueeze(0), spec)
        except (ValueError, RuntimeError):
            continue
        if not bool(masks.newline_target.any()):
            continue
        key = hashlib.sha256(
            f"{ALLOCATION_SEED}\0{record.document_id}\0{start}\0{row_hash}".encode()
        ).digest()
        candidates.append((key, row, record))
    return None if not candidates else min(candidates, key=lambda item: item[0])[1:]


def enumerate_from_sources(
    natural_documents: Iterable[tuple[int, str, str]],
    code_documents: Iterable[tuple[int, str, bytes]],
    encode,
    registry: Mapping[str, object],
    exclusions: contract.HistoricalExclusions,
    *,
    code_revision: str,
) -> tuple[torch.Tensor, tuple[contract.CandidateRecord, ...]]:
    """Pure injectable enumerator used by production and synthetic tests."""

    spec = frozen_mask_spec(registry)
    needs = {
        (role, domain): contract.ROLE_DOMAIN_QUOTAS[role][domain] + EXTRA_CANDIDATES_PER_CELL
        for role in contract.ROLE_ORDER for domain in contract.DOMAIN_ORDER
    }
    rows: list[torch.Tensor] = []; records: list[contract.CandidateRecord] = []
    counts = {key: 0 for key in needs}
    used = {name: set() for name in ("document", "source", "blob", "normalized", "row", "prefix")}

    def admit(row_record):
        if row_record is None:
            return
        row, record = row_record; key = (record.role_license, record.domain.value)
        identities = {
            "document": record.document_id, "source": record.source_file,
            "blob": record.source_blob_sha256,
            "normalized": record.normalized_python_sha256,
            "row": contract.tensor_sha256(row),
            "prefix": contract.tensor_sha256(row[:contract.PREFIX_LENGTH].contiguous()),
        }
        if any(value is not None and value in used[name] for name, value in identities.items()):
            return
        if counts[key] < needs[key]:
            rows.append(row); records.append(record); counts[key] += 1
            for name, value in identities.items():
                if value is not None:
                    used[name].add(value)

    for document_index, document_id, text in natural_documents:
        domain = contract.NewlineDomain.LIST if is_list_table(text) else contract.NewlineDomain.PROSE
        role = role_license(document_id, domain); blob = hashlib.sha256(text.encode()).hexdigest()
        logical_file = f"fineweb:{document_id}"
        if document_id in exclusions.document_ids or logical_file in exclusions.source_files or (
            blob in exclusions.source_blobs or counts[(role, domain.value)] >= needs[(role, domain.value)]
        ):
            continue
        def factory(start, *, _role=role, _domain=domain, _blob=blob):
            return contract.CandidateRecord(
                document_id, document_index, logical_file, FINEWEB_REVISION, _blob, _domain,
                "fineweb_canonical_scored_rows", _role,
                f"{_role.lower()}:{_domain.value}:fineweb-hashfold",
            )
        admit(_row_for_document(encode(text), spec, factory, exclusions))
        if all(counts[(role, domain)] >= needs[(role, domain)]
               for role in contract.ROLE_ORDER for domain in ("prose", "list")):
            break

    for document_index, path, blob_bytes in code_documents:
        domain = contract.NewlineDomain.CODE; document_id = f"git:{path}"
        role = role_license(document_id, domain); blob = hashlib.sha256(blob_bytes).hexdigest()
        normalized = normalized_python_sha256(blob_bytes)
        if document_id in exclusions.document_ids or path in exclusions.source_files or (
            blob in exclusions.source_blobs or normalized in exclusions.normalized_python_sha256s
            or counts[(role, domain.value)] >= needs[(role, domain.value)]
        ):
            continue
        def factory(start, *, _role=role, _blob=blob, _normalized=normalized):
            return contract.CandidateRecord(
                document_id, document_index, path, code_revision, _blob, domain,
                "repository_source_license", _role,
                f"{_role.lower()}:code:git-tree-hashfold", _normalized,
            )
        text = blob_bytes.decode("utf-8", errors="replace")
        admit(_row_for_document([50_256, *encode(text)], spec, factory, exclusions))
        if all(counts[(role, "code")] >= needs[(role, "code")] for role in contract.ROLE_ORDER):
            break
    missing = {f"{role}/{domain}": needs[(role, domain)] - count
               for (role, domain), count in counts.items() if count < needs[(role, domain)]}
    if missing:
        raise RuntimeError(f"newline candidate sources are underpowered: {missing}")
    tensor = torch.stack(rows).contiguous()
    # Exact identity uniqueness is a property of the admitted candidate universe,
    # not something allocation may repair after seeing it.
    identity_fields = (
        [record.document_id for record in records], [record.source_file for record in records],
        [record.source_blob_sha256 for record in records],
        [contract.tensor_sha256(row) for row in tensor],
        [contract.tensor_sha256(row[:contract.PREFIX_LENGTH].contiguous()) for row in tensor],
    )
    if any(len(values) != len(set(values)) for values in identity_fields):
        raise RuntimeError("newline candidate enumeration repeated a protected identity")
    return tensor, tuple(records)


def production_enumeration(
    commit: str, registry: Mapping[str, object], exclusions: contract.HistoricalExclusions,
) -> tuple[torch.Tensor, tuple[contract.CandidateRecord, ...]]:
    import pyarrow.parquet as parquet
    import tiktoken

    encoding = tiktoken.get_encoding(token_registry.ENCODING_NAME)
    parquet_file = parquet.ParquetFile(PARQUET)
    def natural():
        index = 0
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=256):
            for text in batch.column(0).to_pylist():
                document_id = f"{FINEWEB_REVISION}:{FINEWEB_RELATIVE}:{index}"
                yield index, document_id, text
                index += 1
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True,
    ).splitlines()
    def code():
        for index, path in enumerate(sorted(path for path in paths if code_path_is_eligible(path))):
            yield index, path, subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    return enumerate_from_sources(
        natural(), code(), encoding.encode_ordinary, registry, exclusions,
        code_revision=commit,
    )


def _record_payload(record: contract.CandidateRecord) -> dict[str, Any]:
    return {
        "document_id": record.document_id, "source_document_index": record.source_document_index,
        "source_file": record.source_file, "source_revision": record.source_revision,
        "source_blob_sha256": record.source_blob_sha256, "domain": record.domain.value,
        "license_id": record.license_id, "role_license": record.role_license,
        "structural_partition": record.structural_partition,
        "normalized_python_sha256": record.normalized_python_sha256,
    }


def _role_payload(role: contract.FrozenRole) -> dict[str, Any]:
    return {
        "schema": "newline_l12h6_canary_v1_role", "role": role.role,
        "authorization": ROLE_AUTHORIZATIONS[role.role], "rows": role.rows,
        "records": [_record_payload(record) for record in role.records],
        "masks": {name: value for name, value in role.masks.as_mapping().items()},
        "support": dict(role.support),
    }


def _role_from_payload(payload: Mapping[str, Any], spec: NewlineMaskSpec) -> contract.FrozenRole:
    if set(payload) != {"schema", "role", "authorization", "rows", "records", "masks", "support"} \
            or payload.get("schema") != "newline_l12h6_canary_v1_role" or payload.get(
                "authorization"
            ) != ROLE_AUTHORIZATIONS.get(payload.get("role")):
        raise RuntimeError("newline installed role schema/license changed")
    records = tuple(contract.CandidateRecord(
        item["document_id"], item["source_document_index"], item["source_file"],
        item["source_revision"], item["source_blob_sha256"], contract.NewlineDomain(item["domain"]),
        item["license_id"], item["role_license"], item["structural_partition"],
        item["normalized_python_sha256"],
    ) for item in payload["records"])
    rows = payload["rows"]
    replayed = build_newline_masks(rows, spec)
    if set(payload["masks"]) != set(replayed.as_mapping()) or any(
        not torch.equal(payload["masks"][name], replayed.as_mapping()[name])
        for name in replayed.as_mapping()
    ):
        raise RuntimeError("newline installed role masks do not replay")
    support = contract.support_census(replayed, records)
    if payload["support"] != support:
        raise RuntimeError("newline installed role support does not replay")
    return contract.FrozenRole(payload["role"], rows, records, replayed, support)


def _stable_role(path: Path, expected_sha256: str, spec: NewlineMaskSpec) -> contract.FrozenRole:
    before = file_sha256(path)
    if before != expected_sha256:
        raise RuntimeError("newline installed role bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before or not isinstance(payload, Mapping):
        raise RuntimeError("newline installed role changed during load")
    return _role_from_payload(payload, spec)


class RunClaim(NamedTuple):
    descriptor: int
    inode: int
    nonce: str


def acquire_claim(path: Path = LOCK) -> RunClaim:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32); descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (nonce + "\n").encode()); os.fsync(descriptor)
        return RunClaim(descriptor, os.fstat(descriptor).st_ino, nonce)
    except BaseException:
        os.close(descriptor); path.unlink(missing_ok=True); raise


def require_claim(claim: RunClaim, path: Path = LOCK) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(fd); raw = os.read(fd, 4096)
    finally:
        os.close(fd)
    after = path.stat()
    if (stat.st_ino, after.st_ino, raw) != (claim.inode, claim.inode, (claim.nonce + "\n").encode()):
        raise RuntimeError("newline row lock ownership changed")


def release_claim(claim: RunClaim, path: Path = LOCK) -> None:
    try:
        try:
            require_claim(claim, path); path.unlink()
        except (FileNotFoundError, RuntimeError, OSError):
            pass
    finally:
        try: os.close(claim.descriptor)
        except OSError: pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def write_create_only(path: Path, data: bytes, *, before_link) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("xb") as sink:
            sink.write(data); sink.flush(); os.fsync(sink.fileno())
        before_link()
        os.link(temporary, path); _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_snapshot() -> dict[str, Any]:
    paths = {"manifest": MANIFEST, "receipt": RECEIPT, "failure": FAILURE, **ROLE_FILES}
    return {name: {"exists": path.exists(), "sha256": file_sha256(path) if path.is_file() else None}
            for name, path in paths.items()}


def _protected_snapshot(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_hashes": source_closure(authority["source_commit"]),
        "registry_snapshot": registry_snapshot(),
        "source_identity": source_identity(authority["source_commit"]),
    }


def freeze(authority_path: Path = AUTHORITY, audit_path: Path = AUDIT) -> dict[str, Any]:
    if authority_path.resolve() != AUTHORITY.resolve() or audit_path.resolve() != AUDIT.resolve():
        raise RuntimeError("newline canonical authority/audit paths changed")
    authority_payload, authority_sha = _stable_json(authority_path)
    authority = validate_authority(authority_payload, replay_protected=False)
    audit, audit_sha = _stable_json(audit_path)
    validate_audit(audit, authority_sha256=authority_sha, authority=authority)
    authority = validate_authority(authority, replay_protected=True)
    if CACHE.exists() or RECEIPT.exists() or FAILURE.exists() or LOCK.exists():
        raise RuntimeError("newline row namespace is spent or locked")
    frozen_protected = _protected_snapshot(authority)
    claim = acquire_claim()
    try:
        try:
            def guard(*, expect_installed: bool) -> None:
                current_authority, current_authority_sha = _stable_json(AUTHORITY)
                current_audit, current_audit_sha = _stable_json(AUDIT)
                if current_authority_sha != authority_sha or current_authority != authority or (
                    current_audit_sha != audit_sha or current_audit != audit
                ):
                    raise RuntimeError("newline authority/audit changed")
                validate_audit(current_audit, authority_sha256=authority_sha, authority=current_authority)
                if _protected_snapshot(authority) != frozen_protected:
                    raise RuntimeError("newline protected inputs changed")
                expected = expect_installed
                if RECEIPT.exists() or FAILURE.exists() or any(
                    path.exists() != expected for path in (MANIFEST, *ROLE_FILES.values())
                ):
                    raise RuntimeError("newline terminal/artifact state changed")
                require_claim(claim)

            guard(expect_installed=False)
            exclusions = historical_exclusions(authority["registry_snapshot"])
            import tiktoken
            registry = token_registry.build_registry(tiktoken.get_encoding(token_registry.ENCODING_NAME))
            rows, records = production_enumeration(
                authority["source_commit"], registry, exclusions,
            )
            roles = contract.allocate_roles(
                rows, records, frozen_mask_spec(registry), exclusions, seed=ALLOCATION_SEED,
            )
            readiness.build_readiness(roles, registry, frozen_mask_spec(registry), allocation_seed=ALLOCATION_SEED)
            guard(expect_installed=False)
            staging = CACHE.with_name(f".{CACHE.name}.stage.{os.getpid()}.{secrets.token_hex(8)}")
            staging.mkdir(parents=True)
            try:
                role_entries = {}
                for role in roles:
                    staged = staging / ROLE_FILES[role.role].name
                    torch.save(_role_payload(role), staged)
                    with staged.open("rb") as source: os.fsync(source.fileno())
                    role_entries[role.role] = {
                        "filename": staged.name, "file_sha256": file_sha256(staged),
                        **contract.role_summary(role), "authorization": ROLE_AUTHORIZATIONS[role.role],
                    }
                manifest_payload = {
                    "schema": "newline_l12h6_canary_v1_rows_manifest",
                    "authority_sha256": authority_sha, "audit_sha256": audit_sha,
                    "source_commit": authority["source_commit"], "source_hashes": authority["source_hashes"],
                    "registry_snapshot_sha256": _logical_sha(authority["registry_snapshot"]),
                    "source_identity": authority["source_identity"], "roles": role_entries,
                    "outcome_access": False,
                }
                manifest_bytes = (json.dumps(
                    manifest_payload, sort_keys=True, indent=2, allow_nan=False,
                ) + "\n").encode()
                (staging / "manifest.json").write_bytes(manifest_bytes)
                with (staging / "manifest.json").open("rb") as source: os.fsync(source.fileno())
                _fsync_directory(staging)
                guard(expect_installed=False)
                CACHE.mkdir(parents=False)
                for path in (*ROLE_FILES.values(), MANIFEST):
                    os.link(staging / path.name, path)
                _fsync_directory(CACHE); _fsync_directory(CACHE.parent)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            spec = frozen_mask_spec(registry)
            installed_roles = tuple(
                _stable_role(ROLE_FILES[role], role_entries[role]["file_sha256"], spec)
                for role in contract.ROLE_ORDER
            )
            contract.validate_role_disjointness(installed_roles)
            if [contract.role_summary(role) for role in installed_roles] != [
                {key: role_entries[role.role][key] for key in (
                    "role", "rows_sha256", "records_sha256", "document_ids_sha256",
                    "support_sha256", "support",
                )} for role in installed_roles
            ]:
                raise RuntimeError("newline installed role summary changed")
            receipt_payload = {
                "schema": "newline_l12h6_canary_v1_rows_receipt",
                "status": "frozen_before_any_newline_model_forward_receipt_last",
                "authority_sha256": authority_sha, "audit_sha256": audit_sha,
                "source_commit": authority["source_commit"], "source_hashes": authority["source_hashes"],
                "manifest_sha256": file_sha256(MANIFEST), "roles": role_entries,
                "role_authorizations": ROLE_AUTHORIZATIONS, "outcome_access": False,
            }
            receipt_bytes = (json.dumps(
                receipt_payload, sort_keys=True, indent=2, allow_nan=False,
            ) + "\n").encode()
            def receipt_guard() -> None:
                guard(expect_installed=True)
                replayed = json.loads(receipt_bytes)
                if replayed != receipt_payload or file_sha256(MANIFEST) != receipt_payload["manifest_sha256"]:
                    raise RuntimeError("newline receipt/manifest replay changed")
                require_claim(claim)
            write_create_only(RECEIPT, receipt_bytes, before_link=receipt_guard)
            return receipt_payload
        except BaseException as error:
            if not RECEIPT.exists() and not FAILURE.exists():
                try:
                    observed = _artifact_snapshot()
                    failure_payload = {
                        "schema": "newline_l12h6_canary_v1_rows_failure",
                        "status": "terminal_failure_without_success_receipt",
                        "authority_sha256": authority_sha, "audit_sha256": audit_sha,
                        "artifacts": observed, "error_type": type(error).__name__,
                        "error": str(error), "outcome_access": False,
                    }
                    data = (json.dumps(failure_payload, sort_keys=True, indent=2) + "\n").encode()
                    def failure_guard() -> None:
                        if _artifact_snapshot() != observed or RECEIPT.exists() or FAILURE.exists():
                            raise RuntimeError("newline failure inputs/terminals changed")
                        current_authority, current_sha = _stable_json(AUTHORITY)
                        current_audit, current_audit_sha = _stable_json(AUDIT)
                        if current_sha != authority_sha or current_authority != authority or (
                            current_audit_sha != audit_sha or current_audit != audit
                        ):
                            raise RuntimeError("newline failure authority/audit changed")
                        require_claim(claim)
                    write_create_only(FAILURE, data, before_link=failure_guard)
                except BaseException:
                    pass
            raise
    finally:
        release_claim(claim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    arguments = parser.parse_args()
    print(json.dumps(freeze(arguments.authority, arguments.audit), indent=2))


__all__ = (
    "ALLOCATION_SEED", "AUDIT", "AUTHORITY", "CACHE", "FAILURE", "LOCK", "MANIFEST",
    "ROLE_FILES", "SOURCE_PATHS", "acquire_claim", "code_path_is_eligible",
    "code_tree_manifest", "discover_registry_files", "enumerate_from_sources", "file_sha256",
    "freeze", "frozen_mask_spec", "historical_exclusions", "is_list_table",
    "normalized_python_sha256", "registry_snapshot", "release_claim", "require_claim",
    "role_license", "source_closure", "source_identity", "validate_audit",
    "validate_authority", "write_create_only",
)
