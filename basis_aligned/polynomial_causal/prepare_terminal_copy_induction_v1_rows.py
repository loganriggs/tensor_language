#!/usr/bin/env python3
"""Outcome-blind four-role row freezer for terminal copy/induction v1.

No checkpoint or model module is imported.  Natural rows come from distinct ordered
FineWeb source documents; the OOD role comes from file-disjoint tracked Python blobs.
Synthetic positive/control pairs are derived deterministically from each role while
using token banks disjoint across roles and rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tokenize
import secrets
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

import prepare_block3_native_down_behavioral_port_v1_rows as natural
import terminal_copy_induction_v1 as contract


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
FREEZER = Path(__file__).resolve()
TEST = HERE / "test_prepare_terminal_copy_induction_v1_rows.py"
CONTRACT_TEST = HERE / "test_terminal_copy_induction_v1.py"
STREAMING_SCORER = HERE / "terminal_copy_streaming_statistics.py"
STREAMING_SCORER_TEST = HERE / "test_terminal_copy_streaming_statistics.py"
PREREG = HERE / "TERMINAL_COPY_INDUCTION_V1_PREREGISTRATION.md"
CONTRACT = HERE / "terminal_copy_induction_v1.py"
ADAPTER_ADDENDUM = HERE / "TERMINAL_COPY_ATTENTION_ADAPTER_V1_ADDENDUM.md"
SCREENING_AMENDMENT = HERE / "TERMINAL_COPY_INDUCTION_V1_SCREENING_AMENDMENT.md"
AUDIT = HERE / "terminal_copy_induction_v1_rows_audit.json"
RECEIPT_KIND = "terminal_copy_induction_v1_rows"
FAILURE_KIND = "terminal_copy_induction_v1_rows_failure"

START_DOCUMENT_INDEX = 70_000
N_PER_ROLE = 192
NATURAL_ROLES = ("fit_natural", "selection_natural", "final_natural")
OOD_ROLE = "ood_code"
ALL_ROLES = NATURAL_ROLES + (OOD_ROLE,)
SYNTHETIC_PAIRS_PER_ROLE = 32
SYNTHETIC_SEED = "terminal_copy_induction_v1:synthetic-banks:0"
SYNTHETIC_POSITION_TEMPLATES = (
    (8, 32, 80),
    (12, 44, 96),
    (20, 52, 128),
    (28, 60, 160),
)
CODE_SEED = "terminal_copy_induction_v1:ood-code-files:0"
CODE_PRIOR_MANIFESTS = tuple(sorted(HERE.glob("code_oracle_corpus*_manifest.json")))
GPT2_ENCODING_SHA256 = "0be287937901b1baae837369293dd6f63da1bece9609006e6485b57a3de37335"
TIKTOKEN_VERSION = "0.14.0"
PYARROW_VERSION = "25.0.1"
CODE_EXCLUDED_PREFIXES = (
    "archive/", "basis_aligned/bilinear_quotient/",
    "basis_aligned/polynomial_causal/", "data_", "figures/", "logs/",
    "runs/", "runs_", "tests/",
)
CODE_EXCLUDED_PARTS = {
    "__pycache__", "generated", "vendor", "third_party", "runlogs", "results",
}

CACHE = BQ / ".rowcache_terminal_copy_induction_v1"
RECEIPT = BQ / "terminal_copy_induction_v1_rows_receipt.json"
FAILURE = BQ / "terminal_copy_induction_v1_rows_failure.json"
LOCK = Path("/workspace/runs/.terminal_copy_induction_v1_rows.lock")

SOURCE_PATHS = (
    FREEZER, TEST, CONTRACT_TEST, STREAMING_SCORER, STREAMING_SCORER_TEST,
    PREREG, CONTRACT, ADAPTER_ADDENDUM, SCREENING_AMENDMENT,
    HERE / "prepare_block3_native_down_behavioral_port_v1_rows.py",
    HERE / "test_prepare_block3_native_down_behavioral_port_v1_rows.py",
    HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py",
    HERE / "prepare_mlp0_c512_mlp2_compensation_v1_rows.py",
    HERE / "local_fineweb_harvest.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return natural.tensor_sha256(value)


def source_closure(commit: str) -> dict[str, str]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"terminal-copy row source differs from {commit}: {relative}")
        hashes[relative] = digest
    return hashes


def validate_audit(
    commit: str, source_hashes: Mapping[str, str], eligible_code_tree_sha256: str,
) -> dict[str, Any]:
    before = file_sha256(AUDIT)
    payload = json.loads(AUDIT.read_bytes())
    audited_commit = payload.get("source_commit")
    if (
        file_sha256(AUDIT) != before
        or payload.get("approved") is not True
        or not isinstance(audited_commit, str)
        or payload.get("outcome_access") is not False
        or payload.get("source_hashes") != dict(source_hashes)
        or payload.get("eligible_code_tree_sha256") != eligible_code_tree_sha256
    ):
        raise RuntimeError("terminal-copy row audit does not authorize these source bytes")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", audited_commit, commit],
        cwd=ROOT, check=True,
    )
    return payload


def source_identity(
    canonical: Mapping[str, Any], parquet: Path, registry_hashes: Mapping[str, str],
    encoding: Any,
) -> dict[str, Any]:
    """Bind the exact canonical receipt bytes, loader versions, and tokenizer table."""

    import pyarrow
    import tiktoken
    import local_fineweb_harvest as local

    canonical_path = natural.BASE.CANONICAL_RECEIPT.resolve()
    before = file_sha256(canonical_path)
    raw = canonical_path.read_bytes()
    after = file_sha256(canonical_path)
    parsed = json.loads(raw)
    fingerprint = local.encoding_fingerprint(encoding)
    if (
        before != after
        or hashlib.sha256(raw).hexdigest() != before
        or registry_hashes.get(str(canonical_path)) != before
        or parsed != dict(canonical)
        or fingerprint != GPT2_ENCODING_SHA256
        or tiktoken.__version__ != TIKTOKEN_VERSION
        or pyarrow.__version__ != PYARROW_VERSION
        or parquet.stat().st_size != local.PINNED_SIZE
        or file_sha256(parquet) != local.PINNED_SHA256
    ):
        raise RuntimeError("terminal-copy source/tokenizer/loader identity changed")
    return {
        "canonical_receipt_path": str(canonical_path),
        "canonical_receipt_sha256": before,
        "parquet_path": str(parquet.resolve()),
        "parquet_bytes": parquet.stat().st_size,
        "parquet_sha256": local.PINNED_SHA256,
        "tiktoken_version": tiktoken.__version__,
        "gpt2_encoding_sha256": fingerprint,
        "pyarrow_version": pyarrow.__version__,
        "loader_semantics": (
            "pinned parquet canonical document order; GPT-2 encode_ordinary; "
            "search every nonoverlapping 257-token chunk until the first row and "
            "prefix32 collision-free chunk; exactly one row per accepted document"
        ),
    }


def split_natural_rows(
    rows: torch.Tensor, records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, Any]]]]:
    expected = len(NATURAL_ROLES) * N_PER_ROLE
    if tuple(rows.shape) != (expected, contract.ROW_WIDTH) or len(records) != expected:
        raise ValueError("combined natural allocation has the wrong shape")
    role_rows: dict[str, torch.Tensor] = {}
    role_records: dict[str, list[dict[str, Any]]] = {}
    seen_documents: set[str] = set()
    for role_index, role in enumerate(NATURAL_ROLES):
        start, stop = role_index * N_PER_ROLE, (role_index + 1) * N_PER_ROLE
        selected = rows[start:stop].contiguous()
        selected_records = []
        for row_index, record in enumerate(records[start:stop]):
            item = dict(record)
            item["role"] = role
            item["role_row_index"] = row_index
            document = item.get("document_id")
            if not isinstance(document, str) or document in seen_documents:
                raise RuntimeError("natural terminal-copy roles repeat a document")
            seen_documents.add(document)
            selected_records.append(item)
        role_rows[role] = selected
        role_records[role] = selected_records
    return role_rows, role_records


def _walk_json(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_json(item)


def prior_code_paths(manifests: Sequence[Path] = CODE_PRIOR_MANIFESTS) -> tuple[set[str], dict[str, str]]:
    paths: set[str] = set()
    hashes: dict[str, str] = {}
    for manifest in manifests:
        before = file_sha256(manifest)
        payload = json.loads(manifest.read_bytes())
        if file_sha256(manifest) != before:
            raise RuntimeError("prior code manifest changed while reading")
        hashes[str(manifest.resolve())] = before
        for value in _walk_json(payload):
            if not isinstance(value, Mapping):
                continue
            for key in ("path", "source_path", "file_path"):
                path = value.get(key)
                if not isinstance(path, str) or not path.endswith(".py"):
                    continue
                candidate = Path(path)
                if candidate.is_absolute():
                    try:
                        path = str(candidate.resolve().relative_to(ROOT))
                    except ValueError:
                        continue
                paths.add(path)
    return paths, hashes


def code_path_is_eligible(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return (
        path.endswith(".py")
        and not path.startswith(CODE_EXCLUDED_PREFIXES)
        and not any(part in CODE_EXCLUDED_PARTS for part in parts)
        and not name.startswith("test_")
        and not name.endswith("_test.py")
    )


def normalized_python_sha256(blob: bytes) -> str:
    """Hash Python after removing formatting/comments and normalizing literal values."""

    normalized: list[str] = []
    try:
        stream = tokenize.tokenize(io.BytesIO(blob).readline)
        for token in stream:
            if token.type in {
                tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE,
                tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT,
            }:
                continue
            if token.type == tokenize.STRING:
                normalized.append("STRING")
            elif token.type == tokenize.NUMBER:
                normalized.append("NUMBER")
            else:
                normalized.append(token.string)
    except (IndentationError, SyntaxError, tokenize.TokenError, UnicodeDecodeError):
        normalized = [" ".join(blob.decode("utf-8", errors="replace").split())]
    return hashlib.sha256("\0".join(normalized).encode()).hexdigest()


def eligible_code_tree_manifest(commit: str) -> dict[str, Any]:
    """Hash every eligible path and blob so a later commit cannot change OOD sampling."""

    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True,
    ).splitlines()
    entries = []
    for path in sorted(path for path in paths if code_path_is_eligible(path)):
        blob = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        entries.append({
            "path": path,
            "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "normalized_python_sha256": normalized_python_sha256(blob),
        })
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "eligible_file_count": len(entries),
        "entries_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def ordered_code_blobs(commit: str) -> Iterable[tuple[str, bytes]]:
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True,
    ).splitlines()
    candidates = sorted(
        (path for path in paths if code_path_is_eligible(path)),
        key=lambda path: hashlib.sha256(f"{CODE_SEED}\0{path}".encode()).digest(),
    )
    for path in candidates:
        yield path, subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def allocate_code_rows(
    blobs: Iterable[tuple[str, bytes]], encode: Callable[[str], list[int]],
    prior: tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    excluded_paths: set[str], *, excluded_normalized: set[str] | None = None,
    n_rows: int = N_PER_ROLE,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if type(n_rows) is not int or n_rows <= 0:
        raise ValueError("code row count must be positive")
    prior_rows = {row[: contract.ROW_WIDTH] for row in prior[2] if len(row) >= contract.ROW_WIDTH}
    used_prefixes = set(prior[3])
    selected: list[list[int]] = []
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_normalized: set[str] = set() if excluded_normalized is None else set(excluded_normalized)
    for path, blob in blobs:
        if len(selected) == n_rows:
            break
        if not isinstance(path, str) or not path or path in excluded_paths or path in seen_paths or (
            not isinstance(blob, bytes)
        ):
            continue
        seen_paths.add(path)
        normalized_sha256 = normalized_python_sha256(blob)
        if normalized_sha256 in seen_normalized:
            continue
        seen_normalized.add(normalized_sha256)
        tokens = [50256] + encode(blob.decode("utf-8", errors="replace"))
        chunks = len(tokens) // contract.ROW_WIDTH
        if chunks <= 0:
            continue
        digest = hashlib.sha256(f"{CODE_SEED}\0{path}".encode()).digest()
        start_chunk = int.from_bytes(digest[:8], "big") % chunks
        chosen: tuple[list[int], int] | None = None
        for offset in range(chunks):
            chunk = (start_chunk + offset) % chunks
            start = chunk * contract.ROW_WIDTH
            row = tokens[start:start + contract.ROW_WIDTH]
            row_tuple = tuple(int(value) for value in row)
            prefix = row_tuple[: natural.PREFIX_LENGTH]
            if row_tuple in prior_rows or prefix in used_prefixes:
                continue
            chosen = (row, chunk)
            break
        if chosen is None:
            continue
        row, chunk = chosen
        used_prefixes.add(tuple(int(value) for value in row[: natural.PREFIX_LENGTH]))
        selected.append(row)
        records.append({
            "role": OOD_ROLE,
            "role_row_index": len(selected) - 1,
            "path": path,
            "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "normalized_python_sha256": normalized_sha256,
            "chunk_index": chunk,
            "token_start": chunk * contract.ROW_WIDTH,
        })
    if len(selected) != n_rows:
        raise RuntimeError(f"only {len(selected)}/{n_rows} fresh code files were eligible")
    rows = torch.tensor(selected, dtype=torch.long)
    if len({record["path"] for record in records}) != n_rows or len(
        {tuple(row.tolist()) for row in rows}
    ) != n_rows:
        raise RuntimeError("OOD code role is not file/row unique")
    return rows, records


def _token_order(role: str) -> Iterable[int]:
    values = sorted(
        range(50256),
        key=lambda token: hashlib.sha256(
            f"{SYNTHETIC_SEED}\0{role}\0{token}".encode()
        ).digest(),
    )
    yield from values


def build_synthetic_roles(
    role_rows: Mapping[str, torch.Tensor],
    *, pairs_per_role: int = SYNTHETIC_PAIRS_PER_ROLE,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list[list[int]]]]:
    if set(role_rows) != set(ALL_ROLES) or type(pairs_per_role) is not int or (
        pairs_per_role <= 0 or pairs_per_role > min(len(rows) for rows in role_rows.values())
    ):
        raise ValueError("synthetic role source is malformed")
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    banks: dict[str, list[list[int]]] = {}
    globally_used: set[int] = set()
    for role in ALL_ROLES:
        iterator = iter(_token_order(role))
        query_to_y_rows, query_to_z_rows, role_banks = [], [], []
        for index in range(pairs_per_role):
            base = tuple(int(value) for value in role_rows[role][index])
            token_bank: list[int] = []
            while len(token_bank) < 4:
                token = next(iterator)
                if token not in globally_used and token not in base:
                    globally_used.add(token)
                    token_bank.append(token)
            first, reciprocal, query = SYNTHETIC_POSITION_TEMPLATES[
                index % len(SYNTHETIC_POSITION_TEMPLATES)
            ]
            crossover = contract.build_synthetic_association_crossover(
                base,
                first_query_position=first,
                reciprocal_position=reciprocal,
                query_position=query,
                query_token=token_bank[0],
                reciprocal_query=token_bank[1],
                successor_y=token_bank[2],
                successor_z=token_bank[3],
            )
            query_to_y_rows.append(crossover.query_to_y)
            query_to_z_rows.append(crossover.query_to_z)
            role_banks.append(token_bank)
        outputs[role] = {
            "query_to_y": torch.stack(query_to_y_rows).contiguous(),
            "query_to_z": torch.stack(query_to_z_rows).contiguous(),
        }
        banks[role] = role_banks
    return outputs, banks


def fit_token_frequencies(rows: torch.Tensor) -> contract.FitTokenFrequencies:
    if tuple(rows.shape) != (N_PER_ROLE, contract.ROW_WIDTH):
        raise ValueError("fit natural rows have the wrong shape")
    return contract.FitTokenFrequencies.from_rows(rows, vocab_size=50257)


def serialize_copy_cells(cells: contract.CopyCells) -> dict[str, Any]:
    """Make the frozen label artifact explicit and tensor-only."""

    return {
        "all_positive": cells.all_positive,
        "positive": cells.positive,
        "matched_negative": cells.matched_negative,
        "off_target": cells.off_target,
        "pair_indices": cells.pair_indices,
        "unmatched_positive_count": cells.unmatched_positive_count,
        "negative_candidate_count": cells.negative_candidate_count,
        "eligible_stratum_count": cells.eligible_stratum_count,
        "excluded_low_document_stratum_count": cells.excluded_low_document_stratum_count,
    }


def support_census(copy_cells: Mapping[str, contract.CopyCells]) -> dict[str, Any]:
    """Apply the prospective pre-model support gate to every scored role."""

    scored_roles = ("selection_natural", "final_natural", OOD_ROLE)
    census: dict[str, Any] = {}
    for role in scored_roles:
        cells = copy_cells[role]
        positive_indices = torch.nonzero(cells.positive, as_tuple=False)
        negative_indices = torch.nonzero(cells.matched_negative, as_tuple=False)
        positive_documents = len(set(int(value) for value in positive_indices[:, 0]))
        negative_documents = len(set(int(value) for value in negative_indices[:, 0]))
        entry = {
            "positive_positions": len(positive_indices),
            "matched_negative_positions": len(negative_indices),
            "positive_documents": positive_documents,
            "matched_negative_documents": negative_documents,
            "passed": (
                len(positive_indices) >= 48
                and len(negative_indices) >= 48
                and positive_documents >= 24
                and negative_documents >= 24
            ),
        }
        census[role] = entry
    if not all(entry["passed"] for entry in census.values()):
        raise RuntimeError(f"terminal-copy pre-model support gate failed: {census}")
    return census


def verify_code_snapshot(commit: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Re-read every selected code blob from the bound commit before publication."""

    for record in records:
        path = record.get("path")
        if not isinstance(path, str) or not code_path_is_eligible(path):
            raise RuntimeError("terminal-copy OOD code path is no longer eligible")
        blob = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != record.get("blob_sha256") or (
            normalized_python_sha256(blob) != record.get("normalized_python_sha256")
        ):
            raise RuntimeError("terminal-copy OOD code blob changed")


def prior_normalized_code_hashes(commit: str, paths: Iterable[str]) -> set[str]:
    hashes: set[str] = set()
    for path in sorted(set(paths)):
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            hashes.add(normalized_python_sha256(completed.stdout))
    return hashes


def summarize_roles(
    rows: Mapping[str, torch.Tensor], records: Mapping[str, Sequence[Mapping[str, Any]]],
    synthetic: Mapping[str, Mapping[str, torch.Tensor]], banks: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, Any]:
    if set(rows) != set(ALL_ROLES) or set(records) != set(ALL_ROLES):
        raise RuntimeError("terminal-copy role set changed")
    natural_documents = {
        record["document_id"] for role in NATURAL_ROLES for record in records[role]
    }
    code_paths = {record["path"] for record in records[OOD_ROLE]}
    bank_sets = {
        role: {token for sequence in banks[role] for token in sequence} for role in ALL_ROLES
    }
    bank_disjoint = all(
        bank_sets[left].isdisjoint(bank_sets[right])
        for index, left in enumerate(ALL_ROLES) for right in ALL_ROLES[index + 1:]
    )
    gates = {
        "three_natural_roles_have_192_unique_documents_each": (
            len(natural_documents) == len(NATURAL_ROLES) * N_PER_ROLE
            and all(len(records[role]) == N_PER_ROLE for role in NATURAL_ROLES)
        ),
        "ood_has_192_unique_source_files": len(code_paths) == N_PER_ROLE,
        "all_rows_have_width_257": all(
            tuple(value.shape) == (N_PER_ROLE, contract.ROW_WIDTH) for value in rows.values()
        ),
        "synthetic_roles_have_32_pairs": all(
            tuple(synthetic[role][arm].shape) == (SYNTHETIC_PAIRS_PER_ROLE, contract.ROW_WIDTH)
            for role in ALL_ROLES for arm in ("query_to_y", "query_to_z")
        ),
        "synthetic_token_banks_are_cross_role_disjoint": bank_disjoint,
    }
    if not all(gates.values()):
        raise RuntimeError(f"terminal-copy row gates failed: {gates}")
    return {
        "roles": {role: len(rows[role]) for role in ALL_ROLES},
        "synthetic_pairs": {role: len(synthetic[role]["query_to_y"]) for role in ALL_ROLES},
        "unique_natural_documents": len(natural_documents),
        "unique_ood_files": len(code_paths),
        "gates": gates,
    }


def _save_create_only(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        if path.exists():
            raise RuntimeError(f"refusing to overwrite terminal-copy cache file: {path}")
        torch.save(value, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_staged(value: Any, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"terminal-copy staging path already exists: {path}")
    torch.save(value, path)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json_terminal(
    payload: Mapping[str, Any], path: Path, claim: Any, *, lock_path: Path | None = None,
) -> None:
    """Create a terminal JSON once, checking lock ownership at the actual link."""

    lock_path = LOCK if lock_path is None else lock_path

    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    linked = False
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("terminal-copy publication made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        natural.require_claim(claim, lock_path)
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        try:
            _fsync_directory(path.parent)
        except OSError:
            # Once the terminal hard link exists, raising would report failure while
            # leaving an authoritative marker. Exact byte reload makes that state a
            # success; a later receipt validator still rejects any corruption.
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if not linked or path.read_bytes() != encoded:
        raise RuntimeError("terminal-copy terminal JSON failed exact publication")


def _semantic_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_semantic_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_semantic_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def json_normalize(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact JSON-domain value that a receipt reload will produce."""

    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def validate_cached_payload(
    path: Path, expected: Mapping[str, Any], entry: Mapping[str, Any], role: str,
) -> dict[str, Any]:
    before = file_sha256(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if (
        file_sha256(path) != before
        or before != entry.get("file_sha256")
        or not isinstance(loaded, Mapping)
        or not _semantic_equal(loaded, expected)
        or tensor_sha256(loaded["rows"]) != entry.get("rows_tensor_sha256")
        or tensor_sha256(loaded["synthetic"]["query_to_y"]) != entry.get(
            "query_to_y_tensor_sha256"
        )
        or tensor_sha256(loaded["synthetic"]["query_to_z"]) != entry.get(
            "query_to_z_tensor_sha256"
        )
        or tensor_sha256(loaded["copy_cells"]["positive"]) != entry.get(
            "copy_positive_mask_sha256"
        )
        or tensor_sha256(loaded["copy_cells"]["matched_negative"]) != entry.get(
            "copy_matched_negative_mask_sha256"
        )
    ):
        raise RuntimeError(f"terminal-copy cached role failed exact replay: {role}")
    return dict(loaded)


def load_prior_registry(
    registry_files: tuple[Path, ...],
) -> tuple[Any, dict[str, str], dict[str, str], list[dict[str, Any]]]:
    """Load the strict v1 registry; v2 may replace this source-closed hook."""

    prior, registry_hashes, tensor_hashes = natural.load_registry_exclusions(registry_files)
    return prior, registry_hashes, tensor_hashes, []


def verify_prior_snapshot(
    *, commit: str, sources: Mapping[str, str], registry_files: tuple[Path, ...],
    registry_hashes: Mapping[str, str], tensor_hashes: Mapping[str, str],
    prior: Any, parquet: Path, registry_waivers: Sequence[Mapping[str, Any]],
) -> None:
    """Replay the strict v1 registry; v2 may replace this source-closed hook."""

    if registry_waivers:
        raise RuntimeError("strict terminal-copy v1 does not permit registry waivers")
    natural.verify_snapshot(
        commit=commit, sources=sources, registry_files=registry_files,
        registry_hashes=registry_hashes, tensor_hashes=tensor_hashes,
        prior=prior, parquet=parquet,
    )


def freeze() -> dict[str, Any]:
    claim = natural.acquire_claim(LOCK)
    try:
        if CACHE.exists() or RECEIPT.exists() or FAILURE.exists():
            raise RuntimeError("terminal-copy row namespace is spent")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        sources = source_closure(commit)
        code_tree = eligible_code_tree_manifest(commit)
        audit = validate_audit(commit, sources, code_tree["entries_sha256"])
        canonical, parquet = natural.BASE.validate_ordered_source()
        registry_files = natural.discover_registry_files()
        prior, registry_hashes, prior_tensor_hashes, registry_waivers = load_prior_registry(
            registry_files
        )

        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        frozen_source_identity = source_identity(
            canonical, parquet, registry_hashes, encoding,
        )
        combined, combined_records = natural.harvest_fresh_documents(
            natural.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
            start_document_index=START_DOCUMENT_INDEX,
            n_source_documents=len(NATURAL_ROLES) * N_PER_ROLE,
            token_length=contract.ROW_WIDTH,
        )
        role_rows, role_records = split_natural_rows(combined, combined_records)
        excluded_code_paths, code_manifest_hashes = prior_code_paths(
            tuple(sorted(set(CODE_PRIOR_MANIFESTS) | set(registry_files)))
        )
        natural_full_rows = {
            tuple(int(value) for value in row.tolist())
            for role in NATURAL_ROLES for row in role_rows[role]
        }
        natural_prefixes = {
            values[: natural.PREFIX_LENGTH] for values in natural_full_rows
        }
        code_prior = (
            set(prior[0]), set(prior[1]), set(prior[2]) | natural_full_rows,
            set(prior[3]) | natural_prefixes,
        )
        code_rows, code_records = allocate_code_rows(
            ordered_code_blobs(commit), encoding.encode_ordinary, code_prior,
            excluded_code_paths,
            excluded_normalized=prior_normalized_code_hashes(commit, excluded_code_paths),
        )
        role_rows[OOD_ROLE] = code_rows
        role_records[OOD_ROLE] = code_records
        synthetic, banks = build_synthetic_roles(role_rows)
        frequencies = fit_token_frequencies(role_rows["fit_natural"])
        copy_cells: dict[str, contract.CopyCells] = {}
        for role in ALL_ROLES:
            document_ids = tuple(
                str(record.get("document_id") or (
                    f"code:{record['path']}:{record['blob_sha256']}"
                ))
                for record in role_records[role]
            )
            copy_cells[role] = contract.build_copy_cells(
                role_rows[role], frequencies, document_ids,
            )
        frozen_support_census = support_census(copy_cells)
        summary = summarize_roles(role_rows, role_records, synthetic, banks)

        verify_prior_snapshot(
            commit=commit,
            sources=natural.source_closure(commit),
            registry_files=registry_files,
            registry_hashes=registry_hashes,
            tensor_hashes=prior_tensor_hashes,
            prior=prior,
            parquet=parquet,
            registry_waivers=registry_waivers,
        )
        if source_closure(commit) != sources or eligible_code_tree_manifest(commit) != code_tree or (
            validate_audit(commit, sources, code_tree["entries_sha256"]) != audit
        ):
            raise RuntimeError("terminal-copy source/audit closure changed")
        if source_identity(canonical, parquet, registry_hashes, encoding) != frozen_source_identity:
            raise RuntimeError("terminal-copy source identity changed before publication")
        verify_code_snapshot(commit, role_records[OOD_ROLE])

        # Exact outcome-blind semantic replay before any cache becomes visible.
        replay_combined, replay_combined_records = natural.harvest_fresh_documents(
            natural.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
            start_document_index=START_DOCUMENT_INDEX,
            n_source_documents=len(NATURAL_ROLES) * N_PER_ROLE,
            token_length=contract.ROW_WIDTH,
        )
        replay_role_rows, replay_role_records = split_natural_rows(
            replay_combined, replay_combined_records,
        )
        replay_code_rows, replay_code_records = allocate_code_rows(
            ordered_code_blobs(commit), encoding.encode_ordinary, code_prior,
            excluded_code_paths,
            excluded_normalized=prior_normalized_code_hashes(commit, excluded_code_paths),
        )
        replay_role_rows[OOD_ROLE] = replay_code_rows
        replay_role_records[OOD_ROLE] = replay_code_records
        replay_synthetic, replay_banks = build_synthetic_roles(replay_role_rows)
        replay_frequencies = fit_token_frequencies(replay_role_rows["fit_natural"])
        replay_cells: dict[str, contract.CopyCells] = {}
        for role in ALL_ROLES:
            replay_document_ids = tuple(
                str(record.get("document_id") or (
                    f"code:{record['path']}:{record['blob_sha256']}"
                ))
                for record in replay_role_records[role]
            )
            replay_cells[role] = contract.build_copy_cells(
                replay_role_rows[role], replay_frequencies, replay_document_ids,
            )
        if not (
            _semantic_equal(replay_role_rows, role_rows)
            and _semantic_equal(replay_role_records, role_records)
            and _semantic_equal(replay_synthetic, synthetic)
            and _semantic_equal(replay_banks, banks)
            and _semantic_equal(replay_frequencies.query, frequencies.query)
            and _semantic_equal(replay_frequencies.target, frequencies.target)
            and _semantic_equal(
                {role: serialize_copy_cells(replay_cells[role]) for role in ALL_ROLES},
                {role: serialize_copy_cells(copy_cells[role]) for role in ALL_ROLES},
            )
            and support_census(replay_cells) == frozen_support_census
        ):
            raise RuntimeError("terminal-copy exact pre-publication semantic replay changed")
        # This replay is intentionally adjacent to cache installation.  In v2 it
        # rechecks the exact failed-authority/failure pair and every declared absence
        # under the E4-v2 owner lock; in strict v1 it is an ordinary full census.
        verify_prior_snapshot(
            commit=commit,
            sources=natural.source_closure(commit),
            registry_files=registry_files,
            registry_hashes=registry_hashes,
            tensor_hashes=prior_tensor_hashes,
            prior=prior,
            parquet=parquet,
            registry_waivers=registry_waivers,
        )
        natural.require_claim(claim, LOCK)
        staging = CACHE.with_name(
            f".{CACHE.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
        )
        staging.mkdir(parents=False, exist_ok=False)
        entries: dict[str, Any] = {}
        payloads: dict[str, dict[str, Any]] = {}
        try:
            for role in ALL_ROLES:
                payload = {
                    "rows": role_rows[role],
                    "records": role_records[role],
                    "synthetic": synthetic[role],
                    "synthetic_token_banks": banks[role],
                    "synthetic_position_templates": SYNTHETIC_POSITION_TEMPLATES,
                    "copy_cells": serialize_copy_cells(copy_cells[role]),
                }
                payloads[role] = payload
                staged_path = staging / f"{role}.pt"
                _save_staged(payload, staged_path)
                entries[role] = {
                    "path": str((CACHE / staged_path.name).resolve()),
                    "file_sha256": file_sha256(staged_path),
                    "rows_tensor_sha256": tensor_sha256(role_rows[role]),
                    "query_to_y_tensor_sha256": tensor_sha256(
                        synthetic[role]["query_to_y"]
                    ),
                    "query_to_z_tensor_sha256": tensor_sha256(
                        synthetic[role]["query_to_z"]
                    ),
                    "copy_positive_mask_sha256": tensor_sha256(copy_cells[role].positive),
                    "copy_matched_negative_mask_sha256": tensor_sha256(
                        copy_cells[role].matched_negative
                    ),
                }
                validate_cached_payload(staged_path, payload, entries[role], role)
            frequencies_payload = {
                "query": frequencies.query, "target": frequencies.target,
            }
            staged_frequencies = staging / "fit_token_frequencies.pt"
            _save_staged(frequencies_payload, staged_frequencies)
            frequencies_entry = {
                "path": str((CACHE / staged_frequencies.name).resolve()),
                "file_sha256": file_sha256(staged_frequencies),
                "query_tensor_sha256": tensor_sha256(frequencies.query),
                "target_tensor_sha256": tensor_sha256(frequencies.target),
            }
            loaded_frequencies = torch.load(
                staged_frequencies, map_location="cpu", weights_only=True,
            )
            if not _semantic_equal(loaded_frequencies, frequencies_payload):
                raise RuntimeError("terminal-copy staged frequencies failed exact replay")
            _fsync_directory(staging)
            natural.require_claim(claim, LOCK)
            if CACHE.exists():
                raise RuntimeError("terminal-copy cache appeared before install")
            os.replace(staging, CACHE)
            _fsync_directory(CACHE.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        for role in ALL_ROLES:
            validate_cached_payload(
                Path(entries[role]["path"]), payloads[role], entries[role], role,
            )
        installed_frequencies = Path(frequencies_entry["path"])
        before_frequencies = file_sha256(installed_frequencies)
        loaded_frequencies = torch.load(
            installed_frequencies, map_location="cpu", weights_only=True,
        )
        if (
            before_frequencies != frequencies_entry["file_sha256"]
            or file_sha256(installed_frequencies) != before_frequencies
            or not _semantic_equal(loaded_frequencies, frequencies_payload)
        ):
            raise RuntimeError("terminal-copy installed frequencies failed exact replay")

        receipt = json_normalize({
            "schema_version": 1,
            "receipt_kind": RECEIPT_KIND,
            "status": "frozen_before_any_terminal_copy_model_forward",
            "authorized_for_scored_experiments": False,
            "authorized_for_candidate_or_threshold_selection": False,
            "source_commit": commit,
            "source_hashes": sources,
            "audit_file_sha256": file_sha256(AUDIT),
            "audit": audit,
            "selection": {
                "natural_start_document_index": START_DOCUMENT_INDEX,
                "natural_documents_per_role": N_PER_ROLE,
                "ood_files": N_PER_ROLE,
                "tokens_per_row": contract.ROW_WIDTH,
                "score_positions": [contract.SCORE_START, contract.SCORE_STOP],
                "synthetic_pairs_per_role": SYNTHETIC_PAIRS_PER_ROLE,
                "synthetic_position_templates": SYNTHETIC_POSITION_TEMPLATES,
                "copy_label": "nearest prior equal query has current target as successor",
                "minimum_documents_per_polarity_stratum": (
                    contract.MIN_DOCUMENTS_PER_POLARITY_STRATUM
                ),
            },
            "summary": summary,
            "support_census": frozen_support_census,
            "entries": entries,
            "fit_token_frequencies": frequencies_entry,
            "prior_registry_files": registry_hashes,
            "prior_row_tensors": prior_tensor_hashes,
            "failed_unmaterialized_registry_exclusions": registry_waivers,
            "prior_code_manifests": code_manifest_hashes,
            "eligible_code_tree": code_tree,
            "role_licenses": {
                "fit_natural": {
                    "authorized_use": "fit_per_position_head_write_means_only",
                    "requires_receipt": None,
                },
                "selection_natural": {
                    "authorized_use": "candidate_selection_only",
                    "requires_receipt": "terminal_copy_induction_v1_fit_means_receipt",
                },
                "final_natural": {
                    "authorized_use": "one_shot_final_replication_only",
                    "requires_receipt": "terminal_copy_induction_v1_selection_passer_receipt",
                },
                OOD_ROLE: {
                    "authorized_use": "one_shot_single_repository_ood_replication_only",
                    "requires_receipt": "terminal_copy_induction_v1_selection_passer_receipt",
                },
            },
            "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
            "source_identity": frozen_source_identity,
            "outcome_access": {
                "checkpoint_loaded": False,
                "model_imported": False,
                "model_forward_calls": 0,
                "scientific_outcomes_read": False,
            },
        })
        verify_prior_snapshot(
            commit=commit,
            sources=natural.source_closure(commit),
            registry_files=registry_files,
            registry_hashes=registry_hashes,
            tensor_hashes=prior_tensor_hashes,
            prior=prior,
            parquet=parquet,
            registry_waivers=registry_waivers,
        )
        if source_closure(commit) != sources or eligible_code_tree_manifest(commit) != code_tree or (
            validate_audit(commit, sources, code_tree["entries_sha256"]) != audit
        ) or source_identity(canonical, parquet, registry_hashes, encoding) != frozen_source_identity:
            raise RuntimeError("terminal-copy terminal snapshot changed")
        verify_code_snapshot(commit, role_records[OOD_ROLE])
        for role in ALL_ROLES:
            validate_cached_payload(
                Path(entries[role]["path"]), payloads[role], entries[role], role,
            )
        natural.require_claim(claim, LOCK)
        _publish_json_terminal(receipt, RECEIPT, claim)
        reloaded_receipt = json.loads(RECEIPT.read_bytes())
        if reloaded_receipt != receipt or FAILURE.exists():
            raise RuntimeError("terminal-copy terminal receipt failed exact reload")
        return receipt
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            try:
                natural.require_claim(claim, LOCK)
                _publish_json_terminal({
                    "schema_version": 1,
                    "receipt_kind": FAILURE_KIND,
                    "status": "terminal_failure_no_receipt",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "cache_exists": CACHE.exists(),
                    "receipt_exists": RECEIPT.exists(),
                    "outcome_access": {
                        "checkpoint_loaded": False,
                        "model_imported": False,
                        "model_forward_calls": 0,
                    },
                }, FAILURE, claim)
            except BaseException:
                pass
        raise
    finally:
        natural.release_claim(claim, LOCK)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
