#!/usr/bin/env python3
"""Outcome-blind four-role row freezer for terminal copy/induction v1.

No checkpoint or model module is imported.  Natural rows come from distinct ordered
FineWeb source documents; the OOD role comes from file-disjoint tracked Python blobs.
Synthetic positive/control pairs are derived deterministically from each role while
using token banks disjoint across roles and rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
PREREG = HERE / "TERMINAL_COPY_INDUCTION_V1_PREREGISTRATION.md"
CONTRACT = HERE / "terminal_copy_induction_v1.py"
ADAPTER_ADDENDUM = HERE / "TERMINAL_COPY_ATTENTION_ADAPTER_V1_ADDENDUM.md"
AUDIT = HERE / "terminal_copy_induction_v1_rows_audit.json"

START_DOCUMENT_INDEX = 70_000
N_PER_ROLE = 192
NATURAL_ROLES = ("fit_natural", "selection_natural", "final_natural")
OOD_ROLE = "ood_code"
ALL_ROLES = NATURAL_ROLES + (OOD_ROLE,)
SYNTHETIC_PAIRS_PER_ROLE = 32
SYNTHETIC_SEQUENCE_LENGTH = 16
SYNTHETIC_CUT = 8
SYNTHETIC_SEED = "terminal_copy_induction_v1:synthetic-banks:0"
CODE_SEED = "terminal_copy_induction_v1:ood-code-files:0"
CODE_PRIOR_MANIFESTS = tuple(sorted(HERE.glob("code_oracle_corpus*_manifest.json")))

CACHE = BQ / ".rowcache_terminal_copy_induction_v1"
RECEIPT = BQ / "terminal_copy_induction_v1_rows_receipt.json"
LOCK = Path("/workspace/runs/.terminal_copy_induction_v1_rows.lock")

SOURCE_PATHS = (
    FREEZER, TEST, PREREG, CONTRACT, ADAPTER_ADDENDUM, AUDIT,
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


def validate_audit(commit: str) -> dict[str, Any]:
    before = file_sha256(AUDIT)
    payload = json.loads(AUDIT.read_bytes())
    if file_sha256(AUDIT) != before or payload.get("approved") is not True or (
        payload.get("source_commit") != commit
    ) or payload.get("outcome_access") is not False:
        raise RuntimeError("terminal-copy row audit does not authorize this source commit")
    return payload


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


def prior_code_paths(manifests: Sequence[Path] = CODE_PRIOR_MANIFESTS) -> tuple[set[str], dict[str, str]]:
    paths: set[str] = set()
    hashes: dict[str, str] = {}
    for manifest in manifests:
        before = file_sha256(manifest)
        payload = json.loads(manifest.read_bytes())
        if file_sha256(manifest) != before:
            raise RuntimeError("prior code manifest changed while reading")
        hashes[str(manifest.resolve())] = before
        for split_rows in payload.get("files", {}).values():
            for row in split_rows:
                path = row.get("path")
                if isinstance(path, str) and path:
                    paths.add(path)
        for split_rows in payload.get("row_provenance", {}).values():
            for row in split_rows:
                path = row.get("path")
                if isinstance(path, str) and path:
                    paths.add(path)
    return paths, hashes


def ordered_code_blobs(commit: str) -> Iterable[tuple[str, bytes]]:
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True,
    ).splitlines()
    candidates = sorted(
        (path for path in paths if path.endswith(".py")),
        key=lambda path: hashlib.sha256(f"{CODE_SEED}\0{path}".encode()).digest(),
    )
    for path in candidates:
        yield path, subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def allocate_code_rows(
    blobs: Iterable[tuple[str, bytes]], encode: Callable[[str], list[int]],
    prior: tuple[set[str], set[int], set[tuple[int, ...]], set[tuple[int, ...]]],
    excluded_paths: set[str], *, n_rows: int = N_PER_ROLE,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if type(n_rows) is not int or n_rows <= 0:
        raise ValueError("code row count must be positive")
    prior_rows = {row[: contract.ROW_WIDTH] for row in prior[2] if len(row) >= contract.ROW_WIDTH}
    used_prefixes = set(prior[3])
    selected: list[list[int]] = []
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path, blob in blobs:
        if len(selected) == n_rows:
            break
        if not isinstance(path, str) or not path or path in excluded_paths or path in seen_paths or (
            not isinstance(blob, bytes)
        ):
            continue
        seen_paths.add(path)
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
    stem_length = contract.ROW_WIDTH - SYNTHETIC_SEQUENCE_LENGTH - SYNTHETIC_CUT - 1
    for role in ALL_ROLES:
        iterator = iter(_token_order(role))
        positive_rows, control_rows, role_banks = [], [], []
        for index in range(pairs_per_role):
            stem = tuple(int(value) for value in role_rows[role][index, :stem_length])
            sequence: list[int] = []
            while len(sequence) < SYNTHETIC_SEQUENCE_LENGTH:
                token = next(iterator)
                if token not in globally_used and token not in stem:
                    globally_used.add(token)
                    sequence.append(token)
            positive, control = contract.build_synthetic_copy_pair(
                stem, sequence, SYNTHETIC_CUT,
            )
            positive_rows.append(positive)
            control_rows.append(control)
            role_banks.append(sequence)
        outputs[role] = {
            "positive": torch.stack(positive_rows).contiguous(),
            "control": torch.stack(control_rows).contiguous(),
        }
        banks[role] = role_banks
    return outputs, banks


def fit_token_counts(rows: torch.Tensor) -> torch.Tensor:
    if tuple(rows.shape) != (N_PER_ROLE, contract.ROW_WIDTH):
        raise ValueError("fit natural rows have the wrong shape")
    return torch.bincount(
        rows[:, : contract.MODEL_WIDTH].reshape(-1), minlength=50257,
    ).long()


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
            for role in ALL_ROLES for arm in ("positive", "control")
        ),
        "synthetic_token_banks_are_cross_role_disjoint": bank_disjoint,
    }
    if not all(gates.values()):
        raise RuntimeError(f"terminal-copy row gates failed: {gates}")
    return {
        "roles": {role: len(rows[role]) for role in ALL_ROLES},
        "synthetic_pairs": {role: len(synthetic[role]["positive"]) for role in ALL_ROLES},
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


def freeze() -> dict[str, Any]:
    claim = natural.acquire_claim(LOCK)
    try:
        if CACHE.exists() or RECEIPT.exists():
            raise RuntimeError("terminal-copy row namespace is spent")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        sources = source_closure(commit)
        audit = validate_audit(commit)
        canonical, parquet = natural.BASE.validate_ordered_source()
        registry_files = natural.discover_registry_files()
        prior, registry_hashes, prior_tensor_hashes = natural.load_registry_exclusions(registry_files)

        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        combined, combined_records = natural.harvest_fresh_documents(
            natural.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
            start_document_index=START_DOCUMENT_INDEX,
            n_source_documents=len(NATURAL_ROLES) * N_PER_ROLE,
            token_length=contract.ROW_WIDTH,
        )
        role_rows, role_records = split_natural_rows(combined, combined_records)
        excluded_code_paths, code_manifest_hashes = prior_code_paths()
        code_rows, code_records = allocate_code_rows(
            ordered_code_blobs(commit), encoding.encode_ordinary, prior, excluded_code_paths,
        )
        role_rows[OOD_ROLE] = code_rows
        role_records[OOD_ROLE] = code_records
        synthetic, banks = build_synthetic_roles(role_rows)
        counts = fit_token_counts(role_rows["fit_natural"])
        summary = summarize_roles(role_rows, role_records, synthetic, banks)

        natural.verify_snapshot(
            commit=commit,
            sources=natural.source_closure(commit),
            registry_files=registry_files,
            registry_hashes=registry_hashes,
            tensor_hashes=prior_tensor_hashes,
            prior=prior,
            parquet=parquet,
        )
        natural.require_claim(claim, LOCK)
        CACHE.mkdir(parents=False, exist_ok=False)
        entries: dict[str, Any] = {}
        for role in ALL_ROLES:
            payload = {
                "rows": role_rows[role],
                "records": role_records[role],
                "synthetic": synthetic[role],
                "synthetic_token_banks": banks[role],
            }
            path = CACHE / f"{role}.pt"
            _save_create_only(payload, path)
            entries[role] = {
                "path": str(path.resolve()),
                "file_sha256": file_sha256(path),
                "rows_tensor_sha256": tensor_sha256(role_rows[role]),
                "positive_tensor_sha256": tensor_sha256(synthetic[role]["positive"]),
                "control_tensor_sha256": tensor_sha256(synthetic[role]["control"]),
            }
        counts_path = CACHE / "fit_token_counts.pt"
        _save_create_only(counts, counts_path)
        counts_entry = {
            "path": str(counts_path.resolve()),
            "file_sha256": file_sha256(counts_path),
            "tensor_sha256": tensor_sha256(counts),
        }

        receipt = {
            "schema_version": 1,
            "receipt_kind": "terminal_copy_induction_v1_rows",
            "status": "frozen_before_any_terminal_copy_model_forward",
            "authorized_for_scored_experiments": True,
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
            },
            "summary": summary,
            "entries": entries,
            "fit_token_counts": counts_entry,
            "prior_registry_files": registry_hashes,
            "prior_row_tensors": prior_tensor_hashes,
            "prior_code_manifests": code_manifest_hashes,
            "ordered_manifest_gate": canonical["ordered_manifest_local_parquet_identity_gate"],
            "outcome_access": {
                "checkpoint_loaded": False,
                "model_imported": False,
                "model_forward_calls": 0,
                "scientific_outcomes_read": False,
            },
        }
        natural.require_claim(claim, LOCK)
        natural.write_json_create_only(receipt, RECEIPT)
        return receipt
    except BaseException:
        if CACHE.exists() and not RECEIPT.exists():
            # Preserve partial cache for forensic audit; v1 is spent and cannot rerun.
            pass
        raise
    finally:
        natural.release_claim(claim, LOCK)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))

