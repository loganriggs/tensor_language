#!/usr/bin/env python3
"""Freeze fresh, provenance-complete FineWeb rows for affine compiler v1.

This is an additive, experiment-scoped row authority.  It reuses the already
certified ordered-manifest proof for the pinned first FineWeb parquet, but none
of the old oracle row tensors.  The new roles begin after every previously
scored oracle role and must be document/content disjoint from them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "early_mlp_affine_compiler_v1_preregistration.json"
PREREG_SHA256 = "f4da5a8085f9aad3b0cf22f377b04b6dfc39d7950089ba8b9a2c50af587cbc5f"
PRIOR_RECEIPT = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
PRIOR_RECEIPT_SHA256 = "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16"
HARVESTER = HERE / "local_fineweb_harvest.py"
HARVESTER_SHA256 = "87d9abeaf1182811650c35bcae25b0373687d2e87aede895bc9f2bc440b90b04"
DEDUP = BQ / "bilin18_eval_tokens_large.pt"
DEDUP_SHA256 = "bb2b00699e511245bb68069be1fe5559777170fb78a6dc9218830454f38e3cd7"
CENSUS = BQ / "census_lib.py"
CENSUS_SHA256 = "f51c19e83f46dc363a2c5dad1887b55ab5dd9b3684294e940583a6814881cf1f"
CACHE = BQ / ".rowcache_compiler_v1"
RECEIPT = BQ / "early_mlp_affine_compiler_v1_rows_receipt.json"
T_LEN = 513
MODEL_LEN = 257
ROLE_SPECS = {
    "compiler_fit": (480, 15000),
    "compiler_validation": (192, 19000),
    "compiler_final": (192, 23000),
}
SPECS = tuple(ROLE_SPECS.values())


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def logical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def spec_key(spec: tuple[int, int]) -> str:
    return f"n{spec[0]}_skip{spec[1]}"


def require_pinned_sources() -> dict[str, Any]:
    pins = {
        PREREG: PREREG_SHA256,
        PRIOR_RECEIPT: PRIOR_RECEIPT_SHA256,
        HARVESTER: HARVESTER_SHA256,
        DEDUP: DEDUP_SHA256,
        CENSUS: CENSUS_SHA256,
    }
    for path, expected in pins.items():
        if not path.is_file():
            raise RuntimeError(f"pinned input is absent: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"pinned input changed: {path}; expected={expected} observed={observed}"
            )
    relative = PREREG.relative_to(ROOT)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)], cwd=ROOT,
        capture_output=True, text=True,
    )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=ROOT,
    )
    if tracked.returncode != 0 or dirty.returncode != 0:
        raise RuntimeError("compiler preregistration must be committed and clean")

    receipt = json.loads(PRIOR_RECEIPT.read_text())
    gate = receipt.get("ordered_manifest_local_parquet_identity_gate", {})
    required = {
        "passed": True,
        "revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
        "config": "default",
        "ordered_file_count": 27468,
        "ordered_manifest_sha256": (
            "ba5e92b0d157f47cc6f8656eb1c37e46b7aac6957be8be68c1596736b98e6f90"
        ),
        "first_relative_path": "data/CC-MAIN-2013-20/000_00000.parquet",
        "source_size": 2_147_531_358,
        "source_sha256": (
            "c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930"
        ),
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"prior ordered-manifest authority changed at {key}")
    source = Path(gate.get("source_local_path", ""))
    if not source.is_file() or source.stat().st_size != required["source_size"]:
        raise RuntimeError("pinned local FineWeb source is absent or has wrong size")
    if file_sha256(source) != required["source_sha256"]:
        raise RuntimeError("pinned local FineWeb source hash changed")
    return receipt


def prior_document_ids(receipt: Mapping[str, Any]) -> set[str]:
    provenance = receipt.get("document_provenance", {}).get("sets", {})
    if not isinstance(provenance, dict) or not provenance:
        raise RuntimeError("prior receipt lacks document provenance")
    output: set[str] = set()
    for records in provenance.values():
        if not isinstance(records, list):
            raise RuntimeError("prior provenance set is not a list")
        for record in records:
            document = record.get("document_id") if isinstance(record, dict) else None
            if not isinstance(document, str) or not document:
                raise RuntimeError("prior provenance record lacks document_id")
            output.add(document)
    return output


def validate_disjointness(
    tensors: Mapping[tuple[int, int], torch.Tensor],
    provenance: Mapping[tuple[int, int], list[dict[str, Any]]],
    old_documents: set[str],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    documents: dict[str, set[str]] = {}
    full_rows: dict[str, set[str]] = {}
    prefixes: dict[str, set[tuple[int, ...]]] = {}
    for role, spec in ROLE_SPECS.items():
        tensor = tensors[spec]
        records = provenance[spec]
        if tuple(tensor.shape) != (spec[0], T_LEN) or tensor.dtype != torch.long:
            raise RuntimeError(f"invalid {role} tensor: {tensor.shape} {tensor.dtype}")
        if len(records) != spec[0]:
            raise RuntimeError(f"invalid {role} provenance count")
        role_documents = {record["document_id"] for record in records}
        overlap = role_documents.intersection(old_documents)
        if overlap:
            raise RuntimeError(f"{role} overlaps {len(overlap)} prior oracle documents")
        documents[role] = role_documents
        full_rows[role] = {tensor_sha256(row) for row in tensor}
        prefixes[role] = {tuple(row[:32].tolist()) for row in tensor}
        summaries[role] = {
            "request": {"n": spec[0], "skip": spec[1]},
            "shape_full": list(tensor.shape),
            "shape_model_prefix": [spec[0], MODEL_LEN],
            "dtype": str(tensor.dtype),
            "tensor_full_raw_sha256": tensor_sha256(tensor),
            "tensor_prefix257_raw_sha256": tensor_sha256(tensor[:, :MODEL_LEN]),
            "unique_document_count": len(role_documents),
            "document_ids_sha256": logical_json_sha256(
                [record["document_id"] for record in records]
            ),
            "provenance_records_sha256": logical_json_sha256(records),
            "prior_oracle_document_overlap": 0,
        }
    roles = tuple(ROLE_SPECS)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1:]:
            if documents[left].intersection(documents[right]):
                raise RuntimeError(f"document leakage between {left} and {right}")
            if full_rows[left].intersection(full_rows[right]):
                raise RuntimeError(f"full-row content overlap between {left} and {right}")
            if prefixes[left].intersection(prefixes[right]):
                raise RuntimeError(f"prefix-32 content overlap between {left} and {right}")
    return summaries


def save_tensor_atomic(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build() -> dict[str, Any]:
    if CACHE.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite compiler row cache or receipt")
    prior = require_pinned_sources()
    gate = prior["ordered_manifest_local_parquet_identity_gate"]
    source = Path(gate["source_local_path"])

    import tiktoken
    import local_fineweb_harvest as harvest

    reference = torch.load(DEDUP, map_location="cpu", weights_only=True)
    seen = {tuple(reference[row, :32].tolist()) for row in range(reference.shape[0])}
    encoding = tiktoken.get_encoding("gpt2")
    tensors, provenance = harvest.harvest_texts(
        harvest.parquet_texts([source]), SPECS, encoding.encode_ordinary, seen
    )
    summaries = validate_disjointness(tensors, provenance, prior_document_ids(prior))

    staging = CACHE.with_name(f"{CACHE.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"compiler row staging already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        entries = {}
        for role, spec in ROLE_SPECS.items():
            filename = f"fineweb_{spec_key(spec)}.pt"
            save_tensor_atomic(tensors[spec], staging / filename)
            entries[role] = {
                **summaries[role],
                "cache_path": str((CACHE / filename).resolve()),
            }
        os.replace(staging, CACHE)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "receipt_kind": "early_mlp_affine_compiler_v1_rows",
        "status": "frozen_before_predictor_fit",
        "authority": "isolated_compiler_experiment",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": True,
        "training_license_sites": [0, 1],
        "license_scope": "Only early_mlp_affine_compiler_v1 affine maps; no base-model, ship, basis, or other training.",
        "preregistration_path": str(PREREG.resolve()),
        "preregistration_sha256": PREREG_SHA256,
        "source_commit": source_commit,
        "prior_authority": {
            "receipt_path": str(PRIOR_RECEIPT.resolve()),
            "receipt_sha256": PRIOR_RECEIPT_SHA256,
            "ordered_manifest_gate": gate,
            "rule": "The prior receipt supplies source/order identity only; none of its row tensors is reused."
        },
        "loader_semantics": "pinned first parquet in certified dataset order; gpt2 encode_ordinary; census-prefix dedup; range(0,len(tokens)-513,513)",
        "source_identity_rechecked": {
            "path": str(source.resolve()),
            "size": source.stat().st_size,
            "sha256": file_sha256(source),
        },
        "implementation_hashes": {
            "row_preparer": file_sha256(Path(__file__)),
            "local_harvester": HARVESTER_SHA256,
            "census_lib": CENSUS_SHA256,
            "dedup_reference_file": DEDUP_SHA256,
            "gpt2_encoding": harvest.encoding_fingerprint(encoding),
        },
        "entries": entries,
        "document_provenance": {
            "schema_version": 1,
            "sets": {role: provenance[spec] for role, spec in ROLE_SPECS.items()},
        },
        "disjointness_gates": {
            "pairwise_new_document_disjoint": True,
            "pairwise_new_full_row_disjoint": True,
            "pairwise_new_prefix32_disjoint": True,
            "all_new_documents_disjoint_from_prior_oracle_roles": True,
        },
    }
    write_json_atomic(receipt, RECEIPT)
    return receipt


def load_and_validate() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not RECEIPT.is_file():
        raise RuntimeError(f"compiler row receipt is absent: {RECEIPT}")
    receipt = json.loads(RECEIPT.read_text())
    required = {
        "status": "frozen_before_predictor_fit",
        "authority": "isolated_compiler_experiment",
        "authorized_for_scored_experiments": True,
        "authorized_for_training": True,
        "training_license_sites": [0, 1],
        "preregistration_sha256": PREREG_SHA256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"compiler row receipt changed at {key}")
    rows = {}
    for role, spec in ROLE_SPECS.items():
        entry = receipt.get("entries", {}).get(role)
        if not isinstance(entry, dict) or entry.get("request") != {
            "n": spec[0], "skip": spec[1]
        }:
            raise RuntimeError(f"compiler row receipt lacks {role}")
        path = Path(entry["cache_path"])
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != (spec[0], T_LEN) or tensor.dtype != torch.long:
            raise RuntimeError(f"invalid cached compiler rows for {role}")
        if tensor_sha256(tensor) != entry["tensor_full_raw_sha256"]:
            raise RuntimeError(f"cached compiler row hash changed for {role}")
        rows[role] = tensor
    summaries = validate_disjointness(
        {spec: rows[role] for role, spec in ROLE_SPECS.items()},
        {spec: receipt["document_provenance"]["sets"][role]
         for role, spec in ROLE_SPECS.items()},
        prior_document_ids(json.loads(PRIOR_RECEIPT.read_text())),
    )
    for role in ROLE_SPECS:
        if summaries[role]["tensor_full_raw_sha256"] != (
            receipt["entries"][role]["tensor_full_raw_sha256"]
        ):
            raise RuntimeError(f"revalidated compiler rows changed for {role}")
    return receipt, rows


def main() -> None:
    receipt = build()
    load_and_validate()
    print(json.dumps({
        "status": receipt["status"],
        "preregistration_sha256": receipt["preregistration_sha256"],
        "entries": {
            role: {
                "request": row["request"],
                "tensor_prefix257_raw_sha256": row["tensor_prefix257_raw_sha256"],
                "unique_document_count": row["unique_document_count"],
            }
            for role, row in receipt["entries"].items()
        },
        "disjointness_gates": receipt["disjointness_gates"],
    }, indent=2))
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
