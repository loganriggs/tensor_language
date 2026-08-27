#!/usr/bin/env python3
"""Mint a new authoritative FineWeb receipt from the immutable local shadow.

The shadow receipt is never modified or upgraded.  Promotion is allowed only when
the pinned Hugging Face ``datasets`` builder resolves the exact local parquet as
the first file of the pinned default-config train manifest.  Since every requested
document index lies inside that first shard, later files cannot affect the rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
import sys
sys.path.insert(0, str(BQ))
import rowcache  # noqa: E402
import local_fineweb_harvest as shadow  # noqa: E402
import prepare_fineweb_oracle_rows as canonical  # noqa: E402


EXPECTED_ORDERED_FILE_COUNT = 27_468
EXPECTED_ORDERED_MANIFEST_SHA256 = "ba5e92b0d157f47cc6f8656eb1c37e46b7aac6957be8be68c1596736b98e6f90"
EXPECTED_FIRST_URI = (
    "hf://datasets/HuggingFaceFW/fineweb@"
    f"{shadow.PINNED_REVISION}/{shadow.PINNED_RELATIVE_PATH}"
)
EXPECTED_LAST_URI = (
    "hf://datasets/HuggingFaceFW/fineweb@"
    f"{shadow.PINNED_REVISION}/data/CC-MAIN-2025-26/004_00049.parquet"
)
DATASETS_VERSION = "5.0.1"
CANONICAL_RECEIPT_NAME = "fineweb_oracle_v2_receipt.json"


def ordered_manifest_sha256(files: list[str]) -> str:
    payload = json.dumps(files, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_ordered_manifest() -> dict[str, Any]:
    import datasets
    from datasets import load_dataset_builder

    builder = load_dataset_builder(
        "HuggingFaceFW/fineweb", revision=shadow.PINNED_REVISION
    )
    files = [str(path) for path in builder.config.data_files["train"]]
    proof = {
        "datasets_version": datasets.__version__,
        "builder_class": type(builder).__name__,
        "config": builder.config.name,
        "ordered_file_count": len(files),
        "ordered_manifest_sha256": ordered_manifest_sha256(files),
        "first_uri": files[0] if files else None,
        "last_uri": files[-1] if files else None,
    }
    expected = {
        "datasets_version": DATASETS_VERSION,
        "builder_class": "ParquetFineweb",
        "config": "default",
        "ordered_file_count": EXPECTED_ORDERED_FILE_COUNT,
        "ordered_manifest_sha256": EXPECTED_ORDERED_MANIFEST_SHA256,
        "first_uri": EXPECTED_FIRST_URI,
        "last_uri": EXPECTED_LAST_URI,
    }
    if proof != expected:
        raise RuntimeError(f"pinned ordered FineWeb manifest changed: {proof}")
    return proof


def validate_shadow_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if (receipt.get("receipt_kind") != shadow.RECEIPT_KIND
            or receipt.get("authority") != "none"
            or receipt.get("status") != shadow.UNLICENSED_STATUS
            or receipt.get("authorized_for_scored_experiments") is not False):
        raise RuntimeError("input is not the frozen authority-none shadow receipt")
    sources = receipt.get("source_files", [])
    if len(sources) != 1:
        raise RuntimeError("shadow receipt must name exactly one source shard")
    source = Path(sources[0]["local_path"])
    if (sources[0].get("relative_path") != shadow.PINNED_RELATIVE_PATH
            or source.stat().st_size != shadow.PINNED_SIZE
            or shadow.file_sha256(source) != shadow.PINNED_SHA256):
        raise RuntimeError("shadow source shard failed pinned identity")
    current_harvester_hash = shadow.file_sha256(Path(shadow.__file__))
    if receipt["implementation_provenance"]["harvester_source_sha256"] != current_harvester_hash:
        raise RuntimeError("shadow harvester changed after receipt creation")
    maximum_document = max(
        row["dataset_document_index"]
        for rows in receipt["document_provenance"]["sets"].values()
        for row in rows
    )
    if maximum_document >= sources[0]["parquet_num_rows"]:
        raise RuntimeError("shadow rows extend beyond the first pinned parquet")
    return receipt


def promote(
    shadow_receipt_path: Path = shadow.RECEIPT,
    canonical_dir: Path | None = None,
    manifest_resolver: Callable[[], dict[str, Any]] = resolve_ordered_manifest,
) -> dict[str, Any]:
    canonical_dir = Path(rowcache.CACHE) if canonical_dir is None else canonical_dir
    if canonical_dir.exists():
        raise RuntimeError(f"refusing to overwrite canonical rowcache: {canonical_dir}")
    receipt = validate_shadow_receipt(shadow_receipt_path)
    manifest = manifest_resolver()
    required_manifest = {
        "datasets_version": DATASETS_VERSION,
        "builder_class": "ParquetFineweb",
        "config": "default",
        "ordered_file_count": EXPECTED_ORDERED_FILE_COUNT,
        "ordered_manifest_sha256": EXPECTED_ORDERED_MANIFEST_SHA256,
        "first_uri": EXPECTED_FIRST_URI,
        "last_uri": EXPECTED_LAST_URI,
    }
    if manifest != required_manifest:
        raise RuntimeError("manifest resolver did not return the pinned proof")

    staging = canonical_dir.with_name(f"{canonical_dir.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"refusing to overwrite rowcache staging: {staging}")
    staging.mkdir(parents=True)
    try:
        entries = {}
        provenance_sets = {}
        for spec in canonical.SPECS:
            key = shadow.spec_key(spec)
            shadow_entry = receipt["entries"][key]
            tensor = torch.load(shadow_entry["path"], map_location="cpu", weights_only=True)
            if (tuple(tensor.shape) != (spec[0], rowcache.T_LEN)
                    or tensor.dtype != torch.long
                    or canonical.tensor_sha256(tensor) != shadow_entry["tensor_raw_sha256"]):
                raise RuntimeError(f"shadow tensor identity failed for {spec}")
            filename = f"fineweb_n{spec[0]}_skip{spec[1]}.pt"
            staged_path = staging / filename
            rowcache._save_atomic(tensor, str(staged_path))
            entries[canonical.spec_key(*spec)] = {
                "n": spec[0],
                "skip": spec[1],
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_raw_sha256": canonical.tensor_sha256(tensor),
                "cache_path": str((canonical_dir / filename).resolve()),
            }
            provenance_sets[key] = receipt["document_provenance"]["sets"][key]

        source = receipt["source_files"][0]
        promoted = {
            "schema_version": 2,
            "receipt_kind": canonical.ORDERED_MANIFEST_RECEIPT_KIND,
            "authority": "pinned_local_ordered_manifest",
            "authorized_for_scored_experiments": True,
            "dataset": "HuggingFaceFW/fineweb split=train streaming=True",
            "loader_semantics": "census_lib.fineweb_rows; 513-token chunks; census-prefix dedup",
            "ordered_manifest_local_parquet_identity_gate": {
                "passed": True,
                "revision": shadow.PINNED_REVISION,
                "config": "default",
                "ordered_file_count": manifest["ordered_file_count"],
                "ordered_manifest_sha256": manifest["ordered_manifest_sha256"],
                "first_uri": manifest["first_uri"],
                "last_uri": manifest["last_uri"],
                "first_relative_path": shadow.PINNED_RELATIVE_PATH,
                "source_local_path": source["local_path"],
                "source_size": source["size"],
                "source_sha256": source["sha256"],
                "source_parquet_rows": source["parquet_num_rows"],
                "maximum_consumed_document_index": max(
                    row["dataset_document_index"]
                    for rows in provenance_sets.values() for row in rows
                ),
                "datasets_version": manifest["datasets_version"],
                "builder_class": manifest["builder_class"],
            },
            "shadow_receipt_provenance": {
                "path": str(shadow_receipt_path.resolve()),
                "sha256": shadow.file_sha256(shadow_receipt_path),
                "receipt_kind": shadow.RECEIPT_KIND,
                "authority": "none",
                "rule": "The shadow remains unlicensed and unchanged; this is a distinct receipt.",
            },
            "implementation_provenance": {
                "promoter_source_sha256": shadow.file_sha256(Path(__file__)),
                "harvester_source_sha256": receipt["implementation_provenance"]
                ["harvester_source_sha256"],
                "census_lib_source_sha256": receipt["implementation_provenance"]
                ["census_lib_source_sha256"],
                "gpt2_encoding_sha256": receipt["implementation_provenance"]
                ["gpt2_encoding_sha256"],
            },
            "dedup_reference": receipt["dedup_reference"],
            "rowcache_source_sha256": hashlib.sha256(
                (BQ / "rowcache.py").read_bytes()
            ).hexdigest(),
            "entries": entries,
            "document_provenance": {"schema_version": 1, "sets": provenance_sets},
        }
        receipt_path = staging / CANONICAL_RECEIPT_NAME
        receipt_path.write_text(json.dumps(promoted, indent=2) + "\n")
        os.replace(staging, canonical_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    canonical.validate_receipt(canonical_dir / CANONICAL_RECEIPT_NAME)
    return promoted


def main() -> None:
    receipt = promote()
    print(json.dumps({
        "receipt_kind": receipt["receipt_kind"],
        "authority": receipt["authority"],
        "authorized_for_scored_experiments": receipt["authorized_for_scored_experiments"],
        "gate": receipt["ordered_manifest_local_parquet_identity_gate"],
        "entries": {key: value["tensor_raw_sha256"]
                    for key, value in receipt["entries"].items()},
    }, indent=2))
    print(f"wrote authoritative receipt {Path(rowcache.CACHE) / CANONICAL_RECEIPT_NAME}")


if __name__ == "__main__":
    main()
