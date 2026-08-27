#!/usr/bin/env python3
"""Real-stream verify and freeze every FineWeb row set needed by oracle v2.

Run only after all active streaming lanes are clear. The resulting receipt makes
the authoritative FineWeb/code pipeline network-free and content-addressed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(BQ))
import rowcache  # noqa: E402


SPECS = ((96, 80), (480, 80), (96, 1200), (192, 7000), (192, 11000))
VERIFY_SPEC = (8, 40)
RECEIPT = Path(rowcache.CACHE) / "fineweb_oracle_v2_receipt.json"
PINNED_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
PINNED_FIRST_FILE = "data/CC-MAIN-2013-20/000_00000.parquet"
PINNED_LOCAL_FILE_SIZE = 2_147_531_358
PINNED_LOCAL_FILE_SHA256 = "c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930"
ORDERED_MANIFEST_RECEIPT_KIND = "fineweb_oracle_v2_ordered_manifest"


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def spec_key(n: int, skip: int) -> str:
    return f"n{n}_skip{skip}"


def build_receipt(rows: dict[tuple[int, int], torch.Tensor]) -> dict[str, Any]:
    entries = {}
    for spec in SPECS:
        tensor = rows[spec]
        if tuple(tensor.shape) != (spec[0], rowcache.T_LEN) or tensor.dtype != torch.long:
            raise RuntimeError(f"invalid frozen row set {spec}: {tensor.shape} {tensor.dtype}")
        entries[spec_key(*spec)] = {
            "n": spec[0], "skip": spec[1],
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "tensor_raw_sha256": tensor_sha256(tensor),
            "cache_path": str(Path(rowcache._path(*spec)).resolve()),
        }
    return {
        "schema_version": 1,
        "dataset": "HuggingFaceFW/fineweb split=train streaming=True",
        "loader_semantics": "census_lib.fineweb_rows; 513-token chunks; census-prefix dedup",
        "real_stream_bit_identity_gate": {
            "passed": True, "n": VERIFY_SPEC[0], "skip": VERIFY_SPEC[1],
        },
        "rowcache_source_sha256": hashlib.sha256(
            (BQ / "rowcache.py").read_bytes()
        ).hexdigest(),
        "entries": entries,
    }


def validate_receipt(path: Path = RECEIPT) -> tuple[dict[str, Any], dict[tuple[int, int], torch.Tensor]]:
    if not path.exists():
        raise RuntimeError(f"FineWeb oracle row receipt is absent: {path}")
    receipt = json.loads(path.read_text())
    gate = receipt.get("real_stream_bit_identity_gate", {})
    real_stream_gate = gate == {
        "passed": True, "n": VERIFY_SPEC[0], "skip": VERIFY_SPEC[1]
    }
    ordered = receipt.get("ordered_manifest_local_parquet_identity_gate", {})
    ordered_manifest_gate = (
        receipt.get("receipt_kind") == ORDERED_MANIFEST_RECEIPT_KIND
        and receipt.get("authority") == "pinned_local_ordered_manifest"
        and receipt.get("authorized_for_scored_experiments") is True
        and ordered.get("passed") is True
        and ordered.get("revision") == PINNED_REVISION
        and ordered.get("config") == "default"
        and ordered.get("first_relative_path") == PINNED_FIRST_FILE
        and ordered.get("source_size") == PINNED_LOCAL_FILE_SIZE
        and ordered.get("source_sha256") == PINNED_LOCAL_FILE_SHA256
        and isinstance(ordered.get("ordered_file_count"), int)
        and ordered["ordered_file_count"] > 1
        and isinstance(ordered.get("ordered_manifest_sha256"), str)
        and len(ordered["ordered_manifest_sha256"]) == 64
    )
    if not real_stream_gate and not ordered_manifest_gate:
        raise RuntimeError("FineWeb row receipt lacks an accepted identity gate")
    if ordered_manifest_gate:
        source = Path(ordered.get("source_local_path", ""))
        if (not source.is_file() or source.stat().st_size != PINNED_LOCAL_FILE_SIZE
                or file_sha256(source) != PINNED_LOCAL_FILE_SHA256):
            raise RuntimeError("pinned local FineWeb source changed after receipt creation")
        provenance = receipt.get("document_provenance", {})
        if provenance.get("schema_version") != 1:
            raise RuntimeError("ordered-manifest receipt lacks document provenance")
    if receipt.get("rowcache_source_sha256") != hashlib.sha256(
        (BQ / "rowcache.py").read_bytes()
    ).hexdigest():
        raise RuntimeError("rowcache implementation changed after receipt creation")
    rows = {}
    for spec in SPECS:
        entry = receipt.get("entries", {}).get(spec_key(*spec))
        if not isinstance(entry, dict) or entry.get("n") != spec[0] or entry.get("skip") != spec[1]:
            raise RuntimeError(f"receipt lacks required row set {spec}")
        tensor = rowcache._load_checked(entry["cache_path"], spec[0])
        if tensor_sha256(tensor) != entry.get("tensor_raw_sha256"):
            raise RuntimeError(f"frozen FineWeb row hash mismatch for {spec}")
        rows[spec] = tensor
    return receipt, rows


def main() -> None:
    if RECEIPT.exists():
        raise RuntimeError(f"refusing to overwrite existing receipt: {RECEIPT}")
    if not rowcache.verify(*VERIFY_SPEC):
        raise RuntimeError("real-stream rowcache identity gate failed")
    rows = rowcache.multi(list(SPECS))
    receipt = build_receipt(rows)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    validate_receipt(RECEIPT)
    print(json.dumps(receipt, indent=2))
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
