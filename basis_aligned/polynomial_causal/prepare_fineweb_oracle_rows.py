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


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


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
    if gate != {"passed": True, "n": VERIFY_SPEC[0], "skip": VERIFY_SPEC[1]}:
        raise RuntimeError("FineWeb row receipt lacks the real-stream identity gate")
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
