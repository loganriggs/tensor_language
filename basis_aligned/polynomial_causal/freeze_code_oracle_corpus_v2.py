#!/usr/bin/env python3
"""Freeze a file-disjoint, file-contained Python corpus for the code OOD oracle."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import tiktoken
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_COMMIT = "8b9d23e37132051e12016376d2b497e4d53680c9"
TOKENS_PER_ROW = 257
ROWS_PER_FILE_CAP = 4
SPLIT_ROWS = {"basis": 96, "discovery": 192, "heldout": 192}
SPLIT_BUCKETS = {"basis": {0}, "discovery": {1, 2}, "heldout": {3, 4}}
OUT = HERE / "code_oracle_corpus_v2.pt"
MANIFEST = HERE / "code_oracle_corpus_v2_manifest.json"
SPEC = HERE / "SHIP_CONTENT_OOD_ORACLE_SPEC.md"


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE
    ).stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenizer_fingerprint(encoder) -> str:
    digest = hashlib.sha256()
    for token, rank in sorted(encoder._mergeable_ranks.items(), key=lambda row: row[1]):
        digest.update(len(token).to_bytes(4, "little"))
        digest.update(token)
        digest.update(int(rank).to_bytes(4, "little"))
    for token, rank in sorted(encoder._special_tokens.items()):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(int(rank).to_bytes(4, "little"))
    return digest.hexdigest()


def split_for_path(path: str) -> tuple[str, int]:
    bucket = int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big") % 5
    for split, buckets in SPLIT_BUCKETS.items():
        if bucket in buckets:
            return split, bucket
    raise AssertionError(bucket)


def build() -> tuple[torch.Tensor, dict]:
    paths = [
        line.decode().strip()
        for line in git("ls-tree", "-r", "--name-only", SOURCE_COMMIT).splitlines()
        if line.decode().strip().endswith(".py")
    ]
    encoder = tiktoken.get_encoding("gpt2")
    rows: dict[str, list[list[int]]] = {split: [] for split in SPLIT_ROWS}
    provenance: dict[str, list[dict]] = {split: [] for split in SPLIT_ROWS}
    files: dict[str, list[dict]] = {split: [] for split in SPLIT_ROWS}

    for path in paths:
        split, bucket = split_for_path(path)
        if len(rows[split]) >= SPLIT_ROWS[split]:
            continue
        blob = git("show", f"{SOURCE_COMMIT}:{path}")
        tokens = [encoder.eot_token] + encoder.encode_ordinary(
            blob.decode("utf-8", errors="replace")
        )
        available = min(len(tokens) // TOKENS_PER_ROW, ROWS_PER_FILE_CAP)
        take = min(available, SPLIT_ROWS[split] - len(rows[split]))
        if take == 0:
            continue
        blob_hash = sha256(blob)
        files[split].append({
            "path": path,
            "blob_sha256": blob_hash,
            "assignment_bucket": bucket,
            "encoded_tokens_with_eot": len(tokens),
            "available_full_rows": len(tokens) // TOKENS_PER_ROW,
            "rows_used": take,
        })
        for chunk in range(take):
            start = chunk * TOKENS_PER_ROW
            end = start + TOKENS_PER_ROW
            row = tokens[start:end]
            if len(row) != TOKENS_PER_ROW:
                raise AssertionError((path, start, end, len(row)))
            rows[split].append(row)
            provenance[split].append({
                "path": path,
                "blob_sha256": blob_hash,
                "chunk_index": chunk,
                "token_start": start,
                "token_end": end,
            })

    missing = {
        split: wanted - len(rows[split])
        for split, wanted in SPLIT_ROWS.items()
        if len(rows[split]) != wanted
    }
    if missing:
        raise RuntimeError(f"insufficient file-contained rows: {missing}")
    split_paths = {
        split: {row["path"] for row in provenance[split]} for split in SPLIT_ROWS
    }
    for left, right in (("basis", "discovery"), ("basis", "heldout"),
                        ("discovery", "heldout")):
        overlap = split_paths[left] & split_paths[right]
        if overlap:
            raise RuntimeError(f"file overlap {left}/{right}: {sorted(overlap)}")

    ordered = rows["basis"] + rows["discovery"] + rows["heldout"]
    tensor = torch.tensor(ordered, dtype=torch.long)
    if len({tuple(row) for row in ordered}) != len(ordered):
        raise RuntimeError("duplicate token rows in v2 code corpus")
    boundaries = {
        "basis": [0, 96], "discovery": [96, 288], "heldout": [288, 480]
    }
    manifest = {
        "schema_version": 2,
        "source_commit": SOURCE_COMMIT,
        "source": "tracked .py git blobs, hash-assigned to file-disjoint splits",
        "scope": "frozen repository-Python register; not general code",
        "decode_errors": "utf-8 errors=replace",
        "tokenizer": {
            "library": "tiktoken",
            "version": getattr(tiktoken, "__version__", "unknown"),
            "encoding": "gpt2",
            "fingerprint_sha256": tokenizer_fingerprint(encoder),
            "eot_token": encoder.eot_token,
            "valid_token_ids": [0, 50257],
        },
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "tokens_per_row": TOKENS_PER_ROW,
        "rows_per_file_cap": ROWS_PER_FILE_CAP,
        "path_assignment": "uint64_be(sha256(path)[:8]) mod 5",
        "split_buckets": {key: sorted(value) for key, value in SPLIT_BUCKETS.items()},
        "splits": boundaries,
        "split_cluster_counts": {
            split: len(split_paths[split]) for split in SPLIT_ROWS
        },
        "files": files,
        "row_provenance": provenance,
        "no_row_crosses_file_boundary": True,
        "file_disjoint_splits": True,
        "tensor_raw_sha256": sha256(tensor.numpy().tobytes(order="C")),
        "construction_script_sha256": sha256(Path(__file__).read_bytes()),
        "spec_sha256_at_freeze": sha256(SPEC.read_bytes()),
    }
    return tensor, manifest


def main() -> None:
    rows, manifest = build()
    torch.save({"rows": rows, "manifest": manifest}, OUT)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "source_commit": manifest["source_commit"],
        "shape": manifest["shape"],
        "split_cluster_counts": manifest["split_cluster_counts"],
        "tensor_raw_sha256": manifest["tensor_raw_sha256"],
    }, indent=2))
    print(f"wrote {OUT} and {MANIFEST}")


if __name__ == "__main__":
    main()
