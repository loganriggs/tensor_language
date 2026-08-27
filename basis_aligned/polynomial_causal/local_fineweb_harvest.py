#!/usr/bin/env python3
"""Shadow-harvest frozen FineWeb rows from a pinned local parquet shard.

This is an infrastructure recovery path for the unauthenticated streaming failures.
It intentionally does *not* license scored experiments by itself.  It reproduces
``census_lib.fineweb_rows`` tokenization, 513-token chunking, skip, and census-prefix
dedup semantics while additionally preserving document/chunk provenance.  Outputs
are marked ``shadow_unlicensed_pending_remote_bit_identity`` until the registered
remote identity gate compares them bit-for-bit.

The first pinned shard is downloaded separately with the resumable Xet client:

```
hf download HuggingFaceFW/fineweb \
  data/CC-MAIN-2013-20/000_00000.parquet --repo-type dataset \
  --revision 9bb295ddab0e05d785b879661af7260fed5140fc
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Iterator

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
PINNED_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
PINNED_RELATIVE_PATH = "data/CC-MAIN-2013-20/000_00000.parquet"
PINNED_SIZE = 2_147_531_358
PINNED_SHA256 = "c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930"
T_LEN = 513
SPECS = ((8, 40), (96, 80), (480, 80), (96, 1200), (192, 7000), (192, 11000))
SHADOW = BQ / ".rowcache_shadow"
RECEIPT = SHADOW / "fineweb_local_shadow_v1_receipt.json"


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def spec_key(spec: tuple[int, int]) -> str:
    return f"n{spec[0]}_skip{spec[1]}"


def harvest_texts(
    texts: Iterable[tuple[str, str]],
    specs: tuple[tuple[int, int], ...],
    encode: Callable[[str], list[int]],
    seen_prefixes: set[tuple[int, ...]],
    *,
    token_length: int = T_LEN,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], list[dict[str, Any]]]]:
    """Pure multi-spec implementation of the frozen census row semantics.

    ``texts`` yields ``(document_id, text)`` in dataset order.  The document id is
    stored only as provenance and never affects row selection.
    """
    if token_length <= 0:
        raise ValueError("token_length must be positive")
    if not specs or len(set(specs)) != len(specs):
        raise ValueError("specs must be a nonempty unique tuple")
    if any(n <= 0 or skip < 0 for n, skip in specs):
        raise ValueError("every spec needs n>0 and skip>=0")
    active: dict[tuple[int, int], list[list[int]]] = {spec: [] for spec in specs}
    provenance: dict[tuple[int, int], list[dict[str, Any]]] = {spec: [] for spec in specs}
    for document_index, (document_id, text) in enumerate(texts):
        if not isinstance(document_id, str) or not isinstance(text, str):
            raise ValueError("texts must yield string document ids and text")
        if not any(document_index >= skip and len(active[(n, skip)]) < n
                   for n, skip in specs):
            continue
        tokens = encode(text)
        for spec in specs:
            n, skip = spec
            if document_index < skip or len(active[spec]) >= n:
                continue
            for start in range(0, len(tokens) - token_length, token_length):
                row = tokens[start:start + token_length]
                if tuple(row[:32]) in seen_prefixes:
                    continue
                active[spec].append(row)
                provenance[spec].append({
                    "document_id": document_id,
                    "dataset_document_index": document_index,
                    "chunk_id": start // token_length,
                    "token_start": start,
                })
                if len(active[spec]) >= n:
                    break
        if all(len(active[spec]) >= spec[0] for spec in specs):
            break
    output = {}
    for spec in specs:
        tensor = torch.tensor(active[spec], dtype=torch.long)
        if tuple(tensor.shape) != (spec[0], token_length):
            raise RuntimeError(
                f"local FineWeb parquet ended before {spec}: harvested {tuple(tensor.shape)}"
            )
        output[spec] = tensor
    return output, provenance


def parquet_texts(paths: list[Path]) -> Iterator[tuple[str, str]]:
    import pyarrow.parquet as parquet

    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        relative = path.name
        row_index = 0
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=256):
            for text in batch.column(0).to_pylist():
                document_id = f"{PINNED_REVISION}:{relative}:{row_index}"
                yield document_id, text
                row_index += 1


def save_atomic(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_shadow(parquet_paths: list[Path]) -> dict[str, Any]:
    if len(parquet_paths) != 1:
        raise RuntimeError("v1 shadow receipt supports exactly one pinned parquet shard")
    if SHADOW.exists():
        raise RuntimeError(f"refusing to overwrite existing shadow directory: {SHADOW}")
    source = parquet_paths[0]
    if source.stat().st_size != PINNED_SIZE:
        raise RuntimeError(f"pinned parquet size mismatch: {source.stat().st_size} != {PINNED_SIZE}")
    source_hash = file_sha256(source)
    if source_hash != PINNED_SHA256:
        raise RuntimeError(f"pinned parquet SHA256 mismatch: {source_hash}")

    import tiktoken

    reference = torch.load(BQ / "bilin18_eval_tokens_large.pt", map_location="cpu",
                           weights_only=True)
    seen = {tuple(reference[row, :32].tolist()) for row in range(reference.shape[0])}
    encoding = tiktoken.get_encoding("gpt2")
    tensors, provenance = harvest_texts(
        parquet_texts(parquet_paths), SPECS, encoding.encode_ordinary, seen
    )
    staging = SHADOW.with_name(f"{SHADOW.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"refusing to overwrite staging directory: {staging}")
    staging.mkdir(parents=True)
    entries = {}
    provenance_sets = {}
    try:
        for spec in SPECS:
            name = f"fineweb_{spec_key(spec)}.pt"
            staged_path = staging / name
            final_path = SHADOW / name
            save_atomic(tensors[spec], staged_path)
            entries[spec_key(spec)] = {
                "n": spec[0], "skip": spec[1], "shape": list(tensors[spec].shape),
                "dtype": str(tensors[spec].dtype), "tensor_raw_sha256": tensor_sha256(tensors[spec]),
                "path": str(final_path.resolve()),
            }
            provenance_sets[spec_key(spec)] = provenance[spec]
        receipt = {
            "schema_version": 1,
            "status": "shadow_unlicensed_pending_remote_bit_identity",
            "authorized_for_scored_experiments": False,
            "license_rule": "No scored experiment may consume these tensors until n8_skip40 is bit-identical to census_lib.fineweb_rows(8, skip=40) and the registered gate is upgraded.",
            "dataset": "HuggingFaceFW/fineweb split=train",
            "revision": PINNED_REVISION,
            "source_files": [{
                "relative_path": PINNED_RELATIVE_PATH,
                "local_path": str(source.resolve()),
                "size": PINNED_SIZE,
                "sha256": source_hash,
            }],
            "loader_semantics": "pinned parquet row order; gpt2 encode_ordinary; census-prefix dedup; range(0,len(tokens)-513,513)",
            "entries": entries,
            "document_provenance": {"schema_version": 1, "sets": provenance_sets},
        }
        (staging / RECEIPT.name).write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(staging, SHADOW)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet",
        type=Path,
        nargs="?",
        default=Path("/workspace/fineweb_pinned") / PINNED_RELATIVE_PATH,
    )
    args = parser.parse_args()
    receipt = build_shadow([args.parquet.resolve()])
    print(json.dumps(receipt, indent=2))
    print(f"wrote unlicensed shadow receipt {RECEIPT}")


if __name__ == "__main__":
    main()
