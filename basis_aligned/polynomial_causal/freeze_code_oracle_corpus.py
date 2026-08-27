#!/usr/bin/env python3
"""Freeze the preregistered git-object Python corpus for the code OOD oracle."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import tiktoken
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROWS = 480
TOKENS_PER_ROW = 257
OUT = HERE / "code_oracle_corpus.pt"
MANIFEST = HERE / "code_oracle_corpus_manifest.json"


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE
    ).stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build() -> tuple[torch.Tensor, dict]:
    commit = git("rev-parse", "HEAD").decode().strip()
    paths = [
        line.decode().strip()
        for line in git("ls-tree", "-r", "--name-only", commit).splitlines()
        if line.decode().strip().endswith(".py")
    ]
    encoder = tiktoken.get_encoding("gpt2")
    needed = ROWS * TOKENS_PER_ROW
    tokens: list[int] = []
    files = []
    for path in paths:
        blob = git("show", f"{commit}:{path}")
        encoded = encoder.encode(blob.decode("utf-8", errors="replace"))
        if not encoded:
            continue
        take = min(len(encoded), needed - len(tokens))
        tokens.extend(encoded[:take])
        files.append({
            "path": path,
            "blob_sha256": sha256(blob),
            "available_tokens": len(encoded),
            "used_tokens": take,
        })
        if len(tokens) == needed:
            break
    if len(tokens) != needed:
        raise RuntimeError(f"needed {needed} tokens, found {len(tokens)}")
    rows = torch.tensor(tokens, dtype=torch.long).view(ROWS, TOKENS_PER_ROW)
    tensor_bytes = rows.numpy().tobytes(order="C")
    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "source": "tracked .py git objects in lexicographic path order",
        "tokenizer": "tiktoken:gpt2",
        "shape": list(rows.shape),
        "dtype": str(rows.dtype),
        "splits": {
            "basis": [0, 96],
            "discovery": [96, 288],
            "heldout": [288, 480],
        },
        "tensor_raw_sha256": sha256(tensor_bytes),
        "construction_script_sha256": sha256(Path(__file__).read_bytes()),
        "files": files,
    }
    return rows, manifest


def main() -> None:
    rows, manifest = build()
    torch.save({"rows": rows, "manifest": manifest}, OUT)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT} and {MANIFEST}")
    print(json.dumps({key: manifest[key] for key in
                      ("source_commit", "shape", "tensor_raw_sha256")}, indent=2))


if __name__ == "__main__":
    main()
