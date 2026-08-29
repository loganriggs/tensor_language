#!/usr/bin/env python3
"""Materialize frozen MLP2 CMR document roles into masked 257-token rows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Callable, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "MLP2_CMR_V1_PREREGISTRATION.md"
PROTOCOL = HERE / "MLP2_CMR_V1_ROW_PROTOCOL.json"
ADDENDUM = HERE / "MLP2_CMR_V1_TOKEN_MATERIALIZATION_ADDENDUM.md"
ROLE_MANIFEST = HERE / "mlp2_cmr_v1_document_roles_manifest.json"
ROLE_RECEIPT = HERE / "mlp2_cmr_v1_document_roles_receipt.json"
OUTPUT = HERE / "mlp2_cmr_v1_token_rows.pt"
MANIFEST = HERE / "mlp2_cmr_v1_token_rows_manifest.json"
RECEIPT = HERE / "mlp2_cmr_v1_token_rows_receipt.json"
LOCK = HERE / ".mlp2_cmr_v1_token_rows.lock"
PARQUET = Path("/workspace/fineweb_pinned/data/CC-MAIN-2013-20/000_00000.parquet")
ROLE_MANIFEST_SHA256 = "70946691916f3d9de9409087cf8311029c0d9ea9ec26a118d5cacdbf5397d96e"
ROLE_RECEIPT_SHA256 = "0cda2fb909c9e62f8db46f1fc717a2a86fad712346a98d81c307d3b79fde63e1"
PARQUET_SHA256 = "c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930"
PARQUET_SIZE = 2_147_531_358
ENCODING_SHA256 = "0be287937901b1baae837369293dd6f63da1bece9609006e6485b57a3de37335"
ROLES = ("FIT_MEAN", "FIT_SELECTOR", "VALIDATION", "REPLICATION")
DOCUMENTS = 192
WIDTH = 257
SCORE_START = 64
EOT = 50_256
MIN_SUPPORT_DOCUMENTS = 128
MIN_ELIGIBLE_POSITIONS = 16_000
SOURCE_CLOSURE = (
    PREREG, PROTOCOL, ADDENDUM, ROLE_MANIFEST, ROLE_RECEIPT,
    Path(__file__).resolve(), HERE / "test_materialize_mlp2_cmr_v1_token_rows.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def encoding_fingerprint(encoding: Any) -> str:
    digest = hashlib.sha256()
    for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda item: item[1]):
        digest.update(len(token).to_bytes(8, "little"))
        digest.update(token)
        digest.update(int(rank).to_bytes(8, "little"))
    for token, rank in sorted(encoding._special_tokens.items()):
        raw = token.encode()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        digest.update(int(rank).to_bytes(8, "little"))
    return digest.hexdigest()


def tokenize_documents(
    indices: tuple[int, ...], texts: Mapping[int, str], encode: Callable[[str], list[int]],
) -> dict[str, torch.Tensor]:
    if len(indices) != DOCUMENTS or len(set(indices)) != DOCUMENTS:
        raise ValueError("role document indices are malformed")
    rows = torch.full((DOCUMENTS, WIDTH), EOT, dtype=torch.long)
    masks = torch.zeros(DOCUMENTS, WIDTH - 1, dtype=torch.bool)
    original_counts = torch.zeros(DOCUMENTS, dtype=torch.long)
    clipped_counts = torch.zeros(DOCUMENTS, dtype=torch.long)
    for ordinal, index in enumerate(indices):
        text = texts.get(index)
        if not isinstance(text, str):
            raise ValueError(f"missing text for frozen document {index}")
        tokens = encode(text)
        if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token <= EOT
               for token in tokens):
            raise ValueError("tokenizer returned an invalid token ID")
        original_counts[ordinal] = len(tokens)
        clipped = min(len(tokens), WIDTH)
        clipped_counts[ordinal] = clipped
        if clipped:
            rows[ordinal, :clipped] = torch.tensor(tokens[:clipped], dtype=torch.long)
        stop = min(max(clipped - 1, 0), WIDTH - 1)
        if stop > SCORE_START:
            masks[ordinal, SCORE_START:stop] = True
    return {
        "document_indices": torch.tensor(indices, dtype=torch.long),
        "rows": rows,
        "eligible_mask": masks,
        "original_token_counts": original_counts,
        "clipped_token_counts": clipped_counts,
    }


def validate_role(value: Mapping[str, torch.Tensor]) -> dict[str, int]:
    expected = {
        "document_indices": ((DOCUMENTS,), torch.long),
        "rows": ((DOCUMENTS, WIDTH), torch.long),
        "eligible_mask": ((DOCUMENTS, WIDTH - 1), torch.bool),
        "original_token_counts": ((DOCUMENTS,), torch.long),
        "clipped_token_counts": ((DOCUMENTS,), torch.long),
    }
    if set(value) != set(expected):
        raise RuntimeError("token-role tensor keys changed")
    for name, (shape, dtype) in expected.items():
        tensor = value[name]
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise RuntimeError(f"malformed role tensor: {name}")
    rows, masks = value["rows"], value["eligible_mask"]
    clipped = value["clipped_token_counts"]
    if int(rows.min()) < 0 or int(rows.max()) > EOT:
        raise RuntimeError("token row exceeds GPT-2 vocabulary")
    positions = torch.arange(WIDTH - 1).unsqueeze(0)
    expected_mask = (positions >= SCORE_START) & (positions < (clipped - 1).clamp_min(0)[:, None])
    if not torch.equal(masks, expected_mask):
        raise RuntimeError("eligible mask does not match original document extent")
    for ordinal, count in enumerate(clipped.tolist()):
        if count < WIDTH and not bool((rows[ordinal, count:] == EOT).all()):
            raise RuntimeError("short document is not padded exactly with EOT")
    support_documents = int(masks.any(1).sum())
    eligible_positions = int(masks.sum())
    if support_documents < MIN_SUPPORT_DOCUMENTS or eligible_positions < MIN_ELIGIBLE_POSITIONS:
        raise RuntimeError(
            f"role support gate failed: {support_documents} documents, "
            f"{eligible_positions} positions"
        )
    return {
        "documents": DOCUMENTS,
        "support_documents": support_documents,
        "eligible_positions": eligible_positions,
        "shorter_than_257": int((value["original_token_counts"] < WIDTH).sum()),
        "shorter_than_65": int((value["original_token_counts"] <= SCORE_START).sum()),
    }


def read_selected_texts(path: Path, wanted: set[int]) -> dict[int, str]:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    output: dict[int, str] = {}
    offset = 0
    for group in range(source.metadata.num_row_groups):
        count = source.metadata.row_group(group).num_rows
        selected = sorted(index for index in wanted if offset <= index < offset + count)
        if selected:
            values = source.read_row_group(group, columns=["text"]).column(0).to_pylist()
            for index in selected:
                text = values[index - offset]
                if not isinstance(text, str):
                    raise RuntimeError(f"FineWeb text is not a string at {index}")
                output[index] = text
        offset += count
    if set(output) != wanted:
        missing = sorted(wanted - set(output))
        raise RuntimeError(f"pinned Parquet lacks frozen documents: {missing[:3]}")
    return output


def write_create_only(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    hashes = {}
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"source differs from committed bytes: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def load_role_indices() -> dict[str, tuple[int, ...]]:
    if file_sha256(ROLE_MANIFEST) != ROLE_MANIFEST_SHA256 or \
            file_sha256(ROLE_RECEIPT) != ROLE_RECEIPT_SHA256:
        raise RuntimeError("frozen document-role parents changed")
    manifest = json.loads(ROLE_MANIFEST.read_text())
    receipt = json.loads(ROLE_RECEIPT.read_text())
    if receipt.get("manifest_sha256") != ROLE_MANIFEST_SHA256 or \
            receipt.get("authorized_for_document_identity") is not True:
        raise RuntimeError("document-role receipt does not authorize its manifest")
    roles = {
        role: tuple(manifest["roles"][role]["ordered_document_indices"])
        for role in ROLES
    }
    flattened = [index for role in ROLES for index in roles[role]]
    if any(len(roles[role]) != DOCUMENTS for role in ROLES) or \
            len(flattened) != len(set(flattened)):
        raise RuntimeError("frozen document roles are malformed or overlapping")
    return roles


def freeze() -> dict[str, Any]:
    if any(path.exists() for path in (OUTPUT, MANIFEST, RECEIPT)):
        raise RuntimeError("refusing to overwrite token-row namespace")
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        commit, source_hashes = committed_source()
        roles = load_role_indices()
        if PARQUET.stat().st_size != PARQUET_SIZE or file_sha256(PARQUET) != PARQUET_SHA256:
            raise RuntimeError("pinned FineWeb bytes changed")
        wanted = {index for indices in roles.values() for index in indices}
        texts = read_selected_texts(PARQUET, wanted)
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
        if encoding_fingerprint(encoding) != ENCODING_SHA256:
            raise RuntimeError("GPT-2 tokenizer merge table changed")
        bundle = {
            role: tokenize_documents(indices, texts, encoding.encode_ordinary)
            for role, indices in roles.items()
        }
        summaries = {role: validate_role(value) for role, value in bundle.items()}
        # Re-hash source after all text/token work.
        if PARQUET.stat().st_size != PARQUET_SIZE or file_sha256(PARQUET) != PARQUET_SHA256:
            raise RuntimeError("pinned FineWeb changed during tokenization")
        temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.{secrets.token_hex(8)}")
        try:
            torch.save(bundle, temporary)
            os.link(temporary, OUTPUT)
        finally:
            temporary.unlink(missing_ok=True)
        tensor_hashes = {
            role: {name: tensor_sha256(value) for name, value in tensors.items()}
            for role, tensors in bundle.items()
        }
        manifest = {
            "schema_version": 1,
            "experiment_id": "bilin18_mlp2_cmr_v1",
            "status": "token_rows_published_pending_receipt",
            "output_path": str(OUTPUT.relative_to(ROOT)),
            "output_file_sha256": file_sha256(OUTPUT),
            "tensor_hashes": tensor_hashes,
            "role_summaries": summaries,
            "role_manifest_sha256": ROLE_MANIFEST_SHA256,
            "role_receipt_sha256": ROLE_RECEIPT_SHA256,
            "parquet_sha256": PARQUET_SHA256,
            "encoding_sha256": ENCODING_SHA256,
            "source_commit": commit,
            "source_hashes": source_hashes,
            "authorized_for_model_forward": False,
            "authorized_for_scientific_outcomes": False,
        }
        write_create_only(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
        # Reload and replay every tensor before the last-written receipt.
        replay = torch.load(OUTPUT, map_location="cpu", weights_only=True)
        replay_hashes = {
            role: {name: tensor_sha256(value) for name, value in tensors.items()}
            for role, tensors in replay.items()
        }
        if replay_hashes != tensor_hashes or file_sha256(MANIFEST) == "":
            raise RuntimeError("published token bundle failed semantic replay")
        receipt = {
            "schema_version": 1,
            "experiment_id": "bilin18_mlp2_cmr_v1",
            "status": "token_materialization_complete_receipt_last",
            "authority": "token_input_bytes_only",
            "output_path": str(OUTPUT.relative_to(ROOT)),
            "output_file_sha256": file_sha256(OUTPUT),
            "manifest_path": str(MANIFEST.relative_to(ROOT)),
            "manifest_sha256": file_sha256(MANIFEST),
            "role_manifest_sha256": ROLE_MANIFEST_SHA256,
            "role_receipt_sha256": ROLE_RECEIPT_SHA256,
            "role_summaries": summaries,
            "source_commit": commit,
            "source_hashes": source_hashes,
            "authorized_for_token_inputs": True,
            "authorized_for_model_forward": False,
            "authorized_for_fit_or_evaluation": False,
            "authorized_for_scientific_outcomes": False,
            "next_required_authority": (
                "source-closed MLP2 product/response collector binding checkpoint, "
                "role use, selectors, calls, outputs, and three-terminal lifecycle"
            ),
        }
        # Exact parent/source/output checks immediately before receipt link.
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != commit:
            raise RuntimeError("source commit changed during token materialization")
        for path in SOURCE_CLOSURE:
            if file_sha256(path) != source_hashes[str(path.relative_to(ROOT))]:
                raise RuntimeError("source changed before token receipt")
        if file_sha256(ROLE_MANIFEST) != ROLE_MANIFEST_SHA256 or \
                file_sha256(ROLE_RECEIPT) != ROLE_RECEIPT_SHA256 or \
                file_sha256(OUTPUT) != receipt["output_file_sha256"] or \
                file_sha256(MANIFEST) != receipt["manifest_sha256"]:
            raise RuntimeError("token publication parents changed before receipt")
        write_create_only(RECEIPT, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
        return receipt
    finally:
        os.close(lock_fd)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
