#!/usr/bin/env python3
"""Receipt-last checkpoint identity check for the terminal-copy attention adapter."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bilin18_observed_model_facade as facade
from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention


PREREG = HERE / "TERMINAL_COPY_ATTENTION_CHECKPOINT_CHECK_V1_PREREGISTRATION.md"
ADDENDUM = HERE / "TERMINAL_COPY_ATTENTION_ADAPTER_V1_ADDENDUM.md"
ADAPTER = HERE / "terminal_copy_attention_adapter.py"
ADAPTER_TEST = HERE / "test_terminal_copy_attention_adapter.py"
RUNNER = Path(__file__).resolve()
AUTHORITY = HERE / "terminal_copy_attention_checkpoint_check_v1_authority.json"
RESULT = HERE / "terminal_copy_attention_checkpoint_check_v1_result.json"
RECEIPT = HERE / "terminal_copy_attention_checkpoint_check_v1_receipt.json"
FAILURE = HERE / "terminal_copy_attention_checkpoint_check_v1_failure.json"

SOURCE_PATHS = (
    PREREG, ADDENDUM, ADAPTER, ADAPTER_TEST, RUNNER,
    HERE / "bilin18_observed_model_facade.py",
    ROOT / "jacclust" / "tt_model.py", ROOT / "jacclust" / "__init__.py",
)
SEED = 2026082917
LAYERS = (5, 7, 8, 13, 14)
BATCH = 2
SEQUENCE = 32
RELATIVE_TOLERANCE = 2e-3
MAX_ABS_TOLERANCE = 2e-2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_create_only(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as sink:
            descriptor = -1
            sink.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def source_closure() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    hashes = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if digest != file_sha256(path):
            raise RuntimeError(f"checkpoint-check source differs from commit: {relative}")
        hashes[relative] = digest
    return {"commit": commit, "source_hashes": hashes}


def execute() -> dict:
    outputs = (AUTHORITY, RESULT, RECEIPT, FAILURE)
    spent = [str(path) for path in outputs if path.exists()]
    if spent:
        raise RuntimeError(f"checkpoint-check namespace is spent: {spent}")
    started = time.monotonic()
    source = source_closure()
    checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
    authority = {
        "schema_version": 1,
        "status": "frozen_before_checkpoint_weight_load",
        **source,
        "checkpoint": asdict(checkpoint),
        "seed": SEED,
        "layers": list(LAYERS),
        "shape": [BATCH, SEQUENCE, 1152],
        "dtype": "torch.bfloat16",
        "relative_tolerance": RELATIVE_TOLERANCE,
        "max_abs_tolerance": MAX_ABS_TOLERANCE,
        "scientific_claim_authorized": False,
    }
    write_json_create_only(authority, AUTHORITY)
    try:
        model, loaded = facade.load_bilin18(
            device="cuda", dtype=torch.bfloat16, verify_weights_sha256=False,
        )
        if loaded != checkpoint:
            raise RuntimeError("checkpoint load differs from authority")
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        with torch.inference_mode():
            root_state = torch.randn(
                BATCH, SEQUENCE, 1152, generator=generator,
                device="cuda", dtype=torch.bfloat16,
            )
            _, root_bus = model.transformer.h[0].attn(root_state)
            rows = []
            for layer in LAYERS:
                state = torch.randn(
                    BATCH, SEQUENCE, 1152, generator=generator,
                    device="cuda", dtype=torch.bfloat16,
                )
                native_write, native_bus = model.transformer.h[layer].attn(state, root_bus)
                adapter = OwnedPerHeadTensorAttention.from_native(model.transformer.h[layer].attn)
                with adapter.begin(state, root_bus) as transaction:
                    head_sum = transaction.all_heads()
                    full_write = transaction.native_full_write()
                    adapter_bus = transaction.first_value_bus()
                closure = transaction.closure
                native_full_bit_equal = bool(torch.equal(native_write, full_write))
                bus_bit_equal = bool(torch.equal(native_bus, adapter_bus))
                finite = bool(torch.isfinite(head_sum).all() and torch.isfinite(full_write).all())
                passed = (
                    native_full_bit_equal and bus_bit_equal and finite
                    and closure.all_head_recomposition_relative_error <= RELATIVE_TOLERANCE
                    and closure.all_head_recomposition_max_abs_error <= MAX_ABS_TOLERANCE
                )
                rows.append({
                    "layer": layer,
                    "native_full_bit_equal": native_full_bit_equal,
                    "bus_bit_equal": bus_bit_equal,
                    "finite": finite,
                    "all_head_recomposition_relative_error": (
                        closure.all_head_recomposition_relative_error
                    ),
                    "all_head_recomposition_max_abs_error": (
                        closure.all_head_recomposition_max_abs_error
                    ),
                    "price": adapter.price(),
                    "passed": passed,
                })
        result = {
            "schema_version": 1,
            "status": "complete_checkpoint_engineering_check",
            "authority_file_sha256": file_sha256(AUTHORITY),
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "rows": rows,
            "all_layers_passed": all(row["passed"] for row in rows),
            "scientific_claim_authorized": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        if not result["all_layers_passed"]:
            raise RuntimeError("checkpoint adapter identity gate failed: " + json.dumps(rows))
        write_json_create_only(result, RESULT)
        receipt = {
            "schema_version": 1,
            "status": "receipt_last",
            "authority_file_sha256": file_sha256(AUTHORITY),
            "result_file_sha256": file_sha256(RESULT),
            "all_layers_passed": True,
            "scientific_claim_authorized": False,
        }
        write_json_create_only(receipt, RECEIPT)
        return result
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "failed_after_authority",
            "authority_file_sha256": file_sha256(AUTHORITY),
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json_create_only(failure, FAILURE)
        raise


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
