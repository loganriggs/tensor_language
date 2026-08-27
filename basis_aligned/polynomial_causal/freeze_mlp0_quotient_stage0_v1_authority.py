#!/usr/bin/env python3
"""Freeze the final collector closure before any MLP0 v1 evaluation forward."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(HERE))
from prepare_mlp0_quotient_stage0_v1_rows import file_sha256  # noqa: E402


COLLECTOR = HERE / "mlp0_quotient_worst_cell.py"
ROW_RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"
FIT_RECEIPT = BQ / "mlp0_quotient_stage0_v1_fit_receipt.json"
AUTHORITY = BQ / "mlp0_quotient_stage0_v1_collector_authority.json"
RESULT = BQ / "mlp0_quotient_stage0_v1_results.json"
FAILURE = BQ / "mlp0_quotient_stage0_v1_failure.json"
LOCK = Path("/workspace/runs/.bilin18_mlp0_quotient_stage0_v1.lock")
MODEL_REPO = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
SOURCE_CLOSURE = (
    COLLECTOR,
    HERE / "causal_response_quotient.py",
    HERE / "prepare_mlp0_quotient_stage0_v1_rows.py",
    HERE / "MLP0_CAUSAL_QUOTIENT_SPEC.md",
    HERE / "MLP0_QUOTIENT_STAGE0_V1_AMENDMENT.md",
    BQ / "bilin18_joint_removal.py",
    ROOT / "basis_aligned" / "qk_mdl" / "tier2_model.py",
    ROOT / "jacclust" / "tt_model.py",
)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_authority() -> dict[str, Any]:
    if AUTHORITY.exists() or RESULT.exists() or FAILURE.exists() or LOCK.exists():
        raise RuntimeError("v1 collector namespace is already spent or locked")
    fit = json.loads(FIT_RECEIPT.read_text())
    if fit.get("status") != "frozen_before_any_v1_evaluation_model_forward":
        raise RuntimeError("fit constants were not prospectively frozen")
    source_commit = git("rev-parse", "HEAD")
    if source_commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    relative_sources = [str(path.relative_to(ROOT)) for path in SOURCE_CLOSURE]
    if subprocess.run(
        ["git", "diff", "--quiet", "--", *relative_sources], cwd=ROOT
    ).returncode != 0:
        raise RuntimeError("collector source closure is dirty")
    config = Path(hf_hub_download(MODEL_REPO, "config.json", local_files_only=True))
    checkpoint = Path(hf_hub_download(
        MODEL_REPO, "pytorch_model.bin", local_files_only=True
    ))
    return {
        "schema_version": 1,
        "receipt_kind": "mlp0_quotient_stage0_v1_collector_authority",
        "status": "frozen_before_any_v1_model_forward",
        "scope": "single prospective evaluation on frozen skip-21000 rows",
        "source_commit": source_commit,
        "source_hashes": {str(path.resolve()): file_sha256(path) for path in SOURCE_CLOSURE},
        "row_receipt_path": str(ROW_RECEIPT),
        "row_receipt_sha256": file_sha256(ROW_RECEIPT),
        "fit_receipt_path": str(FIT_RECEIPT),
        "fit_receipt_sha256": file_sha256(FIT_RECEIPT),
        "model_repo": MODEL_REPO,
        "model_files": {
            str(config.resolve()): file_sha256(config),
            str(checkpoint.resolve()): file_sha256(checkpoint),
        },
        "response_contract": {
            "fit_constants": fit["constants"],
            "margins": {"kl": 0.01, "ce": 0.0075,
                        "attn1_nrmse": 0.05, "mlp1_nrmse": 0.05},
            "cells": "full positions x fit frequency x previous punctuation/boundary x raw pre-MLP0 residual norm",
            "bootstrap": {"documents": 192, "replicates": 10000,
                          "seed": 20260827, "minimum_documents_per_cell": 30},
        },
        "output_path": str(RESULT),
        "failure_path": str(FAILURE),
        "lock_path": str(LOCK),
        "overwrite_policy": "fail_closed",
        "output_policy": "atomic; sufficient statistics retained",
    }


def main() -> None:
    payload = build_authority()
    write_json_atomic(payload, AUTHORITY)
    print(json.dumps(payload, indent=2))
    print(f"wrote {AUTHORITY}")


if __name__ == "__main__":
    main()
