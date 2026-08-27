#!/usr/bin/env python3
"""Freeze the fit-only MLP0 native-Down compiler before its model forwards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(HERE))
from compile_mlp0_native_down_fit_v1 import (  # noqa: E402
    AUTHORITY,
    FAILURE,
    FIT_RECEIPT,
    LOCK,
    PROGRAMS,
    RIDGE_FRACTION,
    RUNG_RANKS,
    file_sha256,
)


STAGE0_ROW_RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"
STAGE0_FIT_RECEIPT = BQ / "mlp0_quotient_stage0_v2_fit_receipt.json"
MODEL_REPO = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
SOURCE_CLOSURE = (
    HERE / "compile_mlp0_native_down_fit_v1.py",
    HERE / "mlp0_native_down_program.py",
    HERE / "mlp0_quotient_worst_cell.py",
    HERE / "causal_response_quotient.py",
    HERE / "prepare_mlp0_quotient_stage0_v1_rows.py",
    HERE / "MLP0_NATIVE_DOWN_HIERARCHY_SPEC.md",
    BQ / "bilin18_joint_removal.py",
    ROOT / "basis_aligned" / "qk_mdl" / "tier2_model.py",
    ROOT / "jacclust" / "tt_model.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_authority() -> dict:
    if any(path.exists() for path in (AUTHORITY, FIT_RECEIPT, PROGRAMS, FAILURE, LOCK)):
        raise RuntimeError("native-Down fit namespace is already frozen or spent")
    source_commit = git("rev-parse", "HEAD")
    if source_commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    relative = [str(path.relative_to(ROOT)) for path in SOURCE_CLOSURE]
    if subprocess.run(["git", "diff", "--quiet", "--", *relative], cwd=ROOT).returncode:
        raise RuntimeError("fit source closure is dirty")
    stage_fit = json.loads(STAGE0_FIT_RECEIPT.read_text())
    if stage_fit.get("status") != "frozen_before_any_v2_evaluation_model_forward":
        raise RuntimeError("Stage-0 fit receipt is not authoritative")
    config = Path(hf_hub_download(MODEL_REPO, "config.json", local_files_only=True))
    checkpoint = Path(hf_hub_download(
        MODEL_REPO, "pytorch_model.bin", local_files_only=True
    ))
    cuda = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "receipt_kind": "mlp0_native_down_hierarchy_v1_fit_authority",
        "status": "frozen_before_any_native_down_fit_model_forward",
        "scope": "fit-role compiler only; evaluation receipt/path is outside source closure",
        "source_commit": source_commit,
        "source_hashes": {str(path.resolve()): file_sha256(path) for path in SOURCE_CLOSURE},
        "stage0_fit_rows": {
            "receipt_path": str(STAGE0_ROW_RECEIPT.resolve()),
            "receipt_sha256": file_sha256(STAGE0_ROW_RECEIPT),
        },
        "stage0_fit_constants": {
            "receipt_path": str(STAGE0_FIT_RECEIPT.resolve()),
            "receipt_sha256": file_sha256(STAGE0_FIT_RECEIPT),
        },
        "model_repo": MODEL_REPO,
        "model_files": {
            str(config.resolve()): file_sha256(config),
            str(checkpoint.resolve()): file_sha256(checkpoint),
        },
        "solver_contract": {
            "ridge_fraction": RIDGE_FRACTION,
            "continuous_rungs": list(RUNG_RANKS),
            "covariance": "float32 CUDA ordered batch addmm",
            "mean_and_state_sums": "float64 CPU ordered accumulation",
            "eigensolver": "torch.linalg.eigh descending; balanced SVD largest-loading sign",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": cuda.name,
            "compute_capability": f"{cuda.major}.{cuda.minor}",
        },
        "fit_receipt_path": str(FIT_RECEIPT),
        "program_directory": str(PROGRAMS),
        "failure_path": str(FAILURE),
        "lock_path": str(LOCK),
        "output_policy": "fail closed; atomic receipt and directory; no evaluation authority exists",
    }


def main() -> None:
    payload = build_authority()
    write_json_atomic(payload, AUTHORITY)
    print(json.dumps(payload, indent=2))
    print(f"wrote {AUTHORITY}")


if __name__ == "__main__":
    main()
