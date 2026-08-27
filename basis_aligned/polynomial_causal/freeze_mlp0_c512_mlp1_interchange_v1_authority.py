#!/usr/bin/env python3
"""Freeze the complete C512/MLP1 evaluator closure before any evaluation forward."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BQ))

from evaluate_mlp0_c512_mlp1_interchange_v1 import (  # noqa: E402
    AUTHORITY, CODE_REGISTER, FAILURE, FIT_RECEIPT, LOCK, OUT, PROGRAM_KEY,
    ROW_RECEIPT, STAGE0_FIT_RECEIPT, build_unit_identity, closure_sha256,
    expected_call_counts, file_sha256, load_domains, tensor_sha256,
    unit_identity_hashes,
)
from prepare_mlp0_quotient_stage0_v1_rows import load_frozen_role  # noqa: E402
from score_mlp0_c512_mlp1_interchange_v1 import (  # noqa: E402
    ARMS, CONTRASTS, MARGINS, MIN_CODE_FILES_PER_CELL,
    MIN_FINEWEB_DOCUMENTS_PER_CELL, N_BOOTSTRAP, SEED,
)


MODEL_REPO = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
SOURCE_CLOSURE = (
    HERE / "freeze_mlp0_c512_mlp1_interchange_v1_authority.py",
    HERE / "evaluate_mlp0_c512_mlp1_interchange_v1.py",
    HERE / "score_mlp0_c512_mlp1_interchange_v1.py",
    HERE / "mlp0_c512_mlp1_interchange.py",
    HERE / "mlp0_native_down_program.py",
    HERE / "prepare_mlp0_c512_mlp1_interchange_v1_rows.py",
    HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py",
    HERE / "prepare_mlp0_quotient_stage0_v1_rows.py",
    HERE / "local_fineweb_harvest.py",
    HERE / "MLP0_C512_MLP1_INTERCHANGE_SPEC.md",
    BQ / "bilin18_joint_removal.py",
    ROOT / "basis_aligned" / "qk_mdl" / "tier2_model.py",
    ROOT / "jacclust" / "tt_model.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed_head_blob(path: Path) -> None:
    relative = str(path.relative_to(ROOT))
    subprocess.check_call(
        ["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    blob = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != file_sha256(path):
        raise RuntimeError(f"source is not byte-identical to HEAD: {relative}")


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_authority() -> dict:
    if any(path.exists() for path in (AUTHORITY, OUT, FAILURE, LOCK)):
        raise RuntimeError("C512/MLP1 evaluation namespace is already frozen or spent")
    source_commit = git("rev-parse", "HEAD")
    if source_commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    relative = [str(path.relative_to(ROOT)) for path in SOURCE_CLOSURE]
    for path in SOURCE_CLOSURE:
        require_committed_head_blob(path)
    if subprocess.run(["git", "diff", "--quiet", "--", *relative], cwd=ROOT).returncode:
        raise RuntimeError("evaluation source closure is dirty")

    domains, identity, frozen_rows, code_artifact = load_domains()
    row_receipt = json.loads(ROW_RECEIPT.read_text())
    if (row_receipt.get("status") != "frozen_before_any_c512_interchange_model_forward"
            or row_receipt.get("sample_summary", {}).get("n_source_documents") != 384):
        raise RuntimeError("fresh FineWeb receipt is not authoritative")
    rebuilt_identity = build_unit_identity(row_receipt, code_artifact["manifest"])
    if identity != rebuilt_identity:
        raise RuntimeError("unit identity rebuild is not deterministic")

    _, fit_full = load_frozen_role("fit")
    fit_rows = fit_full[:, :257].contiguous()
    if tuple(fit_rows.shape) != (960, 257):
        raise RuntimeError("fit-frozen row shape changed")
    fit_receipt = json.loads(FIT_RECEIPT.read_text())
    if fit_receipt.get("status") != "frozen_before_evaluation_authority":
        raise RuntimeError("C512 fit receipt is not authoritative")
    program_receipt = fit_receipt["programs"][PROGRAM_KEY]
    program_path = Path(program_receipt["path"])
    if (program_path.stat().st_size != program_receipt["bytes"]
            or file_sha256(program_path) != program_receipt["sha256"]):
        raise RuntimeError("C512 program changed before authority")

    config = Path(hf_hub_download(MODEL_REPO, "config.json", local_files_only=True))
    checkpoint = Path(hf_hub_download(MODEL_REPO, "pytorch_model.bin", local_files_only=True))
    source_hashes = {str(path.resolve()): file_sha256(path) for path in SOURCE_CLOSURE}
    bound_hashes = {
        "source_closure_sha256": closure_sha256(source_hashes),
        "row_receipt_sha256": file_sha256(ROW_RECEIPT),
        "row_tensor_sha256": tensor_sha256(frozen_rows),
        "c512_program_sha256": file_sha256(program_path),
        "model_checkpoint_sha256": file_sha256(checkpoint),
        "code_register_sha256": file_sha256(CODE_REGISTER),
    }
    return {
        "schema_version": 1,
        "receipt_kind": "mlp0_c512_mlp1_interchange_v1_eval_authority",
        "status": "frozen_before_any_c512_mlp1_evaluation_forward",
        "scope": "physical MLP0-C512 by MLP1-write 2x2 on fresh FineWeb and frozen code OOD",
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "row_authority": {
            "path": str(ROW_RECEIPT.resolve()),
            "sha256": file_sha256(ROW_RECEIPT),
            "raw_tensor_sha256": tensor_sha256(frozen_rows),
            "domain_rows": {name: len(domain["rows"]) for name, domain in domains.items()},
        },
        "fit_authority": {
            "fit_receipt_sha256": file_sha256(FIT_RECEIPT),
            "stage0_fit_receipt_sha256": file_sha256(STAGE0_FIT_RECEIPT),
            "fit_rows_tensor_sha256": tensor_sha256(fit_rows),
            "fit_rows": len(fit_rows),
        },
        "program_contract": {
            "key": PROGRAM_KEY, "path": str(program_path.resolve()),
            "bytes": program_receipt["bytes"], "sha256": program_receipt["sha256"],
            "rank": 512, "n_centroids": 0,
        },
        "model_repo": MODEL_REPO,
        "model_files": {
            str(config.resolve()): file_sha256(config),
            str(checkpoint.resolve()): file_sha256(checkpoint),
        },
        "assay_contract": {
            "backgrounds": ["live", "mlp2_omit"],
            "arms": list(ARMS), "contrasts": list(CONTRASTS),
            "cells": 16, "margins": MARGINS,
            "shuffle": "deterministic largest-occupancy circular derangement within cell",
            "native_control": "per-position norm-matched exact native MLP1 write",
            "factorial": {"OO": "sO+mO", "CC": "sC+mC", "CO": "sC+mO", "OC": "sO+mC"},
        },
        "integrity_contract": {
            "unit_identity_hashes": unit_identity_hashes(identity),
            "exact_call_counts": expected_call_counts(
                len(fit_rows), {name: len(domain["rows"]) for name, domain in domains.items()}
            ),
            "bound_hashes": bound_hashes,
            "parent_replay_tolerances": {
                "raw_logits_max_abs": 1e-6,
                "capped_logits_max_abs": 1e-6,
                "ce_abs": 1e-7,
            },
            "state_identity_tolerance": 1e-6,
        },
        "response_contract": {
            "bootstrap_replicates": N_BOOTSTRAP, "bootstrap_seed": SEED,
            "fineweb_documents_per_wave": 192,
            "minimum_fineweb_documents_per_cell": MIN_FINEWEB_DOCUMENTS_PER_CELL,
            "code_source_files": 48,
            "minimum_code_files_per_cell": MIN_CODE_FILES_PER_CELL,
            "family": "background x contrast x consumer x cell; one joint source-unit bootstrap",
        },
        "output_path": str(OUT), "failure_path": str(FAILURE), "lock_path": str(LOCK),
        "output_policy": "fail closed; atomic result; raw source-unit ledgers retained",
    }


def main() -> None:
    payload = build_authority()
    write_json_atomic(payload, AUTHORITY)
    print(json.dumps(payload, indent=2))
    print(f"wrote {AUTHORITY}")


if __name__ == "__main__":
    main()
