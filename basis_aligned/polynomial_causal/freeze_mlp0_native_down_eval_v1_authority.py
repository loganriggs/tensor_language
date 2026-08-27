#!/usr/bin/env python3
"""Freeze the complete poison-gated MLP0 evaluation closure before row access."""

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
from evaluate_mlp0_native_down_hierarchy_v1 import (  # noqa: E402
    AUTHORITY,
    FAILURE,
    FIT_RECEIPT,
    LOCK,
    OUT,
    PROGRAM_KEYS,
    ROW_RECEIPT,
    STAGE0_FIT_RECEIPT,
    STAGE0_ROW_RECEIPT,
    file_sha256,
)
from score_mlp0_native_down_hierarchy_v1 import (  # noqa: E402
    MINIMUM_DOCUMENTS_PER_CELL,
    N_BOOTSTRAP,
    SEED,
)


MODEL_REPO = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
SOURCE_CLOSURE = (
    HERE / "freeze_mlp0_native_down_eval_v1_authority.py",
    HERE / "evaluate_mlp0_native_down_hierarchy_v1.py",
    HERE / "score_mlp0_native_down_hierarchy_v1.py",
    HERE / "mlp0_native_down_program.py",
    HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py",
    HERE / "prepare_mlp0_quotient_stage0_v1_rows.py",
    HERE / "MLP0_NATIVE_DOWN_HIERARCHY_SPEC.md",
    BQ / "bilin18_joint_removal.py",
    ROOT / "basis_aligned" / "qk_mdl" / "tier2_model.py",
    ROOT / "jacclust" / "tt_model.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_committed_head_blob(path: Path) -> None:
    relative = str(path.relative_to(ROOT))
    subprocess.check_call(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
        raise RuntimeError("native-Down evaluation namespace is already frozen or spent")
    source_commit = git("rev-parse", "HEAD")
    if source_commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    relative = [str(path.relative_to(ROOT)) for path in SOURCE_CLOSURE]
    for path in SOURCE_CLOSURE:
        require_committed_head_blob(path)
    if subprocess.run(["git", "diff", "--quiet", "--", *relative], cwd=ROOT).returncode:
        raise RuntimeError("evaluation source closure is dirty")
    fit = json.loads(FIT_RECEIPT.read_text())
    if fit.get("status") != "frozen_before_evaluation_authority":
        raise RuntimeError("fit bundle receipt is not frozen")
    if not all(gate["admitted_le_ceiling"] and gate["next_rank_gt_ceiling"]
               for gate in fit["price_gates"].values()):
        raise RuntimeError("fit receipt does not pass every physical price gate")
    nested = fit["authority"]
    stage0_row_hash = file_sha256(STAGE0_ROW_RECEIPT)
    stage0_fit_hash = file_sha256(STAGE0_FIT_RECEIPT)
    if (nested["stage0_fit_rows"]["receipt_sha256"] != stage0_row_hash
            or nested["stage0_fit_constants"]["receipt_sha256"] != stage0_fit_hash):
        raise RuntimeError("Stage-0 evaluation currency changed after native fit")
    row = json.loads(ROW_RECEIPT.read_text())
    if (row.get("status") != "frozen_before_any_native_down_model_forward"
            or row.get("sample_summary", {}).get("n_source_documents") != 384):
        raise RuntimeError("384-document row receipt is not authoritative")
    program_files = {}
    program_contract = {}
    for arm, key in PROGRAM_KEYS.items():
        receipt = fit["programs"][key]
        path = Path(receipt["path"])
        if path.stat().st_size != receipt["bytes"] or file_sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"fit program changed before authority: {arm}")
        program_files[str(path.resolve())] = receipt["sha256"]
        program_contract[arm] = {
            "fit_key": key, "bytes": receipt["bytes"], "sha256": receipt["sha256"],
            "rank": receipt["header"]["rank"], "n_centroids": receipt["header"]["n_centroids"],
        }
    config = Path(hf_hub_download(MODEL_REPO, "config.json", local_files_only=True))
    checkpoint = Path(hf_hub_download(
        MODEL_REPO, "pytorch_model.bin", local_files_only=True
    ))
    return {
        "schema_version": 1,
        "receipt_kind": "mlp0_native_down_hierarchy_v1_eval_authority",
        "status": "frozen_before_any_native_down_evaluation_forward",
        "scope": "one 384-source-document two-wave evaluation of ten immutable programs",
        "source_commit": source_commit,
        "source_hashes": {str(path.resolve()): file_sha256(path) for path in SOURCE_CLOSURE},
        "row_receipt_path": str(ROW_RECEIPT.resolve()),
        "row_receipt_sha256": file_sha256(ROW_RECEIPT),
        "fit_receipt_path": str(FIT_RECEIPT.resolve()),
        "fit_receipt_sha256": file_sha256(FIT_RECEIPT),
        "stage0_row_receipt_path": str(STAGE0_ROW_RECEIPT.resolve()),
        "stage0_row_receipt_sha256": stage0_row_hash,
        "stage0_fit_receipt_path": str(STAGE0_FIT_RECEIPT.resolve()),
        "stage0_fit_receipt_sha256": stage0_fit_hash,
        "program_contract": program_contract,
        "program_files": program_files,
        "model_repo": MODEL_REPO,
        "model_files": {
            str(config.resolve()): file_sha256(config),
            str(checkpoint.resolve()): file_sha256(checkpoint),
        },
        "integrity_contract": {
            "cloned_native": {"logits_max_abs": 1e-6, "ce_abs": 1e-7,
                              "m0_attn1_mlp1_max_abs": 1e-6},
            "candidate_original_down_calls": 0,
            "poison_canary_must_raise_once": True,
            "candidate_source": "hash-verified reloaded bf16 bundle only",
        },
        "response_contract": {
            "windows": ["chunk[0:257]", "chunk[256:513]"],
            "documents": {"wave_A": 192, "wave_B": 192, "pooled": 384},
            "bootstrap_replicates": N_BOOTSTRAP,
            "bootstrap_seed": SEED,
            "minimum_documents_per_cell_each_wave": MINIMUM_DOCUMENTS_PER_CELL,
            "family": "arm x rung x consumer x cell; paired centered max bootstrap",
            "absolute_gate": "wave UCB<1 each; pooled UCB<0.8",
        },
        "output_path": str(OUT),
        "failure_path": str(FAILURE),
        "lock_path": str(LOCK),
        "output_policy": "fail closed; atomic result; raw source-document ledgers retained",
    }


def main() -> None:
    payload = build_authority()
    write_json_atomic(payload, AUTHORITY)
    print(json.dumps(payload, indent=2))
    print(f"wrote {AUTHORITY}")


if __name__ == "__main__":
    main()
