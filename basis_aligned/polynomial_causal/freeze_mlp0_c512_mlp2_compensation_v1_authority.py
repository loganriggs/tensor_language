#!/usr/bin/env python3
"""Freeze the complete MLP2 factorial closure before any evaluation forward."""

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
sys.path[:0] = [str(ROOT), str(BQ), str(HERE)]

from evaluate_mlp0_c512_mlp2_compensation_v1 import (  # noqa: E402
    AUTHORITY, FAILURE, FIT_RECEIPT, INHERITED_RESULT, LOCK, OUT, PROGRAM_KEY,
    ROW_RECEIPT, STAGE0_FIT_RECEIPT, STAGE0_ROW_RECEIPT, V1_AUTHORITY,
    V1_FAILURE, V1_RESULT, closure_sha256, file_sha256,
    inherited_currency_contract, load_domain, repair_amendment_contract,
    tensor_sha256,
)
from mlp0_c512_mlp2_evaluator_contract import (  # noqa: E402
    ARM_CARRIED_PATH, control_contract_sha256, expected_call_contract,
    unit_identity_hashes,
)
from prepare_mlp0_c512_mlp2_compensation_v1_rows import (  # noqa: E402
    load_frozen_rows,
)
from score_mlp0_c512_mlp2_compensation_v1 import (  # noqa: E402
    MARGINS, MIN_DOCUMENTS_PER_CELL, NATIVE_CONTROL_NORM_CONTRACT, N_BOOTSTRAP,
    RAW_ARMS, SEED, frozen_inference_contract,
)


MODEL_REPO = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
SOURCE_CLOSURE = (
    HERE / "freeze_mlp0_c512_mlp2_compensation_v1_authority.py",
    HERE / "evaluate_mlp0_c512_mlp2_compensation_v1.py",
    HERE / "score_mlp0_c512_mlp2_compensation_v1.py",
    HERE / "mlp0_c512_mlp2_evaluator_contract.py",
    HERE / "mlp0_c512_mlp2_compensation.py",
    HERE / "mlp0_c512_mlp1_interchange.py",
    HERE / "evaluate_mlp0_c512_mlp1_interchange_v1.py",
    HERE / "score_mlp0_c512_mlp1_interchange_v1.py",
    HERE / "mlp0_native_down_program.py",
    HERE / "prepare_mlp0_c512_mlp2_compensation_v1_rows.py",
    HERE / "prepare_mlp0_c512_mlp1_interchange_v1_rows.py",
    HERE / "prepare_mlp0_native_down_hierarchy_v1_rows.py",
    HERE / "prepare_mlp0_quotient_stage0_v1_rows.py",
    HERE / "local_fineweb_harvest.py",
    HERE / "MLP0_C512_MLP2_COMPENSATION_SPEC.md",
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


def write_json_create_only(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            descriptor = None
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def build_authority() -> dict:
    if any(path.exists() for path in (AUTHORITY, OUT, FAILURE, LOCK)):
        raise RuntimeError("MLP2 evaluation namespace is already frozen or spent")
    source_commit = git("rev-parse", "HEAD")
    if source_commit != git("rev-parse", "origin/main"):
        raise RuntimeError("HEAD is not synchronized to origin/main")
    relative = [str(path.relative_to(ROOT)) for path in SOURCE_CLOSURE]
    for path in SOURCE_CLOSURE:
        require_committed_head_blob(path)
    if subprocess.run(
        ["git", "diff", "--quiet", "--", *relative], cwd=ROOT
    ).returncode:
        raise RuntimeError("MLP2 evaluation source closure is dirty")

    domain, identity, frozen_rows = load_domain()
    row_receipt, reloaded_rows = load_frozen_rows()
    if (row_receipt.get("status") != "frozen_before_any_c512_mlp2_model_forward"
            or row_receipt.get("sample_summary", {}).get("n_source_documents") != 384
            or not all(row_receipt.get("disjointness_gates", {}).values())
            or not tensor_sha256(reloaded_rows) == tensor_sha256(frozen_rows)):
        raise RuntimeError("fresh MLP2 row authority is invalid")
    n_windows = len(domain["rows"])
    if n_windows != 1256:
        raise RuntimeError("frozen MLP2 window count changed")

    fit_receipt = json.loads(FIT_RECEIPT.read_text())
    if fit_receipt.get("status") != "frozen_before_evaluation_authority":
        raise RuntimeError("C512 fit receipt is not authoritative")
    nested = fit_receipt["authority"]
    if (file_sha256(STAGE0_ROW_RECEIPT)
            != nested["stage0_fit_rows"]["receipt_sha256"]
            or file_sha256(STAGE0_FIT_RECEIPT)
            != nested["stage0_fit_constants"]["receipt_sha256"]):
        raise RuntimeError("C512 fit-authority chain changed")
    program_receipt = fit_receipt["programs"][PROGRAM_KEY]
    program_path = Path(program_receipt["path"])
    if (program_path.stat().st_size != program_receipt["bytes"]
            or file_sha256(program_path) != program_receipt["sha256"]):
        raise RuntimeError("C512 program changed before MLP2 authority")

    inherited, inherited_digest = inherited_currency_contract()
    repair_amendment, repair_amendment_digest = repair_amendment_contract()
    if inherited["fit_rows_tensor_sha256"] != fit_receipt["fit_rows"]["sha256"]:
        raise RuntimeError("C512 fit-row tensor differs from inherited currency chain")
    config = Path(hf_hub_download(MODEL_REPO, "config.json", local_files_only=True))
    checkpoint = Path(hf_hub_download(
        MODEL_REPO, "pytorch_model.bin", local_files_only=True
    ))
    if config.resolve() == checkpoint.resolve():
        raise RuntimeError("config and checkpoint model roles alias")
    source_hashes = {
        str(path.resolve()): file_sha256(path) for path in SOURCE_CLOSURE
    }
    bound_hashes = {
        "source_closure_sha256": closure_sha256(source_hashes),
        "row_receipt_sha256": file_sha256(ROW_RECEIPT),
        "row_tensor_sha256": tensor_sha256(frozen_rows),
        "c512_program_sha256": file_sha256(program_path),
        "model_checkpoint_sha256": file_sha256(checkpoint),
        "model_config_sha256": file_sha256(config),
        "inherited_currency_sha256": inherited_digest,
        "control_contract_sha256": control_contract_sha256(),
        "repair_amendment_sha256": repair_amendment_digest,
    }
    calls = expected_call_contract(n_windows)
    return {
        "schema_version": 1,
        "receipt_kind": "mlp0_c512_mlp2_compensation_v2_eval_authority",
        "status": "frozen_before_any_v2_c512_mlp2_compensation_evaluation_forward",
        "scope": (
            "outcome-blind V2 integrity repair of the physical C512-induced "
            "MLP2 state-by-write factorial"
        ),
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "row_authority": {
            "path": str(ROW_RECEIPT.resolve()),
            "sha256": file_sha256(ROW_RECEIPT),
            "raw_tensor_sha256": tensor_sha256(frozen_rows),
            "raw_shape": list(frozen_rows.shape),
            "evaluation_windows": n_windows,
            "source_documents": 384,
            "wave_documents": {"wave_A": 192, "wave_B": 192},
            "reuse_status": "spent_by_v1_forward_passes_but_outcome_blind",
            "freshness_claim": False,
        },
        "unit_identity": identity,
        "program_contract": {
            "key": PROGRAM_KEY,
            "path": str(program_path.resolve()),
            "bytes": program_receipt["bytes"],
            "sha256": program_receipt["sha256"],
            "rank": 512,
            "n_centroids": 0,
            "fit_receipt_sha256": file_sha256(FIT_RECEIPT),
        },
        "inherited_currency_contract": inherited,
        "repair_amendment": repair_amendment,
        "model_repo": MODEL_REPO,
        "model_files": {
            str(config.resolve()): file_sha256(config),
            str(checkpoint.resolve()): file_sha256(checkpoint),
        },
        "model_file_roles": {
            "config": str(config.resolve()),
            "checkpoint": str(checkpoint.resolve()),
        },
        "assay_contract": {
            "raw_arms": list(RAW_ARMS),
            "arm_carried_path": dict(ARM_CARRIED_PATH),
            "cells": 16,
            "metrics": dict(MARGINS),
            "minimum_documents_per_cell_per_wave": MIN_DOCUMENTS_PER_CELL,
            "control_recipe_sha256": control_contract_sha256(),
            "parent_replays": [
                "exact_live", "candidate_live", "exact_mlp2_omit",
                "candidate_mlp2_omit",
            ],
        },
        "inference_contract": frozen_inference_contract(),
        "integrity_contract": {
            "n_eval_windows": calls["n_eval_windows"],
            "exact_call_counts": calls["exact_call_counts"],
            "exact_phase_site_call_counts": calls["exact_phase_site_call_counts"],
            "bound_hashes": bound_hashes,
            "unit_identity_hashes": unit_identity_hashes(identity),
            "parent_replay_tolerances": {
                "raw_logits_max_abs": 1e-6,
                "capped_logits_max_abs": 1e-6,
                "ce_abs": 1e-7,
            },
            "same_realization_delta_tolerance": 1e-6,
            "carried_state_identity_tolerance": 1e-6,
            "native_control_norm_contract": dict(NATIVE_CONTROL_NORM_CONTRACT),
            "inherited_centered_capped_logit_rms": inherited[
                "centered_capped_logit_rms"
            ],
        },
        "bootstrap": {
            "replicates": N_BOOTSTRAP,
            "seed": SEED,
            "resampling_unit": "source_document",
        },
        "output_path": str(OUT),
        "failure_path": str(FAILURE),
        "lock_path": str(LOCK),
        "output_policy": (
            "authority committed before forward; fail closed; atomic result; "
            "raw source-document ledgers retained"
        ),
    }


def recompute_bound_snapshot(payload: dict) -> dict:
    """Recompute every mutable authority input immediately before publication."""
    if any(path.exists() for path in (AUTHORITY, OUT, FAILURE, LOCK)):
        raise RuntimeError("MLP2 authority/result/failure/lock namespace changed")
    source_commit = payload["source_commit"]
    if (git("rev-parse", "HEAD") != source_commit
            or git("rev-parse", "origin/main") != source_commit):
        raise RuntimeError("git identity changed during MLP2 authority construction")
    sources = {
        str(path.resolve()): file_sha256(path) for path in SOURCE_CLOSURE
    }
    if sources != payload["source_hashes"]:
        raise RuntimeError("source closure changed during MLP2 authority construction")
    domain, identity, frozen_rows = load_domain()
    fit_receipt = json.loads(FIT_RECEIPT.read_text())
    program_receipt = fit_receipt["programs"][PROGRAM_KEY]
    program_path = Path(program_receipt["path"])
    program = {
        "key": PROGRAM_KEY,
        "path": str(program_path.resolve()),
        "bytes": program_path.stat().st_size,
        "sha256": file_sha256(program_path),
        "rank": 512,
        "n_centroids": 0,
        "fit_receipt_sha256": file_sha256(FIT_RECEIPT),
    }
    inherited, inherited_digest = inherited_currency_contract()
    repair_amendment, repair_amendment_digest = repair_amendment_contract()
    roles = payload.get("model_file_roles", {})
    if (set(roles) != {"config", "checkpoint"} or len(set(roles.values())) != 2
            or any(raw not in payload.get("model_files", {}) for raw in roles.values())):
        raise RuntimeError("model roles changed during authority construction")
    model_files = {raw: file_sha256(Path(raw)) for raw in payload["model_files"]}
    bound = {
        "source_closure_sha256": closure_sha256(sources),
        "row_receipt_sha256": file_sha256(ROW_RECEIPT),
        "row_tensor_sha256": tensor_sha256(frozen_rows),
        "c512_program_sha256": file_sha256(program_path),
        "model_checkpoint_sha256": file_sha256(Path(roles["checkpoint"])),
        "model_config_sha256": file_sha256(Path(roles["config"])),
        "inherited_currency_sha256": inherited_digest,
        "control_contract_sha256": control_contract_sha256(),
        "repair_amendment_sha256": repair_amendment_digest,
    }
    return {
        "sources": sources,
        "bound_hashes": bound,
        "model_files": model_files,
        "program_contract": program,
        "identity": identity,
        "identity_hashes": unit_identity_hashes(identity),
        "n_windows": len(domain["rows"]),
        "call_contract": expected_call_contract(len(domain["rows"])),
        "inference_contract": frozen_inference_contract(),
        "control_contract_sha256": control_contract_sha256(),
        "arm_carried_path": dict(ARM_CARRIED_PATH),
        "inherited_currency_contract": inherited,
        "repair_amendment": repair_amendment,
        "v1_namespace": {
            "authority_exists": V1_AUTHORITY.is_file(),
            "failure_exists": V1_FAILURE.is_file(),
            "result_absent": not V1_RESULT.exists(),
        },
        "fit_rows_match": (
            inherited["fit_rows_tensor_sha256"] == fit_receipt["fit_rows"]["sha256"]
        ),
    }


def verify_publication_snapshot(payload: dict) -> None:
    current = recompute_bound_snapshot(payload)
    integrity = payload["integrity_contract"]
    if (current["sources"] != payload["source_hashes"]
            or current["bound_hashes"] != integrity["bound_hashes"]
            or current["model_files"] != payload["model_files"]
            or current["program_contract"] != payload["program_contract"]
            or current["identity"] != payload["unit_identity"]
            or current["identity_hashes"] != integrity["unit_identity_hashes"]
            or current["n_windows"] != integrity["n_eval_windows"]
            or current["call_contract"]["exact_call_counts"]
            != integrity["exact_call_counts"]
            or current["call_contract"]["exact_phase_site_call_counts"]
            != integrity["exact_phase_site_call_counts"]
            or current["inference_contract"] != payload["inference_contract"]
            or current["control_contract_sha256"]
            != payload["assay_contract"]["control_recipe_sha256"]
            or current["arm_carried_path"]
            != payload["assay_contract"]["arm_carried_path"]
            or current["inherited_currency_contract"]
            != payload["inherited_currency_contract"]
            or current["repair_amendment"] != payload["repair_amendment"]
            or current["v1_namespace"] != {
                "authority_exists": True,
                "failure_exists": True,
                "result_absent": True,
            }
            or current["fit_rows_match"] is not True):
        raise RuntimeError("complete MLP2 authority snapshot changed before publication")


def main() -> None:
    payload = build_authority()
    verify_publication_snapshot(payload)
    write_json_create_only(payload, AUTHORITY)
    print(json.dumps(payload, indent=2))
    print(f"wrote {AUTHORITY}")


if __name__ == "__main__":
    main()
