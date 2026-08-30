#!/usr/bin/env python3
"""Deterministic storage/runtime audit for compiler-v2.1's MLP1 Q candidate.

The compiler result prices the rank-64 correction because its evaluation runs inside
the already-built ship.  A standalone C512 x MLP1 x CONTINUE512 composition must also
price and execute the N1 producer that the correction modifies.  This audit reads the
frozen artifacts and reports both prices without loading bilin18 or any evaluation row.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
SHIP = Path("/workspace/runs/bilin18_frozen_ship_v2.pt")
PROGRAMS = BQ / "early_mlp_state_complete_compiler_v21_programs.pt"
BASIS = BQ / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
BASIS_RECEIPT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json"
OUTPUT = HERE / "historical_mlp1_candidate_price_audit.json"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    HERE / "test_historical_mlp1_candidate_price.py",
    HERE / "MLP1_SPARSE_C512_CONTINUE_FACTORIAL_V1_PREREGISTRATION.md",
)

D_MODEL = 1152
HIDDEN = 4608
VOCAB = 50257


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_reals(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel()
    if isinstance(value, (tuple, list)):
        return sum(tensor_reals(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_reals(item) for item in value.values())
    raise TypeError(type(value))


def stable_torch(path: Path) -> tuple[Any, str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    middle = hashlib.sha256(raw).hexdigest()
    after = file_sha256(path)
    if before != middle or middle != after:
        raise RuntimeError(f"tensor parent raced read: {path}")
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return value, before


def stable_json(path: Path) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"JSON parent raced read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON parent is not an object: {path}")
    return value, before


def committed_source_binding() -> dict[str, Any]:
    root = HERE.parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=root, check=True,
    )
    hashes = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(root))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted audit source: {relative}")
        hashes[relative] = digest
    return {"commit": commit, "sha256": hashes}


def audit(
    ship_path: Path = SHIP,
    programs_path: Path = PROGRAMS,
    basis_path: Path = BASIS,
    basis_receipt_path: Path = BASIS_RECEIPT,
    *,
    include_source_binding: bool = False,
) -> dict[str, Any]:
    ship_payload, ship_sha = stable_torch(ship_path)
    program_payload, programs_sha = stable_torch(programs_path)
    basis_payload, basis_sha = stable_torch(basis_path)
    basis_receipt, basis_receipt_sha = stable_json(basis_receipt_path)
    ship = ship_payload["state"]["SHIP"]
    table, ridge = ship["t1"], ship["r1"]
    program = program_payload["programs"]["true"][1]
    correction_price = program_payload["prices"]["true"]["site1"]
    selection = program_payload["selection_receipts"]["true_site1"]
    basis_record = basis_payload["sites"][1]
    basis = basis_record["basis"]

    if tuple(table.shape) != (VOCAB, D_MODEL) or len(ridge) != 3:
        raise RuntimeError("frozen N1 producer shape changed")
    if tuple(ridge[0].shape) != (2 * D_MODEL, D_MODEL) \
            or tuple(ridge[1].shape) != (2 * D_MODEL,) \
            or tuple(ridge[2].shape) != (D_MODEL,):
        raise RuntimeError("frozen N1 ridge shape changed")
    if program.get("interface") != "state_complete_p" \
            or program.get("family") != "B_state_complete_affine_euclidean" \
            or program.get("rank") != 64 \
            or selection.get("selected") != "B_l6_r64" \
            or selection.get("selected_family") != program.get("family"):
        raise RuntimeError("frozen MLP1 correction identity changed")

    if tuple(basis.shape) != (D_MODEL, 64) or basis_record.get(
        "basis_sha256"
    ) != basis_receipt.get("site_basis_sha256", {}).get("1"):
        raise RuntimeError("external MLP1 basis identity changed")
    raw_basis_sha = hashlib.sha256(
        basis.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()
    if raw_basis_sha != basis_record["basis_sha256"] or basis_receipt.get(
        "artifact_sha256"
    ) != basis_sha:
        raise RuntimeError("external MLP1 basis binding changed")

    base_reals = tensor_reals(table) + tensor_reals(ridge)
    predictor_keys = ("mean", "scale", "bias", "left", "right")
    predictor_reals = sum(tensor_reals(program[key]) for key in predictor_keys)
    basis_reals = tensor_reals(basis)
    correction_reals = predictor_reals + basis_reals
    if correction_reals != int(correction_price["total_reals"]) \
            or predictor_reals != int(correction_price["predictor_reals"]) \
            or basis_reals != int(correction_price["basis_reals"]):
        raise RuntimeError("reconstructed correction price differs from frozen price")
    standalone_reals = base_reals + correction_reals
    native_full_reals = int(correction_price["original_mlp_reals"])

    base_multiplies = 2 * D_MODEL * D_MODEL
    correction_multiplies = int(correction_price["inference_multiplies_per_token"])
    native_multiplies = 3 * D_MODEL * HIDDEN

    return {
        "schema": "historical_mlp1_candidate_price_audit_v1",
        "status": "deterministic_artifact_audit_complete",
        "outcome_access": {
            "model_loaded": False,
            "evaluation_rows_loaded": False,
            "model_outputs_opened": False,
        },
        "parents": {
            "frozen_ship": {
                "path": str(ship_path),
                "sha256": ship_sha,
                "realization_sha256": ship_payload["ship_realization_sha256"],
            },
            "compiler_v21_programs": {
                "path": str(programs_path),
                "sha256": programs_sha,
                "selected_true_site1": selection["selected"],
            },
            "external_rank64_basis": {
                "path": str(basis_path),
                "sha256": basis_sha,
                "receipt_path": str(basis_receipt_path),
                "receipt_sha256": basis_receipt_sha,
                "site1_tensor_sha256": raw_basis_sha,
            },
        },
        "semantics": {
            "base_N1": "t1[token] + ym1 + ([attention1_write, live_mlp0_write] - xm1) @ W1",
            "rank64_Q": "replace only the B1-coordinate of base_N1; preserve its orthogonal complement",
            "depends_on_live_mlp0_write": True,
            "is_correction_alone_a_standalone_mlp1_replacement": False,
        },
        "literal_storage_reals": {
            "token_table": tensor_reals(table),
            "ridge": tensor_reals(ridge),
            "base_N1_total": base_reals,
            "rank64_predictor": predictor_reals,
            "rank64_basis": basis_reals,
            "rank64_correction": correction_reals,
            "complete_candidate": standalone_reals,
            "native_full_mlp1": native_full_reals,
            "complete_candidate_over_native": standalone_reals / native_full_reals,
            "correction_only_over_native": correction_reals / native_full_reals,
        },
        "runtime_multiplies_per_token": {
            "base_N1_ridge": base_multiplies,
            "rank64_correction": correction_multiplies,
            "complete_candidate": base_multiplies + correction_multiplies,
            "native_full_mlp1_dense_maps": native_multiplies,
            "complete_candidate_over_native": (
                base_multiplies + correction_multiplies
            ) / native_multiplies,
            "notes": "lookup, additions, RMSNorm, comparisons, and Hadamard products are reported separately",
        },
        "decision": {
            "run_as_current_standalone_simplification": False,
            "reason": (
                "the executable artifact requires the complete frozen N1 producer; "
                "its literal storage is larger than native MLP1"
            ),
            "possible_rehabilitation": (
                "recover and certify the original low-rank-plus-2000-row-exception "
                "table recipe, then rerun the complete price and composition audit"
            ),
        },
        "source_binding": committed_source_binding() if include_source_binding else None,
    }


def write_json_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    value = audit(include_source_binding=True)
    write_json_create_only(OUTPUT, value)
    print(json.dumps(value["literal_storage_reals"], indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
