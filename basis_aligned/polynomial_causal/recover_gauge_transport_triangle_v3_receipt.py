#!/usr/bin/env python3
"""Receipt-only recovery for the complete v2 finite-transport result."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = HERE / "gauge_transport_triangle_results.json"
STATE = HERE / "gauge_transport_triangle_state.pt"
V2_FAILURE = HERE / "gauge_transport_triangle_v2_recovery_failure.json"
V2_AUTHORITY = HERE / "gauge_transport_triangle_v2_recovery_authority.json"
V3_AUTHORITY = HERE / "gauge_transport_triangle_v3_receipt_recovery_authority.json"
V3_RECEIPT = HERE / "gauge_transport_triangle_v3_receipt.json"

EXPECTED_HASHES = {
    "result": "2b79648c9866dfbad51c57c6b8536870962db8998040476f9c636fac5994891b",
    "state": "a85d7cefcc1ea2623dfd0ba42289dbe11b4eb7aec94f923a5492e16a9a069c2e",
    "v2_failure": "4d6c77993af2a4362345b8654ac5ba5d6640f7be0b04cd2d6e17f5b1c081de7b",
    "v2_authority": "5f5785e47dea61db6633c6d65d228946f71905e0baf4b9fa7cc0188377410898",
}
EXPECTED_DECISIONS = {
    "full_oracle_exact": True,
    "projected_u14_sufficient": False,
    "direct_response_transport": False,
    "chain_composes": False,
}
SOURCE_FILES = (
    "basis_aligned/polynomial_causal/recover_gauge_transport_triangle_v3_receipt.py",
    "basis_aligned/polynomial_causal/GAUGE_TRANSPORT_TRIANGLE_V3_RECEIPT_RECOVERY_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/test_recover_gauge_transport_triangle_v3_receipt.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def validate_authority(authority: dict[str, Any]) -> None:
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if authority.get("schema") != "gauge_transport_triangle_v3_receipt_recovery_authority" or (
        authority.get("status") != "source_closed_receipt_only_go"
        or authority.get("authority_sha256") != canonical_sha256(body)
        or tuple(authority.get("source_files", ())) != SOURCE_FILES
        or authority.get("input_sha256s") != EXPECTED_HASHES
        or authority.get("output") != V3_RECEIPT.name
    ):
        raise RuntimeError("v3 receipt authority changed")
    for relative in SOURCE_FILES:
        if sha256_file(ROOT / relative) != authority["source_sha256s"].get(relative):
            raise RuntimeError(f"v3 receipt source changed: {relative}")


def validate_payloads(
    result: dict[str, Any], state: dict[str, Any], failure: dict[str, Any],
    v2_authority: dict[str, Any],
) -> None:
    failure_body = {key: value for key, value in failure.items() if key != "failure_sha256"}
    if failure.get("failure_sha256") != canonical_sha256(failure_body) or (
        failure.get("partial_result_sha256") != EXPECTED_HASHES["result"]
        or failure.get("partial_state_sha256") != EXPECTED_HASHES["state"]
        or failure.get("error") != "triangle result semantic replay failed"
    ):
        raise RuntimeError("v2 failure does not bind the complete partial terminals")
    if result.get("execution_authority_sha256") != v2_authority.get("authority_sha256") or (
        result.get("config", {}).get("model_weights_sha256")
        != "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
        or result.get("config", {}).get("row_receipt_file_sha256")
        != "3f92d8b3aa5e89e6059a010338521bffa0cf440e0815d9d67e1b65aa58a8e102"
        or result.get("config", {}).get("status")
        != "preliminary_screen_failed_no_interface_license"
        or result.get("decisions") != EXPECTED_DECISIONS
        or result.get("screen_passed") is not False
    ):
        raise RuntimeError("v2 result semantics changed")
    if set(state) != {"config", "bases", "supports", "maps"} or (
        state["config"] != result["config"]
        or set(state["bases"]) != {8, 11, 14}
        or set(state["supports"]) != {8, 11, 14}
        or set(state["maps"]) != {"8_11", "8_14", "11_14"}
    ):
        raise RuntimeError("v2 state structure changed")
    tensors = []
    for tensor in state["bases"].values():
        if tuple(tensor.shape) != (1152, 64):
            raise RuntimeError("v2 basis shape changed")
        tensors.append(tensor)
    for tensor in state["supports"].values():
        if tuple(tensor.shape) != (1152, 256):
            raise RuntimeError("v2 support shape changed")
        tensors.append(tensor)
    for tensor in state["maps"].values():
        if tuple(tensor.shape) != (64, 64):
            raise RuntimeError("v2 map shape changed")
        tensors.append(tensor)
    if not all(torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise RuntimeError("v2 state contains a nonfinite or non-tensor value")


def create_only_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if V3_RECEIPT.exists():
        raise RuntimeError("v3 receipt namespace is not fresh")
    authority = json.loads(V3_AUTHORITY.read_text())
    validate_authority(authority)
    commit = authority["source_commit"]
    for ref in ("HEAD", "origin/main"):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, ref],
            cwd=ROOT, check=True,
        )
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCE_FILES,
         str(V3_AUTHORITY.relative_to(ROOT))], cwd=ROOT, text=True,
    ).strip()
    if status:
        raise RuntimeError(f"v3 receipt sources are dirty: {status}")
    paths = {"result": RESULT, "state": STATE, "v2_failure": V2_FAILURE,
             "v2_authority": V2_AUTHORITY}
    if {name: sha256_file(path) for name, path in paths.items()} != EXPECTED_HASHES:
        raise RuntimeError("v3 receipt input hash changed")
    result = json.loads(RESULT.read_text())
    failure = json.loads(V2_FAILURE.read_text())
    v2_authority = json.loads(V2_AUTHORITY.read_text())
    state = torch.load(STATE, map_location="cpu", weights_only=True)
    validate_payloads(result, state, failure, v2_authority)
    normalized_result_sha256 = canonical_sha256(result)
    receipt = {
        "schema": "gauge_transport_triangle_v3_receipt",
        "status": "complete_receipt_only_recovery",
        "v3_authority_sha256": authority["authority_sha256"],
        "input_file_sha256s": EXPECTED_HASHES,
        "normalized_result_sha256": normalized_result_sha256,
        "result_status": result["config"]["status"],
        "screen_passed": result["screen_passed"],
        "decisions": result["decisions"],
        "state_semantics_validated": True,
        "scientific_rerun": False,
        "v2_failure_preserved": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    create_only_json(V3_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
