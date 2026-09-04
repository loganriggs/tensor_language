#!/usr/bin/env python3
# BQLANE: cpu
"""Independent primitive-evidence audit of the one-shot R592 invalid terminal."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OPS = ROOT / "ops"
COMMIT = "7c6be867fcca7a64b3e6dffbff4540e645a32c4e"
TOLERANCE = 1e-5
PAD_TOKEN = 50_256
WIDTH = 30
VOCAB = 50_304

PRODUCER = OPS / "induction_centered_fixed_geometry_rung592.py"
RUNTIME = OPS / "induction_centered_fixed_geometry_rung592_runtime.py"
ADAPTER = OPS / "execute_induction_centered_fixed_geometry_rung592.py"
R585 = OPS / "induction_selector_payload_frozen_factor_rung585.py"
R585_MANIFEST = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
DIAGNOSTIC = ROOT / "induction_centered_fixed_geometry_rung592_invalid_diagnostic.json"
RECEIPT = ROOT / "induction_centered_fixed_geometry_rung592_invalid_receipt.json"
EVIDENCE = ROOT / "induction_centered_fixed_geometry_rung592_invalid_evidence"
RUNLOG = ROOT / "runlogs/execute_induction_centered_fixed_geometry_rung592.log"

FROZEN_SOURCE_SHA256 = {
    PRODUCER: "e625a94216659f4cafb91114b3f253b42844f7e54cb8531b17e0f47614dc5431",
    RUNTIME: "09309b1299b85f2c57689913547fef01f2a9e7b538b2768ac62ff3e48e0f039c",
    ADAPTER: "de8b6e2977551dc19cd00449a1de5c698dbc5978c8d9c23d1ad0d21576e025c5",
    R585: "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b",
    R585_MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    ROWS: "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
}
FROZEN_OUTCOME_SHA256 = {
    DIAGNOSTIC: "e2d858f8e830d25defab60a38bd4ff7a245d2e1ae2460cdbbba64119ec21f8ae",
    RECEIPT: "069f3f65b119d8d0a5884aef7a4c7e4b9a518f8bb801d619b272f5787ad0ca24",
    EVIDENCE / "call_prefix.jsonl": "ca61f9475f42a84d870685fcd78f22ef26a36e19797d813f026eb62ab79a5261",
    RUNLOG: "f06050f4ed6ec59ffe39f89a0c6c6185d28249b2a5e3888a68cc418c7d8a7e5b",
}
NORMAL_NAMESPACES = (
    ROOT / "induction_centered_fixed_geometry_rung592_results.json",
    ROOT / "induction_centered_fixed_geometry_rung592_receipt.json",
    ROOT / "induction_centered_fixed_geometry_rung592_evidence",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def reject(value: str):
        raise ValueError(f"nonfinite JSON constant {value} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def require_finite_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, Mapping):
        for nested in value.values():
            require_finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            require_finite_json(nested)


def load_r585_authority():
    # R585 is used only for its hash-pinned model-free row/endpoint authority.
    spec = importlib.util.spec_from_file_location("r592_postaudit_r585_authority", R585)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load R585 authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_execution_authority()


def audit() -> dict[str, object]:
    if subprocess.check_output(["git", "rev-parse", COMMIT], cwd=REPO, text=True).strip() != COMMIT:
        raise RuntimeError("candidate commit no longer resolves exactly")
    for path, expected in FROZEN_SOURCE_SHA256.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"source byte mismatch: {path}")
    for path in (PRODUCER, RUNTIME, ADAPTER):
        relative = path.relative_to(REPO)
        blob = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"], cwd=REPO)
        if hashlib.sha256(blob).hexdigest() != FROZEN_SOURCE_SHA256[path]:
            raise RuntimeError(f"candidate Git blob mismatch: {relative}")
    for path, expected in FROZEN_OUTCOME_SHA256.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"outcome byte mismatch: {path}")
    if any(path.exists() for path in NORMAL_NAMESPACES):
        raise RuntimeError("normal R592 namespace exists beside invalid terminal")

    diagnostic = strict_json(DIAGNOSTIC)
    receipt = strict_json(RECEIPT)
    runlog = strict_json(RUNLOG)
    require_finite_json(diagnostic)
    require_finite_json(receipt)
    require_finite_json(runlog)
    if receipt["diagnostic_sha256"] != sha256_file(DIAGNOSTIC):
        raise RuntimeError("receipt does not bind diagnostic bytes")
    prefix_path = EVIDENCE / "call_prefix.jsonl"
    if receipt["call_prefix_sha256"] != sha256_file(prefix_path):
        raise RuntimeError("receipt does not bind call prefix")

    observed_files = {
        str(path.relative_to(EVIDENCE)): path
        for path in EVIDENCE.rglob("*") if path.is_file()
    }
    if set(observed_files) != set(receipt["evidence_files"]):
        raise RuntimeError("receipt evidence inventory differs from directory")
    for name, descriptor in receipt["evidence_files"].items():
        path = observed_files[name]
        if path.stat().st_size != descriptor["byte_length"] or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"receipt evidence binding failed: {name}")

    lines = prefix_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeError("invalid terminal did not stop after exactly one call")
    prefix = json.loads(lines[0])
    require_finite_json(prefix)
    if prefix["call_id"] != "FIT:endpoint:0000" or prefix["call_kind"] != "endpoint":
        raise RuntimeError("unexpected first executed call")
    if prefix["storage"] != "raw_current_chunk":
        raise RuntimeError("failed call was incorrectly canonicalized")
    if (EVIDENCE / "FIT/canonical_slice_ledger.jsonl").stat().st_size != 0:
        raise RuntimeError("failed first call has a nonempty canonical ledger")

    authority = load_r585_authority()
    fit_endpoints = [row for row in authority["endpoints"] if row["split"] == "FIT"]
    expected_rows = fit_endpoints[:32]
    expected_ids = [row["endpoint_id"] for row in expected_rows]
    if prefix["authority_row_ids"] != expected_ids:
        raise RuntimeError("first call row membership/order differs from authority")
    call_root = EVIDENCE / "calls/0000_FIT:endpoint:0000"
    arrays = {path.name: np.load(path, allow_pickle=False) for path in call_root.glob("*.npy")}
    if set(arrays) != set(prefix["evidence_files"]):
        raise RuntimeError("first-call array inventory mismatch")
    for name, descriptor in prefix["evidence_files"].items():
        array, path = arrays[name], call_root / name
        if list(array.shape) != descriptor["shape"] or str(array.dtype) != descriptor["dtype"]:
            raise RuntimeError(f"first-call array schema mismatch: {name}")
        if path.stat().st_size != descriptor["byte_length"] or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"prefix does not bind first-call array: {name}")
    if arrays["logits.npy"].shape != (32, VOCAB):
        raise RuntimeError("full-vocabulary logits were not retained")
    if arrays["tokens.npy"].shape != (32, WIDTH):
        raise RuntimeError("fixed-width token evidence changed")
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise RuntimeError("nonfinite first-call primitive")

    expected_tokens = np.full((32, WIDTH), PAD_TOKEN, dtype="<i8")
    expected_query_positions = []
    expected_support = np.empty((32, 4, 2), dtype=np.bool_)
    for index, row in enumerate(expected_rows):
        ids = np.asarray(row["token_ids"], dtype="<i8")
        expected_tokens[index, : len(ids)] = ids
        query = int(row["final_position"])
        expected_query_positions.append(query)
        for site in range(4):
            for role, payload in enumerate(row["payload_positions"]):
                expected_support[index, site, role] = (
                    expected_tokens[index, int(payload) - 1] == expected_tokens[index, query]
                )
    if not np.array_equal(arrays["tokens.npy"], expected_tokens):
        raise RuntimeError("first-call token tensor differs from authority")
    if prefix["query_positions"] != expected_query_positions:
        raise RuntimeError("query positions differ from authority")
    token_sha256 = hashlib.sha256(expected_tokens.tobytes(order="C")).hexdigest()
    if prefix["token_sha256"] != token_sha256:
        raise RuntimeError("token content hash differs from authority")
    if not np.array_equal(arrays["support.npy"], expected_support):
        raise RuntimeError("observed equality support differs from semantic coordinates")

    full_error = np.abs(
        arrays["independent_full_native_write.npy"].astype(np.float64)
        - arrays["native_full_attention_write.npy"].astype(np.float64)
    )
    remainder_error = np.abs(
        arrays["native_equality_term.npy"].astype(np.float64)
        + arrays["native_non_equality_remainder.npy"].astype(np.float64)
        - arrays["native_head_write.npy"].astype(np.float64)
    )
    factorized_error = np.abs(
        arrays["factorized_equality_term.npy"].astype(np.float64)
        - arrays["native_equality_term.npy"].astype(np.float64)
    )
    full_max = float(full_error.max())
    remainder_max = float(remainder_error.max())
    factorized_max = float(factorized_error.max())
    false_support = int((~expected_support).sum())
    if full_max != 0.0 or remainder_max != 5.340576171875e-05 or false_support != 144:
        raise RuntimeError("primitive invalid-predicate values changed")
    if remainder_max <= TOLERANCE:
        raise RuntimeError("registered native decomposition predicate did not fail")
    concurrent_failures = ["native_equality_remainder_reconstruction_failed"]
    if false_support:
        concurrent_failures.append("factor_transport_failed")
    if diagnostic != {
        "schema": "induction_centered_fixed_geometry_rung592_invalid_diagnostic_v1",
        "status": "invalid_diagnostic",
        "failure_predicate": concurrent_failures[0],
        "executed_call_ids": ["FIT:endpoint:0000"],
        "call_prefix_sha256": sha256_file(prefix_path),
        "details": {
            "native_full_write_reconstruction_max_abs": full_max,
            "native_equality_remainder_reconstruction_max_abs": remainder_max,
            "support_false_count": false_support,
        },
        "model_backwards": 0,
        "model_weights_updated": False,
        "final_opened": False,
        "ood_opened": False,
    }:
        raise RuntimeError("diagnostic is not exactly derived from primitive evidence")
    if receipt["executed_call_ids"] != diagnostic["executed_call_ids"]:
        raise RuntimeError("receipt call prefix differs from diagnostic")
    if runlog != {"model_forwards": 1, "status": "invalid_diagnostic"}:
        raise RuntimeError("managed runlog does not match one-call invalid terminal")

    true_per_endpoint = expected_support[:, 0].sum(axis=-1)
    unique, counts = np.unique(true_per_endpoint, return_counts=True)
    support_histogram = {str(int(key)): int(value) for key, value in zip(unique, counts)}
    return {
        "schema": "induction_centered_fixed_geometry_rung592_postexecution_audit_v1",
        "classification": "invalid_instrument",
        "scientific_terminal": None,
        "scientific_claim_permitted": False,
        "primary_failure": concurrent_failures[0],
        "concurrent_failures": concurrent_failures,
        "native_full_write_reconstruction_max_abs": full_max,
        "native_equality_remainder_reconstruction_max_abs": remainder_max,
        "native_equality_remainder_values_over_tolerance": int((remainder_error > TOLERANCE).sum()),
        "factorized_vs_native_equality_max_abs": factorized_max,
        "factorized_vs_native_values_over_tolerance": int((factorized_error > TOLERANCE).sum()),
        "support_false_count": false_support,
        "support_true_count": int(expected_support.sum()),
        "support_true_per_endpoint_histogram": support_histogram,
        "model_forwards": 1,
        "fit_calls_completed": 1,
        "select_calls_completed": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "fit_scored": False,
        "select_opened": False,
        "bootstrap_evaluated": False,
        "endpoint_rows_observed": 32,
        "directed_rows_observed": 0,
        "vocab_size": VOCAB,
        "receipt_evidence_file_count": len(observed_files),
        "receipt_evidence_bytes": sum(path.stat().st_size for path in observed_files.values()),
        "diagnostic_sha256": sha256_file(DIAGNOSTIC),
        "receipt_sha256": sha256_file(RECEIPT),
        "call_prefix_sha256": sha256_file(prefix_path),
        "runlog_sha256": sha256_file(RUNLOG),
        "candidate_commit": COMMIT,
        "source_sha256": {str(path.relative_to(REPO)): digest for path, digest in FROZEN_SOURCE_SHA256.items()},
        "normal_namespaces_absent": True,
        "final_opened": False,
        "ood_opened": False,
    }


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
