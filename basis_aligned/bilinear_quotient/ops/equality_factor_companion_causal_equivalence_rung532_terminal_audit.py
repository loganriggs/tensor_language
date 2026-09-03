#!/usr/bin/env python3
"""Independent artifact, gate, and descriptive audit for completed rung 532."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import bilin18_observed_model_facade as facade
import equality_factor_companion_causal_equivalence_rung532 as rung532


RESULT = ROOT / "equality_factor_companion_causal_equivalence_rung532_results.json"
BUNDLE = ROOT / "equality_factor_companion_causal_equivalence_rung532_bundle.pt"
RUNNER = ROOT / "ops/equality_factor_companion_causal_equivalence_rung532.py"
PREREG = POLY / "EQUALITY_FACTOR_COMPANION_CAUSAL_EQUIVALENCE_RUNG532_PREREGISTRATION.md"
RUNLOG = ROOT / "runlogs/equality_factor_companion_causal_equivalence_rung532.log"
OUT = ROOT / "equality_factor_companion_causal_equivalence_rung532_terminal_audit.json"
EXPECTED = {
    RESULT: "76b7c417a9bceff2f35937f51404c5248bac19b3024fb32ec6891ae70ae4ba2b",
    BUNDLE: "1d7e6cec94250c19159e39b24e156b0c1923fe76e364f23a69ee91b09c5a6bf0",
    RUNNER: "142f4a0f05d582413fb6eac1820654dc6d4491690af9742e0a2d81eac719fdb8",
    PREREG: "5417fd39f3ebe5827276e03e85d73b7791e53dcf85bfde9dc5d41fcaf0c8ec7e",
    RUNLOG: "77934d63cb20918d1d11b0cdef203712102c84557233904af3dfa1ae8f6ef2ff",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_pass(row, arm: str) -> bool:
    report = row["arms"][arm]
    metric = report["member_effect"]
    recovery = report["copy_task_recovery"]
    return bool(
        metric["cosine"] >= 0.85
        and metric["relative_error"] <= 0.60
        and recovery is not None
        and 0.65 <= recovery <= 1.40
        and report["slice_control_mean_abs_ce_change_from_native"] <= 0.01
    )


def _descriptive_summary(contexts):
    """Post-registered diagnostic; never used to rescore the frozen predictions."""
    arms = (
        "product_control", "swapped_first", "direct_first", "permuted_first",
        "swapped_second", "direct_second", "permuted_second",
    )
    summary = {}
    for arm in arms:
        values = [row["arms"][arm] for row in contexts]
        summary[arm] = {
            "base_and_slice_contexts_passing": sum(_base_pass(row, arm) for row in contexts),
            "minimum_member_effect_cosine": min(
                value["member_effect"]["cosine"] for value in values),
            "maximum_member_effect_relative_error": max(
                value["member_effect"]["relative_error"] for value in values),
            "minimum_copy_task_recovery": min(value["copy_task_recovery"] for value in values),
            "maximum_copy_task_recovery": max(value["copy_task_recovery"] for value in values),
            "maximum_slice_control_mean_abs_ce_change_from_native": max(
                value["slice_control_mean_abs_ce_change_from_native"] for value in values),
        }
    summary["registered_failure_reason"] = {
        "swapped_first_beats_permuted_by_0p15_contexts": sum(
            row["arms"]["swapped_first"]["member_effect"]["cosine"]
            >= row["arms"]["permuted_first"]["member_effect"]["cosine"] + 0.15
            for row in contexts),
        "swapped_first_beats_direct_by_0p15_contexts": sum(
            row["arms"]["swapped_first"]["member_effect"]["cosine"]
            >= row["arms"]["direct_first"]["member_effect"]["cosine"] + 0.15
            for row in contexts),
        "swapped_second_beats_permuted_by_0p15_contexts": sum(
            row["arms"]["swapped_second"]["member_effect"]["cosine"]
            >= row["arms"]["permuted_second"]["member_effect"]["cosine"] + 0.15
            for row in contexts),
        "swapped_second_beats_direct_by_0p15_contexts": sum(
            row["arms"]["swapped_second"]["member_effect"]["cosine"]
            >= row["arms"]["direct_second"]["member_effect"]["cosine"] + 0.15
            for row in contexts),
    }
    return summary


def audit():
    observed_hashes = {str(path): sha256(path) for path in EXPECTED}
    if any(observed_hashes[str(path)] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("rung532 terminal artifact hash mismatch")
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "rung532_62_circuit_ce_sufficient_statistics_v1":
        raise RuntimeError("rung532 bundle schema changed")
    if bundle.get("raw_tokens_logits_states_or_per_token_losses_included") is not False:
        raise RuntimeError("rung532 bundle unexpectedly contains raw observations")
    if bundle.get("ood_opened") is not False:
        raise RuntimeError("rung532 opened sealed OOD data")

    reports, contexts = rung532.analyze(bundle["collection"])
    if reports != result["reports"]:
        raise RuntimeError("saved reports do not recompute from sufficient statistics")
    predictions, checks = rung532.score(
        reports, contexts, bundle["diagnostics"], bundle["collection"],
        result["checkpoint"]["weights_sha256"],
    )
    if any(result.get(key) is not value for key, value in predictions.items()):
        raise RuntimeError("saved prediction gates do not independently recompute")
    if checks != result["checks"]:
        raise RuntimeError("saved context counts do not independently recompute")
    strong_null = bool(
        predictions["pred_a_exact_live_interaction_instrument"]
        and predictions["pred_b_product_control_transfers"]
        and not predictions["pred_c_source_second_replaces_target_first"]
        and not predictions["pred_d_source_first_replaces_target_second"])
    if result.get("strong_null") is not strong_null:
        raise RuntimeError("saved strong-null gate changed")

    diagnostics = bundle["diagnostics"]
    calls = diagnostics["direct_native_calls"] + diagnostics["analytical_calls"]
    if calls != rung532.FORWARDS or result["price"]["model_forwards"] != calls:
        raise RuntimeError("forward count does not reconcile")
    if result["checkpoint"]["weights_sha256"] != facade.WEIGHTS_SHA256:
        raise RuntimeError("checkpoint authority changed")
    if tuple(result["documents"]) != rung532.DOCUMENTS or result["price"]["ood_forwards"] != 0:
        raise RuntimeError("row authority changed")

    return {
        "status": "audit_passed",
        "rung": 532,
        "artifact_hashes": observed_hashes,
        "recomputed_predictions": predictions,
        "recomputed_checks": checks,
        "strong_null": strong_null,
        "descriptive_not_rescored": _descriptive_summary(contexts),
        "calls_reconciled": calls,
        "documents_reconfirmed": list(rung532.DOCUMENTS),
        "ood_forwards_reconfirmed": 0,
        "raw_observations_absent": True,
    }


def main():
    report = audit()
    dump(report, OUT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
