#!/usr/bin/env python3
"""Independent artifact, gate, and failure-mode audit for completed rung 533."""

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
import equality_factor_to_slot_exchangeability_rung533 as rung533


RESULT = ROOT / "equality_factor_to_slot_exchangeability_rung533_results.json"
BUNDLE = ROOT / "equality_factor_to_slot_exchangeability_rung533_bundle.pt"
RUNNER = ROOT / "ops/equality_factor_to_slot_exchangeability_rung533.py"
PREREG = POLY / "EQUALITY_FACTOR_TO_SLOT_EXCHANGEABILITY_RUNG533_PREREGISTRATION.md"
RUNLOG = ROOT / "runlogs/equality_factor_to_slot_exchangeability_rung533.log"
OUT = ROOT / "equality_factor_to_slot_exchangeability_rung533_terminal_audit.json"
EXPECTED = {
    RESULT: "5c43872a037f662ab93c64915e74419439513393f026654d8ed16c7bdb7f84d0",
    BUNDLE: "c4a2173ad88624dff33974891f9815d98ae7e7d6e7162d3291d6c7ac0e6ecae4",
    RUNNER: "6ba3a9e5fa4e0fa23c461610451bfc8d65eea909f14fe563131a1441228528fd",
    PREREG: "d5ed32a7a4268768ed170e4a0fdd282fb49e3be97190c077366e77353a6ad1eb",
    RUNLOG: "d8ef4bdf7931f4fd95ffadb02451bfc24b37839f8319fa200b2f62400a16c46c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _failure_modes(row, arm):
    report = row["arms"][arm]
    metric = report["positive_document_effect"]
    recovery = report["positive_task_recovery"]
    return {
        "cosine": metric["cosine"] < 0.85,
        "relative_error": metric["relative_error"] > 0.60,
        "positive_recovery": recovery is None or not 0.65 <= recovery <= 1.40,
        "matched_negative": report["matched_negative_abs_mean_ce_change_from_native"] > 0.01,
        "off_target": report["off_target_abs_mean_ce_change_from_native"] > 0.01,
    }


def _descriptive_summary(reports):
    rows = list(reports.values())
    arms = ("product_control", *rung533.MAPPINGS)
    summary = {}
    for arm in arms:
        arm_reports = [row["arms"][arm] for row in rows]
        modes = [_failure_modes(row, arm) for row in rows]
        summary[arm] = {
            "registered_base_contexts_passing": sum(rung533._base_holds(row, arm) for row in rows),
            "positive_effect_only_contexts_passing": sum(
                report["positive_document_effect"]["cosine"] >= 0.85
                and report["positive_document_effect"]["relative_error"] <= 0.60
                and report["positive_task_recovery"] is not None
                and 0.65 <= report["positive_task_recovery"] <= 1.40
                for report in arm_reports),
            "minimum_positive_effect_cosine": min(
                report["positive_document_effect"]["cosine"] for report in arm_reports),
            "maximum_positive_effect_relative_error": max(
                report["positive_document_effect"]["relative_error"] for report in arm_reports),
            "failure_context_counts": {
                key: sum(mode[key] for mode in modes) for key in modes[0]
            },
        }
        if arm in rung533.MAPPINGS:
            control = rung533.CONTROL_BY_MAPPING[arm]
            summary[arm]["beats_own_key_control_by_0p15_contexts"] = sum(
                row["arms"][arm]["positive_document_effect"]["cosine"]
                >= row["arms"][control]["positive_document_effect"]["cosine"] + 0.15
                for row in rows)
    summary["product_control_by_role_background"] = {
        f"{role}/{background}": sum(
            rung533._base_holds(reports[f"{role}/{background}/half{half}"], "product_control")
            for half in range(2))
        for role in rung533.ROLES for background in rung533.BACKGROUNDS
    }
    return summary


def audit():
    observed_hashes = {str(path): sha256(path) for path in EXPECTED}
    if any(observed_hashes[str(path)] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("rung533 terminal artifact hash mismatch")
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "rung533_cross_corpus_document_ce_sufficient_statistics_v1":
        raise RuntimeError("rung533 bundle schema changed")
    if bundle.get("raw_tokens_logits_hidden_states_or_per_token_losses_included") is not False:
        raise RuntimeError("rung533 bundle unexpectedly contains raw observations")
    if bundle.get("rung532_census_rows_reused") is not False:
        raise RuntimeError("rung533 reused rung532 census rows")

    reports, contexts, stability = rung533.analyze(bundle["collection"])
    if reports != result["reports"] or stability != result["background_stability"]:
        raise RuntimeError("saved reports do not recompute from sufficient statistics")
    predictions, checks = rung533.score(
        contexts, stability, bundle["diagnostics"], result["checkpoint"]["weights_sha256"])
    if any(result.get(key) is not value for key, value in predictions.items()):
        raise RuntimeError("saved prediction gates do not independently recompute")
    if checks != result["checks"]:
        raise RuntimeError("saved context counts do not independently recompute")
    diagnostics = bundle["diagnostics"]
    calls = diagnostics["direct_native_calls"] + diagnostics["analytical_calls"]
    if calls != rung533.FORWARDS or result["price"]["model_forwards"] != calls:
        raise RuntimeError("forward count does not reconcile")
    if result["checkpoint"]["weights_sha256"] != facade.WEIGHTS_SHA256:
        raise RuntimeError("checkpoint authority changed")
    if result["roles"] != list(rung533.ROLES) or result["document_halves"] != [[0, 96], [96, 192]]:
        raise RuntimeError("role or half authority changed")

    return {
        "status": "audit_passed",
        "rung": 533,
        "artifact_hashes": observed_hashes,
        "recomputed_predictions": predictions,
        "recomputed_checks": checks,
        "registered_outcome": "invalid_identification_test_positive_control_failed",
        "descriptive_not_rescored": _descriptive_summary(reports),
        "calls_reconciled": calls,
        "roles_reconfirmed": list(rung533.ROLES),
        "rung532_census_rows_reused": False,
        "raw_observations_absent": True,
    }


def main():
    report = audit()
    dump(report, OUT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
