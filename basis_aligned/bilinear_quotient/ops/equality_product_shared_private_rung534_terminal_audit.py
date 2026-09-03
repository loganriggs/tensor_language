#!/usr/bin/env python3
"""Independent artifact and decision audit for completed rung 534."""

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
import equality_product_shared_private_rung534 as rung534


RESULT = ROOT / "equality_product_shared_private_rung534_results.json"
BUNDLE = ROOT / "equality_product_shared_private_rung534_bundle.pt"
RUNNER = ROOT / "ops/equality_product_shared_private_rung534.py"
PREREG = POLY / "EQUALITY_PRODUCT_SHARED_PRIVATE_RUNG534_PREREGISTRATION.md"
RUNLOG = ROOT / "runlogs/equality_product_shared_private_rung534.log"
OUT = ROOT / "equality_product_shared_private_rung534_terminal_audit.json"
EXPECTED = {
    RESULT: "8804dca2cbd0203a6ef9517a15ec7a4186ed5e69ec8c284b854967c8e13197a7",
    BUNDLE: "77ca551a19004abade5ec5dcc79023a01f3d9c5d97ca693c012ca74f512cef80",
    RUNNER: "fdfb3b0ba8a7a5639cb75677e26e33e24b346f6bd7f45de20f40a70090ab5e88",
    PREREG: "47d738db728c24b8b8d1105e8467905815312fc6d207edad8680a46b8d7de428",
    RUNLOG: "e0e58eeb8b578423446dc956bcbd2b00825c5ee7c24cb177319ee380109d75f4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit():
    observed_hashes = {str(path): sha256(path) for path in EXPECTED}
    if any(observed_hashes[str(path)] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("rung534 terminal artifact hash mismatch")
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "rung534_shared_private_document_ce_sufficient_statistics_v1":
        raise RuntimeError("rung534 bundle schema changed")
    if bundle.get("raw_tokens_logits_hidden_states_or_per_token_losses_included") is not False:
        raise RuntimeError("rung534 bundle unexpectedly contains raw observations")

    reports, contexts = rung534.analyze(bundle["collection"])
    if reports != result["reports"]:
        raise RuntimeError("saved reports do not recompute from sufficient statistics")
    predictions, checks = rung534.score(
        reports, contexts, bundle["diagnostics"], result["checkpoint"]["weights_sha256"])
    if any(result.get(key) is not value for key, value in predictions.items()):
        raise RuntimeError("saved prediction gates do not independently recompute")
    if checks != result["checks"]:
        raise RuntimeError("saved context counts do not independently recompute")

    code_absent = [reports[f"ood_code/donor_absent/half{half}"] for half in range(2)]
    strong_null = bool(
        predictions["pred_a_exact_live_instrument"]
        and predictions["pred_b_shared_signal_premise_reproduces"]
        and any(all(
            not (
                row["cells"][cell]["private_vs_marginal"]["private"]["cosine"] >= 0.80
                and row["cells"][cell]["private_vs_marginal"]["private"]["relative_error"] <= 0.60)
            for row in code_absent) for cell in ("positive", "matched_negative")))
    if strong_null is not result["interaction_only_correction_strong_null"]:
        raise RuntimeError("strong-null decision does not independently recompute")

    diagnostics = bundle["diagnostics"]
    calls = diagnostics["direct_native_calls"] + diagnostics["analytical_calls"]
    if calls != rung534.FORWARDS or result["price"]["model_forwards"] != calls:
        raise RuntimeError("forward count does not reconcile")
    if result["checkpoint"]["weights_sha256"] != facade.WEIGHTS_SHA256:
        raise RuntimeError("checkpoint authority changed")

    compact = {}
    for key, row in reports.items():
        compact[key] = {
            cell: row["cells"][cell]["private_vs_marginal"]["private"]
            for cell in ("positive", "matched_negative")
        }
    return {
        "status": "audit_passed",
        "rung": 534,
        "artifact_hashes": observed_hashes,
        "recomputed_predictions": predictions,
        "recomputed_checks": checks,
        "registered_outcome": "interaction_only_correction_strong_null",
        "strong_null_recomputed": strong_null,
        "private_vs_marginal_metrics": compact,
        "calls_reconciled": calls,
        "raw_observations_absent": True,
    }


def main():
    report = audit()
    dump(report, OUT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
