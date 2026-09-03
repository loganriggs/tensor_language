#!/usr/bin/env python3
"""Independent artifact and gate audit for the completed rung 531 screen."""

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

import equality_score_factor_branch_sharing_rung531 as rung531


RESULT = ROOT / "equality_score_factor_branch_sharing_rung531_results.json"
BUNDLE = ROOT / "equality_score_factor_branch_sharing_rung531_bundle.pt"
RUNNER = ROOT / "ops/equality_score_factor_branch_sharing_rung531.py"
PREREG = POLY / "EQUALITY_SCORE_FACTOR_BRANCH_SHARING_RUNG531_PREREGISTRATION.md"
OUT = ROOT / "equality_score_factor_branch_sharing_rung531_terminal_audit.json"
EXPECTED = {
    RESULT: "016d4e7babaf2fa562ee254e76ea8c354a7448ddb9fb70cf4be6c835c77354ab",
    BUNDLE: "62f3a224eee35b067a79297f967410c2eb342df13e868f9bc31f2ad4de534442",
    RUNNER: "e2eb9bd2674247c1fa1c0e25a50d4e747b899a2883899f3074bf809bc676f71e",
    PREREG: "5bc02afcaffc04aa62baf48f8f8dcbbb727deec86e7722287177055730b785b3",
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
        raise RuntimeError("rung531 terminal artifact hash mismatch")
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "rung531_factor_branch_aggregate_dot_products_v1":
        raise RuntimeError("rung531 bundle schema changed")
    if bundle.get("raw_tokens_logits_states_or_edge_factors_included") is not False:
        raise RuntimeError("rung531 bundle unexpectedly contains raw observations")
    if bundle.get("validation_or_ood_opened") is not False:
        raise RuntimeError("rung531 opened sealed data")
    reports, both, one, gauge = rung531.analyze(bundle["statistics"])
    if reports != result["reports"]:
        raise RuntimeError("saved reports do not recompute from aggregate dot products")
    if both != result["both_factor_candidates"]:
        raise RuntimeError("both-factor candidate list changed")
    if one != result["one_factor_candidates"]:
        raise RuntimeError("one-factor candidate list changed")
    if gauge != result["gauge_consistent_candidates"]:
        raise RuntimeError("gauge candidate list changed")
    diagnostics = bundle["diagnostics"]
    pred_a = bool(
        diagnostics["calls"] == rung531.FORWARDS
        and bool((bundle["statistics"]["edges"] > 0).all())
        and bool((bundle["statistics"]["factor_source2"] > 0).all())
        and bool((bundle["statistics"]["factor_target2"] > 0).all())
        and bool((bundle["statistics"]["product_source2"] > 0).all())
        and bool((bundle["statistics"]["product_target2"] > 0).all())
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["branch_product_max_abs"] == 0.0)
    recomputed = {
        "pred_a_exact_authorized_instrument": pred_a,
        "pred_b_both_score_factors_shared": bool(pred_a and both),
        "pred_c_exactly_one_score_factor_shared": bool(pred_a and one),
        "pred_d_factor_gauges_match_product": bool(pred_a and gauge),
        "strong_null": bool(pred_a and not both and not one),
    }
    if any(result.get(key) is not value for key, value in recomputed.items()):
        raise RuntimeError("saved prediction gates do not independently recompute")
    return {
        "status": "audit_passed",
        "rung": 531,
        "artifact_hashes": observed_hashes,
        "recomputed_predictions": recomputed,
        "both_factor_candidates": both,
        "one_factor_candidates": one,
        "gauge_consistent_candidates": gauge,
        "calls_reconciled": diagnostics["calls"],
        "sealed_rows_reconfirmed": [500, 1000],
        "raw_observations_absent": True,
    }


def main():
    report = audit()
    dump(report, OUT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
