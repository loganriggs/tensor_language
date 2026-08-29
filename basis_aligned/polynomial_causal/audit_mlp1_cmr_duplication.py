#!/usr/bin/env python3
"""Quantify overlap between the proposed local CMR family and spent MLP1 assays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "tensor_bilin18_mlp1_global_gate_bundle.pt"
RESULT = HERE / "tensor_bilin18_mlp1_global_gate_results.json"
OUTPUT = HERE / "mlp1_cmr_duplication_audit_results.json"
BUNDLE_SHA256 = "b8061f3b50b49309d6cc4cc6aa8b6c91c705a3ae91a98a45012ec9985dae372e"
RESULT_SHA256 = "e428887fc0b17b2a374d07ede5752de2c43cad465a17cb891fabd35c4f29ec84"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    left, right = set(a.tolist()), set(b.tolist())
    return len(left & right) / len(left | right)


def rank_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman correlation for deterministic score vectors without material ties."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("score vectors must have the same one-dimensional shape")
    ra = torch.empty_like(a, dtype=torch.float64)
    rb = torch.empty_like(b, dtype=torch.float64)
    order_a = torch.argsort(a.double(), stable=True)
    order_b = torch.argsort(b.double(), stable=True)
    ranks = torch.arange(len(a), dtype=torch.float64)
    ra[order_a], rb[order_b] = ranks, ranks
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()))


def run() -> dict:
    if sha256(BUNDLE) != BUNDLE_SHA256 or sha256(RESULT) != RESULT_SHA256:
        raise RuntimeError("spent MLP1 assay inputs changed")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    result = json.loads(RESULT.read_text())
    if result.get("status") != "no_admitted_support":
        raise RuntimeError("spent MLP1 assay status changed")
    comparisons = {}
    for budget in (32, 128):
        arms = bundle["analysis_bundle"]["budgets"][str(budget)]
        activation = arms["activation_down"]
        row = {}
        for name in ("primary", "response_energy", "factor_product_derangement", "hash_random"):
            other = arms[name]
            row[name] = {
                "support_jaccard_with_activation_down": jaccard(
                    activation["support"], other["support"],
                ),
                "score_rank_spearman_with_activation_down": rank_spearman(
                    activation["selection_scores"], other["selection_scores"],
                ),
            }
        comparisons[str(budget)] = row
    return {
        "schema": "mlp1_cmr_duplication_audit_v1",
        "bundle_sha256": BUNDLE_SHA256,
        "result_sha256": RESULT_SHA256,
        "spent_assay_status": result["status"],
        "exact_relation": (
            "The spent activation_down selector ranks by RMS(a_j)*||D_j||. "
            "Its square is E[a_j^2]*||D_j||^2. The proposed diagonal local CMR "
            "selector is Var(a_j)*||D_j||^2, differing only by the nonnegative "
            "mean-squared term E[a_j]^2*||D_j||^2. Both are immediate-write, "
            "fixed-native-channel scores, not final-logit risk."
        ),
        "comparisons": comparisons,
        "decision": "no_run_mlp1_cmr_due_to_substantial_duplicate_and_prior_negative",
        "claim_boundary": (
            "This audit establishes protocol redundancy. It does not prove that "
            "centered-variance and RMS rankings are identical or that 50% retention fails."
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    output = run()
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
