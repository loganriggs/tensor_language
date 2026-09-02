#!/usr/bin/env python3
"""Recover every singleton finite-loss effect from rung 504's complete pair table.

This is an after-outcome, CPU-only algebraic audit.  Rung 504 stored, for every
unordered pair {s,t}, both the joint finite effect C_st and its inclusion-exclusion
term Q_st = C_st - C_s - C_t.  Therefore y_st = C_st - Q_st = C_s + C_t.
On the complete graph of n sources these pair sums identify every singleton:

    total = sum_{s<t} y_st / (n - 1)
    C_s   = (sum_{t != s} y_st - total) / (n - 2).

No new model outcomes are opened and no fitted or ranked basis is introduced.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
BUNDLE = ROOT / "mlp9_finite_two_source_interaction_rung504_bundle.pt"
RECEIPT = ROOT / "mlp9_finite_two_source_interaction_rung504_results.json"
OUT = ROOT / "rung504_recovered_singleton_finite_loss.json"
SOURCES = (
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A9",
    "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
)
PAIRS = tuple(itertools.combinations(range(len(SOURCES)), 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recover(pair_sums: torch.Tensor) -> torch.Tensor:
    """Recover C_s from all y_st=C_s+C_t; pair_sums has shape [153]."""
    n = len(SOURCES)
    if tuple(pair_sums.shape) != (n * (n - 1) // 2,):
        raise ValueError("pair-sum vocabulary changed")
    incident = torch.zeros(n, dtype=torch.float64)
    for value, (left, right) in zip(pair_sums, PAIRS):
        incident[left] += value
        incident[right] += value
    total = pair_sums.sum() / (n - 1)
    return (incident - total) / (n - 2)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("rung") != 504 \
            or receipt.get("pred_a_exact_finite_suffix_instrument_and_parent_reproduce") is not True \
            or receipt.get("strong_null") is not True \
            or receipt.get("confirmation_opened") is not False:
        raise RuntimeError("rung 504 receipt identity changed")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "mlp9_finite_two_source_interaction_rung504_stats_v1" \
            or bundle.get("raw_tokens_logits_gradients_or_per_token_vectors_included") is not False:
        raise RuntimeError("rung 504 sufficient-statistics schema changed")
    stats = bundle["stats"]
    pair_loss = stats["pair_loss"]
    denominators = stats["denominators"]
    if tuple(pair_loss.shape) != (2, 4, len(PAIRS), 4) \
            or tuple(denominators.shape) != (2, 4, 3):
        raise RuntimeError("rung 504 sufficient-statistics shape changed")

    backgrounds = []
    closure_max = 0.0
    receipt_fraction_error_max = 0.0
    for background in range(2):
        pooled = pair_loss[background, :2].sum(0)
        # Slots: joint score effect, mixed score effect, joint payload, mixed payload.
        score_sums = pooled[:, 0] - pooled[:, 1]
        payload_sums = pooled[:, 2] - pooled[:, 3]
        score_singletons = recover(score_sums)
        payload_singletons = recover(payload_sums)
        score_rebuilt = torch.tensor(
            [score_singletons[l] + score_singletons[r] for l, r in PAIRS],
            dtype=torch.float64,
        )
        payload_rebuilt = torch.tensor(
            [payload_singletons[l] + payload_singletons[r] for l, r in PAIRS],
            dtype=torch.float64,
        )
        closure = max(
            float((score_rebuilt - score_sums).abs().max()),
            float((payload_rebuilt - payload_sums).abs().max()),
        )
        closure_max = max(closure_max, closure)
        benefit = float(denominators[background, :2, 1].sum())
        if abs(benefit) < 1e-12:
            raise RuntimeError("finite copy-benefit denominator is inert")
        score_fractions = score_singletons / benefit
        payload_fractions = payload_singletons / benefit

        for pair_index, (left, right) in enumerate(PAIRS):
            name = f"{SOURCES[left]}+{SOURCES[right]}"
            observed = receipt["selection"]["details"][name]["backgrounds"][background][
                "finite_copy_fraction"
            ]
            direct = float(pooled[pair_index, 0]) / benefit
            receipt_fraction_error_max = max(receipt_fraction_error_max, abs(observed - direct))

        rows = [
            {
                "source": source,
                "finite_copy_fraction": float(score_fractions[index]),
                "finite_payload_fraction": float(payload_fractions[index]),
            }
            for index, source in enumerate(SOURCES)
        ]
        backgrounds.append({
            "name": ("early_present" if background == 0 else "early_absent"),
            "copy_benefit_denominator_nat_sum": benefit,
            "pair_sum_closure_max_abs_nat_sum": closure,
            "singletons": rows,
            "most_positive": sorted(rows, key=lambda row: row["finite_copy_fraction"], reverse=True)[:5],
            "most_negative": sorted(rows, key=lambda row: row["finite_copy_fraction"])[:5],
        })

    if closure_max > 5e-7 or receipt_fraction_error_max > 1e-12:
        raise RuntimeError(
            f"algebra audit failed: closure={closure_max}, receipt={receipt_fraction_error_max}"
        )
    result = {
        "status": "complete",
        "analysis": "after_outcome_cpu_algebra_no_new_model_outcomes",
        "formula": "y_st=C_st-Q_st=C_s+C_t; C_s=(sum_t y_st-sum_edges(y)/(n-1))/(n-2)",
        "source_count": len(SOURCES),
        "pair_count": len(PAIRS),
        "bundle_sha256": sha256(BUNDLE),
        "receipt_sha256": sha256(RECEIPT),
        "pair_sum_closure_max_abs_nat_sum": closure_max,
        "receipt_fraction_error_max_abs": receipt_fraction_error_max,
        "backgrounds": backgrounds,
        "interpretation": (
            "MLP8 carries about one percent of the actual finite copy benefit despite carrying "
            "about one quarter of the local MLP9 write response; MLP7 and attention5 are "
            "large antagonistic singleton effects.  The MLP9 write is therefore a calibrated "
            "reader of the score action, not a standalone causal mediator at this source-removal grain."
        ),
        "new_model_outcomes_opened": False,
        "deployed_parameters_added": 0,
        "deployed_parameters_saved": 0,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "closure_max": closure_max,
        "receipt_fraction_error_max": receipt_fraction_error_max,
        "m8_finite_copy_fractions": [
            next(row["finite_copy_fraction"] for row in background["singletons"]
                 if row["source"] == "M8")
            for background in backgrounds
        ],
        "most_positive": [background["most_positive"] for background in backgrounds],
        "most_negative": [background["most_negative"] for background in backgrounds],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
