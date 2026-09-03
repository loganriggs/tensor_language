#!/usr/bin/env python3
"""Planning-only document budget from the observed MLP0 split-half reliability."""

import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "bilinear_product_das_rung536_power_budget.json"
OBSERVED_RELIABILITY = 0.10593727513452525
DOCUMENTS_PER_OLD_HALF = 124
TARGETS = (0.50, 0.70, 0.80)
BATCH = 4


def multiplier(current, target):
    """Spearman-Brown multiplier: target=m*r/(1+(m-1)*r)."""
    return target * (1.0 - current) / (current * (1.0 - target))


def main():
    budgets = {}
    for target in TARGETS:
        factor = multiplier(OBSERVED_RELIABILITY, target)
        per_half = BATCH * math.ceil(DOCUMENTS_PER_OLD_HALF * factor / BATCH)
        budgets[str(target)] = {
            "estimated_document_multiplier": factor,
            "documents_per_half_rounded_to_batch": per_half,
            "total_documents_for_two_halves": 2 * per_half,
        }
    result = {
        "status": "planning_estimate_complete",
        "rung": 536,
        "method": "Spearman-Brown reliability extrapolation",
        "observed_split_half_reliability": OBSERVED_RELIABILITY,
        "old_documents_per_half": DOCUMENTS_PER_OLD_HALF,
        "budgets": budgets,
        "recommended_minimum_target_reliability": 0.70,
        "recommended_total_documents": budgets["0.7"]["total_documents_for_two_halves"],
        "assumption_warning": (
            "Planning estimate only: assumes independent sampling noise and unchanged signal; "
            "the observed reliability came from a frozen-census circuit fingerprint and cannot "
            "label fresh rows; empirical split-half reliability of each exact portable T/I target "
            "must still be measured before DAS fitting."),
        "model_loaded": False,
        "new_model_forwards": 0,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
