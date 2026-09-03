#!/usr/bin/env python3
"""CPU-only independent audit of R538's row-level sufficient statistics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402
import pending_opener_common_site_rung538 as core  # noqa: E402

RESULT = ROOT / "pending_opener_common_site_rung538_results.json"
INVALID = ROOT / "pending_opener_common_site_rung538_invalid_unverified_checkpoint_results.json"
OUT = ROOT / "pending_opener_common_site_rung538_terminal_audit.json"
RESULT_SHA256 = "f011399614953c958faf2a12ef15e938dcc2f5e3f52ea868763de2a82443a205"


def main() -> None:
    result = json.loads(RESULT.read_text())
    invalid = json.loads(INVALID.read_text())
    reports, passing = core.score(result["raw_donorward_movements"])
    raw_counts_exact = all(
        len(result["raw_donorward_movements"][site][split][family][direction])
        == (48 if split == "FIT" else 16)
        for site in core.SITE_ORDER for split in core.SPLITS
        for family in core.FAMILIES
        for direction in ("base_to_donor", "donor_to_base")
    )
    checks = {
        "result_hash_exact": core.sha256(RESULT) == RESULT_SHA256,
        "row_level_recomputation_exact": reports == result["reports"],
        "raw_cell_counts_exact": raw_counts_exact,
        "passing_order_recomputed_exact": passing == result["passing_sites_in_frozen_order"],
        "selected_site_is_frozen_first_pass": result["selected_site"] == passing[0] == "resid8",
        "calls_exact": result["model_forwards"] == core.EXPECTED_FORWARDS and result["model_backwards"] == 0,
        "forbidden_splits_closed": result["forbidden_splits_opened"] == [],
        "checkpoint_bytes_verified": (
            result["checkpoint"]["weights_sha256"] == facade.WEIGHTS_SHA256
            and result["checkpoint"]["config_sha256"] == facade.CONFIG_SHA256
            and result["checkpoint"]["weights_bytes"] == facade.WEIGHTS_BYTES
        ),
        "corrected_run_reproduces_scientific_matrix": result["reports"] == invalid["reports"],
        "all_selected_site_directions_positive": all(
            value > 0
            for split in core.SPLITS for family in core.FAMILIES
            for direction in ("base_to_donor", "donor_to_base")
            for value in result["raw_donorward_movements"]["resid8"][split][family][direction]
        ),
    }
    audit = {
        "rung": 538, "audit": "terminal_cpu_recomputation",
        "checks": checks, "all_checks_pass": all(checks.values()),
        "selected_site": result["selected_site"],
        "passing_sites_in_frozen_order": passing,
        "minimum_selected_site_bootstrap_lower": min(
            reports["resid8"][split][family][direction]["bootstrap95_lower_mean"]
            for split in core.SPLITS for family in core.FAMILIES
            for direction in ("base_to_donor", "donor_to_base")
        ),
        "minimum_selected_site_positive_fraction": min(
            reports["resid8"][split][family][direction]["positive_movement_fraction"]
            for split in core.SPLITS for family in core.FAMILIES
            for direction in ("base_to_donor", "donor_to_base")
        ),
        "interpretation": (
            "Residual block-8 entry is the earliest tested site with a complete-state causal ceiling "
            "for both counterfactual constructions. This is a site ceiling, not a DAS or dimensionality claim."
        ),
        "next_step": "measure invariance and non-opener full-swap controls at resid8 before projector fitting",
    }
    if not audit["all_checks_pass"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
