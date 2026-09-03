#!/usr/bin/env python3
"""CPU recomputation of every R539 control-ceiling statistic."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
sys.path.insert(0, str(ROOT / "ops"))
import pending_opener_control_ceilings_rung539 as core  # noqa: E402

RESULT = ROOT / "pending_opener_control_ceilings_rung539_results.json"
OUT = ROOT / "pending_opener_control_ceilings_rung539_terminal_audit.json"
RESULT_SHA256 = "d0cf53b6e26df46b113a9a8bf18bc9b86222536b3d2621ff90d690240a8e3a0c"


def main() -> None:
    result = json.loads(RESULT.read_text())
    seed, recomputed = core.SEED, {}
    for split in core.SPLITS:
        recomputed[split] = {}
        for family in core.FAMILIES:
            recomputed[split][family] = {}
            for direction in ("base_to_donor", "donor_to_base"):
                raw = result["raw_sufficient_statistics"][split][family][direction]
                recomputed[split][family][direction] = core.summarize(
                    raw["endpoint_change"], raw["logit_rms"], seed)
                seed += 1
            recomputed[split][family]["causally_testable"] = all(
                recomputed[split][family][direction]["bootstrap95_lower_mean_absolute"] > 0.05
                and recomputed[split][family][direction]["mean_full_vocabulary_logit_rms"] > 0.01
                for direction in ("base_to_donor", "donor_to_base")
            )
    checks = {
        "result_hash_exact": core.sha256(RESULT) == RESULT_SHA256,
        "row_level_recomputation_exact": recomputed == result["reports"],
        "calls_exact": result["model_forwards"] == core.EXPECTED_FORWARDS and result["model_backwards"] == 0,
        "splits_closed": result["forbidden_splits_opened"] == [],
        "checkpoint_exact": result["checkpoint_weights_sha256"]
        == "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "both_control_families_live": result["pred_b_surface_invariance_causally_testable"] is True
        and result["pred_c_nonopener_control_causally_testable"] is True,
    }
    audit = {
        "rung": 539, "checks": checks, "all_checks_pass": all(checks.values()),
        "minimum_endpoint_bootstrap_lower": min(
            recomputed[split][family][direction]["bootstrap95_lower_mean_absolute"]
            for split in core.SPLITS for family in core.FAMILIES
            for direction in ("base_to_donor", "donor_to_base")
        ),
        "minimum_full_vocabulary_logit_rms": min(
            recomputed[split][family][direction]["mean_full_vocabulary_logit_rms"]
            for split in core.SPLITS for family in core.FAMILIES
            for direction in ("base_to_donor", "donor_to_base")
        ),
        "interpretation": (
            "Both answer-preserving families have live full-state effects at resid8, so a later "
            "near-zero projector effect is a falsifiable selectivity claim rather than a zero-ceiling artifact."
        ),
    }
    if not audit["all_checks_pass"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
