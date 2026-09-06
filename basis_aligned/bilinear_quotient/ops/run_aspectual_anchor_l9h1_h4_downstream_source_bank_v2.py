#!/usr/bin/env python3
# BQGATE: A-E and every causal bar are inherited unchanged from v1; only BF16 closure is repaired.
"""Sole numerical-instrument correction for the L9H1/H4 source bank."""

from pathlib import Path

import run_aspectual_anchor_l9h1_h4_downstream_source_bank_v1 as v1


ROOT = Path(__file__).resolve().parent.parent
v1.PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_l9h1_h4_downstream_source_bank_v2.json"
v1.OUT = ROOT / "circuits/followups/aspectual_anchor_l9h1_h4_downstream_source_bank_v2_result.json"
v1.CANDIDATE_ID = "aspectual_anchor.has_vs_had.l9h1_h4_downstream_source_bank_v2"
v1.EXPECTED_PRIOR_SHA256 = "e538aba6f1a2826f935abde03e5c25e086006a8b01773b9751332b9895cbf167"
v1.IDENTITY_TOLERANCE = 0.125

REGISTERED_PREDICTIONS = {
    "pred_a_bf16_source_closure": "all-source and full-pair scored logits agree within 0.125",
    "pred_b_period_determiner_compression": "the two-source core retains at least 60 percent",
    "pred_c_last_extension": "the three-source bank retains at least 80 percent",
    "pred_d_cue_self_negative_control": "cue plus self retains at most 25 percent",
    "pred_e_exact_coverage": "all eight arms and all 64 rows are complete",
}


def main() -> None:
    v1.main()


if __name__ == "__main__":
    main()
