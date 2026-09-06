#!/usr/bin/env python3
"""Engineering-only factor-name mapping repair for v11 transfer v1."""

# BQGATE: EXPERIMENT pred_a_exact_authority_closure_coverage_and_price pred_b_frozen_q8_retains_full_h3_response pred_c_frozen_causal_suffix_closes_q8 pred_d_frozen_factor_signature_recurs pred_e_pre_subject_value_is_causally_zero
import hashlib
from pathlib import Path

import run_temporal_auxiliary_will_had_h3_rank8_v11_causal_suffix_transfer_v1 as impl

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_v11_causal_suffix_transfer_v2.json"
V1_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_v11_causal_suffix_transfer_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_v11_causal_suffix_transfer_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_v11_causal_suffix_transfer_v2"
EXPECTED_PRIOR = "c12d1355b3cf42e5e7ea5173ba402c37354b9be81eb44c5ac6dcc63e421041f5"
EXPECTED_V1_RESULT = "c34ef54c5cc8b486e8d53de240b88b3228646fee3429055b5e08792248bd7938"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha(PRIOR) != EXPECTED_PRIOR or sha(V1_RESULT) != EXPECTED_V1_RESULT:
        raise RuntimeError("v2 repair authority changed")
    impl.PRIOR = PRIOR
    impl.OUT = OUT
    impl.CANDIDATE_ID = CANDIDATE_ID
    impl.EXPECTED = dict(impl.EXPECTED, prior=EXPECTED_PRIOR)
    impl.FACTORS = ("base_pattern_on_value_change", "pattern_on_base_value",
                    "pattern_value_interaction")
    impl.main()


if __name__ == "__main__":
    main()
