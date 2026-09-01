"""RUNG 377 -- TWO-BYTE QK72 + MLP{4,0}@P768 MID-TIER.

Frozen predictions: census<=.012 and >=48 certificates; shifted
mean/p95/max<=.025/.060/.120 and fresh<=.025; exact selection, fits, maps,
dtypes, CEV, corpus, and 516,264,246/1,032,528,492 bill.  Null: census>=.022
or <=40 certificates.  No precision/rank/layer tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
sys.path.insert(0, str(ROOT / "ops"))
REGISTERED_PREDICATES = {
    'pred_a_mid_tier_two_byte_census_and_certificates': "census<=.012 and certificates>=48",
    'pred_b_mid_tier_two_byte_shifted_and_fresh': "OOD and fresh remain under frozen limits",
    'pred_c_mid_tier_two_byte_identity_and_bill': "selection, maps, dtypes, corpus, CEV, and bill exact",
}

import mixed64_bf16_qk_fp16_mlp04_context_p768_ood as harness


def main() -> None:
    harness.OUT = ROOT / "mixed72_bf16_qk_fp16_mlp04_context_p768_ood_results.json"
    harness.CEV = ROOT / "cev_mixed72_bf16_qk_fp16_mlp04_context_p768.pt"
    harness.PARENT = ROOT / "mixed72_context_qk_mlp04_context_p768_ood_results.json"
    harness.WIKI_SKIP = 347_464
    harness.N_ROWS = 120
    harness.SCALARS = 516_264_246
    harness.BYTES = 1_032_528_492
    harness.QK_RANK = 72
    harness.RUNG = 377
    harness.STATUS = "mixed72_bf16_qk_fp16_mlp04_context_p768_ood_complete"
    harness.CLAIM_LEVEL = "physical_two_byte_selected_mlp_qk72_mid_fidelity_tier"
    harness.CENSUS_MAX = .012
    harness.CERTIFICATE_MIN = 48
    harness.OOD_MEAN_MAX = .025
    harness.OOD_P95_MAX = .060
    harness.OOD_MAX = .120
    harness.FRESH_MAX = .025
    harness.NULL_CENSUS = .022
    harness.NULL_CERTIFICATES = 40
    harness.main()


if __name__ == "__main__":
    main()
