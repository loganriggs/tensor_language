"""RUNG 378 -- ORIGINAL-NATIVE SIGNED GATE FOR THE 50-CERT MID-TIER.

Conditional on every rung377 positive.  Frozen bars: exact baseline and bill;
cosine>=.985, normalized error<=.25, norm ratio[.90,1.15], collateral
Spearman>=.98, own median[.90,1.15]. Null cosine<.70 or rho<.75. No tuning.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
sys.path.insert(0, str(ROOT / "ops"))
REGISTERED_PREDICATES = {
    'pred_a_mid_tier_baseline_identity_and_bill': "rung377 baseline, maps, dtypes, and bill reproduce",
    'pred_b_mid_tier_original_native_signed_effect': "cosine>=.985, error<=.25, norm ratio in range",
    'pred_c_mid_tier_circuit_profile': "collateral rho>=.98 and own median in range",
}

import a16_transfer_mixed64_bf16_qk_fp16_mlp04_p768 as harness


def main() -> None:
    harness.OUT = ROOT / "a16_transfer_mixed72_bf16_qk_fp16_mlp04_p768_results.json"
    harness.BASE_RESULT = ROOT / "mixed72_bf16_qk_fp16_mlp04_context_p768_ood_results.json"
    harness.BASE_CEV = ROOT / "cev_mixed72_bf16_qk_fp16_mlp04_context_p768.pt"
    harness.COMP_KO = ROOT / "cev_a16ko_mixed72_bf16_qk_fp16_mlp04_p768.pt"
    harness.QK_RANK = 72
    harness.SCALARS = 516_264_246
    harness.BYTES = 1_032_528_492
    harness.RUNG = 378
    harness.STATUS = "a16_transfer_mixed72_bf16_qk_fp16_mlp04_p768_complete"
    harness.CLAIM_LEVEL = "original_native_signed_two_byte_qk72_mlp04_mid_fidelity_gate"
    harness.CENSUS_MAX = .012
    harness.CERTIFICATE_MIN = 48
    harness.SHIFTED_MAX = .120
    harness.FRESH_MAX = .025
    harness.COSINE_MIN = .985
    harness.ERROR_MAX = .25
    harness.RHO_MIN = .98
    harness.main()


if __name__ == "__main__":
    main()
