"""RUNG 373 -- PROSPECTIVE QK72 + MLP{4,0}@P768 MID-FIDELITY TIER.

Frozen predictions
------------------
pred_a: census <=.011 and >=48 certificates.
pred_b: shifted mean/p95/max <=.025/.060/.120 and fresh max <=.025.
pred_c: exact frozen selection, programs, corpus, CEV, and literal bill.

Null: census >=.022 or <=40 certificates.  No QK-rank interpolation follows.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
sys.path.insert(0, str(ROOT / "ops"))
REGISTERED_PREDICATES = {
    'pred_a_mid_tier_census_and_certificates': "census<=.011 and certificates>=48",
    'pred_b_mid_tier_shifted_and_fresh_transfer': "OOD and fresh remain under the frozen limits",
    'pred_c_mid_tier_selection_identity_and_price': "selection, maps, corpus, CEV, and price are exact",
}

import mixed64_context_qk_mlp04_context_p768_ood as harness


def main() -> None:
    harness.OUT = ROOT / "mixed72_context_qk_mlp04_context_p768_ood_results.json"
    harness.CEV = ROOT / "cev_mixed72_context_qk_mlp04_context_p768.pt"
    harness.QK_RANK = 72
    harness.LAYERS = (0, 4)
    harness.SELECT_COUNT = 2
    harness.EXPECTED_SELECTED = (4, 0)
    harness.WIKI_SKIP = 316_624
    harness.WIKI_STOP = 347_464
    harness.SCALARS = 516_264_246
    harness.BYTES = 1_949_114_988
    harness.CENSUS_MAX = .011
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
