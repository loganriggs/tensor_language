"""RUNG 372 -- ONE-STEP DISTRIBUTED-CUT FALSIFIER.

Use the already-frozen rung366 selection rule, extended from two to exactly
three choices, to install split-B p768 input programs at layers 4, 0, and 2
on context-QK64.  This is one prospective continuation, not a subset sweep.

Frozen predictions
------------------
pred_a: census <=.017 and >=40 certificates.  Only >=43 is a frontier pass.
pred_b: shifted mean/p95/max <=.030/.070/.140 and fresh max <=.035.
pred_c: exact selection, program identities, population, CEV, and price.

Null: census >=.027 or <=32 certificates.  No fourth layer, rank, selection
rule, or bar tuning follows this result tonight.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
sys.path.insert(0, str(ROOT / "ops"))

# Static registration mirror for ops/gate.py; the imported harness evaluates
# these three clauses under the rung-specific thresholds set below.
REGISTERED_PREDICATES = {
    'pred_a_three_cut_census_and_certificate_signal': "census<=.017 and certificates>=40",
    'pred_b_three_cut_shifted_and_fresh_transfer': "OOD mean/p95/max and fresh remain under frozen bars",
    'pred_c_three_cut_selection_identity_and_price': "top-three rule, maps, corpus, CEV, and price are exact",
}

import mixed64_context_qk_mlp04_context_p768_ood as harness


def main() -> None:
    harness.OUT = ROOT / "mixed64_context_qk_mlp042_context_p768_ood_results.json"
    harness.CEV = ROOT / "cev_mixed64_context_qk_mlp042_context_p768.pt"
    harness.LAYERS = (0, 2, 4)
    harness.SELECT_COUNT = 3
    harness.EXPECTED_SELECTED = (4, 0, 2)
    harness.WIKI_SKIP = 285_784
    harness.WIKI_STOP = 316_624
    harness.SCALARS = 509_104_438
    harness.BYTES = 1_920_475_756
    harness.CENSUS_MAX = .017
    harness.CERTIFICATE_MIN = 40
    harness.OOD_MEAN_MAX = .030
    harness.OOD_P95_MAX = .070
    harness.OOD_MAX = .140
    harness.FRESH_MAX = .035
    harness.NULL_CENSUS = .027
    harness.NULL_CERTIFICATES = 32
    harness.main()


if __name__ == "__main__":
    main()
