"""RUNG 341 -- SIGNED a16 ADOPTION GATE FOR CONTEXT-QK80.

Execute the audited signed context-QK harness at rank80 and rescore the exact
receipt under rank80-specific frozen baseline, identity, and price bars.

Frozen predictions
------------------
pred_a_live_baseline_price_and_identity:
    Baseline <=.006/52 certificates, OOD max <=.040, fresh <=.010, exact
    context-RRR rank80/440-map identity and 526,078,262-scalar bill.
pred_b_signed_effect_vector_transfers:
    Cosine >=.90 and normalized vector error <=.60.
pred_c_circuit_effect_profile_transfers:
    Collateral Spearman >=.90 and a16-own median ratio in [.60,1.40].

Null: cosine <.70 or collateral Spearman <.75. Full pass formally adopts the
smallest high-fidelity QK ladder point and motivates rank72.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "a16_transfer_mixed80_context_metric_qk_results.json"
BASE_RESULT = ROOT / "mixed80_context_metric_qk_ood_results.json"
BASE_CEV = ROOT / "cev_mixed80_context_metric_qk.pt"
COMP_KO = ROOT / "cev_a16ko_mixed80_context_metric_qk.pt"
NATIVE_KO = ROOT / "cev_a16ko_native_context_qk80_gate.pt"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 80
SCALARS = 526_078_262
BYTES = 1_988_371_052


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (BASE_RESULT, BASE_CEV, ROOT / "circuits/BATTERY.json",
                     ROOT / f".rowcache/{FIT_CACHE}"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert baseline["census_damage"] <= .006
        assert baseline["certificates_valid"] >= 52
        assert baseline["literal_standalone_scalars"] == SCALARS
        print("A16 TRANSFER CONTEXT-QK80 | dry run: baseline, fit, bill, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    import a16_transfer_mixed88_context_metric_qk as harness

    harness.OUT = OUT
    harness.BASE_RESULT = BASE_RESULT
    harness.BASE_CEV = BASE_CEV
    harness.COMP_KO = COMP_KO
    harness.NATIVE_KO = NATIVE_KO
    harness.FIT_CACHE = FIT_CACHE
    harness.FIT_SLICE = FIT_SLICE
    harness.LAYERS = LAYERS
    harness.RANK = RANK
    harness.SCALARS = SCALARS
    harness.BYTES = BYTES
    harness.main()

    result = json.loads(OUT.read_text())
    pred_a = (result["unablated_census_damage"] <= .006
              and result["unablated_certificates_valid"] >= 52
              and result["unablated_shifted_damage_row_max"] <= .040
              and result["live_unablated_fresh_damage"] <= .010
              and result["qk_metric"] == "context_rrr"
              and result["qk_context_layers"] == list(LAYERS)
              and result["qk_rank"] == RANK
              and result["qk_factorized_maps"] == 440
              and result["literal_standalone_scalars"] == SCALARS
              and result["literal_raw_tensor_bytes"] == BYTES)
    pred_b = (result["effect_cosine"] >= .90
              and result["effect_normalized_error"] <= .60)
    pred_c = (result["collateral_spearman"] >= .90
              and .60 <= result["own_effect_median_ratio"] <= 1.40)
    null = result["effect_cosine"] < .70 or result["collateral_spearman"] < .75
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "a16_transfer_mixed80_context_metric_qk_complete",
        "rung": 341,
        "claim_level": "direct_signed_a16_context_qk80_adoption_gate",
        'pred_a_live_baseline_price_and_identity': bool(pred_a),
        'pred_b_signed_effect_vector_transfers': bool(pred_b),
        'pred_c_circuit_effect_profile_transfers': bool(pred_c),
        "null_signed_context_qk80_transport_fails": bool(null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print(f"rescored rank80 signed receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
