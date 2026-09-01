"""RUNG 349 -- TIGHTENED SIGNED a16 ADOPTION GATE FOR CONTEXT-QK56.

The signed signature erodes monotonically from QK96 to QK64, while the old
bars are only catastrophic guards.  Before observing rank56 intervention data,
freeze a tighter, potentially binding gate.

Frozen predictions
------------------
pred_a_live_baseline_price_and_identity:
    Baseline <=.017/42 certificates, new OOD max <=.075, fresh <=.012,
    exact context-RRR rank56/440-map identity and 512,561,462-scalar bill.
pred_b_tight_signed_effect_vector_transfers:
    Cosine >=.98, normalized vector error <=.30, norm ratio in [.90,1.15].
pred_c_tight_circuit_effect_profile_transfers:
    Collateral Spearman >=.98 and a16-own median ratio in [.90,1.15].

Null: cosine <.70 or collateral Spearman <.75.  A full tightened pass formally
adopts rank56 and licenses exactly one rank48 physical cliff probe.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "a16_transfer_mixed56_context_metric_qk_results.json"
BASE_RESULT = ROOT / "mixed56_context_metric_qk_newcorpus_ood_results.json"
BASE_CEV = ROOT / "cev_mixed56_context_metric_qk.pt"
COMP_KO = ROOT / "cev_a16ko_mixed56_context_metric_qk.pt"
NATIVE_KO = ROOT / "cev_a16ko_native_context_qk56_gate.pt"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 56
SCALARS = 512_561_462
BYTES = 1_934_303_852


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (BASE_RESULT, BASE_CEV, ROOT / "circuits/BATTERY.json",
                     ROOT / f".rowcache/{FIT_CACHE}"):
            assert path.exists(), path
        baseline = json.loads(BASE_RESULT.read_text())
        assert baseline["census_damage"] <= .017
        assert baseline["certificates_valid"] >= 42
        assert baseline["literal_standalone_scalars"] == SCALARS
        print("A16 TRANSFER CONTEXT-QK56 | dry run: baseline, fit, bill, tight bars valid")
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
    pred_a = (result["unablated_census_damage"] <= .017
              and result["unablated_certificates_valid"] >= 42
              and result["unablated_shifted_damage_row_max"] <= .075
              and result["live_unablated_fresh_damage"] <= .012
              and result["qk_metric"] == "context_rrr"
              and result["qk_context_layers"] == list(LAYERS)
              and result["qk_rank"] == RANK
              and result["qk_factorized_maps"] == 440
              and result["literal_standalone_scalars"] == SCALARS
              and result["literal_raw_tensor_bytes"] == BYTES)
    pred_b = (result["effect_cosine"] >= .98
              and result["effect_normalized_error"] <= .30
              and .90 <= result["effect_norm_ratio"] <= 1.15)
    pred_c = (result["collateral_spearman"] >= .98
              and .90 <= result["own_effect_median_ratio"] <= 1.15)
    null = result["effect_cosine"] < .70 or result["collateral_spearman"] < .75
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "a16_transfer_mixed56_context_metric_qk_complete",
        "rung": 349,
        "claim_level": "tight_direct_signed_a16_context_qk56_adoption_gate",
        'pred_a_live_baseline_price_and_identity': bool(pred_a),
        'pred_b_tight_signed_effect_vector_transfers': bool(pred_b),
        'pred_c_tight_circuit_effect_profile_transfers': bool(pred_c),
        "null_signed_context_qk56_transport_fails": bool(null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "own_effect_ratios"}, indent=2), flush=True)
    print(f"rescored rank56 tight signed receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
