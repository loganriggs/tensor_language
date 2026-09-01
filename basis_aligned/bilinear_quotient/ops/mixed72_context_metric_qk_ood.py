"""RUNG 342 -- INDEPENDENT-FIT CONTEXT-QK RANK72 + SHIFTED OOD.

Execute the audited context-QK physical harness at rank72 and rescore its exact
receipt under frozen rank72 bars. This removes another 4,505,600 scalars from
rank80, reaching 521,572,662 scalars.

Frozen predictions
------------------
pred_a_rank72_retains_predictive_census_and_certificates:
    Census <=.008, >=48 certificates, surcharge over rank80 <=.005.
pred_b_new_shifted_ood_mean_and_tails_hold:
    WikiText skip240000 mean/p95/max <=.012/.030/.070.
pred_c_context72_identity_price_dataset_and_fresh_hold:
    Split-B context rank72 at 440 maps/layers2--17, active set, dataset, bill,
    saved CEV, and fresh max <=.015 are exact.

Null: census >=.018 or <=38 certificates. Full pass advances a signed gate and
motivates rank64; failure marks the first QK ladder cliff.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed72_context_metric_qk_ood_results.json"
CEV = ROOT / "cev_mixed72_context_metric_qk.pt"
PARENT = ROOT / "mixed80_context_metric_qk_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 72
WIKI_SKIP = 240000
N_ROWS = 120
SCALARS = 521_572_662
BYTES = 1_970_348_652


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert parent["census_damage"] <= .006 and parent["qk_rank"] == 80
        assert 526_078_262 - 440 * (128 + 1152) * 8 == SCALARS
        assert 1_988_371_052 - 4 * 440 * (128 + 1152) * 8 == BYTES
        print("CONTEXT-QK72 OOD | dry run: parent, fit, dataset, bill, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    import mixed88_context_metric_qk_ood as harness

    harness.OUT = OUT
    harness.CEV = CEV
    harness.PARENT = PARENT
    harness.FIT_CACHE = FIT_CACHE
    harness.FIT_SLICE = FIT_SLICE
    harness.LAYERS = LAYERS
    harness.RANK = RANK
    harness.WIKI_SKIP = WIKI_SKIP
    harness.N_ROWS = N_ROWS
    harness.SCALARS = SCALARS
    harness.BYTES = BYTES
    harness.main()

    result = json.loads(OUT.read_text())
    parent = json.loads(PARENT.read_text())
    census = result["census_damage"]
    certificates = result["certificates_valid"]
    surcharge = census - parent["census_damage"]
    pred_a = census <= .008 and certificates >= 48 and surcharge <= .005
    pred_b = (result["shifted_damage_mean"] <= .012
              and result["shifted_damage_row_p95"] <= .030
              and result["shifted_damage_row_max"] <= .070)
    pred_c = (result["dataset_fingerprint"] == "a46124b21ac53738"
              and result["row_construction"] == {
                  "skip_tokens": WIKI_SKIP, "n_rows": N_ROWS, "tokens_per_row": 257}
              and result["fit_rows_half_open"] == list(FIT_SLICE)
              and result["qk_metric"] == "context_rrr"
              and result["qk_context_layers"] == list(LAYERS)
              and result["qk_rank"] == RANK
              and result["qk_factorized_maps"] == 440
              and result["max_fresh_damage"] <= .015
              and result["literal_standalone_scalars"] == SCALARS
              and result["literal_raw_tensor_bytes"] == BYTES
              and result["saved_census_cev_file"] == CEV.name)
    null = census >= .018 or certificates <= 38
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed72_context_metric_qk_ood_complete",
        "rung": 342,
        "claim_level": "physical_context_qk72_census_ood_price_screen",
        "surcharge_vs_context_qk80": surcharge,
        'pred_a_rank72_retains_predictive_census_and_certificates': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_context72_identity_price_dataset_and_fresh_hold': bool(pred_c),
        "null_context_qk72_crosses_cliff": bool(null),
    })
    result.pop("surcharge_vs_context_qk96", None)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"rescored rank72 receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
