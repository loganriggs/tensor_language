"""RUNG 345 -- INDEPENDENT-FIT CONTEXT-QK RANK64 + SHIFTED OOD.

Execute the audited context-QK physical harness at rank64 and rescore its exact
receipt under frozen rank64 bars. This reaches 517,067,062 scalars.

Frozen predictions
------------------
pred_a_rank64_retains_predictive_census_and_certificates:
    Census <=.012, >=44 certificates, surcharge over rank72 <=.007.
pred_b_new_shifted_ood_mean_and_tails_hold:
    WikiText terminal skip270840 n56 mean/p95/max <=.018/.045/.100.
pred_c_context64_identity_price_dataset_and_fresh_hold:
    Split-B context rank64 at 440 maps/layers2--17, active set, dataset, bill,
    saved CEV, and fresh max <=.020 are exact.

Null: census >=.025 or <=30 certificates. Full pass advances a signed gate and
motivates rank56; failure marks the first QK ladder cliff.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed64_context_metric_qk_ood_results.json"
CEV = ROOT / "cev_mixed64_context_metric_qk.pt"
PARENT = ROOT / "mixed72_context_metric_qk_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 64
WIKI_SKIP = 270840
N_ROWS = 56
SCALARS = 517_067_062
BYTES = 1_952_326_252


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert parent["census_damage"] <= .008 and parent["qk_rank"] == 72
        assert 521_572_662 - 440 * (128 + 1152) * 8 == SCALARS
        assert 1_970_348_652 - 4 * 440 * (128 + 1152) * 8 == BYTES
        print("CONTEXT-QK64 OOD | dry run: parent, fit, dataset, bill, bars valid")
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
    pred_a = census <= .012 and certificates >= 44 and surcharge <= .007
    pred_b = (result["shifted_damage_mean"] <= .018
              and result["shifted_damage_row_p95"] <= .045
              and result["shifted_damage_row_max"] <= .100)
    pred_c = (result["dataset_fingerprint"] == "a46124b21ac53738"
              and result["row_construction"] == {
                  "skip_tokens": WIKI_SKIP, "n_rows": N_ROWS, "tokens_per_row": 257}
              and result["fit_rows_half_open"] == list(FIT_SLICE)
              and result["qk_metric"] == "context_rrr"
              and result["qk_context_layers"] == list(LAYERS)
              and result["qk_rank"] == RANK
              and result["qk_factorized_maps"] == 440
              and result["max_fresh_damage"] <= .020
              and result["literal_standalone_scalars"] == SCALARS
              and result["literal_raw_tensor_bytes"] == BYTES
              and result["saved_census_cev_file"] == CEV.name)
    null = census >= .025 or certificates <= 30
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed64_context_metric_qk_ood_complete",
        "rung": 345,
        "claim_level": "physical_context_qk64_census_ood_price_screen",
        "surcharge_vs_context_qk72": surcharge,
        'pred_a_rank64_retains_predictive_census_and_certificates': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_context64_identity_price_dataset_and_fresh_hold': bool(pred_c),
        "null_context_qk64_crosses_cliff": bool(null),
    })
    result.pop("surcharge_vs_context_qk96", None)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"rescored rank64 receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
