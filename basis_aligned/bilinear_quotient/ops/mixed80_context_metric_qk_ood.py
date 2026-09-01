"""RUNG 339 -- INDEPENDENT-FIT CONTEXT-QK RANK80 + SHIFTED OOD.

Execute the audited rank-QK physical harness at rank80, then rescore its exact
receipt under these rank80-specific frozen bars. This removes another 4,505,600
scalars relative to rank88, reaching 526,078,262 scalars.

Frozen predictions
------------------
pred_a_rank80_retains_high_fidelity_census_and_certificates:
    Census <=.006, >=54 certificates, surcharge over rank88 <=.004.
pred_b_new_shifted_ood_mean_and_tails_hold:
    WikiText skip220000 mean/p95/max <=.010/.025/.060.
pred_c_context80_identity_price_dataset_and_fresh_hold:
    Split-B context rank80 at 440 maps/layers2--17, active set, dataset, bill,
    saved CEV, and fresh max <=.015 are exact.

Null: census >=.015 or <=42 certificates. Full pass advances a signed gate and
motivates rank72/64; failure marks the r80 side of the QK ledge.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed80_context_metric_qk_ood_results.json"
CEV = ROOT / "cev_mixed80_context_metric_qk.pt"
PARENT = ROOT / "mixed88_context_metric_qk_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 80
WIKI_SKIP = 220000
N_ROWS = 120
SCALARS = 526_078_262
BYTES = 1_988_371_052


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert parent["census_damage"] <= .004 and parent["qk_rank"] == 88
        assert 530_583_862 - 440 * (128 + 1152) * 8 == SCALARS
        assert 2_006_393_452 - 4 * 440 * (128 + 1152) * 8 == BYTES
        print("CONTEXT-QK80 OOD | dry run: parent, fit, dataset, bill, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    import mixed88_context_metric_qk_ood as harness

    # The imported harness owns only physical construction/evaluation. All
    # experiment identity and frozen scoring below are rank80-specific.
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
    pred_a = census <= .006 and certificates >= 54 and surcharge <= .004
    pred_b = (result["shifted_damage_mean"] <= .010
              and result["shifted_damage_row_p95"] <= .025
              and result["shifted_damage_row_max"] <= .060)
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
    null = census >= .015 or certificates <= 42
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed80_context_metric_qk_ood_complete",
        "rung": 339,
        "claim_level": "physical_context_qk80_census_ood_price_screen",
        "surcharge_vs_context_qk88": surcharge,
        'pred_a_rank80_retains_high_fidelity_census_and_certificates': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_context80_identity_price_dataset_and_fresh_hold': bool(pred_c),
        "null_context_qk80_crosses_cliff": bool(null),
    })
    result.pop("surcharge_vs_context_qk96", None)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"rescored rank80 receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
