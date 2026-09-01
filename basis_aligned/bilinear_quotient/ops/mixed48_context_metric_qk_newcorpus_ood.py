"""RUNG 351 -- FINAL CONTEXT-QK RANK48 PHYSICAL CLIFF PROBE.

The tightened rank56 signed gate passed with little slack.  Remove one final
fixed block of eight Q/K ranks and evaluate the next non-overlapping 120 rows
of the frozen WikiText-103-train token stream.  No rank40 continuation is
licensed by this result.

Frozen predictions
------------------
pred_a_rank48_retains_useful_census_and_certificates:
    Census <=.024, >=34 certificates, surcharge over rank56 <=.010.
pred_b_new_shifted_ood_mean_and_tails_hold:
    New-corpus mean/p95/max <=.035/.080/.150.
pred_c_context48_identity_price_dataset_and_fresh_hold:
    Split-B context rank48 at 440 maps/layers2--17, exact non-overlap corpus
    receipt, active set, saved CEV, bill, and fresh max <=.035.

Null: census >=.040 or <=24 certificates.  A full physical pass may receive
the same tightened signed gate; any miss stops the pure-QK ladder.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed48_context_metric_qk_newcorpus_ood_results.json"
CEV = ROOT / "cev_mixed48_context_metric_qk.pt"
PARENT = ROOT / "mixed56_context_metric_qk_newcorpus_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 48
N_ROWS = 120
WIKI_SKIP = 160 * 257
DATASET_FINGERPRINT = "7dabb830ac9ebb0d"
TOKEN_COUNT = 675_457
TOKEN_HASH = "4e1ca0fd7f5c6f00"
SCALARS = 508_055_862
BYTES = 1_916_281_452


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert parent["census_damage"] <= .017 and parent["qk_rank"] == 56
        assert 512_561_462 - 440 * (128 + 1152) * 8 == SCALARS
        assert 1_934_303_852 - 4 * 440 * (128 + 1152) * 8 == BYTES
        assert WIKI_SKIP == 41_120 and WIKI_SKIP + N_ROWS * 257 == 71_960
        print("CONTEXT-QK48 FINAL CLIFF PROBE | dry run: parent, corpus, bill, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    import mixed104_online_cv0_ood as source
    import mixed56_context_metric_qk_newcorpus_ood as population
    import mixed88_context_metric_qk_ood as harness

    source.wikitext_rows = population._wikitext103_train_rows
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
    pred_a = census <= .024 and certificates >= 34 and surcharge <= .010
    pred_b = (result["shifted_damage_mean"] <= .035
              and result["shifted_damage_row_p95"] <= .080
              and result["shifted_damage_row_max"] <= .150)
    pred_c = (result["dataset_fingerprint"] == DATASET_FINGERPRINT
              and result["row_construction"] == {
                  "skip_tokens": WIKI_SKIP, "n_rows": N_ROWS, "tokens_per_row": 257}
              and result["fit_rows_half_open"] == list(FIT_SLICE)
              and result["qk_metric"] == "context_rrr"
              and result["qk_context_layers"] == list(LAYERS)
              and result["qk_rank"] == RANK
              and result["qk_factorized_maps"] == 440
              and result["max_fresh_damage"] <= .035
              and result["literal_standalone_scalars"] == SCALARS
              and result["literal_raw_tensor_bytes"] == BYTES
              and result["saved_census_cev_file"] == CEV.name)
    null = census >= .040 or certificates <= 24
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed48_context_metric_qk_newcorpus_ood_complete",
        "rung": 351,
        "claim_level": "final_physical_context_qk48_newcorpus_cliff_probe",
        "shifted_corpus": {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "source_rows_half_open": [100_000, 110_000],
            "token_span_half_open": [WIKI_SKIP, WIKI_SKIP + N_ROWS * 257],
            "token_count": TOKEN_COUNT,
            "token_sha256_prefix": TOKEN_HASH,
        },
        "surcharge_vs_context_qk56": surcharge,
        'pred_a_rank48_retains_useful_census_and_certificates': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_context48_identity_price_dataset_and_fresh_hold': bool(pred_c),
        "null_context_qk48_crosses_cliff": bool(null),
        "stop_rule": "no_rank40_without_new_theory",
    })
    result.pop("surcharge_vs_context_qk96", None)
    result.pop("surcharge_vs_context_qk64", None)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"rescored final rank48 receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
