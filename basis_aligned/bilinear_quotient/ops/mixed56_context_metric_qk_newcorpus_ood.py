"""RUNG 348 -- CONTEXT-QK RANK56 ON A NEW SHIFTED POPULATION.

Continue the fixed split-B context-QK ladder by eight dimensions.  WikiText-2
test is exhausted, and WikiText-103 test contains the same text, so the OOD
population is explicitly switched to WikiText-103 raw TRAIN rows
[100000,110000).  Its first 120 non-overlapping 257-token chunks are evaluation
only; no statistic from this stream fits a program tensor.

Frozen predictions
------------------
pred_a_rank56_retains_predictive_census_and_certificates:
    Census <=.017, >=42 certificates, surcharge over rank64 <=.008.
pred_b_new_shifted_ood_mean_and_tails_hold:
    New-corpus mean/p95/max <=.025/.060/.120.
pred_c_context56_identity_price_dataset_and_fresh_hold:
    Split-B context rank56 at 440 maps/layers2--17, exact new-corpus receipt,
    active set, saved CEV, bill, and fresh max <=.025.

Null: census >=.030 or <=28 certificates.  A full pass advances a signed gate;
failure marks the first QK ladder ledge.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed56_context_metric_qk_newcorpus_ood_results.json"
CEV = ROOT / "cev_mixed56_context_metric_qk.pt"
PARENT = ROOT / "mixed64_context_metric_qk_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
FIT_SLICE = (72, 96)
LAYERS = tuple(range(2, 18))
RANK = 56
N_ROWS = 120
WIKI_TRAIN_ROWS = (100_000, 110_000)
DATASET_FINGERPRINT = "7dabb830ac9ebb0d"
TOKEN_COUNT = 675_457
TOKEN_HASH = "4e1ca0fd7f5c6f00"
SCALARS = 512_561_462
BYTES = 1_934_303_852


def _wikitext103_train_rows(n: int = N_ROWS, width: int = 257,
                            skip: int = 0) -> tuple[torch.Tensor, str, int]:
    from datasets import load_dataset
    import tiktoken

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    lo, hi = WIKI_TRAIN_ROWS
    text = "\n\n".join(row["text"] for row in dataset.select(range(lo, hi))
                       if row["text"].strip())
    tokens = tiktoken.get_encoding("gpt2").encode_ordinary(text)
    digest = hashlib.sha256(b"".join(struct.pack("<I", token) for token in tokens)).hexdigest()[:16]
    stop = skip + n * width
    if (str(dataset._fingerprint) != DATASET_FINGERPRINT or len(tokens) != TOKEN_COUNT
            or digest != TOKEN_HASH or len(tokens) < stop):
        raise RuntimeError("WikiText-103 train population identity changed")
    rows = torch.tensor(tokens[skip:stop], dtype=torch.long).reshape(n, width)
    return rows, str(dataset._fingerprint), len(tokens)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PARENT.exists() and (ROOT / f".rowcache/{FIT_CACHE}").exists()
        parent = json.loads(PARENT.read_text())
        assert parent["census_damage"] <= .012 and parent["qk_rank"] == 64
        assert 517_067_062 - 440 * (128 + 1152) * 8 == SCALARS
        assert 1_952_326_252 - 4 * 440 * (128 + 1152) * 8 == BYTES
        assert WIKI_TRAIN_ROWS == (100_000, 110_000) and N_ROWS * 257 < TOKEN_COUNT
        print("CONTEXT-QK56 NEW-CORPUS OOD | dry run: parent, fit, dataset, bill, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    import mixed104_online_cv0_ood as source
    import mixed88_context_metric_qk_ood as harness

    source.wikitext_rows = _wikitext103_train_rows
    harness.OUT = OUT
    harness.CEV = CEV
    harness.PARENT = PARENT
    harness.FIT_CACHE = FIT_CACHE
    harness.FIT_SLICE = FIT_SLICE
    harness.LAYERS = LAYERS
    harness.RANK = RANK
    harness.WIKI_SKIP = 0
    harness.N_ROWS = N_ROWS
    harness.SCALARS = SCALARS
    harness.BYTES = BYTES
    harness.main()

    result = json.loads(OUT.read_text())
    parent = json.loads(PARENT.read_text())
    census = result["census_damage"]
    certificates = result["certificates_valid"]
    surcharge = census - parent["census_damage"]
    pred_a = census <= .017 and certificates >= 42 and surcharge <= .008
    pred_b = (result["shifted_damage_mean"] <= .025
              and result["shifted_damage_row_p95"] <= .060
              and result["shifted_damage_row_max"] <= .120)
    pred_c = (result["dataset_fingerprint"] == DATASET_FINGERPRINT
              and result["row_construction"] == {
                  "skip_tokens": 0, "n_rows": N_ROWS, "tokens_per_row": 257}
              and result["fit_rows_half_open"] == list(FIT_SLICE)
              and result["qk_metric"] == "context_rrr"
              and result["qk_context_layers"] == list(LAYERS)
              and result["qk_rank"] == RANK
              and result["qk_factorized_maps"] == 440
              and result["max_fresh_damage"] <= .025
              and result["literal_standalone_scalars"] == SCALARS
              and result["literal_raw_tensor_bytes"] == BYTES
              and result["saved_census_cev_file"] == CEV.name)
    null = census >= .030 or certificates <= 28
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed56_context_metric_qk_newcorpus_ood_complete",
        "rung": 348,
        "claim_level": "physical_context_qk56_newcorpus_census_ood_price_screen",
        "shifted_corpus": {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "source_rows_half_open": list(WIKI_TRAIN_ROWS),
            "token_count": TOKEN_COUNT,
            "token_sha256_prefix": TOKEN_HASH,
        },
        "surcharge_vs_context_qk64": surcharge,
        'pred_a_rank56_retains_predictive_census_and_certificates': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_context56_identity_price_dataset_and_fresh_hold': bool(pred_c),
        "null_context_qk56_crosses_cliff": bool(null),
    })
    result.pop("surcharge_vs_context_qk96", None)
    result.pop("surcharge_vs_context_qk72", None)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"rescored rank56 new-corpus receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
