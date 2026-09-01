"""RUNG 352 -- CROSS-FAMILY CONTEXT-QK56 + CONTEXT-MLP0 p512.

Compose two independently gated components at a literal 507,253,046 scalars.
The established cross-family law predicts ~1.02x their additive damage.  This
run tests whether that law persists at the lower QK boundary on the next
non-overlapping WikiText-103-train segment.

Frozen predictions
------------------
pred_a_composition_is_near_additive_predictive_and_certified:
    Census <=.024, >=34 certificates, absolute additive residual <=.004,
    and composition ratio <=1.12.
pred_b_new_shifted_ood_mean_and_tails_hold:
    New-corpus mean/p95/max <=.030/.070/.140.
pred_c_exact_dual_context_identity_price_and_fresh_hold:
    QK context56/440 maps, MLP0 context-p512, fits, corpus, saved CEV, active
    set, exact bill, and fresh max <=.035.

Null: census >=.040, shifted mean >=.050, or composition ratio >=1.25.  A pass
advances the same tightened signed gate; no MLP rank tuning follows a miss.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed56_context_qk_mlp0_context_p512_ood_results.json"
CEV = ROOT / "cev_mixed56_context_qk_mlp0_context_p512.pt"
QK_PARENT = ROOT / "mixed56_context_metric_qk_newcorpus_ood_results.json"
MLP_PARENT = ROOT / "mixed104_mlp0_context_metric_input_frontier_ood_results.json"
FIT_CACHE = "fineweb_n192_skip11000.pt"
QK_FIT = (72, 96)
MLP_FIT = (0, 24)
LAYERS = tuple(range(2, 18))
QK_RANK = 56
MLP_RANK = 512
WIKI_SKIP = 280 * 257
N_ROWS = 120
SCALARS = 507_253_046
BYTES = 1_913_070_188


class _JsonProxy:
    """Present the generic harness's historical `448` parent alias as p512."""

    def __init__(self, module):
        self._module = module

    def loads(self, value, *args, **kwargs):
        result = self._module.loads(value, *args, **kwargs)
        if (isinstance(result, dict)
                and result.get("status") == "mixed104_mlp0_context_metric_input_frontier_ood_complete"):
            result = dict(result)
            result["arms"] = dict(result["arms"])
            result["arms"]["448"] = result["arms"][str(MLP_RANK)]
        return result

    def __getattr__(self, name):
        return getattr(self._module, name)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path in (QK_PARENT, MLP_PARENT, ROOT / f".rowcache/{FIT_CACHE}"):
            assert path.exists(), path
        qk = json.loads(QK_PARENT.read_text())
        mlp = json.loads(MLP_PARENT.read_text())["arms"][str(MLP_RANK)]
        assert qk["qk_rank"] == QK_RANK and qk["census_damage"] <= .017
        assert mlp["census_damage"] <= .012 and mlp["certificates_valid"] >= 47
        assert 512_561_462 - 5_308_416 == SCALARS
        assert 1_934_303_852 - 4 * 5_308_416 == BYTES
        assert WIKI_SKIP == 71_960 and WIKI_SKIP + N_ROWS * 257 == 102_800
        print("QK56 + MLP0-p512 | dry run: parents, non-overlap corpus, bill, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    import mixed104_online_cv0_ood as source
    import mixed56_context_metric_qk_newcorpus_ood as population
    import mixed96_context_qk_mlp0_context_p448_ood as harness

    source.wikitext_rows = population._wikitext103_train_rows
    harness.json = _JsonProxy(json)
    harness.OUT = OUT
    harness.CEV = CEV
    harness.QK_PARENT = QK_PARENT
    harness.MLP_PARENT = MLP_PARENT
    harness.FIT_CACHE = FIT_CACHE
    harness.QK_FIT = QK_FIT
    harness.MLP_FIT = MLP_FIT
    harness.LAYERS = LAYERS
    harness.QK_RANK = QK_RANK
    harness.MLP_RANK = MLP_RANK
    harness.WIKI_SKIP = WIKI_SKIP
    harness.N_ROWS = N_ROWS
    harness.SCALARS = SCALARS
    harness.BYTES = BYTES
    harness.main()

    result = json.loads(OUT.read_text())
    census = result["census_damage"]
    certificates = result["certificates_valid"]
    ratio = result["composition_ratio"]
    pred_a = (census <= .024 and certificates >= 34
              and abs(result["composition_residual"]) <= .004 and ratio <= 1.12)
    pred_b = (result["shifted_damage_mean"] <= .030
              and result["shifted_damage_row_p95"] <= .070
              and result["shifted_damage_row_max"] <= .140)
    pred_c = (result["dataset_fingerprint"] == "7dabb830ac9ebb0d"
              and result["row_construction"] == {
                  "skip_tokens": WIKI_SKIP, "n_rows": N_ROWS, "tokens_per_row": 257}
              and result["qk_fit_rows_half_open"] == list(QK_FIT)
              and result["mlp_fit_rows_half_open"] == list(MLP_FIT)
              and result["qk_metric"] == "context_rrr"
              and result["qk_context_layers"] == list(LAYERS)
              and result["qk_rank"] == QK_RANK
              and result["qk_factorized_maps"] == 440
              and result["mlp0_rank"] == MLP_RANK
              and result["max_fresh_damage"] <= .035
              and result["literal_standalone_scalars"] == SCALARS
              and result["literal_raw_tensor_bytes"] == BYTES
              and result["saved_census_cev_file"] == CEV.name)
    null = census >= .040 or result["shifted_damage_mean"] >= .050 or ratio >= 1.25
    for key in list(result):
        if key.startswith("pred_") or key.startswith("null_"):
            result.pop(key)
    result.update({
        "status": "mixed56_context_qk_mlp0_context_p512_ood_complete",
        "rung": 352,
        "claim_level": "lower_boundary_cross_family_composition_physical_ood_price_screen",
        "shifted_corpus": {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "source_rows_half_open": [100_000, 110_000],
            "token_span_half_open": [WIKI_SKIP, WIKI_SKIP + N_ROWS * 257],
            "token_sha256_prefix": "4e1ca0fd7f5c6f00",
        },
        'pred_a_composition_is_near_additive_predictive_and_certified': bool(pred_a),
        'pred_b_new_shifted_ood_mean_and_tails_hold': bool(pred_b),
        'pred_c_exact_dual_context_identity_price_and_fresh_hold': bool(pred_c),
        "null_lower_boundary_cross_family_composition_fails": bool(null),
    })
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("shifted_damage_by_row", "fresh8")}, indent=2), flush=True)
    print(f"rescored QK56+p512 receipt at {OUT}", flush=True)


if __name__ == "__main__":
    main()
