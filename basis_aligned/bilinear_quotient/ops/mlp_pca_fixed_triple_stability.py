"""RUNG 309 -- LARGE-SAMPLE STABILITY OF THE FIXED-SPACED PCA TRIPLE.

The prospectively frozen {0,8,17} control in rung 308 added +.0725 FineWeb and
+.0334 WikiText while saving 11,501,568 scalars.  Confirm that exact object on
large untouched populations before any whole-program integration.

Fit the unchanged rank-256 PCA bases on the frozen 16 FineWeb fit rows. Evaluate
only layers {0,8,17} on:
  * fineweb_n192_skip7000 rows 16:192 (176 rows),
  * fineweb_n192_skip11000 rows 16:192 (176 rows),
  * 120 WikiText-2 test rows after token skip50000.

Report row-mean damage distributions, not only pooled CE. Price remains three
factorized Down maps, saving 11,501,568 scalars.

Frozen predictions
------------------
pred_a_fixed_triple_mean_stable:
    Mean CE damage <=.08 on all three populations.
pred_b_fixed_triple_tail_stable:
    Row-damage p95 <=.15 and maximum <=.25 on all three populations.
pred_c_fixed_triple_transfers:
    Maximum minus minimum population mean damage <=.025.

Null: any mean damage >=.12 or any row-damage p95 >=.25.  A pass earns only
whole-program integration and certificate testing, not adoption.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp_pca_fixed_triple_stability_results.json"
DEV = "cuda"
LAYERS = (0, 8, 17)
RANK = 256
FIT_ROWS = 16
WIKI_ROWS = 120
WIKI_SKIP = 50000


@torch.no_grad()
def _row_ce(model: torch.nn.Module, rows: torch.Tensor,
            projectors: dict[int, tuple[torch.Tensor, torch.Tensor]], manual_logits) -> torch.Tensor:
    handles = []
    for layer, (basis_cpu, mean_cpu) in projectors.items():
        q, mean = basis_cpu.to(DEV), mean_cpu.to(DEV)

        def hook(_module, _args, output, q=q, mean=mean):
            centered = output.float() - mean
            return (mean + (centered @ q) @ q.T).to(output.dtype)

        handles.append(model.transformer.h[layer].mlp.register_forward_hook(hook))
    values = []
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index, target = batch[:, :-1].to(DEV), batch[:, 1:].to(DEV)
            logits = manual_logits(model, index)
            loss = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
            values.append(loss.mean(1).double().cpu())
    finally:
        for handle in handles:
            handle.remove()
    result = torch.cat(values)
    assert result.shape == (len(rows),) and bool(torch.isfinite(result).all())
    return result


def _summary(damage: torch.Tensor, native: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    return {
        "rows": len(damage),
        "native_ce": float(native.mean()),
        "candidate_ce": float(candidate.mean()),
        "mean_damage": float(damage.mean()),
        "std_row_damage": float(damage.std(unbiased=True)),
        "p50_row_damage": float(torch.quantile(damage, 0.50)),
        "p95_row_damage": float(torch.quantile(damage, 0.95)),
        "min_row_damage": float(damage.min()),
        "max_row_damage": float(damage.max()),
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for name in ("fineweb_n480_skip80.pt", "fineweb_n192_skip7000.pt",
                     "fineweb_n192_skip11000.pt"):
            assert (ROOT / ".rowcache" / name).exists()
        assert len(set(LAYERS)) == 3 and all(0 <= layer < 18 for layer in LAYERS)
        print("MLP PCA FIXED TRIPLE STABILITY | dry run: frozen layers, populations, price, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_activation_pca_four_layer_composition as base
    from mlp0_signed_response_rank_screen import _manual_logits, _wikitext_rows
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == base.D
    fit = base._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    fine_a = base._load_rows(ROOT / ".rowcache/fineweb_n192_skip7000.pt", 192)[16:192]
    fine_b = base._load_rows(ROOT / ".rowcache/fineweb_n192_skip11000.pt", 192)[16:192]
    wiki, fingerprint = _wikitext_rows(WIKI_ROWS, skip=WIKI_SKIP)
    pca = base._fit_pca(base._capture_outputs(model, fit, _manual_logits))
    projectors = {layer: pca[layer] for layer in LAYERS}
    summaries = {}
    for name, rows in (("fineweb_skip7000", fine_a), ("fineweb_skip11000", fine_b),
                       ("wikitext_skip50000", wiki)):
        native = _row_ce(model, rows, {}, _manual_logits)
        candidate = _row_ce(model, rows, projectors, _manual_logits)
        damage = candidate - native
        summaries[name] = _summary(damage, native, candidate)
        print(f"{name}: mean={summaries[name]['mean_damage']:+.4f} "
              f"p95={summaries[name]['p95_row_damage']:+.4f} "
              f"max={summaries[name]['max_row_damage']:+.4f}", flush=True)

    pred_a = all(row["mean_damage"] <= 0.08 for row in summaries.values())
    pred_b = all(row["p95_row_damage"] <= 0.15 and row["max_row_damage"] <= 0.25
                 for row in summaries.values())
    means = [row["mean_damage"] for row in summaries.values()]
    pred_c = max(means) - min(means) <= 0.025
    null = any(row["mean_damage"] >= 0.12 or row["p95_row_damage"] >= 0.25
               for row in summaries.values())
    native_price = 3 * base.H * base.D + base.D
    factor_price = 2 * base.H * base.D + RANK * (base.H + base.D) + base.D
    result = {
        "status": "mlp_pca_fixed_triple_stability_complete",
        "rung": 309,
        "claim_level": "large_untouched_three_population_stability_screen_only",
        "price": {"layers": list(LAYERS), "native_mlp_scalars_each": native_price,
                  "factorized_mlp_scalars_each": factor_price,
                  "total_saving_scalars": len(LAYERS) * (native_price - factor_price)},
        "fit": {"rows": FIT_ROWS, "rank": RANK, "evaluation_used": False},
        "evaluation": {"fineweb_cache_row_range": [16, 191],
                       "fineweb_rows_each": len(fine_a), "wikitext_rows": WIKI_ROWS,
                       "wikitext_skip": WIKI_SKIP, "wikitext_fingerprint": fingerprint},
        "populations": summaries,
        "population_mean_spread": max(means) - min(means),
        'pred_a_fixed_triple_mean_stable': bool(pred_a),
        'pred_b_fixed_triple_tail_stable': bool(pred_b),
        'pred_c_fixed_triple_transfers': bool(pred_c),
        "null_fixed_triple_unstable": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"predicates": [pred_a, pred_b, pred_c], "null": null,
                      "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MLP PCA FIXED TRIPLE STABILITY DONE", flush=True)


if __name__ == "__main__":
    main()
