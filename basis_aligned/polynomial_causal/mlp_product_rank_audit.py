"""Weights-only lower bounds on bilin18 vector-quadratic product complexity.

For each selected bilinear MLP, evaluate its pure quadratic map on two independent
Gaussian designs.  The rank of the resulting output-by-evaluation matrix lower-
bounds the output flattening rank, which lower-bounds the number of shared scalar
products in ANY exact program of the form

    sum_i c_i (a_i dot x) (b_i dot x).

This is a randomized numerical certificate, not an exact symbolic rank proof and
not an activation-distribution compression result.  It exists to bound how much
exact multiplication reduction is even possible before launching a joint compiler.

Registered before execution:
  A. At rtol=1e-6 the lower bound is at least 95% of output width (1095/1152)
     for every audited layer {0,1,2,11,17}.
  B. The two independent sketches differ by at most 5 ranks at rtol=1e-6 for
     every layer.
  C. Therefore the exact product-count interval narrows from [1,4608] to at
     least [1095,4608], ruling out more than 4.21x exact product compression in
     this grammar for each audited full vector map.

Failure is informative: a materially smaller lower bound identifies a layer whose
full quadratic output family has algebraic redundancy worth compiling first.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch


BQ = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = Path(__file__).with_name("mlp_product_rank_audit_results.json")
sys.path.insert(0, str(BQ))
from bilin18_joint_removal import DEV, m  # noqa: E402


LAYERS = (0, 1, 2, 11, 17)
N_EVAL = 1280
SEEDS = (1729, 2718)
RTOLS = (1e-4, 1e-5, 1e-6)


@torch.no_grad()
def sketch_layer(layer: int, seed: int) -> dict:
    block = m.transformer.h[layer]
    left = block.mlp.Left.weight.detach().float().to(DEV)
    right = block.mlp.Right.weight.detach().float().to(DEV)
    down = block.mlp.Down.weight.detach().float().to(DEV)
    hidden, input_dim = left.shape
    output_dim = down.shape[0]

    generator = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(N_EVAL, input_dim, generator=generator, device=DEV) / math.sqrt(input_dim)
    products = (x @ left.T) * (x @ right.T)
    values = products @ down.T
    singular = torch.linalg.svdvals(values).double().cpu()
    top = float(singular[0])
    ranks = {str(rtol): int((singular > top * rtol).sum()) for rtol in RTOLS}
    result = {
        "seed": seed,
        "ranks": ranks,
        "sigma_max": top,
        "sigma_min": float(singular[-1]),
        "sigma_1095_over_max": float(singular[1094] / singular[0]),
        "sigma_1152_over_max": float(singular[1151] / singular[0]),
    }
    del x, products, values, singular, left, right, down
    torch.cuda.empty_cache()
    return result


def main() -> None:
    start = time.time()
    rows = {}
    for layer in LAYERS:
        block = m.transformer.h[layer]
        shape = {
            "input_dim": int(block.mlp.Left.weight.shape[1]),
            "output_dim": int(block.mlp.Down.weight.shape[0]),
            "explicit_products_upper": int(block.mlp.Left.weight.shape[0]),
        }
        sketches = [sketch_layer(layer, seed) for seed in SEEDS]
        lower = min(s["ranks"]["1e-06"] for s in sketches)
        spread = abs(sketches[0]["ranks"]["1e-06"] - sketches[1]["ranks"]["1e-06"])
        rows[str(layer)] = {
            **shape,
            "sketches": sketches,
            "conservative_numerical_lower_rtol_1e-6": lower,
            "sketch_rank_spread": spread,
            "max_exact_compression_consistent_with_bound": shape["explicit_products_upper"] / lower,
        }
        print(f"L{layer}: lower={lower}/{shape['output_dim']} spread={spread} upper={shape['explicit_products_upper']}", flush=True)

    threshold = math.ceil(0.95 * 1152)
    pred_a = all(row["conservative_numerical_lower_rtol_1e-6"] >= threshold for row in rows.values())
    pred_b = all(row["sketch_rank_spread"] <= 5 for row in rows.values())
    pred_c = all(row["max_exact_compression_consistent_with_bound"] <= 4608 / threshold for row in rows.values())
    result = {
        "config": {
            "model": "bilin18",
            "layers": list(LAYERS),
            "n_evaluations": N_EVAL,
            "seeds": list(SEEDS),
            "rtols": list(RTOLS),
            "grammar": "sum_i c_i * (a_i dot x) * (b_i dot x)",
            "certificate_status": "randomized_numerical_lower_bound",
            "distribution_note": "Gaussian design probes coefficient-space rank, not natural activation fidelity",
        },
        "layers": rows,
        "predictions": {
            "A_all_layers_at_least_95pct_output_rank": pred_a,
            "B_two_sketches_within_5_ranks": pred_b,
            "C_rules_out_more_than_4.21x_exact_products": pred_c,
        },
        "runtime_s": round(time.time() - start, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["predictions"], indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
