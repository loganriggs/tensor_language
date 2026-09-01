"""RUNG 383 -- MLP0 PRODUCT-MODE (WIDTH) GRAM SCREEN.

Border-rank-style closure instrument for the joint Tucker-core grid
(MATHEMATICAL_REVIEW_2026-09-01_1000.md): compute the exact 4608x4608
product-mode Gram M[u,v] = (d_u . d_v) * <sym(l_u@r_u), sym(l_v@r_v)>_C
without materializing the 1152^3 contraction, using the same identity as
rung 381.  Its eigenvalue tail lower-bounds the product width k any
Tucker/CP core needs in this metric.

Frozen predictions
------------------
pred_a (instrument): M is PSD (min eig >= -1e-6 * max eig) and the ENTRY
    SUM of M matches the independently computed output-mode Gram trace
    within rel 1e-3 under context-B (corrected identity; rung 382 froze the
    wrong trace form and fired its own null -- see #2479).
pred_b (transfer): eigenvalue mass strictly decreasing across
    k in {576,1152,2304,3456}; context A/B top-2304 subspace overlap >= .70.
pred_c (registered prediction): the product mode is FLATTER than the output
    mode -- context-B retained energy at k=2304 (50% of width) <= .85,
    i.e. the k<=2304 corners of the frozen grid get CLOSED.  Licensing rule
    frozen now: retained@2304 >= .90 keeps k<=2304 corners live; < .80
    closes all k<3456; [.80,.90) licenses only k=3456.

Null: trace mismatch > 1e-2, min eig < -1e-4 * max eig, or split
overlap < .40 (instrument failure).

Price: screen only, no artifact change.  Affected frozen-grid corner
prices restated: (r512,k1152,p768)=3,540,096; (r768,k2304,p768)=7,079,040.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_product_mode_gram_screen_results.json"
CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_A = (0, 24)
FIT_B = (24, 48)
D = 1152
H = 4608
WIDTHS = (576, 1152, 2304, 3456)
CORNER_PRICES = {"r512_k1152_p768": 512 * D + 2 * 1152 * 512 + 768 * 1152 + D * 768 + D,
                 "r768_k2304_p768": 768 * D + 2 * 2304 * 768 + 768 * 2304 + D * 768 + D}


def _sqrt(covariance: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(.5 * (covariance + covariance.T))
    floor = float(values[-1]) * 1e-6
    return (vectors * values.clamp_min(floor).sqrt()) @ vectors.T


@torch.no_grad()
def _product_gram(left, right, down, covariance):
    sqrt = _sqrt(covariance.to(left.device).float())
    l = left @ sqrt
    r = right @ sqrt
    ll = l @ l.T
    rr = r @ r.T
    lr = l @ r.T
    input_gram = .5 * (ll * rr + lr * lr.T)
    output_gram = down.T @ down
    m = input_gram * output_gram
    return .5 * (m + m.T)


@torch.no_grad()
def _output_gram_trace(left, right, down, covariance):
    sqrt = _sqrt(covariance.to(left.device).float())
    l = left @ sqrt
    r = right @ sqrt
    ll = l @ l.T
    rr = r @ r.T
    lr = l @ r.T
    product_gram = .5 * (ll * rr + lr * lr.T)
    return float((down @ product_gram @ down.T).trace())


def _spectrum(gram):
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values = values[order]
    vectors = vectors[:, order]
    total = values.clamp_min(0).sum().clamp_min(1e-30)
    retained = {int(k): float(values[:k].clamp_min(0).sum() / total) for k in WIDTHS}
    return values, retained, vectors


def _overlap(a, b, k):
    return float((a[:, :k].T @ b[:, :k]).square().sum() / k)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert CACHE.exists() and FIT_A[1] == FIT_B[0]
        assert CORNER_PRICES["r512_k1152_p768"] == 3_540_096
        assert CORNER_PRICES["r768_k2304_p768"] == 7_079_040
        print("MLP0 PRODUCT-MODE GRAM | dry run: cache, widths, corner prices, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_all_layer_context_metric_shared_input_screen as M
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    cached = torch.load(CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    rows_a = cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    rows_b = cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    M.LAYERS = (0,)
    context_a = M._covariances(model, rows_a, _manual_logits)[0]
    context_b = M._covariances(model, rows_b, _manual_logits)[0]

    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    assert left.shape == right.shape == (H, D) and down.shape == (D, H)

    gram_b = _product_gram(left, right, down, context_b)
    values_b, retained_b, basis_b = _spectrum(gram_b)
    gram_a = _product_gram(left, right, down, context_a)
    _va, retained_a, basis_a = _spectrum(gram_a)

    trace_product = float(gram_b.trace())
    entry_sum_product = float(gram_b.sum())
    trace_output = _output_gram_trace(left, right, down, context_b)
    trace_rel_err = abs(entry_sum_product - trace_output) / max(abs(trace_output), 1e-30)
    min_eig = float(values_b[-1])
    max_eig = float(values_b[0])
    psd_ok = min_eig >= -1e-6 * max_eig
    split_overlap = _overlap(basis_a, basis_b, 2304)
    decreasing = all(retained_b[WIDTHS[i]] < retained_b[WIDTHS[i + 1]] for i in range(len(WIDTHS) - 1))

    r2304 = retained_b[2304]
    if r2304 >= .90:
        license_rule = "k<=2304 corners remain live"
    elif r2304 < .80:
        license_rule = "all k<3456 corners closed; calibration build must use k>=3456"
    else:
        license_rule = "only k=3456 licensed"

    pred_a = psd_ok and trace_rel_err <= 1e-3
    pred_b = decreasing and split_overlap >= .70
    pred_c = r2304 <= .85
    null = trace_rel_err > 1e-2 or min_eig < -1e-4 * max_eig or split_overlap < .40

    result = {
        "status": "mlp0_product_mode_gram_screen_complete",
        "rung": 383,
        "claim_level": "product_mode_width_gram_closure_screen_only",
        "tensor_definition": "M[u,v]=(d_u.d_v)*<sym(l_u,r_u),sym(l_v,r_v)>_C",
        "fit_cache": CACHE.name, "fit_a": list(FIT_A), "fit_b": list(FIT_B),
        "dimensions": {"d": D, "product_width": H},
        "widths": list(WIDTHS),
        "context_b_retained_energy": retained_b,
        "context_a_retained_energy": retained_a,
        "trace_product_mode_diagonal": trace_product,
        "entry_sum_product_mode": entry_sum_product,
        "trace_output_mode_independent": trace_output,
        "trace_relative_error": trace_rel_err,
        "min_eigenvalue": min_eig, "max_eigenvalue": max_eig,
        "split_top2304_overlap": split_overlap,
        "frozen_grid_corner_prices": CORNER_PRICES,
        "license_determination": license_rule,
        'pred_a_gram_instrument_identities_hold': bool(pred_a),
        'pred_b_spectrum_decreasing_and_split_stable': bool(pred_b),
        'pred_c_product_mode_flatter_than_output_mode': bool(pred_c),
        'null_product_gram_instrument_fails': bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
