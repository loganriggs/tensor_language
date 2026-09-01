"""RUNG 384 -- PRODUCT-MODE GRAM LAYER-GENERALITY SCREEN (LAYERS 4 AND 16).

Rung 383 established (instrument-exact) that MLP0's product mode is the
flattest mode of its invariant tensor: context-B retained energy .659 at
half width, closing k<3456 Tucker corners.  This screen asks whether that
is a layer-0 fact or a LAW: compute the identical corrected product-mode
Gram at the other adopted layer (4) and a late layer (16).

Frozen predictions
------------------
pred_a (instrument): at BOTH layers, entry-sum(M) matches the independent
    output-mode Gram trace within rel 1e-3, and M is PSD
    (min eig >= -1e-6 * max eig).
pred_b (transfer): at both layers the eigenvalue mass is strictly
    increasing in k over {576,1152,2304,3456} and fit-A/B top-2304
    subspace overlap >= .70.
pred_c (registered prediction): product-mode flatness is LAYER-GENERAL --
    context-B retained energy at k=2304 <= .85 at BOTH layers 4 and 16.

Null: instrument failure at either layer (entry-sum rel err > 1e-2,
min eig < -1e-4 * max eig, or split overlap < .40).

Price: screen only, no artifact change.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp_product_mode_gram_layers_screen_results.json"
CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_A = (0, 24)
FIT_B = (24, 48)
D = 1152
H = 4608
LAYERS = (4, 16)
WIDTHS = (576, 1152, 2304, 3456)


def _sqrt(covariance: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(.5 * (covariance + covariance.T))
    floor = float(values[-1]) * 1e-6
    return (vectors * values.clamp_min(floor).sqrt()) @ vectors.T


@torch.no_grad()
def _grams(left, right, down, covariance):
    sqrt = _sqrt(covariance.to(left.device).float())
    l = left @ sqrt
    r = right @ sqrt
    ll = l @ l.T
    rr = r @ r.T
    lr = l @ r.T
    input_gram = .5 * (ll * rr + lr * lr.T)
    m = input_gram * (down.T @ down)
    m = .5 * (m + m.T)
    output_trace = float((down @ input_gram @ down.T).trace())
    return m, output_trace


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
        assert CACHE.exists() and FIT_A[1] == FIT_B[0] and LAYERS == (4, 16)
        print("PRODUCT-MODE LAYER GENERALITY | dry run: cache, layers, widths, bars valid")
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
    M.LAYERS = LAYERS
    cov_a = M._covariances(model, rows_a, _manual_logits)
    cov_b = M._covariances(model, rows_b, _manual_logits)

    per_layer = {}
    inst_ok, trans_ok, flat_ok, null_fired = True, True, True, False
    for layer in LAYERS:
        mlp = model.transformer.h[layer].mlp
        left = mlp.Left.weight.detach().float()
        right = mlp.Right.weight.detach().float()
        down = mlp.Down.weight.detach().float()
        assert left.shape == right.shape == (H, D) and down.shape == (D, H)
        gram_b, out_trace = _grams(left, right, down, cov_b[layer])
        values_b, retained_b, basis_b = _spectrum(gram_b)
        gram_a, _ = _grams(left, right, down, cov_a[layer])
        _va, retained_a, basis_a = _spectrum(gram_a)
        entry_sum = float(gram_b.sum())
        rel = abs(entry_sum - out_trace) / max(abs(out_trace), 1e-30)
        min_eig, max_eig = float(values_b[-1]), float(values_b[0])
        overlap = _overlap(basis_a, basis_b, 2304)
        increasing = all(retained_b[WIDTHS[i]] < retained_b[WIDTHS[i + 1]] for i in range(len(WIDTHS) - 1))
        per_layer[layer] = {
            "context_b_retained_energy": retained_b,
            "context_a_retained_energy": retained_a,
            "entry_sum": entry_sum, "output_trace_independent": out_trace,
            "entry_sum_relative_error": rel,
            "min_eigenvalue": min_eig, "max_eigenvalue": max_eig,
            "split_top2304_overlap": overlap,
        }
        inst_ok &= (rel <= 1e-3 and min_eig >= -1e-6 * max_eig)
        trans_ok &= (increasing and overlap >= .70)
        flat_ok &= (retained_b[2304] <= .85)
        null_fired |= (rel > 1e-2 or min_eig < -1e-4 * max_eig or overlap < .40)

    result = {
        "status": "mlp_product_mode_gram_layers_screen_complete",
        "rung": 384,
        "claim_level": "product_mode_layer_generality_closure_screen_only",
        "tensor_definition": "M[u,v]=(d_u.d_v)*<sym(l_u,r_u),sym(l_v,r_v)>_C per layer",
        "fit_cache": CACHE.name, "fit_a": list(FIT_A), "fit_b": list(FIT_B),
        "layers": list(LAYERS), "widths": list(WIDTHS),
        "per_layer": per_layer,
        "layer0_reference_retained_2304": 0.6587068438529968,
        'pred_a_gram_instrument_identities_hold_both_layers': bool(inst_ok),
        'pred_b_spectra_increasing_and_split_stable_both_layers': bool(trans_ok),
        'pred_c_product_mode_flatness_is_layer_general': bool(flat_ok),
        'null_product_gram_instrument_fails_either_layer': bool(null_fired),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
