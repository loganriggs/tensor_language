"""RUNG 385 -- DEPTH PROFILE OF MODE CONCENTRATION (ALL 18 LAYERS).

Rung 384 falsified layer-general product-mode flatness: layers 0/4 are flat
(retained@2304 .659/.603) while layer 16 is concentrated (.925).  This
screen maps the full depth profile of BOTH non-input modes under the fit-B
context metric: product-mode retained energy at k=2304 and output-mode
retained energy at p=512, for every layer 0..17, with the corrected
entry-sum instrument identity enforced per layer.

Frozen predictions
------------------
pred_a (instrument): at ALL 18 layers, entry-sum(M_product) matches the
    independent output-mode trace within rel 1e-3 and both Grams are PSD
    (min eig >= -1e-6 * max eig).
pred_b (registered prediction): product-mode concentration RISES with
    depth overall -- Spearman(layer index, product retained@2304) >= .60.
pred_c (registered prediction): the late stack is jointly compressible --
    at least 2 layers in {15,16,17} have product retained@2304 >= .90 AND
    output retained@512 >= .85.

Null: instrument failure at any layer (rel err > 1e-2 or min eig
< -1e-4 * max eig).

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
OUT = ROOT / "mlp_mode_concentration_depth_profile_results.json"
CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_B = (24, 48)
D = 1152
H = 4608
LAYERS = tuple(range(18))
K_PRODUCT = 2304
P_OUTPUT = 512


def _sqrt(covariance: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(.5 * (covariance + covariance.T))
    floor = float(values[-1]) * 1e-6
    return (vectors * values.clamp_min(floor).sqrt()) @ vectors.T


def _retained(values: torch.Tensor, k: int) -> float:
    order = torch.argsort(values, descending=True)
    v = values[order]
    total = v.clamp_min(0).sum().clamp_min(1e-30)
    return float(v[:k].clamp_min(0).sum() / total)


def _spearman(x, y):
    import numpy as np
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / ((rx ** 2).sum() ** .5 * (ry ** 2).sum() ** .5))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert CACHE.exists() and LAYERS == tuple(range(18))
        print("MODE CONCENTRATION DEPTH PROFILE | dry run: cache, layers, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_all_layer_context_metric_shared_input_screen as M
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    cached = torch.load(CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    rows_b = cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    M.LAYERS = LAYERS
    cov_b = M._covariances(model, rows_b, _manual_logits)

    profile = {}
    inst_ok, null_fired = True, False
    for layer in LAYERS:
        mlp = model.transformer.h[layer].mlp
        left = mlp.Left.weight.detach().float()
        right = mlp.Right.weight.detach().float()
        down = mlp.Down.weight.detach().float()
        sqrt = _sqrt(cov_b[layer].to(left.device).float())
        l = left @ sqrt
        r = right @ sqrt
        ll = l @ l.T
        rr = r @ r.T
        lr = l @ r.T
        input_gram = .5 * (ll * rr + lr * lr.T)
        m = input_gram * (down.T @ down)
        m = .5 * (m + m.T)
        out_gram = down @ input_gram @ down.T
        out_gram = .5 * (out_gram + out_gram.T)
        mv = torch.linalg.eigvalsh(m)
        ov = torch.linalg.eigvalsh(out_gram)
        entry_sum = float(m.sum()); out_trace = float(out_gram.trace())
        rel = abs(entry_sum - out_trace) / max(abs(out_trace), 1e-30)
        pm_min, pm_max = float(mv.min()), float(mv.max())
        om_min, om_max = float(ov.min()), float(ov.max())
        profile[layer] = {
            "product_retained_2304": _retained(mv, K_PRODUCT),
            "output_retained_512": _retained(ov, P_OUTPUT),
            "entry_sum_relative_error": rel,
        }
        inst_ok &= (rel <= 1e-3 and pm_min >= -1e-6 * pm_max and om_min >= -1e-6 * om_max)
        null_fired |= (rel > 1e-2 or pm_min < -1e-4 * pm_max or om_min < -1e-4 * om_max)
        print(layer, profile[layer], flush=True)

    layers_list = list(LAYERS)
    prod = [profile[l]["product_retained_2304"] for l in layers_list]
    outp = [profile[l]["output_retained_512"] for l in layers_list]
    spearman = _spearman(layers_list, prod)
    late_joint = sum(1 for l in (15, 16, 17)
                     if profile[l]["product_retained_2304"] >= .90 and profile[l]["output_retained_512"] >= .85)

    result = {
        "status": "mlp_mode_concentration_depth_profile_complete",
        "rung": 385,
        "claim_level": "all_layer_mode_concentration_profile_screen_only",
        "fit_cache": CACHE.name, "fit_b": list(FIT_B),
        "layers": layers_list,
        "profile": profile,
        "product_retained_2304_by_layer": prod,
        "output_retained_512_by_layer": outp,
        "depth_spearman_product_2304": spearman,
        "late_jointly_compressible_layers": late_joint,
        'pred_a_instrument_identities_hold_all_layers': bool(inst_ok),
        'pred_b_product_concentration_rises_with_depth': bool(spearman >= .60),
        'pred_c_late_stack_jointly_compressible': bool(late_joint >= 2),
        'null_instrument_fails_any_layer': bool(null_fired),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
