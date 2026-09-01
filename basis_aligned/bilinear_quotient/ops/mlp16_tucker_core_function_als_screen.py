"""RUNG 386 -- L16 TUCKER-CORE FUNCTION-SPACE ALS SCREEN (NO CENSUS, NO ADOPTION).

The depth profile (#2482) showed layer 16 is cheap in every tensor mode.
This screen measures the missing CONVERSION: does a literally priced
(r512,k576,p512) core actually capture the layer FUNCTION on live inputs?
Claim level is function-space only -- no census, no certificates, no
adoption implication; the physical calibration build remains reserved.

Program: f_hat(x) = U C[(A Qx) . (B Qx)] + b_native, with Q FROZEN to the
context-RRR input basis (the proven input-mode object); ALS over A,B and
the rank-p output map (U C fitted by lsq + truncated SVD).  The mode-wise
CP-truncation initialization (top-k products by native importance) is the
matched control: ALS must beat marginal truncation.

Frozen predictions
------------------
pred_a (instrument): >=4096 held-out samples; no NaN; train MSE falls
    >=25% from the initialization over <=15 sweeps.
pred_b (registered prediction): held-out function R2 >= .90 for the ALS
    core on fit-A rows (Euclidean output metric).
pred_c: ALS beats the mode-wise truncation control by >= .05 held-out R2,
    and the literal price is exactly 2,065,536 scalars + native bias
    (vs 15,926,400 native layer).

Null: NaN/divergence, or held-out ALS R2 < .50 (family closed at this
price point at L16).
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp16_tucker_core_function_als_screen_results.json"
CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_A = (0, 24)
FIT_B = (24, 48)
LAYER = 16
D = 1152
H = 4608
R, K, P = 512, 576, 512
SWEEPS = 15
RIDGE = 1e-4
CORE_PRICE = R * D + 2 * K * R + P * K + D * P + D
NATIVE_PRICE = 3 * D * H + D


def _r2(pred, target):
    err = (pred - target).square().sum()
    tot = (target - target.mean(0)).square().sum().clamp_min(1e-30)
    return float(1 - err / tot)


def _fit_out(prods, y, p):
    # ridge lsq W (d x k), then rank-p truncate -> U (d x p), C (p x k)
    g = prods.T @ prods + RIDGE * torch.eye(prods.shape[1], device=prods.device)
    w = torch.linalg.solve(g, prods.T @ y).T
    uu, ss, vv = torch.linalg.svd(w, full_matrices=False)
    u = uu[:, :p] * ss[:p]
    c = vv[:p]
    return u, c


def _fit_factor(z, other, wout, y, k):
    # y ~ wout[(A z).(other z)], linear in A row-wise via weighted lsq per product slot
    # solve jointly: design for product slot j is z * (other z)_j; stack via kron-free trick
    oz = z @ other.T                      # n x k
    n, r = z.shape
    a = torch.empty(k, r, device=z.device)
    # target in product space: t = pinv(wout) y  (ridge)
    g = wout.T @ wout + RIDGE * torch.eye(wout.shape[1], device=z.device)
    t = torch.linalg.solve(g, wout.T @ y.T).T   # n x k
    for j in range(k):
        x = z * oz[:, j:j + 1]
        gj = x.T @ x + RIDGE * torch.eye(r, device=z.device)
        a[j] = torch.linalg.solve(gj, x.T @ t[:, j])
    return a


@torch.no_grad()
def _capture(model, rows, layer):
    xs, ys = [], []
    mlp = model.transformer.h[layer].mlp
    def hook(module, inp, out):
        xs.append(inp[0].detach().reshape(-1, D).float().cpu())
        ys.append(out.detach().reshape(-1, D).float().cpu())
    h = mlp.register_forward_hook(hook)
    dev = next(model.parameters()).device
    for i in range(rows.shape[0]):
        model(rows[i:i + 1].to(dev))
    h.remove()
    return torch.cat(xs), torch.cat(ys)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert CACHE.exists() and CORE_PRICE == 2_065_536 and NATIVE_PRICE == 15_926_400
        print("L16 TUCKER ALS SCREEN | dry run: cache, prices, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    cached = torch.load(CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    rows_a = cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    rows_b = cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    xb, yb = _capture(model, rows_b, LAYER)
    xa, ya = _capture(model, rows_a, LAYER)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xb, yb, xa, ya = (t.to(dev) for t in (xb, yb, xa, ya))
    bias = model.transformer.h[LAYER].mlp.Down.bias
    bias = (bias.detach().float().to(dev) if bias is not None else torch.zeros(D, device=dev))
    yb = yb - bias
    ya = ya - bias

    # frozen Q: context-RRR input basis (top-R right singular basis of X_b)
    cov = xb.T @ xb / xb.shape[0]
    values, vectors = torch.linalg.eigh(.5 * (cov + cov.T))
    q = vectors[:, torch.argsort(values, descending=True)[:R]].T  # R x D

    left = model.transformer.h[LAYER].mlp.Left.weight.detach().float().to(dev)
    right = model.transformer.h[LAYER].mlp.Right.weight.detach().float().to(dev)
    down = model.transformer.h[LAYER].mlp.Down.weight.detach().float().to(dev)
    lq = left @ q.T   # H x R
    rq = right @ q.T
    importance = down.norm(dim=0) * lq.norm(dim=1) * rq.norm(dim=1)
    top = torch.argsort(importance, descending=True)[:K]
    a = lq[top].clone()
    b = rq[top].clone()
    zb = xb @ q.T
    za = xa @ q.T

    def core_pred(z, a_, b_, u_, c_):
        return ((z @ a_.T) * (z @ b_.T)) @ c_.T @ u_.T

    # control: mode-wise truncation (init factors, lsq output once)
    prods0 = (zb @ a.T) * (zb @ b.T)
    u0, c0 = _fit_out(prods0, yb, P)
    control_r2 = _r2(core_pred(za, a, b, u0, c0), ya)

    u, c = u0.clone(), c0.clone()
    losses = []
    init_loss = float((core_pred(zb, a, b, u, c) - yb).square().mean())
    for sweep in range(SWEEPS):
        wout = u @ c
        a = _fit_factor(zb, b, wout, yb, K)
        b = _fit_factor(zb, a, wout, yb, K)
        prods = (zb @ a.T) * (zb @ b.T)
        u, c = _fit_out(prods, yb, P)
        loss = float((core_pred(zb, a, b, u, c) - yb).square().mean())
        losses.append(loss)
        print(f"sweep {sweep} train mse {loss:.6f}", flush=True)
        if not torch.isfinite(torch.tensor(loss)):
            break
    final_loss = losses[-1] if losses else init_loss
    als_r2 = _r2(core_pred(za, a, b, u, c), ya)
    nan_free = all(torch.isfinite(torch.tensor(l)) for l in losses)

    pred_a = xa.shape[0] >= 4096 and nan_free and final_loss <= .75 * init_loss
    pred_b = als_r2 >= .90
    pred_c = (als_r2 - control_r2 >= .05) and CORE_PRICE == 2_065_536
    null = (not nan_free) or als_r2 < .50

    result = {
        "status": "mlp16_tucker_core_function_als_screen_complete",
        "rung": 386,
        "claim_level": "function_space_als_screen_only_no_census_no_adoption",
        "layer": LAYER, "ranks": {"r": R, "k": K, "p": P},
        "fit_cache": CACHE.name, "fit_b_train": list(FIT_B), "fit_a_heldout": list(FIT_A),
        "train_samples": int(xb.shape[0]), "heldout_samples": int(xa.shape[0]),
        "core_price_scalars": CORE_PRICE, "native_layer_price_scalars": NATIVE_PRICE,
        "init_train_mse": init_loss, "final_train_mse": final_loss,
        "train_losses": losses,
        "control_modewise_heldout_r2": control_r2,
        "als_heldout_r2": als_r2,
        'pred_a_als_converges_on_adequate_samples': bool(pred_a),
        'pred_b_core_captures_function_r2_090': bool(pred_b),
        'pred_c_als_beats_modewise_control_and_price_exact': bool(pred_c),
        'null_core_cannot_capture_function': bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
