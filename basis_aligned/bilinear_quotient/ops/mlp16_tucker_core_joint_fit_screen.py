"""RUNG 387 -- L16 TUCKER-CORE JOINT GRADIENT FIT + TRANSFER SCREEN.

Corrects rung 386's mis-keyed null: the ALS optimizer failed, while the
mode-wise truncation CONTROL measured held-out R2 .8145 at the
2,065,536-scalar price.  This rung makes fresh falsifiable claims:
(1) a joint Adam fit from the truncation init must BEAT the observed
control level; (2) the truncation core's .8145 must TRANSFER to cache
rows never used by any fit or evaluation.

Frozen predictions
------------------
pred_a (instrument): Adam training is finite and final train MSE is
    strictly below the truncation core's train MSE (the bar 386's ALS
    failed).
pred_b (registered prediction): joint-fit held-out (fit-A) R2 >= .87 and
    >= control + .05.
pred_c (transfer): truncation-core R2 on FRESH rows 48:72 >= .765
    (within .05 of its fit-A value).

Null: joint fit <= control on fit-A (family's measured level is the
truncation core), or fresh-row transfer < .60 (the .8145 was
split-specific).

Price: screen only; core price restated 2,065,536 scalars.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp16_tucker_core_joint_fit_screen_results.json"
CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_A = (0, 24)
FIT_B = (24, 48)
FRESH = (48, 72)
LAYER = 16
D = 1152
H = 4608
R, K, P = 512, 576, 512
STEPS = 1500
LR = 3e-4
CONTROL_REFERENCE = 0.8144966363906860
CORE_PRICE = R * D + 2 * K * R + P * K + D * P + D


def _r2(pred, target):
    err = (pred - target).square().sum()
    tot = (target - target.mean(0)).square().sum().clamp_min(1e-30)
    return float(1 - err / tot)


@torch.no_grad()
def _capture(model, rows, layer):
    xs, ys = [], []
    mlp = model.transformer.h[layer].mlp
    def hook(module, inp, out):
        xs.append(inp[0].detach().reshape(-1, D).float().cpu())
        ys.append(out.detach().reshape(-1, D).float().cpu())
    h = mlp.register_forward_hook(hook)
    dev = next(model.parameters()).device
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    for i in range(rows.shape[0]):
        _manual_logits(model, rows[i:i + 1, :-1].to(dev))
    h.remove()
    return torch.cat(xs), torch.cat(ys)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert CACHE.exists() and CORE_PRICE == 2_065_536 and FRESH == (48, 72)
        print("L16 TUCKER JOINT FIT | dry run: cache, fresh rows, price, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    from tier2_model import load_elriggs

    cached = torch.load(CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D
    xb, yb = _capture(model, cached[FIT_B[0]:FIT_B[1], :257].long().contiguous(), LAYER)
    xa, ya = _capture(model, cached[FIT_A[0]:FIT_A[1], :257].long().contiguous(), LAYER)
    xf, yf = _capture(model, cached[FRESH[0]:FRESH[1], :257].long().contiguous(), LAYER)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xb, yb, xa, ya, xf, yf = (t.to(dev) for t in (xb, yb, xa, ya, xf, yf))
    bias = model.transformer.h[LAYER].mlp.Down.bias
    bias = (bias.detach().float().to(dev) if bias is not None else torch.zeros(D, device=dev))
    yb, ya, yf = yb - bias, ya - bias, yf - bias

    cov = xb.T @ xb / xb.shape[0]
    values, vectors = torch.linalg.eigh(.5 * (cov + cov.T))
    q = vectors[:, torch.argsort(values, descending=True)[:R]].T

    mlp = model.transformer.h[LAYER].mlp
    left = mlp.Left.weight.detach().float().to(dev)
    right = mlp.Right.weight.detach().float().to(dev)
    down = mlp.Down.weight.detach().float().to(dev)
    lq, rq = left @ q.T, right @ q.T
    importance = down.norm(dim=0) * lq.norm(dim=1) * rq.norm(dim=1)
    top = torch.argsort(importance, descending=True)[:K]
    a0, b0 = lq[top].clone(), rq[top].clone()
    zb, za, zf = xb @ q.T, xa @ q.T, xf @ q.T

    def outfit(a_, b_, z, y):
        prods = (z @ a_.T) * (z @ b_.T)
        g = prods.T @ prods + 1e-4 * torch.eye(K, device=dev)
        w = torch.linalg.solve(g, prods.T @ y).T
        uu, ss, vv = torch.linalg.svd(w, full_matrices=False)
        return uu[:, :P] * ss[:P], vv[:P]

    def pred(z, a_, b_, u_, c_):
        return ((z @ a_.T) * (z @ b_.T)) @ c_.T @ u_.T

    u0, c0 = outfit(a0, b0, zb, yb)
    control_train_mse = float((pred(zb, a0, b0, u0, c0) - yb).square().mean())
    control_a = _r2(pred(za, a0, b0, u0, c0), ya)
    control_fresh = _r2(pred(zf, a0, b0, u0, c0), yf)

    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    u = u0.clone().requires_grad_(True)
    c = c0.clone().requires_grad_(True)
    opt = torch.optim.Adam([a, b, u, c], lr=LR)
    finite = True
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = (pred(zb, a, b, u, c) - yb).square().mean()
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"step {step} train mse {float(loss):.2f}", flush=True)
    with torch.no_grad():
        final_train_mse = float((pred(zb, a, b, u, c) - yb).square().mean())
        joint_a = _r2(pred(za, a, b, u, c), ya)

    pred_a = finite and final_train_mse < control_train_mse
    pred_b = joint_a >= .87 and joint_a >= control_a + .05
    pred_c = control_fresh >= .765
    null = (joint_a <= control_a) or (control_fresh < .60) or (not finite)

    result = {
        "status": "mlp16_tucker_core_joint_fit_screen_complete",
        "rung": 387,
        "claim_level": "function_space_joint_fit_screen_only_no_census_no_adoption",
        "layer": LAYER, "ranks": {"r": R, "k": K, "p": P},
        "fit_cache": CACHE.name, "fit_b_train": list(FIT_B),
        "fit_a_heldout": list(FIT_A), "fresh_rows": list(FRESH),
        "core_price_scalars": CORE_PRICE,
        "control_reference_from_386": CONTROL_REFERENCE,
        "control_train_mse": control_train_mse,
        "control_heldout_r2": control_a,
        "control_fresh_r2": control_fresh,
        "joint_final_train_mse": final_train_mse,
        "joint_heldout_r2": joint_a,
        "adam_steps": STEPS, "lr": LR,
        'pred_a_joint_fit_trains_below_control_mse': bool(pred_a),
        'pred_b_joint_fit_beats_control_and_087': bool(pred_b),
        'pred_c_truncation_core_transfers_to_fresh_rows': bool(pred_c),
        'null_joint_no_better_or_transfer_collapses': bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
