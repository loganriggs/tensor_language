"""DIRECT TEST of the "why not multilinear?" challenge. The reports have said the middle content is "irreducible to
LINEAR/table/bag stand-ins" and quoted a 1.6-nat "ceiling" -- but that is the LINEAR-stand-in ceiling, NOT a
fundamental one. The MLP is literally BILINEAR (y = Down[(Wl x) ⊙ (Wr x)]), so a linear map is the WRONG stand-in.
Match the model's form: fit a RANK-R BILINEAR stand-in of a middle MLP's output from its input and see how much more
it recovers than the linear one (§1005 linear R² ~0.25-0.55). If a modest-rank bilinear map captures the middle, the
content is a BOUNDED-RANK BILINEAR computation we CAN understand -- the "ceiling" was a stand-in-class artifact.

Stand-in: y_hat = x@Wlin + b + ((x@A) ⊙ (x@B)) @ G,  A,B ∈ (D,R), G ∈ (R,D)  -- exactly a rank-R bilinear MLP.
Fit by Adam on TRAIN, held-out R² on TEST. Sweep R = 0 (pure linear baseline), 16, 64, 256. R=0 must match §1005.

REGISTERED PREDICTIONS:
  (0) SANITY: R=0 (linear) held-out R² ≈ §1005's linear reconstruction (~0.25-0.55 front/middle); fit converges.
  (a) LOW-RANK BILINEAR BREAKS THE LINEAR CEILING: rank-R bilinear R² climbs FAR above the linear R² with modest R
      (target: R=256 reaches R² > 0.8 at middle layers where linear was ~0.3) -> the middle content is a
      bounded-rank BILINEAR computation, NOT irreducible; the 1.6-nat "ceiling" was a LINEAR-stand-in artifact;
  (b) report held-out R² vs R per layer + the effective rank (smallest R with R² ≥ 0.9)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilinear_content_standin_results.json'
NEVAL = 220; SEQ = 256; LAYERS = [1, 6, 8, 11, 15]; RANKS = [0, 16, 64, 256]
NTRAIN = 9000; NTEST = 4000; STEPS = 400; LR = 3e-3
CAP = {}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = i_[0] if isinstance(i_, tuple) else i_
                CAP[L] = (x.detach().float().reshape(-1, D), (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    readout(x)
    for h in hs: h.remove()


def fit_bilinear(X, Y, Xte, Yte, R):
    # y_hat = X@Wlin + b + ((X@A)*(X@B))@G  ; fit Adam on train, return held-out R^2
    Dy = Y.shape[1]
    Wlin = torch.zeros(D, Dy, device=DEV, requires_grad=True)
    b = Y.mean(0).clone().detach().requires_grad_(True)
    params = [Wlin, b]
    if R > 0:
        A = (torch.randn(D, R, device=DEV) * (1.0/np.sqrt(D))).requires_grad_(True)
        B = (torch.randn(D, R, device=DEV) * (1.0/np.sqrt(D))).requires_grad_(True)
        G = torch.zeros(R, Dy, device=DEV, requires_grad=True)
        params += [A, B, G]
    opt = torch.optim.Adam(params, lr=LR)
    Yvar = ((Yte - Yte.mean(0))**2).sum()
    for step in range(STEPS):
        opt.zero_grad()
        yh = X @ Wlin + b
        if R > 0: yh = yh + ((X @ A) * (X @ B)) @ G
        loss = ((Y - yh)**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        yh = Xte @ Wlin + b
        if R > 0: yh = yh + ((Xte @ A) * (Xte @ B)) @ G
        r2 = 1.0 - float(((Yte - yh)**2).sum() / Yvar)
    return round(r2, 4)


@torch.no_grad()
def gather(rows, layers, need):
    Xs = {L: [] for L in layers}; Ys = {L: [] for L in layers}
    blocks = rows[:, :SEQ].contiguous()
    for i in range(0, blocks.shape[0], 4):
        forward_capture(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), layers)
        for L in layers:
            x, y = CAP[L]; k = x.shape[0]; idx = torch.randperm(k, device=DEV)[:200]
            Xs[L].append(x[idx].cpu()); Ys[L].append(y[idx].cpu())
        if sum(t.shape[0] for t in Xs[layers[0]]) >= need + 512: break
    return {L: (torch.cat(Xs[L], 0)[:need], torch.cat(Ys[L], 0)[:need]) for L in layers}


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    data = gather(rows, LAYERS, NTRAIN + NTEST)
    out = {'layers': {}, 'ranks': RANKS}
    for L in LAYERS:
        X, Y = data[L]; X = X.to(DEV); Y = Y.to(DEV)
        Xtr, Ytr = X[:NTRAIN], Y[:NTRAIN]; Xte, Yte = X[NTRAIN:], Y[NTRAIN:]
        r2s = {}
        for R in RANKS: r2s[str(R)] = fit_bilinear(Xtr, Ytr, Xte, Yte, R)
        # effective rank: smallest R with R^2>=0.9
        eff = next((R for R in RANKS if r2s[str(R)] >= 0.9), None)
        out['layers'][str(L)] = {'r2_by_rank': r2s, 'eff_rank_90': eff, 'linear_r2': r2s['0'], 'bilinear256_r2': r2s['256'],
                                 'gain_256_over_lin': round(r2s['256'] - r2s['0'], 4)}
        print(f"L{L:>2}: R2 by rank {r2s} | linear {r2s['0']} -> bilinear256 {r2s['256']} (gain {out['layers'][str(L)]['gain_256_over_lin']})", flush=True)
        del X, Y
    mids = [8, 11, 15]
    out['mid_linear_r2'] = round(float(np.mean([out['layers'][str(L)]['linear_r2'] for L in mids])), 3)
    out['mid_bilinear256_r2'] = round(float(np.mean([out['layers'][str(L)]['bilinear256_r2'] for L in mids])), 3)
    out['pred_a_bilinear_breaks_ceiling'] = bool(out['mid_bilinear256_r2'] > out['mid_linear_r2'] + 0.3 and out['mid_bilinear256_r2'] > 0.7)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"MIDDLE (L8/11/15): linear R2 {out['mid_linear_r2']} -> rank-256 bilinear R2 {out['mid_bilinear256_r2']}", flush=True)
    print(f"pred_a bilinear breaks the linear ceiling: {out['pred_a_bilinear_breaks_ceiling']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
