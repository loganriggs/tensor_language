"""SETTLE §990's remaining puzzle by DIRECT TEST (not inference — I've made two wrong inferences here already:
§989's 'near-constant gate', then §989's 'Down reads out the linear part at front', both refuted).

The tension: (§941, certified) the FRONT bilinear-MLP output is ~90-98% LINEARLY recoverable from its input x,
the MIDDLE only ~38%; BUT (§990) the INTERACTION term Down(u*w) DOMINATES the output variance at ALL layers
(~86-98%), front included. The ONLY reconciliation consistent with both: the interaction term Down(u*w), though it
dominates the output VARIANCE, is itself LINEARLY PREDICTABLE from x at the FRONT but NOT at the MIDDLE (a quadratic
in x that collapses to ~linear over the data at the front, genuinely nonlinear in the middle).

TEST IT: with x = mlp input (D=1152), Lx=Left(x), Rx=Right(x), a=mean(Lx), b=mean(Rx), u=Lx-a, w=Rx-b:
  target_int  = Down(u*w)              (the interaction output — no bias)
  target_full = Down(Lx*Rx) + bias     (the full MLP output; §941's object)
Fit a RIDGE linear map x -> target on a train split, report test R^2 (fraction of target variance explained by a
LINEAR function of x). Means a,b estimated on TRAIN only (no leakage).

REGISTERED PREDICTIONS:
  (0) SANITY: R^2 in [<=0, 1]; at the FRONT R^2_full is high (~0.9, replicating §941); at the MIDDLE R^2_full is
      much lower (~0.4, §941).
  (a) INTERACTION IS LINEARLY-PREDICTABLE AT FRONT, NOT MIDDLE: R^2_int at the FRONT (L1) is HIGH (>~0.7) and
      R^2_int(front) >> R^2_int(middle L8/L11) -> the front's near-linearity is that its interaction term collapses
      to ~linear in x, NOT that the interaction is absent (§990) or projected out (§989 refuted). This is the
      corrected mechanism for §941's front linearity;
  (b) report per-layer R^2_int and R^2_full (test split), and the front-vs-middle gap in R^2_int."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'interaction_linear_predictability_results.json'
NEVAL = 220; SEQ = 256; LAYERS = [1, 4, 8, 11, 15]
NTRAIN = 16000; NVAL = 4000; NTEST = 6000  # >> D=1152 so the linear map generalizes
RIDGES = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]  # swept on val, best-full-R2 picked per layer
CAP = {}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = i_[0] if isinstance(i_, tuple) else i_  # mlp input (D)
                CAP[L] = (x.detach().float().reshape(-1, D),
                          mlp.Left(x).detach().float().reshape(-1, mlp.Left.weight.shape[0]),
                          mlp.Right(x).detach().float().reshape(-1, mlp.Right.weight.shape[0]))
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    readout(x)
    for h in hs: h.remove()


def _r2(Y, pred):
    ss_res = float(((Y - pred)**2).sum()); ss_tot = float(((Y - Y.mean(0, keepdim=True))**2).sum())
    return 1.0 - ss_res/max(ss_tot, 1e-9)


@torch.no_grad()
def ridge_fit_r2(Xtr, Ytr, Xva, Yva, Xte, Yte, lam):
    Xtr1 = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=Xtr.device)], 1)
    Xva1 = torch.cat([Xva, torch.ones(Xva.shape[0], 1, device=Xtr.device)], 1)
    Xte1 = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=Xtr.device)], 1)
    A = Xtr1.T @ Xtr1 + lam * torch.eye(Xtr1.shape[1], device=Xtr.device)
    W = torch.linalg.solve(A, Xtr1.T @ Ytr)
    return round(_r2(Yva, Xva1 @ W), 4), round(_r2(Yte, Xte1 @ W), 4)


@torch.no_grad()
def best_ridge_r2(X, Y, ntr, nva):
    # pick lam by best VAL R^2; report TEST R^2 at that lam (no test leakage)
    Xtr, Ytr = X[:ntr], Y[:ntr]; Xva, Yva = X[ntr:ntr+nva], Y[ntr:ntr+nva]; Xte, Yte = X[ntr+nva:], Y[ntr+nva:]
    best_va = -1e9; best_te = None; best_lam = None
    for lam in RIDGES:
        va, te = ridge_fit_r2(Xtr, Ytr, Xva, Yva, Xte, Yte, lam)
        if va > best_va: best_va = va; best_te = te; best_lam = lam
    return best_te, best_lam


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    X_all = {L: [] for L in LAYERS}; L_all = {L: [] for L in LAYERS}; R_all = {L: [] for L in LAYERS}
    need = NTRAIN + NVAL + NTEST
    for i in range(0, nb, 4):
        forward_capture(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), LAYERS)
        for L in LAYERS:
            x, lx, rx = CAP[L]; n = x.shape[0]; idx = torch.randperm(n, device=DEV)[:512]
            X_all[L].append(x[idx].cpu()); L_all[L].append(lx[idx].cpu()); R_all[L].append(rx[idx].cpu())
        if sum(t.shape[0] for t in X_all[LAYERS[0]]) >= need + 1024: break
    out = {'layers': {}}
    for L in LAYERS:
        X = torch.cat(X_all[L], 0)[:need].to(DEV)
        Lx = torch.cat(L_all[L], 0)[:need].to(DEV); Rx = torch.cat(R_all[L], 0)[:need].to(DEV)
        navail = X.shape[0]; ntr = min(NTRAIN, navail - 2000); nva = min(NVAL, (navail-ntr)//2)
        mlp = m.transformer.h[L].mlp
        a = Lx[:ntr].mean(0, keepdim=True); b = Rx[:ntr].mean(0, keepdim=True)  # means on TRAIN only
        u = Lx - a; w = Rx - b
        dn = lambda h: F.linear(h, mlp.Down.weight)
        bias = mlp.Down.bias if mlp.Down.bias is not None else 0
        tgt_int = dn(u*w)                 # interaction output
        tgt_full = dn(Lx*Rx) + bias       # full MLP output (§941 object)
        r2_int, lam_int = best_ridge_r2(X, tgt_int, ntr, nva)
        r2_full, lam_full = best_ridge_r2(X, tgt_full, ntr, nva)
        out['layers'][str(L)] = {'r2_int': r2_int, 'r2_full': r2_full, 'lam_int': lam_int, 'lam_full': lam_full,
                                 'ntrain': ntr, 'nval': nva, 'ntest': navail-ntr-nva}
        print(f"L{L:>2}: R^2 interaction {r2_int} (lam {lam_int}) | R^2 full {r2_full} (lam {lam_full}) [ntr {ntr}]", flush=True)
        del X, Lx, Rx, u, w, tgt_int, tgt_full
    fi = out['layers']['1']['r2_int']; mi = float(np.mean([out['layers'][str(L)]['r2_int'] for L in [8, 11]]))
    ff = out['layers']['1']['r2_full']; mf = float(np.mean([out['layers'][str(L)]['r2_full'] for L in [8, 11]]))
    out['front_L1_r2_int'] = fi; out['mid_L8_11_r2_int'] = round(mi, 3)
    out['front_L1_r2_full'] = ff; out['mid_L8_11_r2_full'] = round(mf, 3)
    # SANITY GATE: the instrument must replicate §941 (front full-output ~linear) or the run is INVALID
    out['sanity_full_replicates_941'] = bool(ff > 0.7 and ff > mf)
    out['pred_a_interaction_linear_at_front'] = bool(out['sanity_full_replicates_941'] and fi > 0.7 and fi > mi + 0.2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"SANITY full replicates §941 (front>0.7 & >mid): {out['sanity_full_replicates_941']}", flush=True)
    print(f"front(L1) R^2_int {fi} vs mid(L8,11) {mi:.3f}  |  R^2_full front {ff} mid {mf:.3f}", flush=True)
    print(f"(a) interaction linearly-predictable at front, not middle: {out['pred_a_interaction_linear_at_front']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
