"""PATH LINEARITY v2 -- HELD-OUT LINEAR PROBE (completes 754c honestly). v1's
scalar-alpha var-explained (0.22 under attn-off) UNDERSTATED linearity because it
forced a single scalar onto the weight-derived prediction W_Left1@(lambda0*mlp0).
The right question is: is the same-position map from Down_0's output to Left_1's
read-change LINEAR at all? Test with a FULL linear map (ridge), fit on half the
tokens, R2 on the held-out half -- the learned map absorbs lambda0, the rms
per-token gain g_t, and the radial projection, so a high held-out R2 means the
same-position path is linear (only attention's cross-position mixing is left out).

Conditions: block-1 attention ON vs ZEROED. Target = Delta(Left_1 input) when
Down_0's mlp output is ablated; regressor = Down_0's mlp output (same position).

REGISTERED PREDICTIONS:
  (0) SANITY: attn-off held-out R2 > attn-on (removing cross-position mixing helps);
  (a) SAME-POSITION PATH IS LINEAR: attn-off held-out R2 >= 0.6 (a full linear map
      explains most of the same-position read-change) -- confirms 754c that the
      composition is mostly linear per position and attention position-mixing is
      the decorrelator, not a hard nonlinearity;
  (b) report attn-on vs attn-off held-out R2 + the shuffled null;
  NULL: fitting on shuffled (mismatched) token pairs gives held-out R2 ~ 0."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'path_linearity_v2_results.json'
NFIT = 64; ATTN_OFF = {'on': False}; KILL0 = {'on': False}


def attn1_hook(mo, i_, o_):
    if not ATTN_OFF['on']: return o_
    return (torch.zeros_like(o_[0]), o_[1]) if isinstance(o_, tuple) else torch.zeros_like(o_)


def mlp0_hook(mo, i_, o_):
    return torch.zeros_like(o_) if KILL0['on'] else o_


@torch.no_grad()
def capture_mlp0(rows, n):
    cap = []
    h = m.transformer.h[0].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def left1_input(rows, n):
    cap = []
    h = m.transformer.h[1].mlp.Left.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


@torch.no_grad()
def measure_delta(rows, n, attn_off):
    ATTN_OFF['on'] = attn_off
    KILL0['on'] = False; rf = left1_input(rows, n)
    KILL0['on'] = True;  rk = left1_input(rows, n)
    KILL0['on'] = False; ATTN_OFF['on'] = False
    return rf - rk


def ridge_r2(X, Y, ridge=1e-2, seed=0):
    """fit Y ~ X M on train half, held-out R2 on test half (center both)."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g, device=X.device)
    ntr = X.shape[0]//2; tr, te = perm[:ntr], perm[ntr:]
    Xtr, Ytr = X[tr], Y[tr]; Xte, Yte = X[te], Y[te]
    mx = Xtr.mean(0, keepdim=True); my = Ytr.mean(0, keepdim=True)
    Xc = Xtr - mx; Yc = Ytr - my
    A = Xc.T @ Xc; A.diagonal().add_(ridge*float(A.diagonal().mean()))
    M = torch.linalg.solve(A, Xc.T @ Yc)
    Yhat = (Xte - mx) @ M + my
    ss_res = ((Yte - Yhat)**2).sum(); ss_tot = ((Yte - Yte.mean(0, keepdim=True))**2).sum()
    return float(1 - ss_res/ss_tot.clamp_min(1e-9))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    ha = m.transformer.h[1].attn.register_forward_hook(attn1_hook)
    hk = m.transformer.h[0].mlp.register_forward_hook(mlp0_hook)
    lam0 = float(m.transformer.h[1].lambdas[0])
    mlp0 = capture_mlp0(rows, NFIT)                    # (N, D) Down_0 write per token

    res = {}
    for tag, off in [('full', False), ('attn_off', True)]:
        dr = measure_delta(rows, NFIT, off)            # (N, D) Left_1 input change
        res[tag] = round(ridge_r2(mlp0, dr), 4)
        print(f'{tag:9s}: held-out linear-probe R2 {res[tag]:.3f}', flush=True)
    # shuffled null under attn-off
    dr = measure_delta(rows, NFIT, True)
    g = torch.Generator(device=DEV).manual_seed(0)
    null = ridge_r2(mlp0[torch.randperm(mlp0.shape[0], generator=g, device=DEV)], dr)
    print(f'shuffled null: R2 {null:.3f}', flush=True)
    ha.remove(); hk.remove()

    p0 = res['attn_off'] > res['full']
    pa = res['attn_off'] >= 0.6
    null_ok = abs(null) < 0.1
    out = {'lambda0': round(lam0, 4), 'held_out_r2': res, 'shuffled_null_r2': round(null, 4),
           'pred_0': bool(p0), 'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(0) attn-off>full: {p0}; (a) same-position path LINEAR (attn-off R2>=0.6): {pa}; NULL shuffled~0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
