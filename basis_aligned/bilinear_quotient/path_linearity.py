"""PATH LINEARITY between Down_0 (write) and Left_1 (read) -- settle whether the
composition is mostly LINEAR (user: lambda folds as a scalar, rms_norm here is a
per-token SCALAR gain with no learnable gamma, attention is bilinear/no-softmax
but MIXES POSITIONS). Corrects 754b's imprecise "nonlinearity dominates": the ~0
centered corr there was measured at Left_1's input -- AFTER block-1 attention
(cross-position mixing) and rms_norm (per-token scale) -- not evidence of a hard
nonlinearity.

Test: predict Left_1's pre-activation CHANGE when Down_0's mlp output is ablated,
from the WEIGHT-ONLY same-position linear coupling  pred_t = W_Left1 @ (lambda0 *
mlp0_out_t), and correlate (centered) under conditions that peel off each piece:
  (full)     block-1 attention ON  -> corr_full   (reproduces 754b ~0)
  (attn-off) block-1 attention ZEROED -> corr_noattn (removes cross-position mixing)
If corr_noattn >> corr_full, ATTENTION position-mixing is the decorrelator and the
SAME-POSITION composition is linear (user's point). Also report variance-explained
of the same-position linear projection, and the same with the WEIGHT-ACTION SAE
recon of Down_0 in place of the true output (weight-only prediction).

REGISTERED PREDICTIONS:
  (0) SANITY: killing mlp_0 changes Left_1's read (nonzero delta);
  (a) MOSTLY LINEAR: corr_noattn >= 0.6 and >> corr_full (attention position-mixing
      is the main decorrelator; same-position path is linear). The weight-only SAE
      prediction tracks the true-output prediction (within 0.1 corr);
  (b) report corr_full, corr_noattn, variance-explained, and the attention vs rms
      split;
  NULL: a shuffled (wrong-token) prediction has corr ~0 under attn-off too."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'path_linearity_results.json'
NFIT = 48; ATTN_OFF = {'on': False}


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def attn1_hook(mo, i_, o_):
    # block-1 attention returns (x1, v1); zero x1 (kill cross-position mixing), keep v1
    if not ATTN_OFF['on']: return o_
    return (torch.zeros_like(o_[0]), o_[1]) if isinstance(o_, tuple) else torch.zeros_like(o_)


KILL0 = {'on': False}
def mlp0_hook(mo, i_, o_):
    return torch.zeros_like(o_) if KILL0['on'] else o_


@torch.no_grad()
def capture_left1_and_mlp0(rows, n):
    """return (mlp0_out per token, left1_input per token) for the full model."""
    mlp0, l1in = [], []
    h1 = m.transformer.h[0].mlp.register_forward_hook(lambda mo, i_, o_: mlp0.append(o_.detach().float().reshape(-1, D)))
    h2 = m.transformer.h[1].mlp.Left.register_forward_hook(lambda mo, i_, o_: l1in.append(i_[0].detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    h1.remove(); h2.remove()
    return torch.cat(mlp0, 0), torch.cat(l1in, 0)


@torch.no_grad()
def left1_input(rows, n):
    cap = []
    h = m.transformer.h[1].mlp.Left.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def ccorr(a, b):
    a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
    return float(torch.corrcoef(torch.stack([a.reshape(-1), b.reshape(-1)]))[0, 1])


def var_explained(pred, meas):
    pred = pred - pred.mean(0, keepdim=True); meas = meas - meas.mean(0, keepdim=True)
    # scalar LS fit meas ~ alpha*pred
    alpha = (pred*meas).sum()/ (pred*pred).sum().clamp_min(1e-9)
    return float(1 - ((meas - alpha*pred)**2).sum()/(meas**2).sum().clamp_min(1e-9))


@torch.no_grad()
def measure_delta(rows, n, attn_off):
    ATTN_OFF['on'] = attn_off
    KILL0['on'] = False; r_full = left1_input(rows, n)
    KILL0['on'] = True;  r_kill = left1_input(rows, n)
    KILL0['on'] = False; ATTN_OFF['on'] = False
    return r_full - r_kill


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT)
    ha = m.transformer.h[1].attn.register_forward_hook(attn1_hook)
    hk = m.transformer.h[0].mlp.register_forward_hook(mlp0_hook)

    mlp0, _ = capture_left1_and_mlp0(rows, NFIT)
    W_L = m.transformer.h[1].mlp.Left.weight.data.float().to(DEV)
    lam0 = float(m.transformer.h[1].lambdas[0])
    # weight-only same-position prediction of Left_1 pre-activation change (up to rms scalar)
    pred = (lam0 * mlp0) @ W_L.T                                  # (N, HID)

    res = {}
    for tag, off in [('full', False), ('attn_off', True)]:
        dr = measure_delta(rows, NFIT, off)                      # (N, D) change in Left_1 input
        dpre = dr @ W_L.T                                        # change in Left_1 pre-activation
        res[tag] = {'corr': round(ccorr(pred, dpre), 4), 'var_expl': round(var_explained(pred, dpre), 4),
                    'delta_norm': round(float(dr.norm()/np.sqrt(dr.shape[0])), 4)}
        print(f'{tag:9s}: corr {res[tag]["corr"]:.3f}  var-expl {res[tag]["var_expl"]:.3f}  |delta| {res[tag]["delta_norm"]:.3f}', flush=True)

    # shuffled null under attn-off
    g = torch.Generator(device=DEV).manual_seed(0)
    dr = measure_delta(rows, NFIT, True); dpre = dr @ W_L.T
    perm = torch.randperm(pred.shape[0], generator=g, device=DEV)
    corr_null = ccorr(pred[perm], dpre)
    print(f'shuffled null (attn-off): corr {corr_null:.3f}', flush=True)
    ha.remove(); hk.remove()

    p0 = res['full']['delta_norm'] > 1e-3
    pa = res['attn_off']['corr'] >= 0.6 and res['attn_off']['corr'] - res['full']['corr'] >= 0.3
    null_ok = abs(corr_null) < 0.15
    out = {'lambda0': round(lam0, 4), 'conditions': res, 'shuffled_null_corr': round(corr_null, 4),
           'pred_0': bool(p0), 'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(0) delta nonzero: {p0}; (a) mostly-linear same-position (attn-off corr>=0.6 & >>full): {pa}; '
          f'NULL shuffled~0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
