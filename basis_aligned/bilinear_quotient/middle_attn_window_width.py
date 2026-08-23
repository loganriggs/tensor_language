"""The last under-characterized band: MIDDLE attention L5-11. §1054 showed a local window [cur+3prev] of the residual
recovers 0.5-0.68 of these layers' (modest) loss -- what is the unrecovered ~30-45%? Test whether it is longer-range
LOCAL routing (a WIDER window closes it) or genuinely non-local/content (wider window saturates below full). Fit ridge
stand-ins from [residual at cur, prev1..W] to each middle attention output for window widths W in {3,7,15}, held-out,
per-layer loss-recovery. Front layers L0-2 included as a contrast (should already be high at W=3).

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate = 0; shuffled-window null ~0.
  (a) LONGER-RANGE ROUTING: if middle-attention recovery RISES markedly from W=3 to W=15 (toward >0.8), the middle
      attention is longer-range LOCAL routing (just a wider window than the front);
  (b) NON-LOCAL SATURATION: if recovery SATURATES below ~0.7 as W grows (wide window no better than narrow), the middle
      attention's residual is genuinely non-local / content-dependent, not window-reconstructible. Report per-layer
      recovery vs W + shuffled null."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_attn_window_width_results.json'
NEVAL = 200; SEQ = 256; LAYERS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]; WS = [3, 7, 15]; RIDGE = 1e3
SUB = {'L': None, 'on': False, 'mode': None, 'W': None, 'M': {}, 'gmean': {}, 'shuf': False}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def resid_window(resin, W):
    # resin: attention INPUT residual (B,T,D); build [cur, prev1..W]
    B, T, _ = resin.shape; feats = [resin]
    for k in range(1, W+1):
        sh = torch.zeros_like(resin); sh[:, k:] = resin[:, :T-k]; feats.append(sh)
    return torch.cat(feats, -1)                                    # (B,T,(W+1)*D)


def sub_hook(L):
    def h(mo, i_, o_):
        resin = (i_[0] if isinstance(i_, tuple) else i_).float()
        if not SUB['on'] or SUB['L'] != L: return None
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if SUB['mode'] == 'mean':
            ny = SUB['gmean'][L].view(1, 1, D).expand(B, T, D)
        else:
            W = SUB['W']; f = resid_window(resin, W).reshape(-1, (W+1)*D)
            if SUB['shuf']: f = f[torch.randperm(f.shape[0], device=DEV)]
            f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
            ny = (f1 @ SUB['M'][(L, W)]).reshape(B, T, D)
        return (ny.to(y.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(y.dtype)
    return h


@torch.no_grad()
def capture(blocks):
    xin = {L: [] for L in LAYERS}; yout = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        a = m.transformer.h[L].attn
        def mk(L):
            def h(mo, i_, o_):
                xin[L].append((i_[0] if isinstance(i_, tuple) else i_).float().cpu())
                yout[L].append((o_[0] if isinstance(o_, tuple) else o_).float().reshape(-1, D))
            return h
        hs.append(a.register_forward_hook(mk(L)))
    SUB['on'] = False
    for i in range(0, blocks.shape[0], 4): fwd(blocks[i:i+4].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    return ({L: torch.cat(xin[L], 0) for L in LAYERS}, {L: torch.cat(yout[L], 0) for L in LAYERS})


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    Xin, Yout = capture(tr)
    for L in LAYERS:
        xin = Xin[L].to(DEV)                     # (nseq,T,D) cpu->gpu
        Yl = Yout[L].to(DEV); SUB['gmean'][L] = Yl.mean(0)
        for W in WS:
            f = resid_window(xin, W).reshape(-1, (W+1)*D)
            f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
            A = f1.T @ f1 + RIDGE*torch.eye((W+1)*D+1, device=DEV)
            SUB['M'][(L, W)] = torch.linalg.solve(A, f1.T @ Yl); del f, f1, A
        del xin, Yl; Xin[L] = None; Yout[L] = None
    hooks = [m.transformer.h[L].attn.register_forward_hook(sub_hook(L)) for L in LAYERS]
    SUB['on'] = False; ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['on'] = True; SUB['L'] = L
        SUB['mode'] = 'mean'; ce_ma = ce(te); denom = max(ce_ma - ce_full, 1e-6)
        rec = {}
        for W in WS:
            SUB['mode'] = 'win'; SUB['W'] = W; SUB['shuf'] = False; rec[f'W{W}'] = round(float((ce_ma - ce(te))/denom), 3)
        SUB['mode'] = 'win'; SUB['W'] = WS[-1]; SUB['shuf'] = True; rec['shuf_null'] = round(float((ce_ma - ce(te))/denom), 3); SUB['shuf'] = False
        SUB['on'] = False
        out['layers'][str(L)] = {'meanabl_cost': round(ce_ma-ce_full, 3), **rec}
        print(f"attn{L}: cost {ce_ma-ce_full:.3f} | W3 {rec['W3']} | W7 {rec['W7']} | W15 {rec['W15']} | null {rec['shuf_null']}", flush=True)
    for h in hooks: h.remove()
    mid = [5, 6, 7, 8, 9, 10, 11]
    out['middle_W3'] = round(float(np.mean([out['layers'][str(L)]['W3'] for L in mid])), 3)
    out['middle_W15'] = round(float(np.mean([out['layers'][str(L)]['W15'] for L in mid])), 3)
    out['middle_gain_W3_to_W15'] = round(out['middle_W15'] - out['middle_W3'], 3)
    out['pred_a_longer_range'] = bool(out['middle_W15'] > 0.8)
    out['pred_b_saturates_nonlocal'] = bool(out['middle_gain_W3_to_W15'] < 0.1 and out['middle_W15'] < 0.75)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"middle W3 {out['middle_W3']} -> W15 {out['middle_W15']} (gain {out['middle_gain_W3_to_W15']}) | pred_a longer-range {out['pred_a_longer_range']} | pred_b nonlocal-saturate {out['pred_b_saturates_nonlocal']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
