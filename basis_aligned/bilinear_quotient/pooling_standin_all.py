"""BOTTOM-UP, the attention lane's MATCHED stand-in. §1047 scored all attention with a LOCAL WINDOW [cur+3prev] and the
middle band came out low (0.03-0.62) -- but the middle attention is a BROAD recency-weighted POOL of the value residual
(§894/930/1039), which a local window structurally cannot represent. Give it the matched stand-in: current token + a
CAUSAL RUNNING MEAN of the attention input (a broad pool). Fit ridge maps for three stand-ins per layer -- window-only,
pool-only [cur + causal cummean], pool+window -- and measure loss-recovery per layer. Expect a DOUBLE DISSOCIATION:
front attention (local routers) is captured by the window and NOT helped by the pool; middle attention (poolers) is
captured by the pool and NOT by the window. If so, the attention lane's low benchmark bars are a stand-in mismatch, and
the middle attention is understood as a broad causal pool -- lifting that lane honestly.

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate = 0; shuffled-feature null ~0.
  (a) FRONT = WINDOW: attn0-2 window-recovery >> pool-recovery (local routing, pool doesn't help);
  (b) MIDDLE = POOL: attn5-14 pool-recovery >> window-recovery (§1047) -> the middle attention IS understood as a broad
      causal pool; pool+window is the best matched stand-in overall;
  (c) report per-layer recovery for window / pool / pool+window + shuffled null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pooling_standin_all_results.json'
NEVAL = 200; SEQ = 256; LAYERS = list(range(18)); RIDGE = 1e3; NPREV = 3
# feature modes: 'win' -> (NPREV+1)*D ; 'pool' -> 2*D (cur + cummean) ; 'poolwin' -> (NPREV+2)*D
DIMS = {'win': (NPREV+1)*D, 'pool': 2*D, 'poolwin': (NPREV+2)*D}
SUB = {'L': None, 'on': False, 'mode': None, 'M': {}, 'gmean': {}, 'cache': {}, 'shuf': False}


def feats(xin):
    # xin: attention input (B,T,D), rms-normed residual at the layer. Build all three feature tensors.
    B, T, _ = xin.shape
    win = [xin] + [torch.cat([torch.zeros(B, k, D, device=DEV), xin[:, :T-k]], 1) for k in range(1, NPREV+1)]
    win = torch.cat(win, -1)                                   # (B,T,(NPREV+1)*D)
    cummean = torch.cumsum(xin, 1) / torch.arange(1, T+1, device=DEV).view(1, T, 1)
    pool = torch.cat([xin, cummean], -1)                      # (B,T,2D)
    poolwin = torch.cat([win, cummean], -1)                   # (B,T,(NPREV+2)D)
    return {'win': win, 'pool': pool, 'poolwin': poolwin}


def forward_logits(idx):
    SUB['cache'] = None
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(L):
    def h(mo, i_, o_):
        xin = (i_[0] if isinstance(i_, tuple) else i_).float()
        if not SUB['on'] or SUB['L'] != L: return None
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if SUB['mode'] == 'mean':
            ny = SUB['gmean'][L].view(1, 1, D).expand(B, T, D)
        else:
            f = feats(xin)[SUB['mode']].reshape(-1, DIMS[SUB['mode']])
            if SUB['shuf']: f = f[torch.randperm(f.shape[0], device=DEV)]
            f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
            ny = (f1 @ SUB['M'][(L, SUB['mode'])]).reshape(B, T, D)
        return (ny.to(y.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(y.dtype)
    return h


@torch.no_grad()
def capture(blocks):
    # store attention input and output per layer as (nseq,T,D) on CPU (preserve T for causal cummean at fit time)
    xins = {L: [] for L in LAYERS}; youts = {L: [] for L in LAYERS}; hs = []
    for L in LAYERS:
        a = m.transformer.h[L].attn
        def mk(L):
            def h(mo_, i_, o_):
                xins[L].append((i_[0] if isinstance(i_, tuple) else i_).float().cpu())
                youts[L].append((o_[0] if isinstance(o_, tuple) else o_).float().cpu())
            return h
        hs.append(a.register_forward_hook(mk(L)))
    SUB['on'] = False
    for i in range(0, blocks.shape[0], 4): forward_logits(blocks[i:i+4].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    return ({L: torch.cat(xins[L], 0) for L in LAYERS}, {L: torch.cat(youts[L], 0) for L in LAYERS})


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    Xin, Yout = capture(tr)
    for L in LAYERS:
        xin = Xin[L].to(DEV)                         # (nseq,T,D)
        Yl = Yout[L].to(DEV).reshape(-1, D); SUB['gmean'][L] = Yl.mean(0)
        ff = feats(xin)
        for mo in DIMS:
            f = ff[mo].reshape(-1, DIMS[mo]); f1 = torch.cat([f, torch.ones(f.shape[0], 1, device=DEV)], 1)
            A = f1.T @ f1 + RIDGE*torch.eye(DIMS[mo]+1, device=DEV)
            SUB['M'][(L, mo)] = torch.linalg.solve(A, f1.T @ Yl); del f, f1, A
        del xin, Yl, ff
        Xin[L] = None; Yout[L] = None
    del Xin, Yout
    hooks = [m.transformer.h[L].attn.register_forward_hook(sub_hook(L)) for L in LAYERS]
    SUB['on'] = False; ce_full = ce(te); print(f"ce_full {ce_full:.4f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'layers': {}}
    for L in LAYERS:
        SUB['on'] = True; SUB['L'] = L
        SUB['mode'] = 'mean'; ce_ma = ce(te); denom = max(ce_ma - ce_full, 1e-6)
        rec = {}
        for mo in ('win', 'pool', 'poolwin'):
            SUB['mode'] = mo; SUB['shuf'] = False; rec[mo] = round(float((ce_ma - ce(te))/denom), 3)
        SUB['mode'] = 'poolwin'; SUB['shuf'] = True; rec['shuf_null'] = round(float((ce_ma - ce(te))/denom), 3); SUB['shuf'] = False
        SUB['on'] = False
        out['layers'][str(L)] = {'meanabl_cost': round(ce_ma-ce_full, 3), **rec}
        print(f"attn{L}: cost {ce_ma-ce_full:.3f} | win {rec['win']} | pool {rec['pool']} | poolwin {rec['poolwin']} | null {rec['shuf_null']}", flush=True)
    for h in hooks: h.remove()
    def band(mo, ls): return round(float(np.mean([out['layers'][str(L)][mo] for L in ls])), 3)
    out['front_L0_2'] = {'win': band('win', [0,1,2]), 'pool': band('pool', [0,1,2])}
    out['middle_L5_14'] = {'win': band('win', range(5,15)), 'pool': band('pool', range(5,15)), 'poolwin': band('poolwin', range(5,15))}
    out['pred_a_front_window'] = bool(out['front_L0_2']['win'] > out['front_L0_2']['pool'])
    out['pred_b_middle_pool'] = bool(out['middle_L5_14']['pool'] > out['middle_L5_14']['win'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front(0-2) win {out['front_L0_2']['win']} pool {out['front_L0_2']['pool']} | middle(5-14) win {out['middle_L5_14']['win']} pool {out['middle_L5_14']['pool']} poolwin {out['middle_L5_14']['poolwin']}", flush=True)
    print(f"pred_a front=window {out['pred_a_front_window']} | pred_b middle=pool {out['pred_b_middle_pool']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
