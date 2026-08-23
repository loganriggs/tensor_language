"""§1050 showed per-module deep-middle stand-ins DON'T COMPOSE: fit on the clean stream, they see a corrupted residual
stream at eval and compound below the mean-ablate floor (negative recovery). The fix is to fit each layer's stand-in ON
THE CORRUPTED STREAM: greedily replace L6, then capture L7's ACTUAL (corrupted) input and true MLP output under that
replacement, fit L7, replace it, capture L8 under 6+7 replaced, ... This is distribution-matched, so it removes the
mismatch and gives the HONEST "deep-middle band understood %" for stand-in = [per-token mean] + [content deviation
projected to rank-K]. If greedy reaches ~0.9, the band IS understood by that structure once composition is handled; if
it plateaus below, the residual is genuine high-rank content.

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate-all-nine = 0; shuffled-input null << 0.
  (a) GREEDY >> CLEAN-FIT (§1050): corrupted-stream fit recovers MUCH more than the clean-fit band replace (which was
      negative at every K) -> the negative §1050 numbers were a composition/distribution-shift artifact, not the band
      being unrecoverable;
  (b) HONEST BAND %: report greedy band loss-recovery vs K. If it plateaus below 0.9 at K=512, the deep-middle band's
      residual is genuine high-rank content (consistent with §1042); if it reaches 0.9, the band is understood as
      token-mean + rank-K content once composition is handled."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bottomup_middle_greedy_results.json'
NEVAL = 200; SEQ = 256; LAYERS = list(range(6, 15)); KS = [128, 512]; RIDGE = 1e2
G = {'replace': set(), 'capture': None, 'mode': 'off', 'K': None, 'shuf': False, 'tok': None,
     'xbar': {}, 'B': {}, 'M': {}, 'gmean': {}, 'capX': None, 'capY': None}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def standin(L, xf, tokf, K):
    xbar = G['xbar'][L][tokf]; dev = xf - xbar
    coords = dev @ G['B'][L][:, :K]
    feat = torch.cat([xbar, coords, torch.ones(xf.shape[0], 1, device=DEV)], 1)
    return feat @ G['M'][(L, K)]


def hook(L):
    def h(mo, i_, o_):
        x = (i_[0] if isinstance(i_, tuple) else i_).float(); o = o_[0] if isinstance(o_, tuple) else o_
        B_, T, _ = o.shape
        if G['capture'] == L:                    # record corrupted input + true output, pass through
            G['capX'].append(x.reshape(-1, D)); G['capY'].append(o.float().reshape(-1, D))
            return None
        if L in G['replace'] and G['mode'] != 'off':
            tokf = G['tok'].reshape(-1); xf = x.reshape(-1, D)
            if G['mode'] == 'mean':
                ny = G['gmean'][L].view(1, 1, D).expand(B_, T, D)
            else:
                if G['shuf']:
                    p = torch.randperm(xf.shape[0], device=DEV); xf = xf[p]; tokf = tokf[p]
                ny = standin(L, xf, tokf, G['K']).reshape(B_, T, D)
            return (ny.to(o.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(o.dtype)
        return None
    return h


@torch.no_grad()
def cap_layer(blocks, L):
    """Forward with layers < L (in LAYERS) already replaced; capture L's corrupted input + true output."""
    G['capture'] = L; G['capX'] = []; G['capY'] = []; toks = []
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); G['tok'] = idx; toks.append(idx.reshape(-1)); forward_logits(idx)
    G['capture'] = None
    return torch.cat(G['capX'], 0), torch.cat(G['capY'], 0), torch.cat(toks, 0)


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous(); G['tok'] = idx
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    hooks = [m.transformer.h[L].mlp.register_forward_hook(hook(L)) for L in LAYERS]
    out = {'perK': {}, 'clean_fit_ref_512': -0.166}  # §1050 shared K=512 for reference
    for K in KS:
        # greedy fit on corrupted stream, largest layer set built incrementally
        G['replace'] = set(); G['mode'] = 'standin'; G['K'] = K
        for L in LAYERS:
            X, Y, tok = cap_layer(tr, L)              # corrupted input, true output, tokens (train)
            xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
            xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
            xbar = xbar / cnts.clamp_min(1).unsqueeze(1); G['xbar'][L] = xbar; G['gmean'][L] = Y.mean(0)
            dev = X - xbar[tok]
            _, _, Vt = torch.linalg.svd(dev - dev.mean(0), full_matrices=False); G['B'][L] = Vt[:K].T.contiguous()
            coords = dev @ G['B'][L][:, :K]
            feat = torch.cat([xbar[tok], coords, torch.ones(X.shape[0], 1, device=DEV)], 1)
            G['M'][(L, K)] = torch.linalg.solve(feat.T @ feat + RIDGE*torch.eye(D+K+1, device=DEV), feat.T @ Y)
            G['replace'].add(L)                        # now replace L for subsequent captures
            del X, Y, dev, feat
        # evaluate: all nine replaced
        G['replace'] = set(LAYERS)
        G['mode'] = 'off'; ce_full = ce(te)
        G['mode'] = 'mean'; ce_ma = ce(te); denom = max(ce_ma - ce_full, 1e-6)
        G['mode'] = 'standin'; G['shuf'] = False; ce_st = ce(te)
        rec = round(float((ce_ma - ce_st)/denom), 3)
        out['perK'][str(K)] = {'recovery_greedy': rec, 'meanabl_cost': round(ce_ma-ce_full, 3)}
        if K == KS[-1]:
            G['shuf'] = True; ce_sh = ce(te); G['shuf'] = False
            out['shuffled_null'] = round(float((ce_ma - ce_sh)/denom), 3)
            out['ce_full'] = round(ce_full, 4)
        print(f"greedy K={K}: recovery {rec} (meanabl cost {ce_ma-ce_full:.3f})", flush=True)
    for h in hooks: h.remove()
    kmax = str(KS[-1])
    out['pred_a_greedy_beats_clean'] = bool(out['perK'][kmax]['recovery_greedy'] > out['clean_fit_ref_512'])
    out['pred_b_band_pct_512'] = out['perK'][kmax]['recovery_greedy']
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a greedy>>clean: {out['pred_a_greedy_beats_clean']} | honest band % @512 = {out['pred_b_band_pct_512']}", flush=True)
    print(f"shuffled null {out.get('shuffled_null')} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
