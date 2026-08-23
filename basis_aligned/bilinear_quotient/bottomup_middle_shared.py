"""BOTTOM-UP culmination for the deep-middle band. §1049 showed the deep-middle MLPs (L6-14) read/write a largely SHARED
content subspace (top-64 overlap mean 0.58, drifting with depth). So treat the band as ONE content object: replace ALL
nine deep-middle MLPs simultaneously with a stand-in = [per-token mean lookup] + [content deviation projected onto a
rank-K content basis], fit closed-form (ridge, held-out). Two bases: (i) PER-LAYER (each layer its own top-K content
PCA) vs (ii) SHARED (one top-K basis from the pooled deviation across all nine layers). Measure BAND-level loss-recovery
vs K. This gives the honest "deep-middle band" understanding number and tests whether sharing the basis compresses the
loss (cross-layer redundancy) the per-layer view missed. Closed-form throughout (avoids the §1036 Adam pitfall).

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablate-all-nine = recovery 0; at large K both bases -> high recovery (K=D would be ~1).
  (a) HIGH-RANK AT BAND LEVEL: band loss-recovery rises with K but needs large K for 90% (consistent with §1042's
      per-layer high rank) -> the deep-middle band is genuinely high-dimensional content, not a low-rank object;
  (b) SHARING BUYS LITTLE: shared-basis recovery ~= per-layer-basis recovery at matched K (the content is high-rank
      WITHIN the shared span, §1049 overlaps only ~0.58) -> the layers share DIRECTIONS but the information is spread,
      so one shared low-K basis does not recover the band. Report band recovery vs K for both bases + shuffled null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bottomup_middle_shared_results.json'
NEVAL = 200; SEQ = 256; LAYERS = list(range(6, 15)); KS = [64, 128, 256, 512]; RIDGE = 1e2
# SUB: per-layer replacement state. mode in {None(off), 'mean', 'stand'}; basis in {'perlayer','shared'}; 'shuf' bool
SUB = {'on': False, 'mode': None, 'basis': 'perlayer', 'K': None, 'shuf': False,
       'xbar': {}, 'B': {}, 'Bshared': None, 'M': {}, 'gmean': {}}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def feat_for(L, x_flat, tok_flat, K, basis):
    xbar = SUB['xbar'][L][tok_flat]                      # (N,D) per-token mean
    dev = x_flat - xbar
    B = (SUB['Bshared'] if basis == 'shared' else SUB['B'][L])[:, :K]  # (D,K)
    coords = dev @ B                                     # (N,K)
    return torch.cat([xbar, coords], 1)                 # (N, D+K)


@torch.no_grad()
def capture(blocks):
    Xs = {L: [] for L in LAYERS}; Ys = {L: [] for L in LAYERS}; toks = []; hs = []
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_):
                Xs[L].append((i_[0] if isinstance(i_, tuple) else i_).float().reshape(-1, D))
                Ys[L].append((o_[0] if isinstance(o_, tuple) else o_).float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    SUB['on'] = False
    for i in range(0, blocks.shape[0], 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1)); forward_logits(idx)
    for h in hs: h.remove()
    return ({L: torch.cat(Xs[L], 0) for L in LAYERS}, {L: torch.cat(Ys[L], 0) for L in LAYERS}, torch.cat(toks, 0))


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        SUB['tokcur'] = idx
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    nb = rows.shape[0]; ntr = int(0.6*nb); tr = rows[:ntr, :SEQ].contiguous(); te = rows[ntr:, :SEQ].contiguous()
    X, Y, tok = capture(tr); V = int(m.lm_head.weight.shape[0])
    # per-token mean tables + per-layer content bases + pooled shared basis
    devs = []
    for L in LAYERS:
        xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X[L]); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1); SUB['xbar'][L] = xbar
        SUB['gmean'][L] = Y[L].mean(0)
        dev = X[L] - xbar[tok]
        _, _, Vt = torch.linalg.svd(dev - dev.mean(0), full_matrices=False)
        SUB['B'][L] = Vt[:max(KS)].T.contiguous(); devs.append(dev)
    devcat = torch.cat(devs, 0)
    _, _, Vts = torch.linalg.svd(devcat - devcat.mean(0), full_matrices=False)
    SUB['Bshared'] = Vts[:max(KS)].T.contiguous(); del devs, devcat
    # fit stand-in maps per (layer, basis, K)
    for L in LAYERS:
        Yl = Y[L]
        for basis in ('perlayer', 'shared'):
            for K in KS:
                feat = feat_for(L, X[L], tok, K, basis)
                f1 = torch.cat([feat, torch.ones(feat.shape[0], 1, device=DEV)], 1)
                A = f1.T @ f1 + RIDGE*torch.eye(D+K+1, device=DEV)
                SUB['M'][(L, basis, K)] = torch.linalg.solve(A, f1.T @ Yl)
        del Yl
    del X, Y
    ACTIVE = set(LAYERS)
    def band_hook(L):
        def h(mo, i_, o_):
            if not SUB['on'] or L not in ACTIVE: return None
            x = (i_[0] if isinstance(i_, tuple) else i_).float(); o = o_[0] if isinstance(o_, tuple) else o_
            B_, T, _ = o.shape
            if SUB['mode'] == 'mean':
                ny = SUB['gmean'][L].view(1, 1, D).expand(B_, T, D)
            else:
                tok_ = SUB['tokcur'].reshape(-1); xf = x.reshape(-1, D)
                if SUB['shuf']:
                    p = torch.randperm(xf.shape[0], device=DEV); xf = xf[p]; tok_ = tok_[p]
                feat = feat_for(L, xf, tok_, SUB['K'], SUB['basis'])
                ny = (torch.cat([feat, torch.ones(feat.shape[0], 1, device=DEV)], 1) @ SUB['M'][(L, SUB['basis'], SUB['K'])]).reshape(B_, T, D)
            return (ny.to(o.dtype),) + tuple(o_[1:]) if isinstance(o_, tuple) else ny.to(o.dtype)
        return h
    hooks = [m.transformer.h[L].mlp.register_forward_hook(band_hook(L)) for L in LAYERS]

    SUB['on'] = False; ce_full = ce(te)
    SUB['on'] = True; SUB['mode'] = 'mean'; ce_ma = ce(te); denom = max(ce_ma - ce_full, 1e-6)
    print(f"ce_full {ce_full:.4f} | meanablate-all-nine cost {ce_ma-ce_full:.3f}", flush=True)
    out = {'ce_full': round(ce_full, 4), 'band_meanablate_cost': round(ce_ma-ce_full, 3),
           'perlayer': {}, 'shared': {}, 'shuffled_null': {}}
    SUB['mode'] = 'stand'
    for basis in ('perlayer', 'shared'):
        for K in KS:
            SUB['basis'] = basis; SUB['K'] = K; SUB['shuf'] = False
            ce_st = ce(te); rec = round(float((ce_ma - ce_st)/denom), 3)
            out[basis][str(K)] = rec
            print(f"band {basis} K={K}: recovery {rec}", flush=True)
    # shuffled null at largest K, shared
    SUB['basis'] = 'shared'; SUB['K'] = max(KS); SUB['shuf'] = True
    ce_sh = ce(te); out['shuffled_null'][str(max(KS))] = round(float((ce_ma - ce_sh)/denom), 3); SUB['shuf'] = False
    SUB['on'] = False
    for h in hooks: h.remove()
    kmax = str(max(KS))
    out['pred_a_high_rank_band'] = bool(out['shared'][kmax] < 0.9)   # needs even more than max K for 90%
    out['pred_b_sharing_buys_little'] = bool(abs(out['shared'][kmax] - out['perlayer'][kmax]) < 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a high-rank band (shared K={kmax} <0.9): {out['pred_a_high_rank_band']}", flush=True)
    print(f"pred_b sharing~=perlayer: {out['pred_b_sharing_buys_little']} | shuffled null {out['shuffled_null'][kmax]}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
