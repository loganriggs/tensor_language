"""Causal capstone for the high-rank claim. §1055 named the content (interpretable topic/register axes) and showed the
top-10 PCs are only ~12% of variance. If the content is GENUINELY high-rank, its loss impact should be spread ~evenly
across its dimensions -- no privileged few -- so loss-increase-per-variance-removed should be ~FLAT across top/mid/bottom
content-PC bands. If instead a few top directions carry most of the loss, the content would be effectively low-rank and
compressible. Test causally: project a band of content directions OUT of the deep-middle residual stream (after each
block, L6-14) and measure the loss increase, for top/mid/bottom bands of the content-deviation PCA + a random-direction
control. loss-per-variance flat -> uniformly load-bearing high-rank content (explains why no low-rank stand-in works).

REGISTERED PREDICTIONS:
  (0) SANITY: no ablation = ce_full; a random-K projection removes little variance and costs little.
  (a) UNIFORMLY LOAD-BEARING: loss-increase-per-variance-removed is roughly FLAT across top/mid/bottom content bands
      (within ~2-3x), NOT concentrated in the top band -> the content is genuinely high-rank, every dimension pulls its
      weight, which is why a low-rank stand-in fails (§1042/1050/1051);
  (b) report per-band variance-removed, loss-increase, and loss-per-variance; top-band should still cost more in
      ABSOLUTE loss (it removes more variance) but not disproportionately per unit variance."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_ablation_results.json'
NEVAL = 200; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL_RANGE = list(range(6, 15)); KB = 64
CAP = {}
PROJ = {'U': None}   # (D,K) directions to project OUT after each block in ABL_RANGE; None = off


def forward_capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    fwd(idx)
    for h in hs: h.remove()


def fwd(idx, ablate=False):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ablate and PROJ['U'] is not None and li in ABL_RANGE:
            U = PROJ['U']; x = x - (x @ U) @ U.T
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def ce(blocks, ablate):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx, ablate).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    # build content-deviation PCA from pooled L8-12 mlp-input deviation
    for L in REF_LAYERS: CAP[L] = []
    idsL = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); forward_capture(idx, REF_LAYERS)
    tok = torch.cat(idsL, 0); devsum = None
    for L in REF_LAYERS:
        X = torch.cat(CAP[L], 0); xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X, dv; CAP[L] = []
    dev = devsum / len(REF_LAYERS); devc = dev - dev.mean(0)
    U_full, S, Vt = torch.linalg.svd(devc, full_matrices=False)  # Vt (min(N,D), D)
    S2 = S**2; tot = float(S2.sum())
    bands = {'top': (0, KB), 'mid': (544, 544+KB), 'bottom': (D-KB, D)}
    # random orthonormal control
    g = torch.Generator(device=DEV).manual_seed(0)
    Rnd = torch.linalg.qr(torch.randn(D, KB, generator=g, device=DEV))[0]
    te = blocks  # eval on all (ablation is train-free)
    ce_full = ce(te, ablate=False)
    out = {'ce_full': round(ce_full, 4), 'KB': KB, 'abl_range': [ABL_RANGE[0], ABL_RANGE[-1]], 'bands': {}}
    for name, (a, b) in bands.items():
        PROJ['U'] = Vt[a:b].T.contiguous()
        ce_a = ce(te, ablate=True); var_removed = float(S2[a:b].sum())/tot
        inc = ce_a - ce_full
        out['bands'][name] = {'var_removed': round(var_removed, 4), 'loss_increase': round(inc, 4),
                              'loss_per_var': round(inc/max(var_removed, 1e-6), 3)}
        print(f"{name} PCs[{a}:{b}]: var_removed {var_removed:.4f} | loss+ {inc:.4f} | loss/var {inc/max(var_removed,1e-6):.2f}", flush=True)
    PROJ['U'] = Rnd; ce_r = ce(te, ablate=True)
    out['random_control'] = {'loss_increase': round(ce_r - ce_full, 4)}
    print(f"random-{KB} control: loss+ {ce_r-ce_full:.4f}", flush=True)
    PROJ['U'] = None
    lpv = [out['bands'][n]['loss_per_var'] for n in ('top', 'mid', 'bottom')]
    ratio = max(lpv)/max(min(lpv), 1e-6)
    out['loss_per_var_spread_ratio'] = round(ratio, 2)
    out['pred_a_uniformly_loadbearing'] = bool(ratio < 4.0)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"loss/var top,mid,bottom = {lpv} | spread {ratio:.2f}x | pred_a uniform {out['pred_a_uniformly_loadbearing']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
