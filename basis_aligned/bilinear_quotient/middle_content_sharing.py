"""Is the deep-middle ONE content computation on a SHARED subspace, or many independent high-rank maps? §1042 showed
each middle MLP multiplies a high-rank content×content. If the middle layers read/write the SAME content subspace, the
band is a single content object (more unified/understood) rather than 10 independent computations. Measure the
subspace OVERLAP of the content deviation (x_ctx = input minus per-token mean) across deep-middle layers: top-K PCA
subspace per layer, pairwise overlap = ||U_a^T U_b||_F^2 / K (1 = identical, 0 = orthogonal), vs a random-subspace
null.

REGISTERED PREDICTIONS:
  (0) SANITY: self-overlap = 1; random-subspace null overlap ~ K/D (small, ~0.06 for K=64).
  (a) SHARED CONTENT SUBSPACE: pairwise content-subspace overlap across deep-middle layers is HIGH (>~0.5, far above
      the random null) -> the deep-middle reads/writes ONE shared content manifold; the band is a single (high-rank)
      content computation, not 10 independent ones -> more unified/understood;
  (b) report the pairwise overlap matrix + mean, and the random null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_content_sharing_results.json'
NEVAL = 96; SEQ = 256; LAYERS = [6, 8, 10, 12, 14]; K = 64
CAP = {}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def forward_capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): CAP[L] = (i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D)
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    forward_logits(idx)
    for h in hs: h.remove()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d_tok = None
    blocks = rows[:, :SEQ].contiguous()
    import tiktoken  # noqa
    Xs = {L: [] for L in LAYERS}; toks = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); forward_capture(idx, LAYERS); toks.append(idx.reshape(-1))
        for L in LAYERS: Xs[L].append(CAP[L])
    tok = torch.cat(toks, 0); V = int(m.lm_head.weight.shape[0])
    U = {}
    for L in LAYERS:
        X = torch.cat(Xs[L], 0)
        # content deviation = X - per-token mean
        xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        xdev = X - xbar[tok]
        _, _, Vt = torch.linalg.svd(xdev - xdev.mean(0), full_matrices=False)
        U[L] = Vt[:K].T.contiguous()   # (D,K) top-K content directions
        del X, xdev
    def overlap(A, B): return float((A.T @ B).pow(2).sum() / K)
    ov = {}
    for a in LAYERS:
        for b in LAYERS:
            if a < b: ov[f'{a}-{b}'] = round(overlap(U[a], U[b]), 3)
    # random null
    g = torch.Generator(device=DEV).manual_seed(0)
    R1 = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    R2 = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    out = {'pairwise_overlap': ov, 'mean_overlap': round(float(np.mean(list(ov.values()))), 3),
           'random_null_overlap': round(overlap(R1, R2), 3), 'K': K}
    out['pred_a_shared_subspace'] = bool(out['mean_overlap'] > 0.5 and out['mean_overlap'] > 5*out['random_null_overlap'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pairwise content-subspace overlap: {ov}", flush=True)
    print(f"mean overlap {out['mean_overlap']} | random null {out['random_null_overlap']} (K={K})", flush=True)
    print(f"pred_a shared content subspace: {out['pred_a_shared_subspace']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
