"""RSPD BLOCK0.ATTN CORE -- name block0.attn's rank-2 core (699: r80=2,
rank-2 recovers 82% of its 1.28-nat contribution). Two directions; name
each by output (unembedding) and by which current tokens drive its per-
token coefficient. Uses fast A-SVD (700).

NULL REDESIGN (701 correction): peakedness was confounded by the massive-
activation dims in the attention output. Here the null is a random unit
direction drawn in the COMPLEMENT of X's top massive-variance dims (so it
cannot spuriously spike on them). Structure evidence = (i) rank-2 CE
recovery (causal, unconfounded) and (ii) token-conditional readout.

REGISTERED PREDICTIONS:
  (0) SANITY: rank-2 fast-A-SVD recovers >=0.75 of block0.attn's benefit
      (matches 699 rec@2=0.82);
  (a) INTERPRETABLE: for at least one of the 2 directions, the top-|s|
      current tokens share a human-readable character (report the lists);
  (b) report a1,a2 unembedding boost/suppress and top-|s| tokens each;
  NULL: a random direction in the complement of X's massive dims yields
      top-|s| tokens with LOWER identity-concentration (top-token share)
      than the real directions."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_block0_attn_core_results.json'
NCAP = 48
NEVAL = 48
R = 2


def asvd_fast(W, X, eps=1e-3):
    U, S, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * S
    G = X.T @ X; G.diagonal().add_(eps)
    B = torch.linalg.solve(G, (Vh @ X).T).T
    return A, B


@torch.no_grad()
def forward_ce(rows, n, mod=None, Wsub=None):
    orig = None
    if mod is not None and Wsub is not None:
        orig = mod.weight.data
        mod.weight.data = torch.zeros_like(orig) if Wsub == 'ablate' else Wsub.to(orig.dtype)
    ce_s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1)
        ce_s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1),
                                 reduction='mean')) * idx.shape[0]; nn += idx.shape[0]
    if orig is not None:
        mod.weight.data = orig
    return ce_s / nn


@torch.no_grad()
def capture(mod, rows, n, in_dim):
    cap = []; toks = []
    h = mod.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0), torch.cat(toks, 0)


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


def top_token_share(s, tk, topn=200):
    """identity-concentration: among the topn highest-|s| positions, the
    fraction taken by the single most common token id."""
    idx = np.argsort(-np.abs(s))[:topn]
    c = Counter(tk[idx].tolist())
    return c.most_common(1)[0][1] / topn, [d1(t) for t, _ in c.most_common(6)]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NCAP + NEVAL)
    ev = rows[NCAP:NCAP + NEVAL]
    mod = m.transformer.h[0].attn.c_proj
    W = mod.weight.data.float().to(DEV)
    X, toks = capture(mod, rows[:NCAP], NCAP, 1152); tk = toks.numpy()
    A, B = asvd_fast(W, X)

    ce_full = forward_ce(ev, NEVAL)
    ce_abl = forward_ce(ev, NEVAL, mod, 'ablate')
    W2 = A[:, :R] @ B[:R, :]
    rec2 = (ce_abl - forward_ce(ev, NEVAL, mod, W2)) / (ce_abl - ce_full)
    print(f'benefit {ce_abl-ce_full:.3f}  rank-{R} recovered {rec2:.3f}', flush=True)

    W_U = m.lm_head.weight.data.float().to(DEV)
    dirs = []
    for j in range(R):
        aj = A[:, j] / A[:, j].norm(); bj = B[j, :] / B[j, :].norm()
        shift = (W_U @ aj).cpu().numpy(); order = np.argsort(-shift)
        s = (X @ bj).cpu().numpy()
        share, toptoks = top_token_share(s, tk)
        dd = {'j': j, 'boost': [d1(t) for t in order[:8]],
              'suppress': [d1(t) for t in order[::-1][:8]],
              'top_s_tokens': toptoks, 'top_token_share': round(float(share), 3)}
        dirs.append(dd)
        print(f'dir {j}: boost {dd["boost"][:5]}', flush=True)
        print(f'   top-|s| tokens {toptoks} (share {share:.2f})', flush=True)

    # NULL: random dir in complement of X's top-32 massive-variance dims
    var = X.var(0); massive = torch.argsort(-var)[:32]
    g = torch.Generator(device='cpu').manual_seed(0)
    rd = torch.randn(1152, generator=g).to(DEV); rd[massive] = 0.0; rd = rd / rd.norm()
    s_rand = (X @ rd).cpu().numpy()
    share_rand, _ = top_token_share(s_rand, tk)
    print(f'\nNULL random(complement) top_token_share {share_rand:.3f}', flush=True)

    p0 = rec2 >= 0.75
    max_share = max(dd['top_token_share'] for dd in dirs)
    null_ok = max_share > share_rand
    print(f'(0) rank-{R} recovers >=0.75: {p0}; NULL (real more concentrated '
          f'than random {max_share:.2f}>{share_rand:.2f}): {null_ok}', flush=True)

    out = {'benefit': round(ce_abl - ce_full, 4), 'rank2_recovered': round(float(rec2), 4),
           'directions': dirs, 'null_random_share': round(float(share_rand), 3),
           'pred_0': bool(p0), 'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
