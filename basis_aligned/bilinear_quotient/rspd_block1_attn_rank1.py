"""RSPD BLOCK1.ATTN RANK-1 -- the standout lead from 699: block1's
attention output map (c_proj) has r80=1; a single rank-1 direction recovers
93% of its 2.06-nat contribution. Name it. block1.attn's whole functional
effect ~= outer(a1, b1): a1 (residual, 1152) is WHERE it writes; the per-
token coefficient s = X @ b1 is HOW MUCH each token fires it. Characterize
by (i) a1's unembedding readout (rough -- 16 blocks downstream, flagged),
(ii) which current tokens have the largest |s| (what drives the component),
(iii) the position profile of s (is it positional?).

Uses the fast A-SVD (normal-equations right-inverse; 700) -- verified to
match the library to ~1e-3.

REGISTERED PREDICTIONS:
  (0) SANITY: rank-1 fast-A-SVD recovers >=0.85 of block1.attn's benefit
      (matches 699's rec@1=0.93 within tolerance);
  (a) the per-token coefficient s is STRUCTURED, not noise: the top-|s|
      tokens share an interpretable character (report them; success = a
      human-readable pattern, judged from the token lists);
  (b) report a1 top unembedding tokens, top-|s| current tokens, and the
      correlation of s with position;
  NULL: a RANDOM unit direction in the same input space yields a coefficient
      whose top tokens are NOT interpretable and whose |s| distribution is
      flatter (max/median ratio much smaller than the real component's)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_block1_attn_rank1_results.json'
NCAP = 48
NEVAL = 48


def asvd_fast(W, X, eps=1e-3):
    tgtT = W @ X.T
    U, S, Vh = torch.linalg.svd(tgtT, full_matrices=False)
    A = U * S
    G = X.T @ X; G.diagonal().add_(eps)
    B = torch.linalg.solve(G, (Vh @ X).T).T
    return A, B


@torch.no_grad()
def forward_ce(rows, n, mod=None, Wsub=None):
    orig = None
    if mod is not None:
        orig = mod.weight.data
        mod.weight.data = (torch.zeros_like(orig) if Wsub == 'ablate'
                           else Wsub.to(orig.dtype)) if Wsub is not None else orig
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
    if mod is not None:
        mod.weight.data = orig
    return ce_s / nn


@torch.no_grad()
def capture(mod, rows, n, in_dim):
    cap = []; toks = []
    h = mod.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, in_dim)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0), torch.cat(toks, 0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NCAP + NEVAL)
    ev = rows[NCAP:NCAP + NEVAL]
    mod = m.transformer.h[1].attn.c_proj
    W = mod.weight.data.float().to(DEV)

    X, toks = capture(mod, rows[:NCAP], NCAP, 1152)
    A, B = asvd_fast(W, X)
    a1 = A[:, 0]; b1 = B[0, :]

    ce_full = forward_ce(ev, NEVAL)
    ce_abl = forward_ce(ev, NEVAL, mod, 'ablate')
    W1 = A[:, :1] @ B[:1, :]
    ce_r1 = forward_ce(ev, NEVAL, mod, W1)
    benefit = ce_abl - ce_full
    rec1 = (ce_abl - ce_r1) / benefit
    print(f'benefit {benefit:.3f}  rank-1 recovered {rec1:.3f}', flush=True)

    # (i) a1 unembedding readout (rough proxy)
    W_U = m.lm_head.weight.data.float().to(DEV)
    shift = (W_U @ (a1 / a1.norm())).cpu().numpy()
    order = np.argsort(-shift)
    def d1(t):
        try: return cl.d1(int(t))
        except Exception: return f'<{t}>'
    a1_boost = [d1(t) for t in order[:10]]
    a1_supp = [d1(t) for t in order[::-1][:10]]

    # (ii) per-token coefficient s = X @ b1
    s = (X @ (b1 / b1.norm())).cpu().numpy()
    tk = toks.numpy()
    # top current-tokens by mean |s|
    absS = np.abs(s)
    order_s = np.argsort(-absS)
    top_tok_examples = [d1(tk[i]) for i in order_s[:15]]
    peak = float(absS.max() / (np.median(absS) + 1e-9))
    # position correlation
    pos = np.tile(np.arange(256), NCAP)[:len(s)]
    pos_corr = float(np.corrcoef(s, pos)[0, 1])

    # NULL: random input direction
    g = torch.Generator(device='cpu').manual_seed(0)
    rd = torch.randn(1152, generator=g); rd = (rd / rd.norm()).to(DEV)
    s_rand = (X @ rd).cpu().numpy()
    peak_rand = float(np.abs(s_rand).max() / (np.median(np.abs(s_rand)) + 1e-9))

    print(f'a1 boost:    {a1_boost}', flush=True)
    print(f'a1 suppress: {a1_supp}', flush=True)
    print(f'top |s| current-token examples: {top_tok_examples}', flush=True)
    print(f'|s| peak (max/median) {peak:.1f}x  (random dir {peak_rand:.1f}x)', flush=True)
    print(f's-vs-position corr {pos_corr:+.3f}', flush=True)

    p0 = rec1 >= 0.85
    null_ok = peak > 1.5 * peak_rand
    print(f'\n(0) rank-1 recovers >=0.85: {p0}; NULL (real peakier than random): {null_ok}',
          flush=True)

    out = {'benefit': round(benefit, 4), 'rank1_recovered': round(float(rec1), 4),
           'a1_boost': a1_boost, 'a1_suppress': a1_supp,
           'top_abs_s_tokens': top_tok_examples, 's_peak': round(peak, 2),
           's_peak_random': round(peak_rand, 2), 's_pos_corr': round(pos_corr, 4),
           'pred_0': bool(p0), 'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
