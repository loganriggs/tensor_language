"""RSPD MLP16 RANK-1 -- name mlp16's single direction (713: mlp16 r80=1, a
rank-1 decoder recovers 80%+ of its 0.88-nat benefit -- the back-end analog
of block1.attn's rank-1 core, 701). mlp16 is 1 block from the readout so
its unembedding readout is a GOOD proxy (unlike mlp1). Report: (i) what a1
writes (unembedding), (ii) which current tokens drive the coefficient
s = gate @ b1, (iii) confound-free structure via token-identity
concentration vs a random direction in the complement of the gate's massive
dims. Uses the fixed regime-aware fast A-SVD (712).

REGISTERED PREDICTIONS:
  (0) SANITY: rank-1 recovers >=0.75 of mlp16's benefit (matches 713 r80=1);
  (a) INTERPRETABLE: a1's top unembedding tokens and/or the top-|s| current
      tokens share a human-readable character (report);
  (b) report a1 writes, top-|s| tokens, token-identity concentration;
  NULL: a random direction in the complement of the gate's massive dims has
      LOWER token-identity concentration than the real b1."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; HID = 4608; LAYER = 16
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_mlp16_rank1_results.json'
NFIT = 24; NEVAL = 48


def asvd_fast(W, X, eps=1e-3):
    U, S, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    A = U * S; N, din = X.shape
    if N >= din:
        G = X.T @ X; G.diagonal().add_(eps)
        B = torch.linalg.solve(G, (Vh @ X).T).T
    else:
        Gn = X @ X.T; Gn.diagonal().add_(eps)
        B = Vh @ torch.linalg.solve(Gn, X)
    return A, B


@torch.no_grad()
def forward_ce(rows, n, W=None):
    mod = m.transformer.h[LAYER].mlp.Down; orig = mod.weight.data
    if W is not None:
        mod.weight.data = torch.zeros_like(orig) if W == 'ablate' else W.to(orig.dtype)
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)
        lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean'))*idx.shape[0]; nn += idx.shape[0]
    if W is not None: mod.weight.data = orig
    return s/nn


@torch.no_grad()
def capture(rows, n):
    cap = []; toks = []
    h = m.transformer.h[LAYER].mlp.Down.register_forward_pre_hook(
        lambda mo, inp: cap.append(inp[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        toks.append(idx.reshape(-1).cpu())
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove()
    return torch.cat(cap, 0), torch.cat(toks, 0)


def d1(t):
    try: return cl.d1(int(t))
    except Exception: return f'<{t}>'


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    ev = rows[NFIT:NFIT+NEVAL]
    W = m.transformer.h[LAYER].mlp.Down.weight.data.float().to(DEV)
    X, toks = capture(rows[:NFIT], NFIT); tk = toks.numpy()
    A, B = asvd_fast(W, X)
    a1 = A[:, 0]; b1 = B[0, :]

    ce_full = forward_ce(ev, NEVAL); ce_abl = forward_ce(ev, NEVAL, 'ablate')
    rec1 = (ce_abl - forward_ce(ev, NEVAL, A[:, :1] @ B[:1, :])) / (ce_abl - ce_full)
    print(f'benefit {ce_abl-ce_full:.3f}  rank-1 recovered {rec1:.3f}', flush=True)

    W_U = m.lm_head.weight.data.float().to(DEV)
    shift = (W_U @ (a1/a1.norm())).cpu().numpy(); order = np.argsort(-shift)
    boost = [d1(t) for t in order[:10]]; supp = [d1(t) for t in order[::-1][:10]]
    s = (X @ (b1/b1.norm())).cpu().numpy()
    top_idx = np.argsort(-np.abs(s))[:200]
    conc = Counter(tk[top_idx].tolist()).most_common(1)[0][1]/200
    toptoks = [d1(t) for t, _ in Counter(tk[top_idx].tolist()).most_common(8)]

    var = X.var(0); massive = torch.argsort(-var)[:32]
    g = torch.Generator().manual_seed(0); rd = torch.randn(HID, generator=g).to(DEV)
    rd[massive] = 0; rd = rd/rd.norm()
    s_r = (X @ rd).cpu().numpy(); conc_r = Counter(tk[np.argsort(-np.abs(s_r))[:200]].tolist()).most_common(1)[0][1]/200

    print(f'a1 writes:   {boost}', flush=True)
    print(f'a1 suppress: {supp}', flush=True)
    print(f'top-|s| tokens {toptoks} (concentration {conc:.2f} vs random {conc_r:.2f})', flush=True)

    p0 = rec1 >= 0.75; null_ok = conc > 1.5*conc_r
    print(f'\n(0) rank-1 recovers>=0.75: {p0}; NULL real>random concentration: {null_ok}', flush=True)
    out = {'benefit': round(float(ce_abl-ce_full), 4), 'rank1_recovered': round(float(rec1), 4),
           'a1_writes': boost, 'a1_suppress': supp, 'top_s_tokens': toptoks,
           'concentration': round(float(conc), 3), 'concentration_random': round(float(conc_r), 3),
           'pred_0': bool(p0), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
