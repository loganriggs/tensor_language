"""RSPD DEPTH RANK MAP -- place the front findings in full-depth context.
CE-priced functional rank (r80) + benefit for ALL 18 MLP Down layers, via
the fast A-SVD (700). Shows the depth profile of functional rank: where are
the low-rank vs high-rank MLPs?

REGISTERED PREDICTIONS:
  (0) SANITY: full-rank A-SVD reproduces baseline per layer;
  (a) DEPTH PROFILE: the edges are low-rank (mlp0 r80~8, mlp17 r80~4) and
      some middle layers are high-rank (mlp1/mlp2 r80>=128 from 699);
      register the open question of what the FULL profile looks like
      (monotone? U-shaped? scattered?) -- report it;
  (b) report r80 + benefit for all 18 layers;
  NULL: random rank-r projection recovers ~0 for a spot-checked layer."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rspd_depth_rank_map_results.json'
NFIT = 12; NEVAL = 48
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def asvd_fast(W, X, eps=1e-3):
    U, S, Vh = torch.linalg.svd(W @ X.T, full_matrices=False)
    G = X.T @ X; G.diagonal().add_(eps)
    return U * S, torch.linalg.solve(G, (Vh @ X).T).T


@torch.no_grad()
def forward_ce(rows, n, mod=None, W=None):
    orig = None
    if mod is not None:
        orig = mod.weight.data
        mod.weight.data = (torch.zeros_like(orig) if W == 'ablate' else W.to(orig.dtype)) if W is not None else orig
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(lg.float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='mean')) * idx.shape[0]; nn += idx.shape[0]
    if orig is not None: mod.weight.data = orig
    return s / nn


@torch.no_grad()
def capture_all(rows, n):
    caps = {li: [] for li in range(len(m.transformer.h))}
    hooks = []
    for li, blk in enumerate(m.transformer.h):
        hooks.append(blk.mlp.Down.register_forward_pre_hook(
            (lambda li: lambda mo, inp: caps[li].append(inp[0].detach().float().reshape(-1, HID)))(li)))
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    for h in hooks: h.remove()
    return {li: torch.cat(caps[li], 0) for li in caps}


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    fit, ev = rows[:NFIT], rows[NFIT:NFIT + NEVAL]
    NL = len(m.transformer.h)
    X = capture_all(fit, NFIT)
    ce_full = forward_ce(ev, NEVAL)
    print(f'baseline CE {ce_full:.4f}', flush=True)

    prof = {}
    for li in range(NL):
        mod = m.transformer.h[li].mlp.Down; W = mod.weight.data.float().to(DEV)
        ce_abl = forward_ce(ev, NEVAL, mod, 'ablate'); ben = ce_abl - ce_full
        A, B = asvd_fast(W, X[li])
        r80 = None
        for r in RANKS:
            rec = (ce_abl - forward_ce(ev, NEVAL, mod, A[:, :r] @ B[:r, :])) / max(ben, 1e-6)
            if rec >= 0.80: r80 = r; break
        if r80 is None: r80 = RANKS[-1]
        prof[li] = {'benefit': round(float(ben), 4), 'r80': int(r80)}
        print(f'mlp{li:2d}: benefit {ben:+.3f}  r80 {r80}', flush=True)

    # null spot-check on a high-rank layer (mlp1)
    mod = m.transformer.h[1].mlp.Down; W = mod.weight.data.float().to(DEV)
    ce_abl = forward_ce(ev, NEVAL, mod, 'ablate'); ben = ce_abl - ce_full
    g = torch.Generator().manual_seed(0); Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g)); Q = Q.to(DEV)
    rec_rand8 = (ce_abl - forward_ce(ev, NEVAL, mod, Q[:, :8] @ (Q[:, :8].T @ W))) / ben
    print(f'\nnull: mlp1 random rank-8 recovered {rec_rand8:+.3f}', flush=True)

    r80s = [prof[li]['r80'] for li in range(NL)]
    print(f'r80 by depth: {r80s}', flush=True)
    out = {'baseline_ce': round(ce_full, 4), 'profile': prof,
           'r80_by_depth': r80s, 'null_mlp1_rand8': round(float(rec_rand8), 4),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
