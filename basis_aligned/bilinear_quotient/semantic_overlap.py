"""SEMANTIC SUBSPACE OVERLAP across components (do attention and MLP write token-
class along the SAME residual directions, or COMPLEMENTARY ones?). 768/771: the
MLP's token-semantic subspace is LEXICAL (determiners/punct/numbers), attention's
is STRUCTURAL (conjunctions/clause-boundaries). Both live in the shared residual
space, so we can measure whether they occupy the SAME directions (redundant/
reinforcing) or DIFFERENT ones (the front splits token processing into two
complementary canonical subspaces). Compute the top-64 token-semantic subspace of
attn0, mlp0, attn4, mlp4 and measure pairwise principal-angle overlap vs random.

REGISTERED PREDICTIONS:
  (0) SANITY: each self-overlap = 1; random-subspace overlap near the r/D floor;
  (a) COMPLEMENTARY-ish: attention vs MLP semantic subspace overlap is well ABOVE
      random (both encode token-class) but clearly BELOW 1 and below the within-
      component-type cross-layer overlap -- i.e. attention and MLP share some token-
      class directions but each has its own (structural vs lexical), so the front
      uses partly-distinct residual real estate for the two;
  (b) report the full pairwise overlap matrix + random baseline;
  NULL: random subspace overlaps all ~ the r/D floor (~0.2 at r=64,D=1152)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'semantic_overlap_results.json'
NEVAL = 48; MINCOUNT = 5; RSEM = 64


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows, n, which, L):
    """which: 'mlp' -> mlp output; 'attn' -> attention x1."""
    cap = []; toks = []
    if which == 'mlp':
        hk = lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D))
        h = m.transformer.h[L].mlp.register_forward_hook(hk)
    else:
        def hk(mo, i_, o_):
            x1 = o_[0] if isinstance(o_, tuple) else o_; cap.append(x1.detach().float().reshape(-1, D))
        h = m.transformer.h[L].attn.register_forward_hook(hk)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        forward_logits(idx)
    h.remove(); return torch.cat(cap, 0), torch.cat(toks).numpy()


def semantic_subspace(O, toks, r=RSEM):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(toks):
        mk = toks == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    Vh = torch.linalg.svd(M, full_matrices=False)[2]
    return Vh[:r].T.contiguous()                            # (D, r) orthonormal


def overlap(A, B):
    return float(torch.linalg.svdvals(A.T @ B).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    comps = [('attn', 0), ('mlp', 0), ('attn', 4), ('mlp', 4)]
    subs = {}
    for which, L in comps:
        O, toks = capture(rows, NEVAL, which, L); subs[f'{which}{L}'] = semantic_subspace(O, toks)
    keys = list(subs.keys())
    g = torch.Generator(device=DEV).manual_seed(0)
    Rr = torch.linalg.qr(torch.randn(D, RSEM, generator=g, device=DEV))[0]
    rand_floor = overlap(subs['mlp0'], Rr)

    mat = {}
    for a in keys:
        for b in keys:
            if a < b: mat[f'{a}|{b}'] = round(overlap(subs[a], subs[b]), 4)
    for k, v in mat.items(): print(f'{k}: {v}', flush=True)
    print(f'random floor {rand_floor:.3f}', flush=True)

    attn_mlp_0 = mat['attn0|mlp0']
    within_mlp = mat['mlp0|mlp4']; within_attn = mat['attn0|attn4']
    pa = attn_mlp_0 > rand_floor + 0.1 and attn_mlp_0 < 0.85 and attn_mlp_0 < max(within_mlp, within_attn)
    null_ok = rand_floor < 0.4
    out = {'rsem': RSEM, 'pairwise_overlap': mat, 'random_floor': round(rand_floor, 4),
           'attn0_mlp0': attn_mlp_0, 'within_mlp_0_4': within_mlp, 'within_attn_0_4': within_attn,
           'pred_a_complementary': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) attn & MLP semantic subspaces partly-distinct (above random, below 1 & below within-type): {pa}; NULL: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
