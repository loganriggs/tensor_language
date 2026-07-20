"""Rank-then-VQ for QK selection (Logan 2026-07-21): the composed code that
should dominate both pure rank and pure vector-quantization. Selection lives in
a low-rank subspace (rank ~16-32, geometry) and needs a ~256-class alphabet
(cardinality); so project each word's [q|k] factor row to r dimensions, then
vector-quantize the r-dim coefficients into k classes. Keeps VQ's cheap
log2(k)-bit codes while shrinking centroids from k*256 to k*r floats.

Bits-honest comparison (per head-branch, joint [q|k] table is V x 256):
  pure VQ-k     : k*256 float + V*log2(k) index
  pure rank-r   : r*256 float (basis) + V*r float (dense coeffs)
  rank-r + VQ-k : r*256 float (basis) + k*r float (centroids) + V*log2(k) index
Real ΔCE (layer-0 score patch), all heads/branches."""
import json
import math
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
qh1, kh1 = branch_factors(m, 1, dtype=torch.float32)
qh2, kh2 = branch_factors(m, 2, dtype=torch.float32)
BR = [('q1', 'k1', qh1, kh1), ('q2', 'k2', qh2, kh2)]


def kmeans(Xc, k, iters=12, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    C = Xc[torch.randperm(len(Xc), generator=g)[:k].to(Xc.device)].clone()
    for _ in range(iters):
        a = torch.empty(len(Xc), dtype=torch.long, device=Xc.device)
        for i in range(0, len(Xc), 4096):
            xx = Xc[i:i + 4096]
            a[i:i + 4096] = ((xx * xx).sum(1, True) - 2 * xx @ C.T + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); c2 = torch.zeros(k, device=Xc.device)
        Cn.index_add_(0, a, Xc); c2.index_add_(0, a, torch.ones(len(Xc), device=Xc.device))
        nz = c2 > 0; C[nz] = Cn[nz] / c2[nz][:, None]
    return a, C


@torch.no_grad()
def audit(tabs):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD).to(s1.dtype)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD).to(s2.dtype)
            return n1, n2
        lg = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        lg = reference_forward(m, b[:, :-1], 'bf16').float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
CE0 = tot / n
res = {'baseline_ce': CE0, 'arms': {}}
print(f'baseline {CE0:.4f}', flush=True)
DIM = 2 * HD


def empty():
    return {'q1': torch.empty_like(qh1), 'k1': torch.empty_like(kh1),
            'q2': torch.empty_like(qh2), 'k2': torch.empty_like(kh2)}


def report(tag, tabs, bits):
    for kk in tabs:
        tabs[kk] = tabs[kk].to(DEV)
    d = audit(tabs) - CE0
    res['arms'][tag] = {'dce': round(d, 4), 'Mbits': round(bits / 1e6, 2)}
    print(f'{tag}: dCE {d:+.4f}  {bits/1e6:.2f}Mbits', flush=True)
    json.dump(res, open(f'{QK}/qk_rank_then_vq.json', 'w'), indent=2)


# pure VQ-256
tabs = empty(); bits = 0
for hh in range(NH):
    for qn, kn, qt, kt in BR:
        X = torch.cat([qt[:, hh], kt[:, hh]], 1).to(DEV)
        a, C = kmeans(X, 256, seed=hh)
        tabs[qn][:, hh] = C[a][:, :HD].cpu(); tabs[kn][:, hh] = C[a][:, HD:].cpu()
        bits += 256 * DIM * 32 + V * math.log2(256)
report('pure VQ k=256', tabs, bits)

# pure rank-16
tabs = empty(); bits = 0
for hh in range(NH):
    for qn, kn, qt, kt in BR:
        X = torch.cat([qt[:, hh], kt[:, hh]], 1).to(DEV).double()
        mu = X.mean(0); U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
        Xr = ((U[:, :16] * S[:16]) @ Vh[:16] + mu).float()
        tabs[qn][:, hh] = Xr[:, :HD].cpu(); tabs[kn][:, hh] = Xr[:, HD:].cpu()
        bits += 16 * DIM * 32 + V * 16 * 32
report('pure rank r=16', tabs, bits)

# rank-r THEN VQ-k
for r, k in [(16, 256), (32, 256), (16, 1024)]:
    tabs = empty(); bits = 0
    for hh in range(NH):
        for qn, kn, qt, kt in BR:
            X = torch.cat([qt[:, hh], kt[:, hh]], 1).to(DEV).double()
            mu = X.mean(0); U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
            coef = (U[:, :r] * S[:r]).float()                  # (V, r) subspace coords
            a, C = kmeans(coef, k, seed=hh)                    # VQ inside subspace
            rec = (C[a].double() @ Vh[:r] + mu).float()        # back to 256-dim
            tabs[qn][:, hh] = rec[:, :HD].cpu(); tabs[kn][:, hh] = rec[:, HD:].cpu()
            bits += r * DIM * 32 + k * r * 32 + V * math.log2(k)
    report(f'rank r={r} + VQ k={k}', tabs, bits)
print('qk rank then vq done', flush=True)
