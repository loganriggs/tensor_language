"""QK selection: clustering vs rank reduction (Logan 2026-07-21).
Question: is the number of clusters the same as the rank? No — they are
different reductions. This measures both, head to head, on the layer-0 query/key
factor tables (V x 128 per head-branch, rank <= 128 = head dim).

For each head, both branches, reduce the [q|k] factor rows by:
  - vq-k CLUSTERING (k discrete centroids; k in {16,64,128,256,1024})
  - rank-r SVD (continuous r-dim subspace; r in {8,16,32,64,128})
and audit the real ΔCE (patch layer-0 scores with the reduced factors). Also
report the effective rank of each clustered table (how many singular values of
the centroid table are non-trivial) to show clusters != rank numerically."""
import json
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
# per-branch folded factor tables (V, NH, HD), branch 1 and 2
qh1, kh1 = branch_factors(m, 1, dtype=torch.float32)
qh2, kh2 = branch_factors(m, 2, dtype=torch.float32)


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
def audit(qk_tabs):
    """qk_tabs: dict with q1,k1,q2,k2 each (V,NH,HD). Patch layer-0 scores."""
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(qk_tabs['q1'], qk_tabs['k1'], idx, HD).to(s1.dtype)
            n2 = scores_from_factors(qk_tabs['q2'], qk_tabs['k2'], idx, HD).to(s2.dtype)
            return n1, n2
        from tier2_model import reference_forward
        lg = reference_forward(m, idx, 'bf16', score_patch=patch).float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


from tier2_model import reference_forward
CE0 = 0.0
tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        lg = reference_forward(m, b[:, :-1], 'bf16').float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
CE0 = tot / n
res = {'baseline_ce': CE0, 'clustering': {}, 'rank': {}, 'eff_rank_of_clusters': {}}
print(f'baseline {CE0:.4f}', flush=True)


def eff_rank(M):
    s = torch.linalg.svdvals(M.double())
    p = s / s.sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))   # entropy-based effective rank


# CLUSTERING sweep (shared [q|k] partition per head-branch)
for k in (16, 64, 128, 256, 1024):
    tabs = {'q1': torch.empty_like(qh1), 'k1': torch.empty_like(kh1),
            'q2': torch.empty_like(qh2), 'k2': torch.empty_like(kh2)}
    erank = []
    for hh in range(NH):
        for (qn, kn, qt, kt) in [('q1', 'k1', qh1, kh1), ('q2', 'k2', qh2, kh2)]:
            X = torch.cat([qt[:, hh], kt[:, hh]], 1).to(DEV)   # (V, 2*HD)
            a, C = kmeans(X, k, seed=hh)
            tabs[qn][:, hh] = C[a][:, :HD].cpu()
            tabs[kn][:, hh] = C[a][:, HD:].cpu()
            if hh == 0 and qn == 'q1':
                erank.append(eff_rank(C[a][:, :HD]))
    for kk in tabs:
        tabs[kk] = tabs[kk].to(DEV)
    d = audit(tabs) - CE0
    res['clustering'][f'k={k}'] = round(d, 4)
    res['eff_rank_of_clusters'][f'k={k}'] = round(erank[0], 1)
    print(f'CLUSTER k={k}: dCE {d:+.4f}  (eff-rank of clustered q1-h0 table {erank[0]:.1f})', flush=True)
    json.dump(res, open(f'{QK}/qk_cluster_vs_rank.json', 'w'), indent=2)

# RANK sweep (SVD of the [q|k] rows, truncate)
for r in (8, 16, 32, 64, 128):
    tabs = {'q1': torch.empty_like(qh1), 'k1': torch.empty_like(kh1),
            'q2': torch.empty_like(qh2), 'k2': torch.empty_like(kh2)}
    for hh in range(NH):
        for (qn, kn, qt, kt) in [('q1', 'k1', qh1, kh1), ('q2', 'k2', qh2, kh2)]:
            X = torch.cat([qt[:, hh], kt[:, hh]], 1).to(DEV).double()
            mu = X.mean(0)
            U, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
            Xr = ((U[:, :r] * S[:r]) @ Vh[:r] + mu).float()
            tabs[qn][:, hh] = Xr[:, :HD].cpu()
            tabs[kn][:, hh] = Xr[:, HD:].cpu()
    for kk in tabs:
        tabs[kk] = tabs[kk].to(DEV)
    d = audit(tabs) - CE0
    res['rank'][f'r={r}'] = round(d, 4)
    print(f'RANK r={r}: dCE {d:+.4f}', flush=True)
    json.dump(res, open(f'{QK}/qk_cluster_vs_rank.json', 'w'), indent=2)
print('qk cluster vs rank done', flush=True)
