"""MED PHASE 10b: are the extracted filters reproducible across seeds?

Extract the top-K MLP-0 pixel-space texture units from two independently-trained
models (seed 0, seed 1). Compare: (A) per-filter best-match cosine of the preferred
pixel patterns (sign-invariant); (B) principal angles between the feature SUBSPACES
(span of top-K {a_j, b_j} in 147-dim pixel space). Null: seed-0 vs a random
orthonormal subspace of equal dimension. High match => filters are task features,
not one-run artifacts.
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from medmnist import PathMNIST

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
ROOT = '/workspace/tensor_language/medmnist_data'
d = PathMNIST(split='train', root=ROOT)
Xtr = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2).to(DEV) / 255.0


def extract(ckname, K=64):
    ck = torch.load(f'{QK}/{ckname}', map_location=DEV)
    cfg = ck['cfg']
    D, PS, NP, INNER = cfg['D'], cfg['PS'], cfg['NP'], cfg['INNER']
    MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
    W = ck['state']
    PXD = 3 * PS * PS
    X = (Xtr - MEAN) / STD

    def patchify(x):
        B = x.shape[0]
        p = x.unfold(2, PS, PS).unfold(3, PS, PS)
        return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)

    We = W['embed.weight']
    A = We.T @ W['blocks.1.L.weight'].T           # (PXD, INNER)
    Bm = We.T @ W['blocks.1.R.weight'].T
    with torch.no_grad():
        P = patchify(X[:2048])
        imp = ((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).abs().mean((0, 1))
               * W['blocks.1.Dn.weight'].norm(dim=0)).cpu().numpy()
    order = list(np.argsort(-imp))[:K]
    # preferred pattern per unit = top eigenvector of symmetric pixel form
    pats, basis = [], []
    for j in order:
        a_j, b_j = A[:, j], Bm[:, j]
        S = 0.5 * (torch.outer(a_j, b_j) + torch.outer(b_j, a_j))
        ev, evec = torch.linalg.eigh(S)
        pats.append(evec[:, ev.abs().argmax()])
        basis += [a_j / a_j.norm(), b_j / b_j.norm()]
    return torch.stack(pats), torch.stack(basis), PXD


P0, B0, PXD = extract('med_bvit2.pt')
P1, B1, _ = extract('med_seed1.pt')


def principal_cos(U, V):
    # mean cosine of principal angles between row-spaces of U (m,d), V (n,d)
    Qu = torch.linalg.qr(U.T)[0]           # (d, m)
    Qv = torch.linalg.qr(V.T)[0]
    s = torch.linalg.svdvals(Qu.T @ Qv)
    return float(s.mean()), float(s[:8].mean())


# (A) per-filter best-match cosine (sign-invariant)
C = (P0 @ P1.T).abs()                       # (K,K)
best = C.max(1).values
res = {'per_filter_match_mean': round(float(best.mean()), 3),
       'per_filter_match_median': round(float(best.median()), 3),
       'per_filter_match_frac_over_0.7': round(float((best > 0.7).float().mean()), 3)}

# (B) subspace principal angles vs seed-1 and vs random null
m_all, m_top = principal_cos(B0, B1)
rnd = torch.linalg.qr(torch.randn(B1.shape[1], B1.shape[0], device=DEV))[0].T
r_all, r_top = principal_cos(B0, rnd)
res['subspace_meancos_seed1'] = round(m_all, 3)
res['subspace_meancos_random'] = round(r_all, 3)
res['subspace_top8cos_seed1'] = round(m_top, 3)
res['subspace_top8cos_random'] = round(r_top, 3)
res['seed0_acc'] = 0.857
res['seed1_acc'] = 0.853
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/med_reproduce.json', 'w'), indent=2)
print('MED REPRODUCE DONE', flush=True)
