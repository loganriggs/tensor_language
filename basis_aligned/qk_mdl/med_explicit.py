"""MED PHASE 5: the standalone explicit pipeline (extraction fidelity headline).

Build a fully explicit, auditable classifier from the extracted components ONLY:
  patch -> {quadratic texture features (a_j.p)(b_j.p) for MLP-0 units j} -> mean-pool
  over patches -> linear head (9 classes).
Everything upstream of the head is the frozen extracted pixel-space bilinear forms
(a_j = We^T L_j, b_j = We^T R_j); only the final linear head is fit. No attention,
no residual stack, no rms_norm — a genuinely explicit surrogate. Report test
accuracy versus number of units K, against full model 85.7% / attn-removed 78.1%.
Controls: (a) linear-only head on pooled raw patch means (no quadratic features) —
the floor; (b) full 192 units.
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
ck = torch.load(f'{QK}/med_bvit2.pt', map_location=DEV)
cfg = ck['cfg']
D, NH, HD, NL, PS, NP, INNER = (cfg[k] for k in ('D', 'NH', 'HD', 'NL', 'PS', 'NP', 'INNER'))
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
W = ck['state']
PXD = 3 * PS * PS


def load(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1)
    return ((X.to(DEV) - MEAN) / STD), y.to(DEV)


def patchify(x):
    B = x.shape[0]
    p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


Xtr, ytr = load('train')
Xte, yte = load('test')

# extracted pixel-space bilinear forms for ALL 192 units, ranked by importance
We = W['embed.weight']                        # (D, PXD)
A = We.T @ W['blocks.1.L.weight'].T           # (PXD, INNER): columns a_j
Bm = We.T @ W['blocks.1.R.weight'].T          # (PXD, INNER): columns b_j
# importance ranking from phase 4 (recompute quickly on a slice)
with torch.no_grad():
    P = patchify(Xtr[:2048])
    fa = torch.einsum('bnp,pj->bnj', P, A)
    fb = torch.einsum('bnp,pj->bnj', P, Bm)
    act = fa * fb                             # (B,N,INNER) explicit unit activations
    imp = (act.abs().mean((0, 1)) * W['blocks.1.Dn.weight'].norm(dim=0)).cpu().numpy()
order = list(np.argsort(-imp))


@torch.no_grad()
def features(X, units):
    cols = torch.tensor(units, device=DEV)
    Au, Bu = A[:, cols], Bm[:, cols]
    outs = []
    for i in range(0, len(X), 4096):
        P = patchify(X[i:i + 4096])
        f = torch.einsum('bnp,pj->bnj', P, Au) * torch.einsum('bnp,pj->bnj', P, Bu)
        outs.append(f.mean(1))                # mean-pool over patches
    return torch.cat(outs)


def fit_head(Ftr, Fte, din):
    head = torch.nn.Linear(din, 9).to(DEV)
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
    Ftr2, Fte2 = (Ftr - mu) / sd, (Fte - mu) / sd
    opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-3)
    for step in range(3000):
        bi = torch.randint(0, len(Ftr2), (1024,), device=DEV)
        loss = F.cross_entropy(head(Ftr2[bi]), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((head(Fte2).argmax(1) == yte).float().mean())


res = {'full_model': 0.857, 'attn_removed': 0.781}
# control: linear on pooled raw patch means (no quadratic features)
lin_tr = patchify(Xtr).mean(1)
lin_te = patchify(Xte).mean(1)
res['linear_floor'] = round(fit_head(lin_tr, lin_te, PXD), 4)
print(f'linear floor (pooled raw pixels): {res["linear_floor"]:.4f}', flush=True)
for K in (8, 16, 32, 64, 128, 192):
    units = order[:K]
    Ftr, Fte = features(Xtr, units), features(Xte, units)
    a = fit_head(Ftr, Fte, K)
    res[f'explicit_K{K}'] = round(a, 4)
    print(f'explicit quadratic pipeline, K={K} units: {a:.4f}', flush=True)
    json.dump(res, open(f'{QK}/med_explicit.json', 'w'), indent=2)
print('MED EXPLICIT DONE', flush=True)
