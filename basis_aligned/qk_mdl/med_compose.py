"""MED PHASE 8: is the composition gap explicit spatial statistics?

The explicit patch-local pipeline (13d) mean-pools texture features and plateaus
(0.716 path / 0.821 blood) below full (0.857 / 0.943). Test whether the gap is
captured by RICHER explicit pooling of the SAME extracted quadratic-texture
features: mean, std, max, min across the 16 patches (second-order + extremal
spatial statistics), then a linear head. No new learned features, no attention —
just more pooling statistics. If this closes the gap, "composition" = explicit
spatial statistics; if not, it is genuinely deep/non-statistical.
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from medmnist import PathMNIST, BloodMNIST

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
ROOT = '/workspace/tensor_language/medmnist_data'


def run(dsname, ckname, ncls, full_acc):
    DS = {'path': PathMNIST, 'blood': BloodMNIST}[dsname]
    ck = torch.load(f'{QK}/{ckname}', map_location=DEV)
    cfg = ck['cfg']
    D, PS, NP, INNER = cfg['D'], cfg['PS'], cfg['NP'], cfg['INNER']
    MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
    W = ck['state']
    PXD = 3 * PS * PS

    def load(split):
        d = DS(split=split, root=ROOT)
        X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2) / 255.0
        y = torch.from_numpy(d.labels).long().squeeze(1)
        return ((X.to(DEV) - MEAN) / STD), y.to(DEV)

    def patchify(x):
        B = x.shape[0]
        p = x.unfold(2, PS, PS).unfold(3, PS, PS)
        return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)

    Xtr, ytr = load('train')
    Xte, yte = load('test')
    We = W['embed.weight']
    A = We.T @ W['blocks.1.L.weight'].T
    Bm = We.T @ W['blocks.1.R.weight'].T

    @torch.no_grad()
    def patchfeat(X):                       # (N, 16, INNER) explicit texture features
        outs = []
        for i in range(0, len(X), 4096):
            P = patchify(X[i:i + 4096])
            outs.append(torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm))
        return torch.cat(outs)

    Ftr, Fte = patchfeat(Xtr), patchfeat(Xte)

    def pool(F_, kind):
        if kind == 'mean':
            return F_.mean(1)
        if kind == 'meanstd':
            return torch.cat([F_.mean(1), F_.std(1)], 1)
        if kind == 'full4':
            return torch.cat([F_.mean(1), F_.std(1), F_.amax(1), F_.amin(1)], 1)

    def fit(Ftr2, Fte2):
        din = Ftr2.shape[1]
        head = torch.nn.Linear(din, ncls).to(DEV)
        mu, sd = Ftr2.mean(0, keepdim=True), Ftr2.std(0, keepdim=True).clamp_min(1e-6)
        Ftr2, Fte2 = (Ftr2 - mu) / sd, (Fte2 - mu) / sd
        opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-3)
        for s in range(3000):
            bi = torch.randint(0, len(Ftr2), (1024,), device=DEV)
            loss = F.cross_entropy(head(Ftr2[bi]), ytr[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return float((head(Fte2).argmax(1) == yte).float().mean())

    out = {'full': full_acc}
    for kind in ('mean', 'meanstd', 'full4'):
        out[kind] = round(fit(pool(Ftr, kind), pool(Fte, kind)), 4)
        print(f'{dsname} {kind}: {out[kind]:.4f}', flush=True)
    return out


res = {'path': run('path', 'med_bvit2.pt', 9, 0.857),
       'blood': run('blood', 'med_blood.pt', 8, 0.943)}
json.dump(res, open(f'{QK}/med_compose.json', 'w'), indent=2)
print(json.dumps(res, indent=1), flush=True)
print('MED COMPOSE DONE', flush=True)
