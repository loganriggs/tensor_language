"""MED PHASE 6: filter-to-class attribution — which texture filter votes which tissue.

Fit the explicit head on the top-32 extracted quadratic-texture features (the ones
rendered in the artifact), then read the head weights: for each of the 9 tissue
classes, which filters are its strongest positive detectors. Turns the filter
gallery into a labeled dictionary (filter u_j -> tissue types it signals).
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
D, PS, NP, INNER = cfg['D'], cfg['PS'], cfg['NP'], cfg['INNER']
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
W = ck['state']
PXD = 3 * PS * PS
LABELS = {0: 'adipose', 1: 'background', 2: 'debris', 3: 'lymphocytes', 4: 'mucus',
          5: 'smooth muscle', 6: 'normal mucosa', 7: 'cancer stroma', 8: 'adenocarcinoma'}


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
We = W['embed.weight']
A = We.T @ W['blocks.1.L.weight'].T
Bm = We.T @ W['blocks.1.R.weight'].T
saved = torch.load(f'{QK}/med_mlp0_filters.pt', map_location=DEV)
units = saved['units']                       # top-32 units, importance order


@torch.no_grad()
def features(X):
    cols = torch.tensor(units, device=DEV)
    Au, Bu = A[:, cols], Bm[:, cols]
    outs = []
    for i in range(0, len(X), 4096):
        P = patchify(X[i:i + 4096])
        f = torch.einsum('bnp,pj->bnj', P, Au) * torch.einsum('bnp,pj->bnj', P, Bu)
        outs.append(f.mean(1))
    return torch.cat(outs)


Ftr, Fte = features(Xtr), features(Xte)
mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
head = torch.nn.Linear(32, 9).to(DEV)
opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-3)
for step in range(4000):
    bi = torch.randint(0, len(Ftr), (1024,), device=DEV)
    loss = F.cross_entropy(head(Ftr[bi]), ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    acc = float((head(Fte).argmax(1) == yte).float().mean())
Wc = head.weight.detach().cpu().numpy()      # (9, 32)
res = {'top32_head_acc': round(acc, 4), 'per_class_top_filters': {}}
for c in range(9):
    top = np.argsort(-Wc[c])[:4]
    res['per_class_top_filters'][LABELS[c]] = [
        {'unit': int(units[j]), 'weight': round(float(Wc[c, j]), 3)} for j in top]
# per-filter dominant class
res['per_filter_class'] = {}
for j in range(32):
    c = int(np.argmax(Wc[:, j]))
    res['per_filter_class'][f'u{units[j]}'] = LABELS[c]
print(json.dumps(res['per_class_top_filters'], indent=1), flush=True)
print(f'top-32 head acc {acc:.4f}', flush=True)
json.dump(res, open(f'{QK}/med_attrib.json', 'w'), indent=2)
print('MED ATTRIB DONE', flush=True)
