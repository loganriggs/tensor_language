"""MED PHASE 16: is the validate-by-generalization loop a real METHOD (not a
one-class anecdote)? Leakage-free protocol:
- SELECT features using only VAL (in-domain institutions): either by clean-val
  strength (baseline A) or by robustness across a controllable nuisance = clean AND
  stain-shifted val (method B). Test institution is NEVER used for selection.
- FIT a 9-way head on TRAIN using the selected features.
- EVALUATE on the untouched TEST institution (natural shift) and on stain-shifted
  TEST. If B beats A on the shifted evaluations, selecting for generalization is a
  usable robustness recipe.
"""
import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import PathMNIST

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
ROOT = '/workspace/tensor_language/medmnist_data'
ck = torch.load(f'{QK}/med_bvit2.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, PS, NP, INNER = cfg['D'], cfg['PS'], cfg['NP'], cfg['INNER']
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
PXD = 3 * PS * PS


def raw(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2).to(DEV) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1).to(DEV)
    return X, y


Xtr, ytr = raw('train'); Xva, yva = raw('val'); Xte, yte = raw('test')
A = W['embed.weight'].T @ W['blocks.1.L.weight'].T
Bm = W['embed.weight'].T @ W['blocks.1.R.weight'].T


def patchify(x):
    B = x.shape[0]; p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


def stain(x, eps, g):
    gain = 1 + eps * torch.randn(len(x), 3, 1, 1, device=DEV, generator=g)
    bias = eps * 0.5 * torch.randn(len(x), 3, 1, 1, device=DEV, generator=g)
    return (x * gain + bias).clamp(0, 1)


@torch.no_grad()
def feats(Xraw):
    outs = []
    for i in range(0, len(Xraw), 8192):
        P = patchify((Xraw[i:i+8192] - MEAN) / STD)
        outs.append((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).mean(1))
    return torch.cat(outs)


g = torch.Generator(device=DEV).manual_seed(1)
Ftr = feats(Xtr)
Fva = feats(Xva); Fva_s = feats(stain(Xva, 0.1, g))
Fte = feats(Xte); Fte_s = feats(stain(Xte, 0.1, g))


def auc_all(Fsel, y):
    # per-filter per-class |AUC-0.5|, return max over classes (best discriminative role)
    R = torch.argsort(torch.argsort(Fsel, 0), 0).float() + 1  # ranks per filter
    out = torch.zeros(Fsel.shape[1], device=DEV)
    for c in range(9):
        pos = (y == c); npos = pos.sum().float(); nneg = (~pos).sum().float()
        if npos == 0 or nneg == 0:
            continue
        a = (R[pos].sum(0) - npos*(npos+1)/2) / (npos*nneg)
        out = torch.maximum(out, (a - 0.5).abs())
    return out


s_clean = auc_all(Fva, yva)
s_stain = auc_all(Fva_s, yva)
s_gen = torch.minimum(s_clean, s_stain)


def fit_eval(units):
    cols = torch.tensor(units, device=DEV)
    head = nn.Linear(len(units), 9).to(DEV)
    Xf = Ftr[:, cols]; mu, sd = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True).clamp_min(1e-6)
    opt = torch.optim.AdamW(head.parameters(), 5e-3, weight_decay=1e-3)
    for s in range(3000):
        bi = torch.randint(0, len(Xf), (1024,), device=DEV)
        l = F.cross_entropy(head((Xf[bi]-mu)/sd), ytr[bi]); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        te = float((head((Fte[:, cols]-mu)/sd).argmax(1) == yte).float().mean())
        te_s = float((head((Fte_s[:, cols]-mu)/sd).argmax(1) == yte).float().mean())
    return round(te, 4), round(te_s, 4)


K = 32
sel_A = torch.argsort(-s_clean)[:K].tolist()     # baseline: strongest in-domain
sel_B = torch.argsort(-s_gen)[:K].tolist()       # method: robust across stain nuisance
overlap = len(set(sel_A) & set(sel_B))
A_te, A_te_s = fit_eval(sel_A)
B_te, B_te_s = fit_eval(sel_B)
res = {'K': K, 'overlap_AB': overlap,
       'baseline_strength': {'test': A_te, 'test_stained': A_te_s},
       'method_generalization': {'test': B_te, 'test_stained': B_te_s}}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/med_loop_method.json', 'w'), indent=2)
print('MED LOOP METHOD DONE', flush=True)
