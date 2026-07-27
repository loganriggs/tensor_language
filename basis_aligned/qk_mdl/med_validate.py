"""MED PHASE 15: the extract -> validate-by-generalization loop, on the REAL cancer
class. The fold gives exact candidate features; cross-institution generalization
(PathMNIST val = train institutions, test = held-out institution) separates true
(causal, generalizing) features from train-specific/spurious ones.

For adenocarcinoma (class 8): per extracted texture filter, discriminative power
(AUC, filter activation vs class-8-vs-rest) on VAL (in-domain) and TEST (shifted).
True-candidate features = high on both; spurious = high val, low test. Render both,
plus a robustness check: a cancer detector built from generalizing filters vs from
train-specific filters, evaluated on the shift.
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
CANCER = 8
ck = torch.load(f'{QK}/med_bvit2.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, PS, NP, INNER = cfg['D'], cfg['PS'], cfg['NP'], cfg['INNER']
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
PXD = 3 * PS * PS


def load(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2).to(DEV) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1).to(DEV)
    return ((X - MEAN) / STD), y


Xtr, ytr = load('train'); Xva, yva = load('val'); Xte, yte = load('test')


def patchify(x):
    B = x.shape[0]; p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


A = W['embed.weight'].T @ W['blocks.1.L.weight'].T
Bm = W['embed.weight'].T @ W['blocks.1.R.weight'].T


@torch.no_grad()
def feats(X):
    outs = []
    for i in range(0, len(X), 8192):
        P = patchify(X[i:i+8192])
        outs.append((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).mean(1))
    return torch.cat(outs)


Ftr, Fva, Fte = feats(Xtr), feats(Xva), feats(Xte)
imp = (Ftr.abs().mean(0) * W['blocks.1.Dn.weight'].norm(dim=0)).cpu().numpy()
order = list(np.argsort(-imp))[:64]


def auc(scores, labels):
    # AUC of scores for positive class via rank statistic (labels bool)
    order_ = torch.argsort(scores)
    ranks = torch.empty_like(order_, dtype=torch.float)
    ranks[order_] = torch.arange(1, len(scores)+1, device=DEV, dtype=torch.float)
    npos = labels.sum().float(); nneg = (~labels).sum().float()
    if npos == 0 or nneg == 0:
        return 0.5
    return float((ranks[labels].sum() - npos*(npos+1)/2) / (npos*nneg))


va_pos = (yva == CANCER); te_pos = (yte == CANCER)
rows = []
for j in order:
    a_va = auc(Fva[:, j], va_pos); a_te = auc(Fte[:, j], te_pos)
    rows.append({'unit': int(j), 'val_auc': round(a_va, 3), 'test_auc': round(a_te, 3),
                 'val_strength': round(abs(a_va-0.5), 3), 'test_strength': round(abs(a_te-0.5), 3)})
va = np.array([r['val_auc'] for r in rows]); te = np.array([r['test_auc'] for r in rows])
corr = float(np.corrcoef(np.abs(va-0.5), np.abs(te-0.5))[0, 1])

# rank by generalizing strength (min of the two) and by train-specificity (val-test gap)
for r in rows:
    r['generalizes'] = round(min(r['val_strength'], r['test_strength']), 3)
    r['train_specific_gap'] = round(r['val_strength'] - r['test_strength'], 3)
gen = sorted(rows, key=lambda r: -r['generalizes'])[:6]
spec = sorted(rows, key=lambda r: -r['train_specific_gap'])[:6]

# robustness check: cancer detector (class-8-vs-rest) from generalizing vs train-specific filters
def detect_auc(units, Ffit, yfit, Feval, yeval):
    cols = torch.tensor(units, device=DEV)
    head = nn.Linear(len(units), 1).to(DEV)
    Xf = Ffit[:, cols]; mu, sd = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True).clamp_min(1e-6)
    opt = torch.optim.Adam(head.parameters(), 1e-2)
    yb = (yfit == CANCER).float()
    for s in range(1500):
        bi = torch.randint(0, len(Xf), (2048,), device=DEV)
        l = F.binary_cross_entropy_with_logits(head(((Xf[bi]-mu)/sd)).squeeze(1), yb[bi])
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        sc = head(((Feval[:, cols]-mu)/sd)).squeeze(1)
    return auc(sc, yeval == CANCER)


gen_units = [r['unit'] for r in gen]; spec_units = [r['unit'] for r in spec]
rob = {
    'generalizing_filters_test_auc': round(detect_auc(gen_units, Ftr, ytr, Fte, yte), 3),
    'train_specific_filters_test_auc': round(detect_auc(spec_units, Ftr, ytr, Fte, yte), 3),
}

# render candidate-true and spurious filter patterns
def pattern(j):
    a_j, b_j = A[:, j], Bm[:, j]
    S = 0.5*(torch.outer(a_j, b_j)+torch.outer(b_j, a_j))
    ev, evec = torch.linalg.eigh(S)
    v = evec[:, ev.abs().argmax()] * torch.sign(ev[ev.abs().argmax()])
    return v.reshape(3, PS, PS).cpu()


torch.save({'generalizing': {r['unit']: pattern(r['unit']) for r in gen},
            'train_specific': {r['unit']: pattern(r['unit']) for r in spec}},
           f'{QK}/med_validate_patterns.pt')
res = {'cancer_class': CANCER, 'auc_generalization_corr': round(corr, 3),
       'top_generalizing': gen, 'top_train_specific': spec, 'robustness': rob,
       'n_filters': len(rows)}
print(json.dumps({'corr': corr, 'robustness': rob,
                  'best_generalizing': gen[0], 'most_train_specific': spec[0]}, indent=1), flush=True)
json.dump(res, open(f'{QK}/med_validate.json', 'w'), indent=2)
print('MED VALIDATE DONE', flush=True)
