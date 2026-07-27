"""MED PHASE 14: plant confounders, train, verify reliance. Then (phase 14b) a
detection bake-off: does the tensor fold find them, and do we NEED it vs standard
baselines (color / saliency / causal ablation)?

Confounder = a color-NEUTRAL localized marker (so a global color baseline is blind
to it): a 4x4 gray square at a fixed corner, stamped on ALL training images of one
class. Simplicity bias should make the model exploit it. Verify reliance by
stamping the marker onto OTHER classes' test images and measuring prediction flips.
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
CLS = 0            # confounded class (adipose)
MVAL = 0.5         # gray marker value (color-neutral)
MPOS = (0, 4)      # top-left 4x4 corner (inside patch 0)


def rawload(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2).to(DEV) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1).to(DEV)
    return X, y


def stamp(x):
    x = x.clone()
    x[:, :, MPOS[0]:MPOS[1], MPOS[0]:MPOS[1]] = MVAL
    return x


Xtr, ytr = rawload('train')
Xva, yva = rawload('val')
Xte, yte = rawload('test')
# plant marker on all training + val images of CLS
Xtr = Xtr.clone(); Xtr[ytr == CLS] = stamp(Xtr[ytr == CLS])
Xva = Xva.clone(); Xva[yva == CLS] = stamp(Xva[yva == CLS])
MEAN = Xtr.mean((0, 2, 3), keepdim=True); STD = Xtr.std((0, 2, 3), keepdim=True)
norm = lambda x: (x - MEAN) / STD
D, NH, HD, NL, PS, NP, INNER = 96, 6, 16, 3, 7, 16, 192


class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        for n in ('q', 'k', 'q2', 'k2', 'v', 'proj'):
            setattr(s, n, nn.Linear(D, D, bias=False))
    def forward(s, x):
        B, T, _ = x.shape; h = F.rms_norm(x, (D,))
        def hd(l): return F.rms_norm(l(h).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd(s.q), hd(s.k), hd(s.q2), hd(s.k2)
        v = s.v(h).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        return x + s.proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.L = nn.Linear(D, INNER, bias=False); s.R = nn.Linear(D, INNER, bias=False); s.Dn = nn.Linear(INNER, D, bias=False)
    def forward(s, x):
        h = F.rms_norm(x, (D,)); return x + s.Dn(s.L(h)*s.R(h))


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.embed = nn.Linear(3*PS*PS, D); s.pos = nn.Parameter(torch.randn(1, NP, D)*0.02)
        s.blocks = nn.ModuleList([m for _ in range(NL) for m in (Attn(), MLP())])
        s.head = nn.Linear(D, 9)
    def patch(s, x):
        B = x.shape[0]; p = x.unfold(2, PS, PS).unfold(3, PS, PS)
        return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, 3*PS*PS)
    def forward(s, x):
        h = s.embed(s.patch(x)) + s.pos
        for b in s.blocks: h = b(h)
        return s.head(F.rms_norm(h, (D,)).mean(1))


net = Net().to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=8000, pct_start=0.1)
def aug(x):
    if torch.rand(1).item() < .5: x = x.flip(3)
    if torch.rand(1).item() < .5: x = x.flip(2)
    return x
for step in range(8000):
    bi = torch.randint(0, len(Xtr), (256,), device=DEV)
    loss = F.cross_entropy(net(norm(aug(Xtr[bi]))), ytr[bi])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()


@torch.no_grad()
def acc(X, y):
    net.eval(); o = torch.cat([net(norm(X[i:i+4096])).argmax(1) for i in range(0, len(X), 4096)]); net.train()
    return float((o == y).float().mean())


torch.save({'state': net.state_dict(), 'mean': MEAN, 'std': STD, 'CLS': CLS,
            'MPOS': MPOS, 'MVAL': MVAL,
            'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'PS': PS, 'NP': NP, 'INNER': INNER}},
           f'{QK}/med_confound.pt')
# reliance tests
clean_test = acc(Xte, yte)
# class-CLS test recall WITHOUT marker (should be low if model depends on marker)
mc = yte == CLS
recall_nomark = acc(Xte[mc], yte[mc])
recall_mark = acc(stamp(Xte[mc]), yte[mc])
# stamp marker on OTHER classes -> flip to CLS?
other = yte != CLS
with torch.no_grad():
    net.eval()
    pred_stamped = torch.cat([net(norm(stamp(Xte[other])[i:i+4096])).argmax(1) for i in range(0, int(other.sum()), 4096)])
flip_to_cls = float((pred_stamped == CLS).float().mean())
res = {'clean_test': round(clean_test, 4), 'CLS': CLS,
       'recall_CLS_no_marker': round(recall_nomark, 3), 'recall_CLS_with_marker': round(recall_mark, 3),
       'flip_to_CLS_when_stamped_on_others': round(flip_to_cls, 3)}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/med_confound.json', 'w'), indent=2)
print('MED CONFOUND DONE', flush=True)
