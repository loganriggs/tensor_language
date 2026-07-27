"""ECG STAGE 2 / PHASE C (discovery): sex-from-ECG. Unlike diagnosis, the ECG feature
encoding sex is only partly known -- a genuine discovery target (the ECG analog of
sex-from-retinal-fundus). Train a foldable model to predict sex on PTB-XL, so we can
later extract and cross-cohort-validate WHAT it uses. Same exact architecture/patching
as the diagnostic model. Reference: sex-from-ECG reaches AUC ~0.90 in the literature.
"""
import ast, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id')
fold = df.strat_fold.values
sex = df.sex.values.astype(np.float32)          # 0=male, 1=female (predict female)


def load(split, mask):
    X = torch.from_numpy(np.load(f'{OUT}/ecg_X_{split}.npy')).to(DEV)
    y = torch.from_numpy(sex[mask]).to(DEV)
    return X, y


Xtr, ytr = load('train', fold <= 8)
Xva, yva = load('val', fold == 9)
Xte, yte = load('test', fold == 10)
MU = Xtr.mean((0, 2), keepdim=True); SD = Xtr.std((0, 2), keepdim=True).clamp_min(1e-6)
norm = lambda x: (x - MU) / SD
NLEAD, TLEN, PT = 12, 1000, 50
NP = TLEN // PT; PXD = NLEAD * PT
D, NH, HD, NL, INNER = 96, 6, 16, 3, 192


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
        s.embed = nn.Linear(PXD, D); s.pos = nn.Parameter(torch.randn(1, NP, D)*0.02)
        s.blocks = nn.ModuleList([m for _ in range(NL) for m in (Attn(), MLP())])
        s.head = nn.Linear(D, 1)
    def patch(s, x):
        B = x.shape[0]
        return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)
    def forward(s, x):
        h = s.embed(s.patch(x)) + s.pos
        for b in s.blocks: h = b(h)
        return s.head(F.rms_norm(h, (D,)).mean(1)).squeeze(1)


net = Net().to(DEV)
P = sum(p.numel() for p in net.parameters())
print(f'params {P/1e6:.3f}M, female prevalence {ytr.mean():.3f}', flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=10000, pct_start=0.1)


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


@torch.no_grad()
def ev(X, y):
    net.eval(); s = torch.cat([net(norm(X[i:i+2048])) for i in range(0, len(X), 2048)]); net.train()
    return auc(s, y.bool())


best = 0.0
for step in range(10000):
    bi = torch.randint(0, len(Xtr), (128,), device=DEV)
    x = norm(Xtr[bi])
    if torch.rand(1).item() < 0.5:
        x = torch.roll(x, int(torch.randint(-50, 50, (1,))), dims=2)
    loss = F.binary_cross_entropy_with_logits(net(x), ytr[bi])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    if (step+1) % 1000 == 0:
        va = ev(Xva, yva)
        if va > best:
            best = va
            torch.save({'state': net.state_dict(), 'MU': MU, 'SD': SD,
                        'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'INNER': INNER,
                                'PT': PT, 'NP': NP, 'PXD': PXD, 'NLEAD': NLEAD, 'NCLS': 1}},
                       f'{QK}/ecg_sex_model.pt')
        print(f'step {step+1}: loss {loss.item():.3f} val AUC {va:.4f} (best {best:.4f})', flush=True)
ck = torch.load(f'{QK}/ecg_sex_model.pt', map_location=DEV); net.load_state_dict(ck['state'])
te = ev(Xte, yte)
res = {'params_M': round(P/1e6, 4), 'best_val_auc': round(best, 4), 'test_auc': round(te, 4),
       'reference_auc': 0.90}
print(json.dumps(res), flush=True)
json.dump(res, open(f'{QK}/ecg_sex.json', 'w'), indent=2)
print('ECG SEX DONE', flush=True)
