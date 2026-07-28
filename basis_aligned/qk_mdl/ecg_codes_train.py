"""ECG fine-grained: train on the SPECIFIC diagnostic SCP codes (not 5 superclasses),
and FIRST establish per-code capability (which codes the model can actually predict)
before any decomposition. This is the "account for whether the model is capable"
step: we will only decompose codes the model genuinely learns.
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
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
agg = pd.read_csv(f'{OUT}/scp_statements.csv', index_col=0)
DIAG = list(agg[agg.diagnostic == 1].index)                 # specific diagnostic codes
fold = df.strat_fold.values

# multi-hot over diagnostic codes; keep codes with >=40 train-positive records
present = np.zeros((len(df), len(DIAG)), dtype=np.float32)
for i, codes in enumerate(df.scp_codes.values):
    for j, c in enumerate(DIAG):
        if c in codes:
            present[i, j] = 1.0
tr_mask = fold <= 8
prev = present[tr_mask].sum(0)
keep = prev >= 40
CODES = [DIAG[j] for j in range(len(DIAG)) if keep[j]]
Y = present[:, keep]
print(f'{len(DIAG)} diagnostic codes -> {len(CODES)} with >=40 train positives', flush=True)


def load(split, mask):
    X = torch.from_numpy(np.load(f'{OUT}/ecg_X_{split}.npy')).to(DEV)
    return X, torch.from_numpy(Y[mask]).to(DEV)


Xtr, Ytr = load('train', fold <= 8)
Xva, Yva = load('val', fold == 9)
Xte, Yte = load('test', fold == 10)
MU = Xtr.mean((0, 2), keepdim=True); SD = Xtr.std((0, 2), keepdim=True).clamp_min(1e-6)
norm = lambda x: (x - MU) / SD
NLEAD, TLEN, PT = 12, 1000, 50
NP = TLEN // PT; PXD = NLEAD * PT
D, NH, HD, NL, INNER = 96, 6, 16, 3, 192
NCLS = len(CODES)


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
        s.head = nn.Linear(D, NCLS)
    def patch(s, x):
        B = x.shape[0]
        return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)
    def forward(s, x):
        h = s.embed(s.patch(x)) + s.pos
        for b in s.blocks: h = b(h)
        return s.head(F.rms_norm(h, (D,)).mean(1))


net = Net().to(DEV)
P = sum(p.numel() for p in net.parameters())
print(f'params {P/1e6:.3f}M, {NCLS} codes', flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=12000, pct_start=0.1)


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


@torch.no_grad()
def per_code_auc(X, Y):
    net.eval(); s = torch.cat([net(norm(X[i:i+2048])) for i in range(0, len(X), 2048)]).float(); net.train()
    return [auc(s[:, c], Y[:, c].bool()) for c in range(NCLS)]


best = 0.0
for step in range(12000):
    bi = torch.randint(0, len(Xtr), (128,), device=DEV)
    x = norm(Xtr[bi])
    if torch.rand(1).item() < 0.5:
        x = torch.roll(x, int(torch.randint(-50, 50, (1,))), dims=2)
    loss = F.binary_cross_entropy_with_logits(net(x), Ytr[bi])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    if (step+1) % 2000 == 0:
        m = float(np.mean(per_code_auc(Xva, Yva)))
        if m > best:
            best = m
            torch.save({'state': net.state_dict(), 'MU': MU, 'SD': SD, 'codes': CODES,
                        'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'INNER': INNER,
                                'PT': PT, 'NP': NP, 'PXD': PXD, 'NLEAD': NLEAD, 'NCLS': NCLS}},
                       f'{QK}/ecg_codes_model.pt')
        print(f'step {step+1}: loss {loss.item():.3f} val macro-AUC {m:.4f} (best {best:.4f})', flush=True)
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV); net.load_state_dict(ck['state'])
te = per_code_auc(Xte, Yte)
codecap = sorted([(CODES[c], round(te[c], 3), int(Y[fold == 10][:, c].sum())) for c in range(NCLS)],
                 key=lambda t: -t[1])
capable = [c for c, a, n in codecap if a >= 0.75 and n >= 10]
res = {'params_M': round(P/1e6, 4), 'n_codes': NCLS, 'test_macro_auc': round(float(np.mean(te)), 4),
       'n_capable_auc>=0.75': len(capable), 'per_code': [{'code': c, 'auc': a, 'test_pos': n} for c, a, n in codecap]}
print(json.dumps({'macro': res['test_macro_auc'], 'n_capable': len(capable),
                  'top': res['per_code'][:8], 'bottom': res['per_code'][-6:]}, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_codes_train.json', 'w'), indent=2)
print('ECG CODES TRAIN DONE', flush=True)
