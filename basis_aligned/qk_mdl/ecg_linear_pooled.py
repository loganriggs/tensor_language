"""ECG FAIR linear baseline: linear on shift-tolerant POOLED features (per lead x
time-patch: mean, std, peak-abs) so temporal misalignment is not the bottleneck.
This isolates whether the task genuinely needs NONLINEARITY vs just temporal handling.
Compare per-code AUC to the foldable model.
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
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV)
CODES = ck['codes']; NCLS = len(CODES)
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
NP, PT = 20, 50


def pooled(X):
    # X (N,12,1000) normalized -> per (lead,patch) mean/std/peak -> (N, 12*20*3)
    x = ((X - MU) / SD).reshape(len(X), 12, NP, PT)
    return torch.cat([x.mean(-1), x.std(-1), x.abs().amax(-1)], 1).reshape(len(X), -1)


def labels(mask):
    Y = np.zeros((mask.sum(), NCLS), dtype=np.float32)
    for i, cc in enumerate(df.scp_codes.values[mask]):
        for j, c in enumerate(CODES):
            if c in cc:
                Y[i, j] = 1.0
    return torch.from_numpy(Y).to(DEV)


def load(split, mask):
    X = torch.from_numpy(np.load(f'{OUT}/ecg_X_{split}.npy')).to(DEV)
    return pooled(X), labels(mask)


Xtr, Ytr = load('train', fold <= 8)
Xte, Yte = load('test', fold == 10)
# standardize features
fmu, fsd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
Xtr, Xte = (Xtr - fmu) / fsd, (Xte - fmu) / fsd
DIM = Xtr.shape[1]
print(f'pooled feature dim {DIM}', flush=True)
lin = nn.Linear(DIM, NCLS).to(DEV)
opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=1e-3)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=10000, pct_start=0.1)


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


for step in range(10000):
    bi = torch.randint(0, len(Xtr), (256,), device=DEV)
    loss = F.binary_cross_entropy_with_logits(lin(Xtr[bi]), Ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
with torch.no_grad():
    s = torch.cat([lin(Xte[i:i+4096]) for i in range(0, len(Xte), 4096)]).float()
lin_auc = {CODES[c]: round(auc(s[:, c], Yte[:, c].bool()), 3) for c in range(NCLS)}
model_auc = {d['code']: d['auc'] for d in json.load(open(f'{QK}/ecg_codes_train.json'))['per_code']}
cap = [c for c in CODES if model_auc.get(c, 0) >= 0.75 and int(Yte[:, CODES.index(c)].sum()) >= 10]
rows = sorted([(c, model_auc[c], lin_auc[c], round(model_auc[c] - lin_auc[c], 3)) for c in cap], key=lambda r: -r[3])
res = {'feature_dim': DIM,
       'pooled_linear_macro': round(float(np.mean([lin_auc[c] for c in cap])), 4),
       'model_macro': round(float(np.mean([model_auc[c] for c in cap])), 4),
       'mean_gap': round(float(np.mean([r[3] for r in rows])), 4),
       'codes_model_beats_by>=0.05': sum(1 for r in rows if r[3] >= 0.05),
       'codes_within_0.03': sum(1 for r in rows if r[3] <= 0.03),
       'biggest_gaps': [{'code': r[0], 'model': r[1], 'pooled_linear': r[2], 'gap': r[3]} for r in rows[:6]],
       'smallest_gaps': [{'code': r[0], 'model': r[1], 'pooled_linear': r[2], 'gap': r[3]} for r in rows[-6:]]}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_linear_pooled.json', 'w'), indent=2)
print('ECG LINEAR POOLED DONE', flush=True)
