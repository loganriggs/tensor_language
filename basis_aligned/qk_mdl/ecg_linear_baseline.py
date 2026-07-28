"""ECG linear baseline (Logan's question): can a plain LINEAR model on the raw signal
match the foldable bilinear model? If yes, the nonlinearity/circuit story is oversold;
if no, the bilinear MLP is doing real work. Linear probe = single linear layer on the
flattened normalized 12x1000 signal, multi-label over the 35 codes. Compare per-code
AUC to the foldable model (ecg_codes_model.pt).
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


def labels(mask):
    Y = np.zeros((mask.sum(), NCLS), dtype=np.float32)
    for i, cc in enumerate(df.scp_codes.values[mask]):
        for j, c in enumerate(CODES):
            if c in cc:
                Y[i, j] = 1.0
    return torch.from_numpy(Y).to(DEV)


def load(split, mask):
    X = torch.from_numpy(np.load(f'{OUT}/ecg_X_{split}.npy')).to(DEV)
    X = ((X - MU) / SD).reshape(len(X), -1)                 # flatten 12x1000 -> 12000
    return X, labels(mask)


Xtr, Ytr = load('train', fold <= 8)
Xte, Yte = load('test', fold == 10)
DIM = Xtr.shape[1]
lin = nn.Linear(DIM, NCLS).to(DEV)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)   # L2-regularized
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=8000, pct_start=0.1)


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


for step in range(8000):
    bi = torch.randint(0, len(Xtr), (256,), device=DEV)
    x = Xtr[bi]
    if torch.rand(1).item() < 0.5:                          # same time-shift aug (on flattened -> reshape)
        x = torch.roll(x.view(-1, 12, 1000), int(torch.randint(-50, 50, (1,))), dims=2).reshape(len(bi), -1)
    loss = F.binary_cross_entropy_with_logits(lin(x), Ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
with torch.no_grad():
    s = torch.cat([lin(Xte[i:i+4096]) for i in range(0, len(Xte), 4096)]).float()
lin_auc = {CODES[c]: round(auc(s[:, c], Yte[:, c].bool()), 3) for c in range(NCLS)}

# model per-code AUC (from capability json) for comparison
model_auc = {d['code']: d['auc'] for d in json.load(open(f'{QK}/ecg_codes_train.json'))['per_code']}
cap = [c for c in CODES if model_auc.get(c, 0) >= 0.75 and int(Yte[:, CODES.index(c)].sum()) >= 10]
rows = [(c, model_auc[c], lin_auc[c], round(model_auc[c] - lin_auc[c], 3)) for c in cap]
rows.sort(key=lambda r: -r[3])
res = {'linear_macro_capable': round(float(np.mean([lin_auc[c] for c in cap])), 4),
       'model_macro_capable': round(float(np.mean([model_auc[c] for c in cap])), 4),
       'mean_model_minus_linear': round(float(np.mean([r[3] for r in rows])), 4),
       'codes_model_beats_linear_by>=0.05': sum(1 for r in rows if r[3] >= 0.05),
       'biggest_gaps': [{'code': r[0], 'model': r[1], 'linear': r[2], 'gap': r[3]} for r in rows[:8]],
       'smallest_gaps': [{'code': r[0], 'model': r[1], 'linear': r[2], 'gap': r[3]} for r in rows[-5:]]}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_linear_baseline.json', 'w'), indent=2)
print('ECG LINEAR BASELINE DONE', flush=True)
