"""Tier-2 step 2: distill the ECG-AGE teacher into a FOLDABLE student (regression). Student mimics
the teacher's predicted ECG-age (MSE) on PTB-XL inputs. Verify student==teacher (corr, MAE) and
student-vs-true age. Then the foldable student carries the mortality-linked ECG-age biomarker and
can be exactly decomposed to see WHAT makes an ECG look older.
"""
import ast, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
NLEAD, TLEN, PT = 12, 1000, 50
NP = TLEN // PT; PXD = NLEAD * PT
D, NH, HD, NL, INNER = 96, 6, 16, 3, 192

Atr = torch.from_numpy(np.load(f'{QK}/age_soft_train.npy')).to(DEV)   # teacher ECG-age
Ate = torch.from_numpy(np.load(f'{QK}/age_soft_test.npy')).to(DEV)
Xtr = torch.from_numpy(np.load(f'{OUT}/ecg_X_train.npy')).to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
MU = Xtr.mean((0, 2), keepdim=True); SD = Xtr.std((0, 2), keepdim=True).clamp_min(1e-6)
norm = lambda x: (x - MU) / SD
amu, asd = float(Atr.mean()), float(Atr.std())                        # standardize age target
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
true_age = torch.from_numpy(df['age'].values.astype(np.float32)).to(DEV)[fold == 10]
valid = ((true_age >= 18) & (true_age <= 89)).cpu().numpy()


class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        for n in ('q', 'k', 'q2', 'k2', 'v', 'proj'): setattr(s, n, nn.Linear(D, D, bias=False))
    def forward(s, x):
        B, T, _ = x.shape; h = F.rms_norm(x, (D,))
        def hd(l): return F.rms_norm(l(h).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd(s.q), hd(s.k), hd(s.q2), hd(s.k2); v = s.v(h).view(B, T, NH, HD)
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
        s.blocks = nn.ModuleList([m for _ in range(NL) for m in (Attn(), MLP())]); s.head = nn.Linear(D, 1)
    def patch(s, x):
        B = x.shape[0]; return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)
    def forward(s, x):
        h = s.embed(s.patch(norm(x))) + s.pos
        for b in s.blocks: h = b(h)
        return s.head(F.rms_norm(h, (D,)).mean(1)).squeeze(1)


net = Net().to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=12000, pct_start=0.1)
tgt = (Atr - amu) / asd
for step in range(12000):
    bi = torch.randint(0, len(Xtr), (128,), device=DEV)
    loss = F.mse_loss(net(Xtr[bi]), tgt[bi])
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if step % 3000 == 0:
        with torch.no_grad():
            p = torch.cat([net(Xte[i:i+1024]) for i in range(0, len(Xte), 1024)])*asd + amu
        r = float(torch.corrcoef(torch.stack([p, Ate]))[0, 1])
        print(f'  step {step} loss {loss.item():.3f} student-vs-teacher age corr {r:.3f}', flush=True)

with torch.no_grad():
    pste = (torch.cat([net(Xte[i:i+1024]) for i in range(0, len(Xte), 1024)])*asd + amu).float()
corr_st = float(torch.corrcoef(torch.stack([pste, Ate]))[0, 1])
mae_st = float((pste - Ate).abs().mean())
pv, tv, av = pste.cpu().numpy()[valid], true_age.cpu().numpy()[valid], Ate.cpu().numpy()[valid]
res = {'student_vs_teacher_corr': round(corr_st, 3), 'student_vs_teacher_mae': round(mae_st, 2),
       'student_vs_true_corr': round(float(np.corrcoef(pv, tv)[0, 1]), 3),
       'teacher_vs_true_corr': round(float(np.corrcoef(av, tv)[0, 1]), 3),
       'student_vs_true_mae': round(float(np.mean(np.abs(pv-tv))), 2)}
torch.save({'state': net.state_dict(), 'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'INNER': INNER,
            'PT': PT, 'NP': NP, 'PXD': PXD, 'NLEAD': NLEAD, 'NCLS': 1}, 'amu': amu, 'asd': asd,
            'MU': MU, 'SD': SD}, f'{QK}/ecg_age_student_model.pt')
json.dump(res, open(f'{QK}/ecg_age_student_distill.json', 'w'), indent=2)
print(json.dumps(res, indent=1), flush=True)
print('ECG AGE STUDENT DISTILL DONE', flush=True)
