"""Tier-2 step 4 (resolve the §51 caveat): distill the teacher's AGE-GAP directly. In §51 the
student matched raw ECG-age (corr 0.906) but the mortality-relevant age-gap (teacher_age - true_age,
a ~1y signal on a 16y prediction) washed out / REVERSED. Fix: make the gap the TARGET. Train the
foldable student to predict (teacher_age - true_age) from the ECG. Verify it preserves the mortality
direction (pathology gap > normal gap), unlike the raw-age student.
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

TAtr = np.load(f'{QK}/age_soft_train.npy')   # teacher predicted age
TAte = np.load(f'{QK}/age_soft_test.npy')
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
age = df['age'].values.astype(np.float32)
true_tr, true_te = age[fold <= 8], age[fold == 10]
gap_tr = TAtr - true_tr                       # teacher age-gap (mortality signal)
gap_te = TAte - true_te
vtr = (true_tr >= 18) & (true_tr <= 89)
vte = (true_te >= 18) & (true_te <= 89)
Xtr = torch.from_numpy(np.load(f'{OUT}/ecg_X_train.npy')).to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
MU = Xtr.mean((0, 2), keepdim=True); SD = Xtr.std((0, 2), keepdim=True).clamp_min(1e-6)
norm = lambda x: (x - MU) / SD
# train only on valid-age records; standardize gap target
idx_tr = np.where(vtr)[0]
gmu, gsd = float(gap_tr[vtr].mean()), float(gap_tr[vtr].std())
Gt = torch.from_numpy(((gap_tr - gmu)/gsd).astype(np.float32)).to(DEV)
idx_tr_t = torch.from_numpy(idx_tr).to(DEV)
norm_te = np.array(['NORM' in cc for cc in df.scp_codes.values[fold == 10]])


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
for step in range(12000):
    bi = idx_tr_t[torch.randint(0, len(idx_tr_t), (128,), device=DEV)]
    loss = F.mse_loss(net(Xtr[bi]), Gt[bi])
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if step % 3000 == 0:
        with torch.no_grad():
            p = (torch.cat([net(Xte[i:i+1024]) for i in range(0, len(Xte), 1024)])*gsd + gmu).cpu().numpy()
        r = float(np.corrcoef(p[vte], gap_te[vte])[0, 1])
        print(f'  step {step} loss {loss.item():.3f} student-gap vs teacher-gap corr {r:.3f}', flush=True)

with torch.no_grad():
    pg = (torch.cat([net(Xte[i:i+1024]) for i in range(0, len(Xte), 1024)])*gsd + gmu).cpu().numpy()
v = vte
corr = float(np.corrcoef(pg[v], gap_te[v])[0, 1])
# mortality direction: pathology vs normal predicted gap
nm = norm_te[v]; sg = pg[v]
res = {'student_gap_vs_teacher_gap_corr': round(corr, 3),
       'student_gap_pathology': round(float(sg[~nm].mean()), 3), 'student_gap_normal': round(float(sg[nm].mean()), 3),
       'student_gap_diff_path_minus_norm': round(float(sg[~nm].mean()-sg[nm].mean()), 3),
       'teacher_gap_pathology': round(float(gap_te[v][~nm].mean()), 3), 'teacher_gap_normal': round(float(gap_te[v][nm].mean()), 3),
       'teacher_gap_diff': round(float(gap_te[v][~nm].mean()-gap_te[v][nm].mean()), 3),
       'raw_age_student_diff_from_51': -0.46}
torch.save({'state': net.state_dict(), 'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'INNER': INNER,
            'PT': PT, 'NP': NP, 'PXD': PXD, 'NLEAD': NLEAD, 'NCLS': 1}, 'gmu': gmu, 'gsd': gsd,
            'MU': MU, 'SD': SD}, f'{QK}/ecg_agegap_student_model.pt')
json.dump(res, open(f'{QK}/ecg_age_gap_distill.json', 'w'), indent=2)
print(json.dumps(res, indent=1), flush=True)
print('ECG AGE GAP DISTILL DONE', flush=True)
