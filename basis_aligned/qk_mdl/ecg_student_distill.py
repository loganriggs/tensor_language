"""Path 2 step 2: distill the SOTA teacher into a FOLDABLE student. The student is our
no-softmax bilinear model (patched-in-time, exactly foldable) trained to MIMIC the teacher's
6 soft outputs on PTB-XL inputs (our 100Hz format) -- teacher provides the labels, so we
capture its capability WITHOUT its training data. Then verify the student matches the teacher
(agreement + soft-output correlation), so the downstream interpretability is of the SOTA behavior.
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
TCLASSES = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
NCLS = 6
NLEAD, TLEN, PT = 12, 1000, 50
NP = TLEN // PT; PXD = NLEAD * PT
D, NH, HD, NL, INNER = 96, 6, 16, 3, 192

Str = torch.from_numpy(np.load(f'{QK}/teacher_soft_train.npy')).to(DEV)
Sva = torch.from_numpy(np.load(f'{QK}/teacher_soft_val.npy')).to(DEV)
Ste = torch.from_numpy(np.load(f'{QK}/teacher_soft_test.npy')).to(DEV)
Xtr = torch.from_numpy(np.load(f'{OUT}/ecg_X_train.npy')).to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
MU = Xtr.mean((0, 2), keepdim=True); SD = Xtr.std((0, 2), keepdim=True).clamp_min(1e-6)
norm = lambda x: (x - MU) / SD
# PTB-XL true labels for the 3 overlapping classes
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
def ptl(code): return torch.tensor([1.0 if code in cc else 0.0 for cc in df.scp_codes.values[fold == 10]], device=DEV)
TRUE = {'1dAVb': ptl('1AVB'), 'RBBB': ptl('CRBBB'), 'LBBB': ptl('CLBBB')}


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
        s.blocks = nn.ModuleList([m for _ in range(NL) for m in (Attn(), MLP())]); s.head = nn.Linear(D, NCLS)
    def patch(s, x):
        B = x.shape[0]; return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)
    def forward(s, x):
        h = s.embed(s.patch(norm(x))) + s.pos
        for b in s.blocks: h = b(h)
        return s.head(F.rms_norm(h, (D,)).mean(1))


net = Net().to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=12000, pct_start=0.1)


def auc(s, y):
    y = y.bool(); p = y.sum().float(); n = (~y).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(s)).float()+1
    return float((r[y].sum()-p*(p+1)/2)/(p*n))


for step in range(12000):
    bi = torch.randint(0, len(Xtr), (128,), device=DEV)
    loss = F.binary_cross_entropy_with_logits(net(Xtr[bi]), Str[bi])   # soft-target distillation
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if step % 3000 == 0:
        with torch.no_grad():
            s = torch.sigmoid(torch.cat([net(Xte[i:i+1024]) for i in range(0, len(Xte), 1024)]))
        fid = np.mean([auc(s[:, j], (Ste[:, j] > 0.5)) for j in range(6)])
        print(f'  step {step} loss {loss.item():.4f} teacher-agreement AUC {fid:.3f}', flush=True)

with torch.no_grad():
    ste = torch.sigmoid(torch.cat([net(Xte[i:i+1024]) for i in range(0, len(Xte), 1024)])).float()
# fidelity: agreement with teacher hard labels + soft correlation; and vs PTB-XL truth
fidelity = {}
for j, c in enumerate(TCLASSES):
    th = (Ste[:, j] > 0.5)
    corr = float(torch.corrcoef(torch.stack([ste[:, j], Ste[:, j]]))[0, 1]) if th.sum() > 0 else None
    fidelity[c] = {'teacher_agreement_auc': round(auc(ste[:, j], th), 3),
                   'soft_corr': None if corr is None else round(corr, 3),
                   'n_teacher_pos': int(th.sum()),
                   'student_vs_true_auc': round(auc(ste[:, j], TRUE[c].bool()), 3) if c in TRUE else None,
                   'teacher_vs_true_auc': round(auc(Ste[:, j], TRUE[c].bool()), 3) if c in TRUE else None}
    print(f'  {c}: agree {fidelity[c]["teacher_agreement_auc"]} corr {fidelity[c]["soft_corr"]} '
          f'| student-vs-true {fidelity[c]["student_vs_true_auc"]} teacher-vs-true {fidelity[c]["teacher_vs_true_auc"]}', flush=True)

torch.save({'state': net.state_dict(), 'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'INNER': INNER,
            'PT': PT, 'NP': NP, 'PXD': PXD, 'NLEAD': NLEAD, 'NCLS': NCLS}, 'classes': TCLASSES,
            'MU': MU, 'SD': SD}, f'{QK}/ecg_student_model.pt')
res = {'teacher_classes': TCLASSES,
       'mean_teacher_agreement_auc': round(float(np.mean([fidelity[c]['teacher_agreement_auc'] for c in TCLASSES])), 3),
       'mean_soft_corr': round(float(np.mean([fidelity[c]['soft_corr'] for c in TCLASSES if fidelity[c]['soft_corr'] is not None])), 3),
       'fidelity': fidelity}
json.dump(res, open(f'{QK}/ecg_student_distill.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('mean_teacher_agreement_auc', 'mean_soft_corr')}, indent=1), flush=True)
print('ECG STUDENT DISTILL DONE', flush=True)
