"""ECG STAGE 1 train: foldable patched bilinear model on PTB-XL 5-superclass task.
Same exact architecture as the medical ViT (no softmax, bilinear MLP) but the 12-lead
signal is patched along TIME (12 leads = channels, like RGB). Goal: competitive macro
AUC vs the ~0.93 reference, proving the foldable architecture transfers off images.
Keep small (~0.4M params); scale only if accuracy forces it.
"""
import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'


def load(split):
    X = torch.from_numpy(np.load(f'{OUT}/ecg_X_{split}.npy'))          # (N,12,1000)
    Y = torch.from_numpy(np.load(f'{OUT}/ecg_Y_{split}.npy'))
    return X.to(DEV), Y.to(DEV)


Xtr, Ytr = load('train'); Xva, Yva = load('val'); Xte, Yte = load('test')
# per-lead standardization from train
MU = Xtr.mean((0, 2), keepdim=True); SD = Xtr.std((0, 2), keepdim=True).clamp_min(1e-6)
norm = lambda x: (x - MU) / SD
NLEAD, TLEN = 12, 1000
PT = 50                       # time patch length -> 20 patches
NP = TLEN // PT
PXD = NLEAD * PT             # 600 per patch
D, NH, HD, NL, INNER = 96, 6, 16, 3, 192
NCLS = 5


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
print(f'params {P/1e6:.3f}M', flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=10000, pct_start=0.1)


def auc(scores, labels):
    o = torch.argsort(torch.argsort(scores)); r = o.float() + 1
    npos = labels.sum().float(); nneg = (~labels).sum().float()
    if npos == 0 or nneg == 0: return 0.5
    return float((r[labels].sum() - npos*(npos+1)/2)/(npos*nneg))


@torch.no_grad()
def macro_auc(X, Y):
    net.eval()
    sc = torch.cat([net(norm(X[i:i+2048])) for i in range(0, len(X), 2048)]).float()
    net.train()
    return float(np.mean([auc(sc[:, c], Y[:, c].bool()) for c in range(NCLS)]))


best = 0.0
for step in range(10000):
    bi = torch.randint(0, len(Xtr), (128,), device=DEV)
    x = norm(Xtr[bi])
    if torch.rand(1).item() < 0.5:                 # time-shift augmentation
        x = torch.roll(x, int(torch.randint(-50, 50, (1,))), dims=2)
    loss = F.binary_cross_entropy_with_logits(net(x), Ytr[bi])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    if (step+1) % 1000 == 0:
        va = macro_auc(Xva, Yva)
        if va > best:
            best = va
            torch.save({'state': net.state_dict(), 'MU': MU, 'SD': SD,
                        'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'INNER': INNER,
                                'PT': PT, 'NP': NP, 'PXD': PXD, 'NLEAD': NLEAD, 'NCLS': NCLS}},
                       f'{QK}/ecg_model.pt')
        print(f'step {step+1}: loss {loss.item():.3f} val macroAUC {va:.4f} (best {best:.4f})', flush=True)
ckpt = torch.load(f'{QK}/ecg_model.pt', map_location=DEV); net.load_state_dict(ckpt['state'])
te = macro_auc(Xte, Yte)
res = {'params_M': round(P/1e6, 4), 'best_val_macroAUC': round(best, 4),
       'test_macroAUC': round(te, 4), 'reference_macroAUC': 0.93}
print(json.dumps(res), flush=True)
json.dump(res, open(f'{QK}/ecg_train.json', 'w'), indent=2)
print('ECG TRAIN DONE', flush=True)
