"""MED PHASE 1 (Logan redirect 2026-07-27): train a FOLDABLE bilinear-attention ViT
on PathMNIST, then apply the qk_mdl toolkit to extract its algorithm.

Architecture mirrors bilin18 so every technique ports:
- patch embed 7x7 -> 16 patches, D-dim; learned pos emb; NO cls token (mean-pool).
- attention pattern = (q1.k1)(q2.k2)/d^2, NO softmax, NO causal mask (bidirectional).
- bilinear MLP: Down(Left(x) * Right(x)).
- rms_norm pre-attn and pre-mlp; residual stream.
Target: match MedLiT-nano-class accuracy (~90%+) at <1M params so the fold is
tractable. This tick just trains + checkpoints; folding follows in later ticks.
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


def load(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2) / 255.0   # (N,3,28,28)
    y = torch.from_numpy(d.labels).long().squeeze(1)
    return X, y


Xtr, ytr = load('train')
Xva, yva = load('val')
Xte, yte = load('test')
MEAN = Xtr.mean((0, 2, 3), keepdim=True)
STD = Xtr.std((0, 2, 3), keepdim=True)
Xtr = ((Xtr - MEAN) / STD)
Xva = ((Xva - MEAN) / STD)
Xte = ((Xte - MEAN) / STD)
print(f'train {Xtr.shape} val {Xva.shape} test {Xte.shape}', flush=True)

D = 96
NH = 6
HD = D // NH
NL = 3
PS = 7                                       # patch size -> 4x4 = 16 patches
NP = (28 // PS) ** 2
INNER = 192


class BilinAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.q2 = nn.Linear(D, D, bias=False)
        self.k2 = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        h = F.rms_norm(x, (D,))

        def hd(lin):
            return F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd(self.q), hd(self.k), hd(self.q2), hd(self.k2)
        v = self.v(h).view(B, T, NH, HD)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = s1 * s2                                        # no softmax, no mask
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D)
        return x + self.proj(y)


class BilinMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = nn.Linear(D, INNER, bias=False)
        self.R = nn.Linear(D, INNER, bias=False)
        self.Dn = nn.Linear(INNER, D, bias=False)

    def forward(self, x):
        h = F.rms_norm(x, (D,))
        return x + self.Dn(self.L(h) * self.R(h))


class MedBViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3 * PS * PS, D)
        self.pos = nn.Parameter(torch.randn(1, NP, D) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(NL):
            self.blocks.append(BilinAttn())
            self.blocks.append(BilinMLP())
        self.head = nn.Linear(D, 9)

    def patchify(self, x):
        B = x.shape[0]
        p = x.unfold(2, PS, PS).unfold(3, PS, PS)          # B,3,4,4,7,7
        p = p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, 3 * PS * PS)
        return p

    def forward(self, x):
        h = self.embed(self.patchify(x)) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = F.rms_norm(h, (D,)).mean(1)
        return self.head(h)


net = MedBViT().to(DEV)
P = sum(p.numel() for p in net.parameters())
print(f'params {P/1e6:.3f}M', flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=8000, pct_start=0.1)
Xtr_d, ytr_d = Xtr.to(DEV), ytr.to(DEV)
Xva_d, yva_d = Xva.to(DEV), yva.to(DEV)


@torch.no_grad()
def acc(X, y):
    net.eval()
    outs = []
    for i in range(0, len(X), 2048):
        outs.append(net(X[i:i + 2048]).argmax(1))
    net.train()
    return float((torch.cat(outs) == y).float().mean())


best = 0.0
for step in range(8000):
    bi = torch.randint(0, len(Xtr_d), (256,), device=DEV)
    logits = net(Xtr_d[bi])
    loss = F.cross_entropy(logits, ytr_d[bi])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    sched.step()
    if (step + 1) % 1000 == 0:
        va = acc(Xva_d, yva_d)
        if va > best:
            best = va
            torch.save({'state': net.state_dict(), 'mean': MEAN, 'std': STD,
                        'cfg': {'D': D, 'NH': NH, 'HD': HD, 'NL': NL, 'PS': PS,
                                'NP': NP, 'INNER': INNER}},
                       f'{QK}/med_bvit.pt')
        print(f'step {step+1}: loss {loss.item():.3f} val {va:.4f} (best {best:.4f})',
              flush=True)
ckpt = torch.load(f'{QK}/med_bvit.pt', map_location=DEV)
net.load_state_dict(ckpt['state'])
te = acc(Xte.to(DEV), yte.to(DEV))
out = {'params_M': round(P / 1e6, 4), 'best_val': round(best, 4), 'test_acc': round(te, 4)}
print(json.dumps(out), flush=True)
json.dump(out, open(f'{QK}/med_bvit.json', 'w'), indent=2)
print('MED BVIT DONE', flush=True)
