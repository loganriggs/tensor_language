"""MED PHASE 9: the efficiency number (Logan's third named goal).

Compare the trained foldable ViT against the extracted explicit pipeline on
parameters, estimated FLOPs, and measured wall-clock latency, at their respective
accuracies. Closes "improving efficiency" with a real figure.
"""
import sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import PathMNIST

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
ROOT = '/workspace/tensor_language/medmnist_data'
ck = torch.load(f'{QK}/med_bvit2.pt', map_location=DEV)
cfg = ck['cfg']
D, NH, HD, NL, PS, NP, INNER = (cfg[k] for k in ('D', 'NH', 'HD', 'NL', 'PS', 'NP', 'INNER'))
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
W = ck['state']
PXD = 3 * PS * PS


def load(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1)
    return ((X.to(DEV) - MEAN) / STD), y.to(DEV)


Xte, yte = load('test')


def patchify(x):
    B = x.shape[0]
    p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


@torch.no_grad()
def full_forward(x):
    h = patchify(x) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'
        hn = F.rms_norm(h, (D,))
        B, T, _ = hn.shape
        def hd(nm):
            return F.rms_norm((hn @ W[aw + nm + '.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw + 'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / HD) * \
              (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw + 'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'
        hn2 = F.rms_norm(h, (D,))
        h = h + (((hn2 @ W[mw + 'L.weight'].T) * (hn2 @ W[mw + 'R.weight'].T)) @ W[mw + 'Dn.weight'].T)
    h = F.rms_norm(h, (D,)).mean(1)
    return h @ W['head.weight'].T + W['head.bias']


# explicit pipeline weights: a_j, b_j (147 x 192), head (192 -> 9)
We = W['embed.weight']
A = (We.T @ W['blocks.1.L.weight'].T).contiguous()      # (147, 192)
Bm = (We.T @ W['blocks.1.R.weight'].T).contiguous()
# fit head once (deterministic)
Xtr, ytr = load('train')
@torch.no_grad()
def feats(X):
    outs = []
    for i in range(0, len(X), 8192):
        P = patchify(X[i:i + 8192])
        outs.append((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).mean(1))
    return torch.cat(outs)
Ftr, Fte = feats(Xtr), feats(Xte)
mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
head = nn.Linear(INNER, 9).to(DEV)
opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-3)
for s in range(3000):
    bi = torch.randint(0, len(Ftr), (1024,), device=DEV)
    loss = F.cross_entropy(head((Ftr[bi] - mu) / sd), ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step()
Whead, bhead = head.weight.detach(), head.bias.detach()


@torch.no_grad()
def explicit_forward(x):
    P = patchify(x)
    f = ((P @ A) * (P @ Bm)).mean(1)
    return ((f - mu) / sd) @ Whead.T + bhead


@torch.no_grad()
def timed(fn, X, iters=20):
    for i in range(0, len(X), 8192):        # warmup
        fn(X[i:i + 8192])
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        for i in range(0, len(X), 8192):
            fn(X[i:i + 8192])
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000     # ms per full test pass


@torch.no_grad()
def acc(fn):
    out = []
    for i in range(0, len(Xte), 8192):
        out.append(fn(Xte[i:i + 8192]).argmax(1))
    return float((torch.cat(out) == yte).float().mean())


full_params = sum(v.numel() for v in W.values())
expl_params = A.numel() + Bm.numel() + Whead.numel() + bhead.numel() + 2 * INNER
# FLOPs per image (multiply-adds): full ~ patch embed + per layer (4 qkv proj + attn + proj + mlp)
def full_flops():
    f = NP * PXD * D                        # embed
    for _ in range(NL):
        f += NP * (5 * D * D)               # q,k,q2,k2,v projections
        f += 2 * NH * NP * NP * HD          # scores
        f += NH * NP * NP * HD              # weighted v
        f += NP * D * D                     # out proj
        f += NP * (2 * D * INNER + INNER * D)  # bilinear mlp
    f += D * 9
    return f
expl_flops = NP * PXD * INNER * 2 + INNER * 9    # two bilinear projections + head

res = {
    'full': {'params': full_params, 'flops_per_img': full_flops(),
             'test_acc': round(acc(full_forward), 4), 'ms_per_pass': round(timed(full_forward, Xte), 2)},
    'explicit': {'params': int(expl_params), 'flops_per_img': int(expl_flops),
                 'test_acc': round(acc(explicit_forward), 4), 'ms_per_pass': round(timed(explicit_forward, Xte), 2)},
}
res['param_ratio'] = round(full_params / expl_params, 2)
res['flop_ratio'] = round(full_flops() / expl_flops, 2)
res['speedup'] = round(res['full']['ms_per_pass'] / res['explicit']['ms_per_pass'], 2)
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/med_efficiency.json', 'w'), indent=2)
print('MED EFFICIENCY DONE', flush=True)
