"""MED PHASE 7b: does the PathMNIST structure generalize? Full probe on BloodMNIST.
Layer/head/MLP mean-ablation importance + attn-removed patch-local test + standalone
explicit quadratic-texture pipeline decomposition. Compare to PathMNIST (13a-13d).
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from medmnist import BloodMNIST

torch.manual_seed(0)
DEV = 'cuda'
NCLS = 8
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
ROOT = '/workspace/tensor_language/medmnist_data'
ck = torch.load(f'{QK}/med_blood.pt', map_location=DEV)
cfg = ck['cfg']
D, NH, HD, NL, PS, NP, INNER = (cfg[k] for k in ('D', 'NH', 'HD', 'NL', 'PS', 'NP', 'INNER'))
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
W = ck['state']
PXD = 3 * PS * PS


def load(split):
    d = BloodMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1)
    return ((X.to(DEV) - MEAN) / STD), y.to(DEV)


def patchify(x):
    B = x.shape[0]
    p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


def embed(x):
    return patchify(x) @ W['embed.weight'].T + W['embed.bias'] + W['pos']


Xtr, ytr = load('train')
Xte, yte = load('test')


@torch.no_grad()
def forward(x, kill=None, kill_mlp=None, kill_attn0=False):
    h = embed(x)
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
        if kill and kill[1] == li:
            if kill[0] == 'layer':
                pat = pat.mean(0, keepdim=True).expand_as(pat)
            else:
                pat = pat.clone(); pat[:, kill[2]] = pat[:, kill[2]].mean(0, keepdim=True)
        yo = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw + 'proj.weight'].T
        if kill_attn0 and li == 0:
            yo = torch.zeros_like(yo)
        h = h + yo
        mw = f'blocks.{2*li+1}.'
        hn2 = F.rms_norm(h, (D,))
        mo = ((hn2 @ W[mw + 'L.weight'].T) * (hn2 @ W[mw + 'R.weight'].T)) @ W[mw + 'Dn.weight'].T
        if kill_mlp is not None and li == kill_mlp:
            mo = mo.mean(0, keepdim=True).expand_as(mo)
        h = h + mo
    h = F.rms_norm(h, (D,)).mean(1)
    return h @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def acc(**kw):
    out = []
    for i in range(0, len(Xte), 4096):
        out.append(forward(Xte[i:i + 4096], **kw).argmax(1))
    return float((torch.cat(out) == yte).float().mean())


base = acc()
res = {'base': round(base, 4), 'layers': {}, 'mlps': {}, 'heads': {}}
print(f'base {base:.4f}', flush=True)
for li in range(NL):
    res['layers'][li] = round(base - acc(kill=('layer', li)), 4)
    res['mlps'][li] = round(base - acc(kill_mlp=li), 4)
print('layer loads', res['layers'], flush=True)
print('mlp loads', res['mlps'], flush=True)
hload = {}
for li in range(NL):
    for h in range(NH):
        hload[f'{li}.{h}'] = round(base - acc(kill=('head', li, h)), 4)
res['heads'] = hload
res['n_heads_pos'] = sum(1 for v in hload.values() if v > 0.003)
res['attn0_removed'] = round(acc(kill_attn0=True), 4)
print(f'heads with load>0.003: {res["n_heads_pos"]}/18; attn0-removed {res["attn0_removed"]:.4f}',
      flush=True)

# explicit pipeline decomposition
We = W['embed.weight']
A = We.T @ W['blocks.1.L.weight'].T
Bm = We.T @ W['blocks.1.R.weight'].T
with torch.no_grad():
    P = patchify(Xtr[:2048])
    imp = ((torch.einsum('bnp,pj->bnj', P, A) * torch.einsum('bnp,pj->bnj', P, Bm)).abs().mean((0, 1))
           * W['blocks.1.Dn.weight'].norm(dim=0)).cpu().numpy()
order = list(np.argsort(-imp))


@torch.no_grad()
def feats(X, units):
    cols = torch.tensor(units, device=DEV)
    Au, Bu = A[:, cols], Bm[:, cols]
    outs = []
    for i in range(0, len(X), 4096):
        P = patchify(X[i:i + 4096])
        outs.append((torch.einsum('bnp,pj->bnj', P, Au) *
                     torch.einsum('bnp,pj->bnj', P, Bu)).mean(1))
    return torch.cat(outs)


def fit(Ftr, Fte, din):
    head = torch.nn.Linear(din, NCLS).to(DEV)
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-3)
    for s in range(3000):
        bi = torch.randint(0, len(Ftr), (1024,), device=DEV)
        loss = F.cross_entropy(head(Ftr[bi]), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((head(Fte).argmax(1) == yte).float().mean())


res['linear_floor'] = round(fit(patchify(Xtr).mean(1), patchify(Xte).mean(1), PXD), 4)
res['explicit_K192'] = round(fit(feats(Xtr, order[:192]), feats(Xte, order[:192]), 192), 4)
print(f'linear floor {res["linear_floor"]:.4f}  explicit-192 {res["explicit_K192"]:.4f}',
      flush=True)
json.dump(res, open(f'{QK}/med_blood_probe.json', 'w'), indent=2)
print('MED BLOOD PROBE DONE', flush=True)
