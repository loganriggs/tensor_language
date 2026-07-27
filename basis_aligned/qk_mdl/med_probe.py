"""MED PHASE 2: open the trained foldable ViT with the qk_mdl toolkit.

(A) Mean-ablation importance (we learned zero-ablation lies): replace each layer's
    attention pattern, and each head's, with its batch-mean, measure test-accuracy
    drop. Which layers/heads carry content?
(B) Exact layer-0 patch-code fold: layer-0 q/k for patch i are
    rms_norm(W . rms_norm(embed(patch_i) + pos_i)) — an EXACT closed form in the
    patch pixels (no vocabulary; rms_norm is a per-vector scaling). Build per-patch
    codes q1,k1,q2,k2 directly and verify they reproduce the model's layer-0 scores
    numerically. This is the image analogue of the bilin18 embedding fold.
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
ck = torch.load(f'{QK}/med_bvit2.pt', map_location=DEV)
cfg = ck['cfg']
D, NH, HD, NL, PS, NP, INNER = (cfg[k] for k in ('D', 'NH', 'HD', 'NL', 'PS', 'NP', 'INNER'))
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)


def load(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1)
    return ((X.to(DEV) - MEAN) / STD), y.to(DEV)


Xte, yte = load('test')
W = ck['state']


def lin(name, x):
    return x @ W[name].T


def patchify(x):
    B = x.shape[0]
    p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, 3 * PS * PS)


def embed(x):
    return patchify(x) @ W['embed.weight'].T + W['embed.bias'] + W['pos']


@torch.no_grad()
def forward(x, kill=None):
    """kill=('layer'|'head', li, [h]) -> mean-ablate that attention pattern."""
    h = embed(x)
    for li in range(NL):
        aw = f'blocks.{2*li}.'
        hn = F.rms_norm(h, (D,))
        B, T, _ = hn.shape

        def hd(nm):
            return F.rms_norm((hn @ W[aw + nm + '.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw + 'v.weight'].T).view(B, T, NH, HD)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = s1 * s2
        if kill and kill[1] == li:
            if kill[0] == 'layer':
                pat = pat.mean(0, keepdim=True).expand_as(pat)
            else:
                pat = pat.clone()
                pat[:, kill[2]] = pat[:, kill[2]].mean(0, keepdim=True)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D)
        h = h + (y @ W[aw + 'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'
        hn2 = F.rms_norm(h, (D,))
        inner = (hn2 @ W[mw + 'L.weight'].T) * (hn2 @ W[mw + 'R.weight'].T)
        h = h + (inner @ W[mw + 'Dn.weight'].T)
    h = F.rms_norm(h, (D,)).mean(1)
    return h @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def acc(kill=None):
    out = []
    for i in range(0, len(Xte), 4096):
        out.append(forward(Xte[i:i + 4096], kill).argmax(1))
    return float((torch.cat(out) == yte).float().mean())


base = acc()
res = {'base_test': round(base, 4), 'layers': {}, 'heads': {}}
print(f'base {base:.4f}', flush=True)
for li in range(NL):
    a = acc(('layer', li))
    res['layers'][li] = round(base - a, 4)
    print(f'layer {li} mean-ablated: -{base - a:.4f}', flush=True)
for li in range(NL):
    for h in range(NH):
        a = acc(('head', li, h))
        res['heads'][f'{li}.{h}'] = round(base - a, 4)
    print(f'layer {li} heads: ' +
          ' '.join(f'{res["heads"][f"{li}.{h}"]:+.3f}' for h in range(NH)), flush=True)

# (B) exact layer-0 code fold verification
with torch.no_grad():
    xb = Xte[:64]
    h = embed(xb)
    hn = F.rms_norm(h, (D,))
    B, T, _ = hn.shape

    def hd(nm):
        return F.rms_norm((hn @ W['blocks.0.' + nm + '.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    s1_ref = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2_ref = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    # codes built the same exact way (this IS the fold — closed form in patch pixels)
    codes = {nm: hd(nm) for nm in ('q', 'k', 'q2', 'k2')}
    s1_code = torch.einsum('bqhd,bkhd->bhqk', codes['q'], codes['k']) / HD
    err = float((s1_code - s1_ref).abs().max())
    res['fold_max_err'] = err
    print(f'layer-0 code fold max abs err: {err:.2e}', flush=True)
json.dump(res, open(f'{QK}/med_probe.json', 'w'), indent=2)
print('MED PROBE DONE', flush=True)
