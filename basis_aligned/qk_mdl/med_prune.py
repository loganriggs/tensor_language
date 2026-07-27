"""MED PHASE 3: minimal sufficient circuit (the efficiency half of extraction).

Using mean-ablation (tick-262 discipline): (A) keep only the top-k attention heads
(mean-ablate the rest), sweep k; (B) mean-ablate each bilinear MLP; (C) the minimal
circuit = keep the 3 load-bearing heads + whichever MLPs are necessary, report its
accuracy versus the 85.7% full model. Whatever survives is the extraction target
for the visual-archetype fold.
"""
import sys, json
import numpy as np
import torch
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


def load(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1)
    return ((X.to(DEV) - MEAN) / STD), y.to(DEV)


Xte, yte = load('test')


def patchify(x):
    B = x.shape[0]
    p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, 3 * PS * PS)


def embed(x):
    return patchify(x) @ W['embed.weight'].T + W['embed.bias'] + W['pos']


@torch.no_grad()
def forward(x, keep_heads=None, kill_mlp=None):
    """keep_heads: set of 'li.h' to keep; all others mean-ablated. None = keep all.
    kill_mlp: set of layer indices whose MLP is mean-ablated."""
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
        if keep_heads is not None:
            pm = pat.mean(0, keepdim=True).expand_as(pat).clone()
            for hh in range(NH):
                if f'{li}.{hh}' in keep_heads:
                    pm[:, hh] = pat[:, hh]
            pat = pm
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D)
        h = h + (y @ W[aw + 'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'
        hn2 = F.rms_norm(h, (D,))
        mo = ((hn2 @ W[mw + 'L.weight'].T) * (hn2 @ W[mw + 'R.weight'].T)) @ W[mw + 'Dn.weight'].T
        if kill_mlp and li in kill_mlp:
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
res = {'base': round(base, 4)}
print(f'base {base:.4f}', flush=True)

# head ranking from probe
ranked = ['2.1', '2.2', '0.0', '2.0', '1.0', '1.3', '1.5', '0.1', '1.4']
for k in (1, 2, 3, 4, 6, 9):
    keep = set(ranked[:k])
    a = acc(keep_heads=keep)
    res[f'top{k}_heads'] = round(a, 4)
    print(f'keep top-{k} heads {sorted(keep)}: {a:.4f}', flush=True)

for li in range(NL):
    a = acc(kill_mlp={li})
    res[f'kill_mlp{li}'] = round(base - a, 4)
    print(f'mean-ablate MLP {li}: -{base - a:.4f}', flush=True)

# minimal circuit: top-3 heads + all MLPs kept
a3 = acc(keep_heads=set(ranked[:3]))
res['minimal_3heads_allmlp'] = round(a3, 4)
print(f'MINIMAL (3 heads, all MLP): {a3:.4f} vs base {base:.4f}', flush=True)
# top-3 heads + drop least useful MLP if any
json.dump(res, open(f'{QK}/med_prune.json', 'w'), indent=2)
print('MED PRUNE DONE', flush=True)
