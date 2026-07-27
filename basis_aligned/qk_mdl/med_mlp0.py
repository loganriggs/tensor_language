"""MED PHASE 4: dissect MLP-0 (the classifier engine) into visual filters.

(A) Patch-local hypothesis: layer-0 attention is nearly content-free (-0.018), so
    MLP-0's input approximates rms_norm(patch_embedding) per patch. Test: run the
    model with layer-0 attention removed entirely (attn0 output zeroed) — if
    accuracy barely drops, MLP-0 acts patch-wise on raw pixels.
(B) Inner-unit pruning: mean-ablate each of the 192 bilinear units; keep top-k sweep.
(C) Pixel-space filters: each unit j computes (L_j.h)(R_j.h). With h ~ embed =
    We@patch, this is patch^T (a_j b_j^T) patch, a_j = We^T L_j, b_j = We^T R_j.
    The top eigenvector of the symmetric part is the unit's preferred 7x7x3 pixel
    pattern. Save top units' filters for the visualization artifact.
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
def forward(x, kill_attn0=False, keep_units=None):
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
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D)
        yo = y @ W[aw + 'proj.weight'].T
        if kill_attn0 and li == 0:
            yo = torch.zeros_like(yo)
        h = h + yo
        mw = f'blocks.{2*li+1}.'
        hn2 = F.rms_norm(h, (D,))
        inner = (hn2 @ W[mw + 'L.weight'].T) * (hn2 @ W[mw + 'R.weight'].T)
        if keep_units is not None and li == 0:
            mask = torch.zeros(INNER, device=DEV)
            mask[list(keep_units)] = 1.0
            inner = inner * mask
        h = h + (inner @ W[mw + 'Dn.weight'].T)
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
res['kill_attn0'] = round(acc(kill_attn0=True), 4)
print(f'base {base:.4f}  attn0-removed {res["kill_attn0"]:.4f}', flush=True)

# unit importance: mean-ablate each unit (set its inner activation to batch mean)
# approx via zeroing contribution and measuring; cheaper: rank by output-weight norm x activation std
with torch.no_grad():
    Xs = Xte[:2048]
    h = embed(Xs)
    aw = 'blocks.0.'
    hn = F.rms_norm(h, (D,))
    B, T, _ = hn.shape
    def hd(nm):
        return F.rms_norm((hn @ W[aw + nm + '.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    v = (hn @ W[aw + 'v.weight'].T).view(B, T, NH, HD)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q, k) / HD) * (torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD)
    yo = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw + 'proj.weight'].T
    h1 = h + yo
    mw = 'blocks.1.'
    hn2 = F.rms_norm(h1, (D,))
    act = (hn2 @ W[mw + 'L.weight'].T) * (hn2 @ W[mw + 'R.weight'].T)   # (B,T,INNER)
    dn_norm = W[mw + 'Dn.weight'].norm(dim=0)                          # (INNER,)
    importance = (act.abs().mean((0, 1)) * dn_norm).cpu().numpy()
order = list(np.argsort(-importance))
for k_ in (8, 16, 32, 64, 128, 192):
    a = acc(keep_units=set(order[:k_]))
    res[f'top{k_}_units'] = round(a, 4)
    print(f'keep top-{k_} MLP-0 units: {a:.4f}', flush=True)

# pixel-space filters for top-32 units
We = W['embed.weight']                       # (D, 147)
Lw = W['blocks.1.L.weight']                  # (INNER, D)
Rw = W['blocks.1.R.weight']
filters = []
for j in order[:32]:
    a_j = We.T @ Lw[j]                       # (147,)
    b_j = We.T @ Rw[j]
    S = 0.5 * (torch.outer(a_j, b_j) + torch.outer(b_j, a_j))
    ev, evec = torch.linalg.eigh(S)
    top = evec[:, ev.abs().argmax()]
    filters.append((top * torch.sign(ev[ev.abs().argmax()])).reshape(3, PS, PS).cpu())
torch.save({'filters': torch.stack(filters), 'units': [int(o) for o in order[:32]],
            'importance': importance[order[:32]].tolist()}, f'{QK}/med_mlp0_filters.pt')
res['n_filters_saved'] = 32
json.dump(res, open(f'{QK}/med_mlp0.json', 'w'), indent=2)
print('MED MLP0 DONE', flush=True)
