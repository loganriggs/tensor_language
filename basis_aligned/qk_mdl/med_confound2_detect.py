"""MED PHASE 14b: confounder-detection bake-off. Marker is in patch 0 (top-left).
Each method produces a per-patch (16) importance for the confounded class CLS;
we score whether it localizes the marker (patch 0). The question: does the fold
add over standard, architecture-general methods (global color / saliency / causal
occlusion)?
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
ck = torch.load(f'{QK}/med_confound2.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, NH, HD, NL, PS, NP, INNER = (cfg[k] for k in ('D', 'NH', 'HD', 'NL', 'PS', 'NP', 'INNER'))
MEAN, STD = ck['mean'].to(DEV), ck['std'].to(DEV)
CLS, MPOS, MVAL = ck['CLS'], ck['MPOS'], ck['MVAL']
PXD = 3 * PS * PS


def rawload(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2).to(DEV) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1).to(DEV)
    return X, y


_cb = torch.zeros(7,7,device=DEV)
for _i in range(7):
    for _j in range(7):
        _cb[_i,_j] = 0.35 if (_i+_j)%2==0 else -0.35
def stamp(x):
    x = x.clone(); x[:,:,0:7,0:7] = (x[:,:,0:7,0:7]+_cb).clamp(0,1); return x


Xte, yte = rawload('test')
Xtr, ytr = rawload('train')
norm = lambda x: (x - MEAN) / STD


def patchify(x):
    B = x.shape[0]; p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


def forward(x, need_grad=False):
    h = patchify(norm(x)) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + (((hn2 @ W[mw+'L.weight'].T)*(hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


# use a batch of marker-stamped images (model predicts CLS on these)
Xs = stamp(Xte[:1024])
scores = {}

# METHOD 1: global color baseline (can it even detect CLS at all, non-spatially?)
gc_tr = norm(Xtr).mean((2, 3))                    # (N,3) global color
gc_te = norm(stamp(Xte)).mean((2, 3))
probe = nn.Linear(3, 9).to(DEV); opt = torch.optim.Adam(probe.parameters(), 1e-2)
for s in range(2000):
    bi = torch.randint(0, len(gc_tr), (2048,), device=DEV)
    l = F.cross_entropy(probe(gc_tr[bi]), ytr[bi]); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad():
    gc_cls_recall = float((probe(gc_te[yte == CLS]).argmax(1) == CLS).float().mean())
scores['global_color_CLS_recall'] = round(gc_cls_recall, 3)   # spatial? NO. detects CLS globally?

# METHOD 2: gradient saliency, per patch, for CLS logit
Xg = Xs.clone().requires_grad_(True)
out = forward(Xg)[:, CLS].sum()
g, = torch.autograd.grad(out, Xg)
sal_pix = g.abs().mean(0)                          # (3,28,28)
sal_patch = sal_pix.unfold(1, PS, PS).unfold(2, PS, PS).mean((0, 3, 4)).reshape(-1)  # (16,)
scores['saliency_patch'] = sal_patch.tolist()

# METHOD 3: causal occlusion, per patch (replace patch with dataset-mean, measure CLS drop)
with torch.no_grad():
    base = forward(Xs)[:, CLS].mean()
    meanpatch = patchify(norm(Xtr)).mean(0)        # (16, PXD) mean patch content
    occ = []
    for n in range(NP):
        Xn = norm(Xs); P = patchify(Xn); P = P.clone(); P[:, n] = meanpatch[n]
        # reconstruct image-space is awkward; instead run a patch-level forward from P
        h = P @ W['embed.weight'].T + W['embed.bias'] + W['pos']
        for li in range(NL):
            aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
            def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
            q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
            v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
            h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
            mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
            h = h + (((hn2 @ W[mw+'L.weight'].T)*(hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T)
        occ.append(float(base - (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias'])[:, CLS].mean()))
scores['occlusion_patch'] = [round(o, 3) for o in occ]

# METHOD 4 (FOLD): exact per-patch contribution to CLS logit via the folded pipeline.
# each patch's attention-out + mlp path is dense, but the EXACT per-patch texture
# feature -> CLS readout is computable. Use the explicit patch feature * head_CLS.
A = (W['embed.weight'].T @ W['blocks.1.L.weight'].T)
Bm = (W['embed.weight'].T @ W['blocks.1.R.weight'].T)
with torch.no_grad():
    Pf = patchify(norm(Xs))                        # (B,16,PXD)
    feat = (Pf @ A) * (Pf @ Bm)                    # (B,16,INNER) exact texture features per patch
    # fit exact linear readout feat(mean-pooled) -> CLS on train, then per-patch contribution
    Ptr = patchify(norm(Xtr)); ftr = ((Ptr @ A)*(Ptr @ Bm)).mean(1)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp_min(1e-6)
hd_ = nn.Linear(INNER, 9).to(DEV); opt = torch.optim.Adam(hd_.parameters(), 5e-3)
for s in range(2000):
    bi = torch.randint(0, len(ftr), (1024,), device=DEV)
    l = F.cross_entropy(hd_((ftr[bi]-mu)/sd), ytr[bi]); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad():
    wcls = hd_.weight[CLS]                          # (INNER,)
    contrib = (((feat - mu)/sd) * wcls).sum(-1).mean(0)   # (16,) per-patch CLS contribution
scores['fold_patch'] = contrib.tolist()


def loc(v):
    v = np.array(v); r = int((v > v[0]).sum()) + 1   # rank of patch 0 (1=highest)
    return {'patch0': round(float(v[0]), 3), 'rank_of_patch0': r, 'argmax': int(v.argmax())}


res = {'global_color_CLS_recall': scores['global_color_CLS_recall'],
       'saliency': loc(scores['saliency_patch']),
       'occlusion': loc(scores['occlusion_patch']),
       'fold': loc(scores['fold_patch']),
       'marker_patch': 0}
print(json.dumps(res, indent=1), flush=True)
json.dump({**res, 'raw': scores}, open(f'{QK}/med_confound2_detect.json', 'w'), indent=2)
print('MED CONFOUND2 DETECT DONE', flush=True)
