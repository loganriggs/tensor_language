"""MED PHASE 12: does the extracted color-reliance predict stain-shift fragility,
and does it give an actionable robustness lever? (Logan's "actually useful" test.)

Simulated H&E stain shift = per-channel affine on raw pixels (gain+bias, the
dominant stain-intensity/balance nuisance; a proxy, not full color deconvolution).
Fragility curves (accuracy vs shift) for: bilinear ViT, softmax ViT, the extracted
explicit texture pipeline WITH color, and a COLOR-REMOVED texture pipeline
(per-patch DC subtracted). Plus a per-tile test: does color-reliance predict which
tiles flip under shift?
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


def raw(split):
    d = PathMNIST(split=split, root=ROOT)
    X = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2).to(DEV) / 255.0
    y = torch.from_numpy(d.labels).long().squeeze(1).to(DEV)
    return X, y


Xtr_raw, ytr = raw('train')
Xte_raw, yte = raw('test')
ckb = torch.load(f'{QK}/med_bvit2.pt', map_location=DEV)
cks = torch.load(f'{QK}/med_softmax.pt', map_location=DEV)
cfg = ckb['cfg']
D, NH, HD, NL, PS, NP, INNER = (cfg[k] for k in ('D', 'NH', 'HD', 'NL', 'PS', 'NP', 'INNER'))
MEAN, STD = ckb['mean'].to(DEV), ckb['std'].to(DEV)
PXD = 3 * PS * PS


def patchify(x):
    B = x.shape[0]
    p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


def stain(x, eps, g):
    gain = 1 + eps * torch.randn(len(x), 3, 1, 1, device=DEV, generator=g)
    bias = eps * 0.5 * torch.randn(len(x), 3, 1, 1, device=DEV, generator=g)
    return (x * gain + bias).clamp(0, 1)


@torch.no_grad()
def bilinear_fwd(xraw):
    W = ckb['state']
    h = patchify((xraw - MEAN) / STD) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + (((hn2 @ W[mw+'L.weight'].T)*(hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T)
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias'])


@torch.no_grad()
def softmax_fwd(xraw):
    W = cks['state']
    h = patchify((xraw - MEAN) / STD) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        q = (hn @ W[aw+'q.weight'].T).view(B, T, NH, HD)
        k = (hn @ W[aw+'k.weight'].T).view(B, T, NH, HD)
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).softmax(-1)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + (F.gelu(hn2 @ W[mw+'L.weight'].T) @ W[mw+'Dn.weight'].T)
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias'])


# explicit texture pipelines (with and without absolute color)
W = ckb['state']
A = (ckb['state']['embed.weight'].T @ W['blocks.1.L.weight'].T).contiguous()
Bm = (ckb['state']['embed.weight'].T @ W['blocks.1.R.weight'].T).contiguous()


def tex_feats(xraw, remove_color):
    P = patchify((xraw - MEAN) / STD)
    if remove_color:
        P = P - P.mean(2, keepdim=True)          # subtract per-patch DC (absolute color)
    return ((P @ A) * (P @ Bm)).mean(1)


def fit_head(feat_fn):
    Ftr = feat_fn(Xtr_raw)
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True).clamp_min(1e-6)
    head = nn.Linear(INNER, 9).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-3)
    for s in range(3000):
        bi = torch.randint(0, len(Ftr), (1024,), device=DEV)
        loss = F.cross_entropy(head((Ftr[bi]-mu)/sd), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda xraw: head((feat_fn(xraw)-mu)/sd)


tex_color = fit_head(lambda x: tex_feats(x, False))
tex_nocol = fit_head(lambda x: tex_feats(x, True))


@torch.no_grad()
def acc(fn, xraw):
    out = []
    for i in range(0, len(xraw), 8192):
        out.append(fn(xraw[i:i+8192]).argmax(1))
    return torch.cat(out)


models = {'bilinear': bilinear_fwd, 'softmax': softmax_fwd,
          'explicit_color': tex_color, 'explicit_nocolor': tex_nocol}
EPS = [0.0, 0.05, 0.1, 0.2, 0.3]
g = torch.Generator(device=DEV).manual_seed(42)
curves = {m: [] for m in models}
for eps in EPS:
    for m, fn in models.items():
        accs = []
        for _ in range(4 if eps > 0 else 1):
            xs = stain(Xte_raw, eps, g) if eps > 0 else Xte_raw
            accs.append(float((acc(fn, xs) == yte).float().mean()))
        curves[m].append(round(float(np.mean(accs)), 4))
    print(f'eps {eps}: ' + '  '.join(f'{m} {curves[m][-1]:.3f}' for m in models), flush=True)

# per-tile: does color-reliance predict failure under shift?
with torch.no_grad():
    clean_pred = acc(bilinear_fwd, Xte_raw)
    xs = stain(Xte_raw, 0.2, g)
    shift_pred = acc(bilinear_fwd, xs)
    flipped = (clean_pred != shift_pred)
    # color-reliance proxy: prediction changes when color is removed (texture-only disagrees)
    tex_pred = acc(tex_nocol, Xte_raw)
    color_reliant = (tex_pred != clean_pred)
    # among correct-clean tiles, do color-reliant ones flip more under shift?
    correct = (clean_pred == yte)
    cr = color_reliant & correct
    ncr = (~color_reliant) & correct
    res = {
        'fragility_curves': {m: curves[m] for m in models}, 'eps': EPS,
        'softmax_val_test_gap': [0.966, 0.830], 'bilinear_val_test_gap': [0.967, 0.857],
        'flip_rate_color_reliant': round(float(flipped[cr].float().mean()), 3),
        'flip_rate_non_color_reliant': round(float(flipped[ncr].float().mean()), 3),
        'n_color_reliant': int(cr.sum()), 'n_non': int(ncr.sum()),
    }
print(json.dumps({k: res[k] for k in ('flip_rate_color_reliant', 'flip_rate_non_color_reliant')},
                 indent=1), flush=True)
json.dump(res, open(f'{QK}/med_stain.json', 'w'), indent=2)
print('MED STAIN DONE', flush=True)
