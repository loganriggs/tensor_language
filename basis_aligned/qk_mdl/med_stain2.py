"""MED PHASE 13b: fragility curve of the stain-invariant model vs the original.
Confirms whether per-image color standardization (the mechanism-correct fix)
flattens the stain-shift curve."""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from medmnist import PathMNIST

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
ROOT = '/workspace/tensor_language/medmnist_data'
d = PathMNIST(split='test', root=ROOT)
Xte = torch.from_numpy(d.imgs).float().permute(0, 3, 1, 2).to(DEV) / 255.0
yte = torch.from_numpy(d.labels).long().squeeze(1).to(DEV)
ckb = torch.load(f'{QK}/med_bvit2.pt', map_location=DEV)
cki = torch.load(f'{QK}/med_staininv.pt', map_location=DEV)
cfg = ckb['cfg']
D, NH, HD, NL, PS, NP, INNER = (cfg[k] for k in ('D', 'NH', 'HD', 'NL', 'PS', 'NP', 'INNER'))
MEAN, STD = ckb['mean'].to(DEV), ckb['std'].to(DEV)
PXD = 3 * PS * PS


def patchify(x):
    B = x.shape[0]
    p = x.unfold(2, PS, PS).unfold(3, PS, PS)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, NP, PXD)


def perimg(x):
    m = x.mean((2, 3), keepdim=True); s = x.std((2, 3), keepdim=True).clamp_min(1e-4)
    return (x - m) / s


def stain(x, eps, g):
    gain = 1 + eps * torch.randn(len(x), 3, 1, 1, device=DEV, generator=g)
    bias = eps * 0.5 * torch.randn(len(x), 3, 1, 1, device=DEV, generator=g)
    return (x * gain + bias).clamp(0, 1)


@torch.no_grad()
def fwd(ck, xraw, norm):
    W = ck['state']
    xn = norm(xraw)
    h = patchify(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
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


dsnorm = lambda x: (x - MEAN) / STD
g = torch.Generator(device=DEV).manual_seed(42)
EPS = [0.0, 0.05, 0.1, 0.2, 0.3]
curves = {'original': [], 'stain_invariant': []}
for eps in EPS:
    for name, ck, norm in (('original', ckb, dsnorm), ('stain_invariant', cki, perimg)):
        accs = []
        for _ in range(4 if eps > 0 else 1):
            xs = stain(Xte, eps, g) if eps > 0 else Xte
            with torch.no_grad():
                pred = torch.cat([fwd(ck, xs[i:i+8192], norm).argmax(1) for i in range(0, len(xs), 8192)])
            accs.append(float((pred == yte).float().mean()))
        curves[name].append(round(float(np.mean(accs)), 4))
    print(f'eps {eps}: original {curves["original"][-1]:.3f}  stain_invariant {curves["stain_invariant"][-1]:.3f}', flush=True)
res = {'eps': EPS, 'curves': curves,
       'retention_at_0.1': {k: round(curves[k][2] / curves[k][0], 3) for k in curves}}
print(json.dumps(res['retention_at_0.1'], indent=1), flush=True)
json.dump(res, open(f'{QK}/med_stain2.json', 'w'), indent=2)
print('MED STAIN2 DONE', flush=True)
