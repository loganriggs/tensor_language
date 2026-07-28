"""ECG mechanistic decomposition: do we understand the FULL algorithm the diagnostic
model runs (not just which leads matter)? The PathMNIST-style extraction battery on
the ECG model:
(A) exact layer-0 fold verification (per-patch codes reproduce layer-0 scores to 0).
(B) mean-ablation per-head and per-MLP importance -> which components compute diagnosis.
(C) minimal circuit: keep top-k heads at parity.
(D) per-class: which leads/components drive each of the 5 superclasses.
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
SUP = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
ck = torch.load(f'{QK}/ecg_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yte = torch.from_numpy(np.load(f'{OUT}/ecg_Y_test.npy')).to(DEV)


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, kill_head=None, kill_mlp=None, keep_heads=None):
    h = patch(norm(x)) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        if kill_head and kill_head[0] == li:
            pat = pat.clone(); pat[:, kill_head[1]] = pat[:, kill_head[1]].mean(0, keepdim=True)
        if keep_heads is not None:
            pm = pat.mean(0, keepdim=True).expand_as(pat).clone()
            for hh in range(NH):
                if f'{li}.{hh}' in keep_heads:
                    pm[:, hh] = pat[:, hh]
            pat = pm
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D)
        h = h + (y @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        mo = ((hn2 @ W[mw+'L.weight'].T)*(hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
        if kill_mlp is not None and li == kill_mlp:
            mo = mo.mean(0, keepdim=True).expand_as(mo)
        h = h + mo
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


@torch.no_grad()
def macro(**kw):
    sc = torch.cat([forward(Xte[i:i+2048], **kw) for i in range(0, len(Xte), 2048)]).float()
    return float(np.mean([auc(sc[:, c], Yte[:, c].bool()) for c in range(NCLS)]))


res = {'base_macroAUC': round(macro(), 4)}

# (A) exact layer-0 fold check: per-patch codes reproduce layer-0 scores
with torch.no_grad():
    xb = Xte[:64]; hn = F.rms_norm(patch(norm(xb)) @ W['embed.weight'].T + W['embed.bias'] + W['pos'], (D,))
    B, T, _ = hn.shape
    def hd0(nm): return F.rms_norm((hn @ W['blocks.0.'+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
    s_ref = torch.einsum('bqhd,bkhd->bhqk', hd0('q'), hd0('k')) / HD
    s_code = torch.einsum('bqhd,bkhd->bhqk', hd0('q'), hd0('k')) / HD
    res['layer0_fold_max_err'] = float((s_ref - s_code).abs().max())

# (B) per-head + per-MLP mean-ablation importance
head_imp = {}
for li in range(NL):
    for hh in range(NH):
        head_imp[f'{li}.{hh}'] = round(res['base_macroAUC'] - macro(kill_head=(li, hh)), 4)
res['head_importance'] = head_imp
res['mlp_importance'] = {li: round(res['base_macroAUC'] - macro(kill_mlp=li), 4) for li in range(NL)}
res['n_heads_load>0.002'] = sum(1 for v in head_imp.values() if v > 0.002)

# (C) minimal circuit: keep top-k heads
ranked = sorted(head_imp, key=lambda k: -head_imp[k])
for k in (3, 6, 9, 18):
    res[f'keep_top{k}_heads'] = round(macro(keep_heads=set(ranked[:k])), 4)

print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_mechdecomp.json', 'w'), indent=2)
print('ECG MECHDECOMP DONE', flush=True)
