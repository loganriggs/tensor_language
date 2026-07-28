"""ECG layer-wise causal localization: where does each code's computation actually
live? Block-0 MLP ablation cost only 0.051 macro (mechdecomp), so the minimal circuit
is NOT in block-0 alone. Ablate each of the 3 MLP layers and 3 attention layers per
code -> per-code AUC drop -> dominant layer. Then per-unit ablation in ALL 3 MLP layers
-> minimal circuit in the RIGHT layer. Correlate dominant-layer depth with the
linear-baseline gap: do MORPHOLOGY codes (high gap, need nonlinearity) recruit DEEPER
layers (more composition) than AMPLITUDE codes (near-linear)?
"""
import ast, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV)
# linear-baseline gap per code (morphology vs amplitude) for correlation
lp = json.load(open(f'{QK}/ecg_linear_pooled.json'))
GAP = {}
for d in lp['biggest_gaps'] + lp['smallest_gaps']:
    GAP[d['code']] = d['gap']


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, kill_mlp=None, kill_attn=None, kill_unit=None):
    """kill_mlp/kill_attn: layer idx to ablate. kill_unit: (layer, unit) to zero."""
    xn = norm(x)
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        if kill_attn != li:
            def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
            q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
            v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
            h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        if kill_mlp != li:
            inner = (hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)
            if kill_unit is not None and kill_unit[0] == li:
                inner = inner.clone(); inner[:, :, kill_unit[1]] = 0.0
            h = h + (inner @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def all_auc(**kw):
    s = torch.cat([forward(Xte[i:i+2048], **kw) for i in range(0, len(Xte), 2048)]).float()
    R = torch.argsort(torch.argsort(s, 0), 0).float() + 1
    out = np.zeros(NCLS)
    for c in range(NCLS):
        lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
        out[c] = 0.5 if p == 0 or n == 0 else float((R[lab, c].sum()-p*(p+1)/2)/(p*n))
    return out


base = all_auc()
capable = [c for c in range(NCLS) if base[c] >= 0.75 and int(Yte[:, c].sum()) >= 10]
print(f'{len(capable)} capable codes', flush=True)

# whole-layer ablations
mlp_drop = np.stack([base - all_auc(kill_mlp=L) for L in range(NL)])      # (NL, NCLS)
attn_drop = np.stack([base - all_auc(kill_attn=L) for L in range(NL)])    # (NL, NCLS)
print('macro MLP-layer drops:', [round(float(mlp_drop[L, capable].mean()), 3) for L in range(NL)], flush=True)
print('macro Attn-layer drops:', [round(float(attn_drop[L, capable].mean()), 3) for L in range(NL)], flush=True)

# per-unit ablation in all 3 MLP layers -> (NL, INNER, NCLS)
unit_drop = np.zeros((NL, INNER, NCLS))
for L in range(NL):
    for j in range(INNER):
        unit_drop[L, j] = base - all_auc(kill_unit=(L, j))
    print(f'  unit ablation layer {L} done', flush=True)
np.save(f'{QK}/ecg_unit_drop_alllayers.npy', unit_drop)

# per code: dominant MLP layer, minimal circuit in that layer, depth vs linear-gap
per_code = {}
depth_gap = []
for c in capable:
    code = CODES[c]
    ld = mlp_drop[:, c]
    dom = int(np.argmax(ld))
    ud = unit_drop[dom, :, c]
    circ = [int(j) for j in np.argsort(-ud) if ud[j] > 0.003][:12]
    # depth centroid: weighted mean layer by positive drop (0..2)
    w = np.clip(ld, 0, None)
    depth = float((w * np.arange(NL)).sum() / w.sum()) if w.sum() > 0 else 0.0
    per_code[code] = {'auc': round(float(base[c]), 3),
                      'mlp_layer_drops': [round(float(mlp_drop[L, c]), 3) for L in range(NL)],
                      'attn_layer_drops': [round(float(attn_drop[L, c]), 3) for L in range(NL)],
                      'dominant_mlp_layer': dom, 'depth_centroid': round(depth, 2),
                      'circuit_units_dom_layer': circ, 'n_units': len(circ),
                      'linear_gap': GAP.get(code)}
    if code in GAP:
        depth_gap.append((depth, GAP[code], code))

# correlation: depth centroid vs linear gap (morphology should be deeper)
if len(depth_gap) >= 4:
    dd = np.array([x[0] for x in depth_gap]); gg = np.array([x[1] for x in depth_gap])
    r = float(np.corrcoef(dd, gg)[0, 1])
else:
    r = None

res = {'n_capable': len(capable),
       'macro_mlp_layer_drops': [round(float(mlp_drop[L, capable].mean()), 3) for L in range(NL)],
       'macro_attn_layer_drops': [round(float(attn_drop[L, capable].mean()), 3) for L in range(NL)],
       'dominant_layer_histogram': {int(L): int(sum(per_code[CODES[c]]['dominant_mlp_layer'] == L for c in capable)) for L in range(NL)},
       'depth_vs_lineargap_corr': None if r is None else round(r, 3),
       'depth_gap_points': [{'code': c, 'depth': round(d, 2), 'linear_gap': g} for d, g, c in sorted(depth_gap)],
       'per_code': per_code}
json.dump(res, open(f'{QK}/ecg_layer_circuit.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('macro_mlp_layer_drops', 'macro_attn_layer_drops',
      'dominant_layer_histogram', 'depth_vs_lineargap_corr')}, indent=1), flush=True)
print('depth vs gap (morphology should be deeper):', json.dumps(res['depth_gap_points'], indent=1), flush=True)
print('ECG LAYER CIRCUIT DONE', flush=True)
