"""ECG per-code circuits: for each CAPABLE diagnostic code (AUC>=0.75), find its
minimal circuit -- which block-0 MLP units and which leads causally compute it -- and
measure shared vs code-specific structure. Validate top leads against known cardiology.
"""
import sys, json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# known diagnostic-lead expectations for validation (code -> set of key leads)
KNOWN = {'CRBBB': {'V1', 'V2'}, 'CLBBB': {'V1', 'V6', 'I'}, 'IRBBB': {'V1', 'V2'},
         'LAFB': {'I', 'aVL', 'III', 'aVF'}, 'LPFB': {'III', 'aVF', 'I'},
         'INJAS': {'V1', 'V2', 'V3'}, 'INJAL': {'I', 'aVL', 'V5', 'V6'},
         'IMI': {'II', 'III', 'aVF'}, 'AMI': {'V1', 'V2', 'V3', 'V4'},
         'ISCAN': {'V1', 'V2', 'V3', 'V4'}, 'ISCIN': {'II', 'III', 'aVF'}}
ck = torch.load(f'{QK}/ecg_codes_big.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD

import ast, pandas as pd
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
te_codes = df.scp_codes.values[fold == 10]
for i, cc in enumerate(te_codes):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV)


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, kill_unit=None, occ_lead=None):
    xn = norm(x)
    if occ_lead is not None:
        xn = xn.clone(); xn[:, occ_lead] = 0.0
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
        v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        inner = (hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)
        if kill_unit is not None and li == 0:
            inner = inner.clone(); inner[:, :, kill_unit] = 0.0
        h = h + (inner @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def all_auc(kill_unit=None, occ_lead=None):
    s = torch.cat([forward(Xte[i:i+2048], kill_unit, occ_lead) for i in range(0, len(Xte), 2048)]).float()
    R = torch.argsort(torch.argsort(s, 0), 0).float() + 1
    out = []
    for c in range(NCLS):
        lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
        out.append(0.5 if p == 0 or n == 0 else float((R[lab, c].sum()-p*(p+1)/2)/(p*n)))
    return np.array(out)


base = all_auc()
capable = [c for c in range(NCLS) if base[c] >= 0.75 and int(Yte[:, c].sum()) >= 10]
print(f'{len(capable)} capable codes; computing unit + lead importance...', flush=True)

# per-unit ablation -> (INNER, NCLS) AUC-drop
unit_drop = np.zeros((INNER, NCLS))
for j in range(INNER):
    unit_drop[j] = base - all_auc(kill_unit=j)
    if j % 48 == 0:
        print(f'  unit {j}/{INNER}', flush=True)
# per-lead ablation -> (12, NCLS)
lead_drop = np.zeros((NLEAD, NCLS))
for L in range(NLEAD):
    lead_drop[L] = base - all_auc(occ_lead=L)

# per-code circuit + feature
per_code = {}
phys_hits = 0; phys_total = 0
for c in capable:
    code = CODES[c]
    ud = unit_drop[:, c]; ld = lead_drop[:, c]
    circ = [int(j) for j in np.argsort(-ud) if ud[j] > 0.003][:12]
    top_leads = [LEADS[i] for i in np.argsort(-ld)[:3]]
    entry = {'auc': round(float(base[c]), 3), 'test_pos': int(Yte[:, c].sum()),
             'circuit_units': circ, 'n_units': len(circ), 'top_leads': top_leads}
    if code in KNOWN:
        hit = len(set(top_leads) & KNOWN[code]) > 0
        entry['physiology_match'] = hit; entry['expected_leads'] = sorted(KNOWN[code])
        phys_hits += hit; phys_total += 1
    per_code[code] = entry

# shared vs code-specific: unit generality (how many codes each unit serves) + circuit overlap
sig = (unit_drop[:, capable] > 0.003)               # (INNER, n_capable) boolean
unit_generality = sig.sum(1)                          # how many codes each unit serves
res = {'n_capable': len(capable), 'per_code': per_code,
       'physiology_top_lead_match': f'{phys_hits}/{phys_total}',
       'mean_circuit_size': round(float(np.mean([per_code[CODES[c]]['n_units'] for c in capable])), 1),
       'generalist_units_serve>=5_codes': int((unit_generality >= 5).sum()),
       'specialist_units_serve_1_code': int((unit_generality == 1).sum()),
       'units_used_by_any_code': int((unit_generality >= 1).sum()),
       'max_code_overlap': None}
# mean pairwise circuit Jaccard across codes (shared structure)
circs = [set(per_code[CODES[c]]['circuit_units']) for c in capable]
jac = []
for i in range(len(circs)):
    for k in range(i+1, len(circs)):
        u = circs[i] | circs[k]
        if u:
            jac.append(len(circs[i] & circs[k]) / len(u))
res['mean_pairwise_circuit_jaccard'] = round(float(np.mean(jac)), 3)
np.save(f'{QK}/ecg_unit_drop_big.npy', unit_drop)
print(json.dumps({k: res[k] for k in ('n_capable', 'physiology_top_lead_match', 'mean_circuit_size',
      'generalist_units_serve>=5_codes', 'specialist_units_serve_1_code', 'units_used_by_any_code',
      'mean_pairwise_circuit_jaccard')}, indent=1), flush=True)
print('sample codes:', json.dumps({CODES[c]: per_code[CODES[c]] for c in capable[:5]}, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_code_circuits_big.json', 'w'), indent=2)
print('ECG CODE CIRCUITS BIG DONE', flush=True)
