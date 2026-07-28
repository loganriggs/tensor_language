"""Export the interpretable feature waveforms + code maps + causal dose-response for the
artifact (Logan: illustrate the waveform that IS each feature, and how they interact)."""
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
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); fc = ib['feat_code_auc']
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV); Xte_n = (Xte - MU) / SD
NTE = int((fold == 10).sum())


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def hn0_of(xn):
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    aw = 'blocks.0.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
    def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
    h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W['blocks.0.proj.weight'].T)
    return F.rms_norm(h, (D,))


with torch.no_grad():
    HN = torch.cat([hn0_of(Xte_n[i:i+2048]) for i in range(0, len(Xte_n), 2048)])
    act = (HN @ Ahat).pow(2)


def template(r, k=300):
    flat = act[:, :, r].reshape(-1); kk = min(k, int((flat > 0).sum()))
    topi = torch.topk(flat, kk).indices; ex = topi // NP; pos = topi % NP
    T = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()):
        T += Xte_n[e, :, p*PT:(p+1)*PT]
    return (T / kk).cpu().numpy()


inj = json.load(open(f'{QK}/ecg_input_inject.json'))
ctrl = json.load(open(f'{QK}/ecg_inject_control.json'))
# choose features to show: the 10 shared multi-code features + a few strong single-code
feats = {}
for r in range(A.shape[1]):
    served = sorted([(CODES[c], round(float(fc[r, c]), 3)) for c in range(NCLS) if fc[r, c] >= 0.72], key=lambda t: -t[1])
    if len(served) >= 2:
        feats[r] = {'leads_peak': None, 'serves': served}
# add top single-code features for codes not already covered
covered = set(cc for f in feats.values() for cc, _ in f['serves'])
for c in range(NCLS):
    if fc[:, c].max() >= 0.75:
        r = int(np.argmax(fc[:, c]))
        if r not in feats:
            feats[r] = {'serves': [(CODES[c], round(float(fc[r, c]), 3))]}
export = {'leads': LEADS, 'features': {}}
for r in feats:
    T = template(r)                                   # (12,50)
    peaks = {LEADS[L]: round(float(np.abs(T[L]).max()), 2) for L in range(NLEAD)}
    topleads = sorted(peaks, key=peaks.get, reverse=True)[:3]
    export['features'][str(r)] = {'template': [[round(float(v), 3) for v in T[L]] for L in range(NLEAD)],
                                  'top_leads': topleads, 'serves': feats[r]['serves']}
# causal dose-response for CLBBB (+ a couple more strong ones)
dose = {}
for code in ['CLBBB', 'LVH', 'ISC_', '1AVB']:
    if code in inj['per_code']:
        pc = inj['per_code'][code]
        dose[code] = {'alphas': [0.0] + inj['alphas'],
                      'target_negprob': [pc['insert_negprob_base']] + [d['target_negprob'] for d in pc['insert_dose']],
                      'top_feature': pc['top_feature']}
export['dose_response'] = dose
export['specificity'] = {k: {'real': ctrl['per_code'][k]['target_rise_real'],
                             'scrambled': ctrl['per_code'][k]['target_rise_scrambled'],
                             'rank': ctrl['per_code'][k]['specificity_rank']} for k in ctrl['per_code']}
export['headline'] = {'n_features_shown': len(export['features']),
                      'behavioral_rank': '32-64 (of 192 neurons)',
                      'model_macro_auc': 0.925}
json.dump(export, open(f'{QK}/ecg_viz_data.json', 'w'))
print(f"exported {len(export['features'])} feature waveforms, {len(dose)} dose curves", flush=True)
print('ECG EXPORT VIZ DONE', flush=True)
