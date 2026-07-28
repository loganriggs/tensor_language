"""ECG input-space causal test (Logan: insert/remove features -> predictable diagnosis
changes on the test set; correlated dx share features). Internal ablation is buffered by
redundancy (§33/§34), so intervene at the INPUT/waveform level, which bypasses it and is
physiological. For each code's top interaction feature we render the morphology TEMPLATE it
reads (mean raw waveform of top-activating patches), then on the held-out TEST set:
  INSERT: add alpha*template to every time-window of real NEGATIVE ECGs -> prob rise?
  REMOVE: subtract the template's projection from real POSITIVE ECGs -> prob drop?
  DOSE-RESPONSE: sweep alpha; monotone? CORRELATED SPILLOVER: do feature-sharing codes move together?
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
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); Rk = ib['rank']; fc = ib['feat_code_auc']
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
Xte_n = (Xte - MU) / SD
Yte = np.zeros((int((fold == 10).sum()), NCLS), dtype=np.float32)
for i, cc in enumerate(df.scp_codes.values[fold == 10]):
    for j, c in enumerate(CODES):
        if c in cc:
            Yte[i, j] = 1.0
Yte = torch.from_numpy(Yte).to(DEV); NTE = Yte.shape[0]


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward_xn(xn):                                    # takes NORMALIZED input
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
        h = h + (inner @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


@torch.no_grad()
def probs_of(xn):
    return torch.cat([torch.sigmoid(forward_xn(xn[i:i+2048])) for i in range(0, len(xn), 2048)]).float()


def auc_col(s, c):
    lab = Yte[:, c].bool(); p = lab.sum().float(); n = (~lab).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(s[:, c])).float() + 1
    return float((r[lab].sum()-p*(p+1)/2)/(p*n))


# ---- feature activations on test to render templates ----
@torch.no_grad()
def hn0_of(xn):
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    aw = 'blocks.0.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
    def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
    h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W['blocks.0.proj.weight'].T)
    return F.rms_norm(h, (D,))                          # (B,NP,D)


with torch.no_grad():
    HN = torch.cat([hn0_of(Xte_n[i:i+2048]) for i in range(0, len(Xte_n), 2048)])   # (NTE,NP,D)
    act = (HN @ Ahat).pow(2)                            # (NTE,NP,R) feature activation per patch


def template(r, k=300):
    flat = act[:, :, r].reshape(-1)
    kk = min(k, int((flat > 0).sum()))
    topi = torch.topk(flat, kk).indices; ex = topi // NP; pos = topi % NP
    T = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()):
        T += Xte_n[e, :, p*PT:(p+1)*PT]
    return T / kk                                       # (12,50) normalized morphology


base_p = probs_of(Xte_n)
base = np.array([auc_col(base_p, c) for c in range(NCLS)])
capable = [c for c in range(NCLS) if base[c] >= 0.75 and int(Yte[:, c].sum()) >= 10]
ALPHAS = [0.5, 1.0, 2.0, 4.0]

results = {}
for c in capable:
    code = CODES[c]
    r = int(np.argmax(fc[:, c]))
    T = template(r)                                     # (12,50)
    served = [c2 for c2 in capable if c2 != c and fc[r, c2] >= 0.72]   # correlated (shared-feature) codes
    negmask = ~Yte[:, c].bool(); posmask = Yte[:, c].bool()
    # INSERT into negatives: add alpha*T to every time-window
    ins = []
    for a in ALPHAS:
        xn = Xte_n.clone().reshape(NTE, NLEAD, NP, PT)
        xn = (xn + a * T[None, :, None, :]).reshape(NTE, NLEAD, NP*PT)
        p = probs_of(xn)
        ins.append({'alpha': a, 'target_negprob': round(float(p[negmask, c].mean()), 3),
                    'spillover': {CODES[c2]: round(float(p[negmask, c2].mean() - base_p[negmask, c2].mean()), 3) for c2 in served}})
    # REMOVE from positives: subtract projection of each window onto unit template
    Tn = T / T.norm().clamp_min(1e-6)
    xn = Xte_n.clone().reshape(NTE, NLEAD, NP, PT)
    coef = (xn * Tn[None, :, None, :]).sum((1, 3), keepdim=True)        # projection coeff per (ecg,window)
    xn_rm = (xn - coef * Tn[None, :, None, :]).reshape(NTE, NLEAD, NP*PT)
    p_rm = probs_of(xn_rm)
    results[code] = {
        'top_feature': r, 'auc': round(float(base[c]), 3),
        'served_codes': [CODES[c2] for c2 in served],
        'insert_negprob_base': round(float(base_p[negmask, c].mean()), 3),
        'insert_dose': ins,
        'remove_posprob_base': round(float(base_p[posmask, c].mean()), 3),
        'remove_posprob_after': round(float(p_rm[posmask, c].mean()), 3),
        'remove_drop': round(float(base_p[posmask, c].mean() - p_rm[posmask, c].mean()), 3),
    }
    dose = [x['target_negprob'] for x in ins]
    mono = all(dose[i] <= dose[i+1] for i in range(len(dose)-1))
    results[code]['insert_monotone'] = mono
    print(f'  {code}: neg {results[code]["insert_negprob_base"]} -> {dose} (mono {mono}); '
          f'remove pos {results[code]["remove_posprob_base"]}->{results[code]["remove_posprob_after"]}', flush=True)

# aggregate
rises = [results[CODES[c]]['insert_dose'][-1]['target_negprob'] - results[CODES[c]]['insert_negprob_base'] for c in capable]
drops = [results[CODES[c]]['remove_drop'] for c in capable]
monos = [results[CODES[c]]['insert_monotone'] for c in capable]
# spillover concordance: for codes with served set, did spillover move SAME direction as target at max dose?
concord = []
for c in capable:
    sp = results[CODES[c]]['insert_dose'][-1]['spillover']
    if sp:
        concord.append(float(np.mean([1.0 if v > 0.01 else 0.0 for v in sp.values()])))
res = {'n_capable': len(capable), 'alphas': ALPHAS,
       'mean_insert_rise_maxdose': round(float(np.mean(rises)), 3),
       'frac_insert_monotone': round(float(np.mean(monos)), 3),
       'mean_remove_drop': round(float(np.mean(drops)), 3),
       'codes_insert_rise>=0.1': [CODES[c] for c in capable if (results[CODES[c]]['insert_dose'][-1]['target_negprob']-results[CODES[c]]['insert_negprob_base']) >= 0.1],
       'mean_spillover_concordance': round(float(np.mean(concord)), 3) if concord else None,
       'per_code': results}
json.dump(res, open(f'{QK}/ecg_input_inject.json', 'w'), indent=2)
print(json.dumps({k: res[k] for k in ('mean_insert_rise_maxdose', 'frac_insert_monotone', 'mean_remove_drop',
      'codes_insert_rise>=0.1', 'mean_spillover_concordance')}, indent=1), flush=True)
print('ECG INPUT INJECT DONE', flush=True)
