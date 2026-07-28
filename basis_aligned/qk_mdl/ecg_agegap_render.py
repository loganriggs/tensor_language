"""Tier-2 visual payoff: render WHAT a prematurely-old ECG looks like. Within matched true-age
bands, contrast the median beat of ECGs the foldable age-gap student reads as prematurely-OLD
(high predicted gap) vs normally-aged (low gap). The difference morphology = the mortality-linked
'premature aging' waveform signature (age controlled, so it's not just 'old'). Save for a figure.
"""
import ast, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ck = torch.load(f'{QK}/ecg_agegap_student_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; gmu, gsd = ck['gmu'], ck['gsd']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
Xte = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
MUn, SDn = MU.cpu().numpy(), SD.cpu().numpy()


def patch(xn):
    B = xn.shape[0]; return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def gap_pred(x):
    xn = (x - MU) / SD
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    for li in range(NL):
        aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
        def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
        q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2'); v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
        pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
        h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
        mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
        h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
    return (F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).squeeze(1) * gsd + gmu


GAP = torch.cat([gap_pred(Xte[i:i+2048]) for i in range(0, len(Xte), 2048)]).cpu().numpy()
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values; true_age = df['age'].values.astype(np.float32)[fold == 10]

Xn = ((Xte - MU) / SD).cpu().numpy()
def med_beat(i):
    sig = Xn[i, 1]; pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    beats = [Xn[i, :, p-50:p+50] for p in pk if p-50 >= 0 and p+50 < 1000]
    return np.median(np.stack(beats), 0) if beats else Xn[i, :, 450:550]
BEATS = np.stack([med_beat(i) for i in range(len(Xn))])   # (N,12,100)

# within age band 50-75, high-gap (top 25%) vs low-gap (bottom 25%)
band = (true_age >= 50) & (true_age <= 75)
gb = GAP[band]; bb = BEATS[band]; ab = true_age[band]
hi = gb >= np.quantile(gb, 0.75); lo = gb <= np.quantile(gb, 0.25)
old_beat = np.median(bb[hi], 0); young_beat = np.median(bb[lo], 0)    # (12,100)
diff = old_beat - young_beat
per_lead_rms = {LEADS[L]: round(float(np.sqrt((diff[L]**2).mean())), 3) for L in range(NLEAD)}
res = {'band': '50-75y', 'n_high_gap': int(hi.sum()), 'n_low_gap': int(lo.sum()),
       'mean_gap_high': round(float(gb[hi].mean()), 2), 'mean_gap_low': round(float(gb[lo].mean()), 2),
       'mean_true_age_high': round(float(ab[hi].mean()), 1), 'mean_true_age_low': round(float(ab[lo].mean()), 1),
       'per_lead_diff_rms': dict(sorted(per_lead_rms.items(), key=lambda kv: -kv[1]))}
# export beats for a figure (unnormalize back to mV for display: *SD + MU won't apply per-sample; keep normalized)
np.savez(f'{QK}/ecg_agegap_render.npz', old_beat=old_beat, young_beat=young_beat, diff=diff, leads=np.array(LEADS))
json.dump(res, open(f'{QK}/ecg_agegap_render.json', 'w'), indent=2)
print(json.dumps(res, indent=1), flush=True)
print('ECG AGEGAP RENDER DONE', flush=True)
