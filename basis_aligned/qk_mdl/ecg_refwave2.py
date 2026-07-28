"""ECG reference waveforms — EXPANDED diagnoses + TEMPLATE-MATCH BASELINE (Logan). Three
levels compared cross-cohort (Germany -> US Georgia), no refitting on US:
  BASELINE  = template-match: build the diagnostic template (aligned median beat,
              positive-minus-normal) on Germany, classify US ECGs by best-shift cosine of
              their median beat to it. NO model. This is 'just the average waveform'.
  FEATURE   = our one interaction feature's activation (a direction in the Germany model).
  MODEL     = the full model's per-code logit (ceiling).
Diagnoses span conduction (LBBB/RBBB/LAFB/1AVB), amplitude (LVH), morphology (ST-elevation
injury, ST-depression & T-inversion ischemia, Q-wave/MI, T-abnormal). Also exports the real
US median beats for the atlas.
"""
import glob, json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MAP = {
 'CLBBB': ('164909002', 'Complete LBBB', 'conduction'), 'CRBBB': ('713427006', 'Complete RBBB', 'conduction'),
 'LAFB':  ('445118002', 'Left ant. fascicular block', 'conduction'), '1AVB': ('270492004', 'First-degree AV block', 'conduction'),
 'LVH':   ('164873001', 'LV hypertrophy', 'amplitude'),
 'INJAS': ('164931005', 'ST elevation (injury)', 'morphology'), 'ISC_': ('429622005', 'ST depression (ischemia)', 'morphology'),
 'ISCIN': ('59931005',  'T inversion (ischemia)', 'morphology'), 'NDT': ('164934002', 'T-wave abnormal', 'morphology'),
 'AMI':   ('164917005', 'Q-wave abnormal (prior MI)', 'morphology'),
}
NORM_SN = '426783006'
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']; NCLS = len(CODES)
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
ib = torch.load(f'{QK}/ecg_interaction_basis.pt', map_location=DEV, weights_only=False)
A = ib['A'].to(DEV); R = ib['rank']; fc = ib['feat_code_auc']
Ahat = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
MUc = MU.cpu().numpy()[0]; SDc = SD.cpu().numpy()[0]


def patch(xn):
    B = xn.shape[0]
    return xn.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def hn0(xn):
    h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
    aw = 'blocks.0.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
    def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
    q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
    v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
    h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W['blocks.0.proj.weight'].T)
    return F.rms_norm(h, (D,))


@torch.no_grad()
def full_logits(X):
    out = []
    for i in range(0, len(X), 2048):
        xn = (X[i:i+2048] - MU) / SD
        h = patch(xn) @ W['embed.weight'].T + W['embed.bias'] + W['pos']
        for li in range(NL):
            aw = f'blocks.{2*li}.'; hn = F.rms_norm(h, (D,)); B, T, _ = hn.shape
            def hd(nm): return F.rms_norm((hn @ W[aw+nm+'.weight'].T).view(B, T, NH, HD), (HD,))
            q, k, q2, k2 = hd('q'), hd('k'), hd('q2'), hd('k2')
            v = (hn @ W[aw+'v.weight'].T).view(B, T, NH, HD)
            pat = (torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)
            h = h + (torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D) @ W[aw+'proj.weight'].T)
            mw = f'blocks.{2*li+1}.'; hn2 = F.rms_norm(h, (D,))
            h = h + ((hn2 @ W[mw+'L.weight'].T) * (hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T
        out.append((F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']).float())
    return torch.cat(out)


def feats_all(X):
    out = []
    for i in range(0, len(X), 2048):
        out.append((hn0((X[i:i+2048]-MU)/SD) @ Ahat).pow(2).mean(1))
    return torch.cat(out)


def auc_np(score, lab):
    lab = lab.astype(bool); p = lab.sum(); n = (~lab).sum()
    if p == 0 or n == 0: return 0.5
    order = np.argsort(np.argsort(score)) + 1
    return float((order[lab].sum() - p*(p+1)/2) / (p*n))


def med_beat_one(x, win=100, detlead=1):
    """x: (12,1000) normalized. Return (12,win) R-aligned median beat, or None."""
    sig = x[detlead]
    pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    beats = [x[:, p-win//2:p+win//2] for p in pk if p-win//2 >= 0 and p+win//2 < 1000]
    if len(beats) < 1:
        return x[:, 450:550]
    return np.median(np.stack(beats), 0)


def cohort_medbeats(X):
    Xn = ((X - MU) / SD).cpu().numpy()
    return np.stack([med_beat_one(Xn[i]) for i in range(len(Xn))])   # (N,12,100)


# Germany feature templates + test beats
Xpt = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV); Xpt_n = (Xpt - MU) / SD
HN = torch.cat([hn0(Xpt_n[i:i+2048]) for i in range(0, len(Xpt_n), 2048)])
ACT = (HN @ Ahat).pow(2)
def template(r, k=300):
    flat = ACT[:, :, r].reshape(-1); kk = min(k, int((flat > 0).sum()))
    topi = torch.topk(flat, kk).indices; ex = topi // NP; pos = topi % NP
    T = torch.zeros(NLEAD, PT, device=DEV)
    for e, p in zip(ex.tolist(), pos.tolist()): T += Xpt_n[e, :, p*PT:(p+1)*PT]
    return (T / kk).cpu().numpy()
print('extracting Germany-test median beats...', flush=True)
PTbeats = cohort_medbeats(Xpt)                                       # (Npt,12,100)

# PTB-XL test labels
import ast, pandas as pd
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
te_codes = df.scp_codes.values[fold == 10]
def ptlabel(code): return np.array([1.0 if code in cc else 0.0 for cc in te_codes])

# Georgia
recs = sorted(glob.glob(f'{OUT}/georgia/*/*.hea'))
dx = []
for hp in recs:
    codes = ''
    for line in open(hp):
        if line.lower().replace(' ', '').startswith('#dx'): codes = line.split(':', 1)[1]
    dx.append(codes)
Xg = torch.from_numpy(np.load(f'{OUT}/georgia_X.npy')).to(DEV)
assert len(dx) == len(Xg)
def glabel(sn): return np.array([1.0 if sn in d else 0.0 for d in dx])
print('extracting US (Georgia) median beats...', flush=True)
GAbeats = cohort_medbeats(Xg)
Fg = feats_all(Xg).cpu().numpy()
Lg = full_logits(Xg).cpu().numpy()
gNORM = glabel(NORM_SN)


def tmatch_scores(beats, T_diag, topL):
    """best-shift cosine of each beat (diagnostic leads) to template T_diag (12,win)."""
    B = beats[:, topL, :]                                             # (N,k,100)
    Td = T_diag[topL]                                                 # (k,100)
    N, k, wl = B.shape
    Bc = B - B.mean(2, keepdims=True); Tc = Td - Td.mean(1, keepdims=True)
    scores = np.full(N, -1.0)
    for off in range(-8, 9):                                          # small shift tolerance
        Ts = np.roll(Tc, off, axis=1)
        num = (Bc * Ts[None]).sum((1, 2))
        den = np.sqrt((Bc**2).sum((1, 2)) * (Ts**2).sum() + 1e-9)
        scores = np.maximum(scores, num/den)
    return scores


rows = {}; beats_out = {}
for code, (sn, human, cat) in MAP.items():
    if code not in CODES: continue
    c = CODES.index(code); r = int(np.argmax(fc[:, c]))
    gl = glabel(sn); npos = int(gl.sum())
    if npos < 20:
        continue
    # diagnostic template from GERMANY test: positive-minus-normal median beat
    ptl = ptlabel(code)
    ptnorm = ptlabel('NORM')
    if ptl.sum() >= 10:
        T_diag = np.median(PTbeats[ptl.astype(bool)], 0) - np.median(PTbeats[ptnorm.astype(bool)], 0)
    else:
        T_diag = None
    Tfeat = template(r)
    u = np.abs(Tfeat).max(1); topL = list(np.argsort(-u)[:4])
    # three cross-cohort AUCs on US
    feat_auc = auc_np(Fg[:, r], gl)
    model_auc = auc_np(Lg[:, c], gl)
    tm_auc = auc_np(tmatch_scores(GAbeats, T_diag, topL), gl) if T_diag is not None else None
    # US real median beat + cosine of our feature to it
    usbeat = np.median(GAbeats[gl.astype(bool)], 0)
    ub = usbeat[topL]; tf = Tfeat[topL]
    best = -1
    for off in range(0, 100-PT+1):
        wf = ub[:, off:off+PT].flatten(); a = tf.flatten()-tf.mean(); b = wf-wf.mean()
        best = max(best, float(abs((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))))
    rows[code] = {'human': human, 'category': cat, 'feature': r, 'us_records': npos,
                  'germany_feature_auc': round(float(fc[r, c]), 3),
                  'us_template_match_auc': None if tm_auc is None else round(tm_auc, 3),
                  'us_feature_auc': round(feat_auc, 3), 'us_model_auc': round(model_auc, 3),
                  'feature_vs_template': None if tm_auc is None else round(feat_auc - tm_auc, 3),
                  'us_beat_shape_cosine': round(best, 3), 'top_leads': [LEADS[L] for L in topL]}
    beats_out[code] = {'human': human, 'category': cat, 'us_beat': [[round(float(v), 3) for v in usbeat[L]] for L in range(NLEAD)],
                       'germany_feature_template': [[round(float(v), 3) for v in Tfeat[L]] for L in range(NLEAD)],
                       'top_leads': [LEADS[L] for L in topL], 'shape_cosine': round(best, 3),
                       'us_template_match_auc': None if tm_auc is None else round(tm_auc, 3),
                       'us_feature_auc': round(feat_auc, 3), 'us_model_auc': round(model_auc, 3)}
    print(f'  {code:6s}[{cat[:4]}] {human[:26]:26s} US n={npos:4d} | tmatch {tm_auc if tm_auc is None else round(tm_auc,3)} '
          f'| feature {feat_auc:.3f} | model {model_auc:.3f}', flush=True)

cats = {}
for cat in ('conduction', 'amplitude', 'morphology'):
    rr = [v for v in rows.values() if v['category'] == cat and v['us_template_match_auc'] is not None]
    if rr:
        cats[cat] = {'mean_tmatch': round(float(np.mean([v['us_template_match_auc'] for v in rr])), 3),
                     'mean_feature': round(float(np.mean([v['us_feature_auc'] for v in rr])), 3),
                     'mean_model': round(float(np.mean([v['us_model_auc'] for v in rr])), 3)}
summary = {'n_diagnoses': len(rows), 'by_category': cats,
           'mean_feature_minus_template': round(float(np.mean([v['feature_vs_template'] for v in rows.values() if v['feature_vs_template'] is not None])), 3)}
json.dump({'leads': LEADS, 'rows': rows, 'beats': beats_out}, open(f'{QK}/ecg_refwave2_data.json', 'w'))
json.dump(summary, open(f'{QK}/ecg_refwave2.json', 'w'), indent=2)
print(json.dumps(summary, indent=1), flush=True)
print('ECG REFWAVE2 DONE', flush=True)
