"""ECG clinical-criterion baselines (Logan): the honest per-diagnosis baseline is the clinical
MEASUREMENT that defines each diagnosis. Extract hand-crafted clinical features from the aligned
median beat -- Sokolow-Lyon & Cornell VOLTAGE (LVH), QRS WIDTH (BBB), PR-interval proxy (AV
block), ST elevation/depression (injury/ischemia), T amplitude, frontal AXIS -- fit a logistic
on them, and compare per-diagnosis AUC to the atomic basis, the full model, and the shape template.
Bar: does the deep model beat the criterion a cardiologist already uses?
"""
import ast, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
L = {n: i for i, n in enumerate(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])}
WIN = 100
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
CODES = ck['codes']; NCLS = len(CODES); MU = ck['MU'].cpu().numpy(); SD = ck['SD'].cpu().numpy()
model_auc = {d['code']: d['auc'] for d in json.load(open(f'{QK}/ecg_codes_train.json'))['per_code']}
GA_MAP = {'CLBBB': '164909002', 'CRBBB': '713427006', 'LAFB': '445118002', '1AVB': '270492004',
          'LVH': '164873001', 'INJAS': '164931005', 'ISC_': '429622005', 'ISCIN': '59931005',
          'NDT': '164934002', 'AMI': '164917005'}


def med_beat(x, detlead=1):
    sig = x[detlead]
    pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    beats = [x[:, p-WIN//2:p+WIN//2] for p in pk if p-WIN//2 >= 0 and p+WIN//2 < 1000]
    return np.median(np.stack(beats), 0) if beats else x[:, 450:550]


def clin_feats(b):
    """b: (12,100) normalized median beat, R near sample 50. Return clinical feature vector."""
    base = b[:, 25:40].mean(1)                                    # PR-segment baseline per lead
    R = np.maximum(b[:, 42:58].max(1) - base, 0)                  # R amplitude per lead
    S = np.maximum(base - b[:, 48:72].min(1), 0)                  # S depth per lead
    ST = b[:, 64:70].mean(1) - base                               # ST level (J+~50ms)
    T = b[:, 80:92].mean(1) - base                                # T amplitude
    dev = np.abs(b[:, 40:72] - base[:, None])
    qrs_w = (dev > 0.25*dev.max(1, keepdims=True)).sum(1).mean()  # QRS width proxy (samples)
    # PR proxy: P-peak position before QRS in lead II
    ii = b[L['II'], 15:42]; ppos = 15 + int(np.argmax(np.abs(ii - base[L['II']])))
    pr = 50 - ppos
    ant = [L['V1'], L['V2'], L['V3']]; lat = [L['I'], L['V5'], L['V6']]; inf = [L['II'], L['III'], L['aVF']]
    return np.array([
        S[L['V1']] + R[L['V5']],                 # Sokolow-Lyon
        R[L['aVL']] + S[L['V3']],                # Cornell
        R[[L['V5'], L['V6']]].max(),             # max lateral R
        qrs_w, pr,
        ST[ant].mean(), ST[lat].mean(), ST[inf].mean(),
        T[ant].mean(), T[lat].mean(), T[inf].mean(),
        (R[L['I']]-S[L['I']]),                   # net QRS lead I (axis)
        (R[L['aVF']]-S[L['aVF']]),               # net QRS aVF (axis)
    ], dtype=np.float32)


def cohort_clin(path):
    X = np.load(path); Xn = (X - MU) / SD
    return np.stack([clin_feats(med_beat(Xn[i])) for i in range(len(Xn))])


print('extracting clinical features...', flush=True)
Ftr = cohort_clin(f'{OUT}/ecg_X_train.npy'); Fte = cohort_clin(f'{OUT}/ecg_X_test.npy'); Fga = cohort_clin(f'{OUT}/georgia_X.npy')
FEATNAMES = ['Sokolow-Lyon', 'Cornell', 'maxLatR', 'QRSwidth', 'PR', 'ST_ant', 'ST_lat', 'ST_inf',
             'T_ant', 'T_lat', 'T_inf', 'axisI', 'axisaVF']

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
def lab(mask):
    Y = np.zeros((mask.sum(), NCLS), np.float32)
    for i, cc in enumerate(df.scp_codes.values[mask]):
        for j, c in enumerate(CODES):
            if c in cc: Y[i, j] = 1.0
    return torch.from_numpy(Y).to(DEV)
Ytr, Yte = lab(fold <= 8), lab(fold == 10)
recs = sorted(glob.glob(f'{OUT}/georgia/*/*.hea')); dx = []
for hp in recs:
    codes = ''
    for line in open(hp):
        if line.lower().replace(' ', '').startswith('#dx'): codes = line.split(':', 1)[1]
    dx.append(codes)
GAlab = {code: np.array([1.0 if sn in d else 0.0 for d in dx]) for code, sn in GA_MAP.items()}

Ftr_t = torch.from_numpy(Ftr).to(DEV); Fte_t = torch.from_numpy(Fte).to(DEV); Fga_t = torch.from_numpy(Fga).to(DEV)
mu, sd = Ftr_t.mean(0, keepdim=True), Ftr_t.std(0, keepdim=True).clamp_min(1e-6)
Xtr, Xte, Xga = (Ftr_t-mu)/sd, (Fte_t-mu)/sd, (Fga_t-mu)/sd
capable = [c for c in range(NCLS) if int(Yte[:, c].sum()) >= 10 and model_auc.get(CODES[c], 0) >= 0.75]


def auc(sc, y):
    y = y.bool(); p = y.sum().float(); n = (~y).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(sc)).float() + 1
    return float((r[y].sum()-p*(p+1)/2)/(p*n))


lin = nn.Linear(13, NCLS).to(DEV)
opt = torch.optim.Adam(lin.parameters(), lr=5e-3, weight_decay=1e-3)
for _ in range(5000):
    bi = torch.randint(0, len(Xtr), (512,), device=DEV)
    loss = F.binary_cross_entropy_with_logits(lin(Xtr[bi]), Ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    ste, sga = lin(Xte), lin(Xga)
de_macro = float(np.mean([auc(ste[:, c], Yte[:, c]) for c in capable]))

# baselines to compare against (Georgia)
rw = json.load(open(f'{QK}/ecg_refwave2_data.json'))['rows']
atom_ga = json.load(open(f'{QK}/ecg_atomic_basis.json'))['K_sweep']['24']['georgia_mapped']

rows = {}
for code in GA_MAP:
    if code not in CODES or GAlab[code].sum() < 20: continue
    c = CODES.index(code)
    clin_ga = round(auc(sga[:, c], torch.from_numpy(GAlab[code]).to(DEV)), 3)
    rows[code] = {'clinical_features_us': clin_ga, 'atomic_us': atom_ga.get(code),
                  'template_us': rw[code]['us_template_match_auc'], 'model_us': rw[code]['us_model_auc'],
                  'clinical_de': round(float(auc(ste[:, c], Yte[:, c])), 3), 'model_de': round(model_auc[CODES[c]], 3)}
    print(f'  {code:6s}: clinical {clin_ga} | atomic {atom_ga.get(code)} | template {rw[code][\"us_template_match_auc\"]} | model {rw[code][\"us_model_auc\"]}', flush=True)

# which clinical feature drives each diagnosis (interpretable)
Wc = lin.weight.detach().cpu().numpy()
drivers = {CODES[c]: FEATNAMES[int(np.argmax(np.abs(Wc[c])))] for c in capable}
res = {'clinical_features_germany_macro': round(de_macro, 3), 'model_germany_macro': round(float(np.mean([model_auc[CODES[c]] for c in capable])), 3),
       'n_clinical_features': 13, 'per_code_us': rows, 'top_clinical_driver_per_dx': drivers}
json.dump(res, open(f'{QK}/ecg_clinical_baseline.json', 'w'), indent=2)
print(json.dumps({'clinical_de_macro': res['clinical_features_germany_macro'], 'model_de_macro': res['model_germany_macro'],
                  'drivers_sample': {k: drivers[k] for k in list(drivers)[:8]}}, indent=1), flush=True)
print('ECG CLINICAL BASELINE DONE', flush=True)
