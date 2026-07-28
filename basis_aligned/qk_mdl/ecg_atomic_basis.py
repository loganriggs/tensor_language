"""ECG atomic compositional basis (Logan): learn a SMALL shared dictionary of atomic WAVEFORM
primitives on aligned median beats, with a SPARSE readout so each diagnosis = composition of a
few reused atoms. Beats keep amplitude (unlike the cosine template baseline), so the basis CAN
represent voltage (LVH) and shape. Test: how few atoms compose all diagnoses? are they reused
+ interpretable? does it beat the template-match baseline and transfer to the US cohort?
Compares K-atom basis vs template-match (§42) vs full model; random-atom control.
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
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
NLEAD = 12
WIN = 100
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV, weights_only=False)
CODES = ck['codes']; NCLS = len(CODES)
MU, SD = ck['MU'], ck['SD']
MUnp = MU.cpu().numpy(); SDnp = SD.cpu().numpy()   # (1,12,1)
model_auc = {d['code']: d['auc'] for d in json.load(open(f'{QK}/ecg_codes_train.json'))['per_code']}
GA_MAP = {'CLBBB': '164909002', 'CRBBB': '713427006', 'LAFB': '445118002', '1AVB': '270492004',
          'LVH': '164873001', 'INJAS': '164931005', 'ISC_': '429622005', 'ISCIN': '59931005',
          'NDT': '164934002', 'AMI': '164917005'}


def med_beat_one(x, detlead=1):
    sig = x[detlead]
    pk, _ = find_peaks(np.abs(sig), distance=40, height=np.percentile(np.abs(sig), 90))
    beats = [x[:, p-WIN//2:p+WIN//2] for p in pk if p-WIN//2 >= 0 and p+WIN//2 < 1000]
    return np.median(np.stack(beats), 0) if beats else x[:, 450:550]


def cohort_beats(path):
    X = np.load(path)
    Xn = (X - MUnp) / SDnp
    return np.stack([med_beat_one(Xn[i]) for i in range(len(Xn))]).reshape(len(Xn), -1).astype(np.float32)  # (N,1200)


print('extracting median beats (train/test/georgia)...', flush=True)
Btr = cohort_beats(f'{OUT}/ecg_X_train.npy'); Bte = cohort_beats(f'{OUT}/ecg_X_test.npy'); Bga = cohort_beats(f'{OUT}/georgia_X.npy')
np.save(f'{QK}/ecg_medbeats_test.npy', Bte)   # for rendering
DIM = Btr.shape[1]

df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
def lab(mask):
    Y = np.zeros((mask.sum(), NCLS), np.float32)
    for i, cc in enumerate(df.scp_codes.values[mask]):
        for j, c in enumerate(CODES):
            if c in cc: Y[i, j] = 1.0
    return torch.from_numpy(Y).to(DEV)
Ytr, Yte = lab(fold <= 8), lab(fold == 10)
# georgia mapped labels
recs = sorted(glob.glob(f'{OUT}/georgia/*/*.hea')); dx = []
for hp in recs:
    codes = ''
    for line in open(hp):
        if line.lower().replace(' ', '').startswith('#dx'): codes = line.split(':', 1)[1]
    dx.append(codes)
GAlab = {code: np.array([1.0 if sn in d else 0.0 for d in dx]) for code, sn in GA_MAP.items()}

Btr_t = torch.from_numpy(Btr).to(DEV); Bte_t = torch.from_numpy(Bte).to(DEV); Bga_t = torch.from_numpy(Bga).to(DEV)
mu, sd = Btr_t.mean(0, keepdim=True), Btr_t.std(0, keepdim=True).clamp_min(1e-6)
Xtr, Xte, Xga = (Btr_t-mu)/sd, (Bte_t-mu)/sd, (Bga_t-mu)/sd
capable = [c for c in range(NCLS) if int(Yte[:, c].sum()) >= 10 and model_auc.get(CODES[c], 0) >= 0.75]


def auc(sc, y):
    y = y.bool(); p = y.sum().float(); n = (~y).sum().float()
    if p == 0 or n == 0: return 0.5
    r = torch.argsort(torch.argsort(sc)).float() + 1
    return float((r[y].sum()-p*(p+1)/2)/(p*n))


class Atomic(nn.Module):
    def __init__(s, K):
        super().__init__()
        s.enc = nn.Linear(DIM, K); s.dec = nn.Linear(K, DIM, bias=False); s.read = nn.Linear(K, NCLS)
    def forward(s, x):
        z = F.relu(s.enc(x)); return s.dec(z), s.read(z), z


def fit(K, l1=3e-3, recon_w=0.3, steps=6000):
    net = Atomic(K).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for st in range(steps):
        bi = torch.randint(0, len(Xtr), (512,), device=DEV)
        rec, lg, z = net(Xtr[bi])
        loss = F.binary_cross_entropy_with_logits(lg, Ytr[bi]) + recon_w*(rec-Xtr[bi]).pow(2).mean() + l1*net.read.weight.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, lte, _ = net(Xte); _, lga, _ = net(Xga)
        de = float(np.mean([auc(lte[:, c], Yte[:, c]) for c in capable]))
        ga = {code: round(auc(lga[:, CODES.index(code)], torch.from_numpy(GAlab[code]).to(DEV)), 3)
              for code in GA_MAP if code in CODES and GAlab[code].sum() >= 20}
        Wr = net.read.weight.detach().abs()            # (NCLS,K)
        thr = 0.15 * Wr.max()
        atoms_per_dx = float((Wr[capable] > thr).sum(1).float().mean())
        dx_per_atom = float((Wr[capable] > thr).sum(0).float().mean())
    return net, de, ga, atoms_per_dx, dx_per_atom


res = {'model_macro': round(float(np.mean([model_auc[CODES[c]] for c in capable])), 3), 'K_sweep': {}}
nets = {}
for K in [6, 8, 12, 16, 24, 32]:
    net, de, ga, apd, dpa = fit(K)
    res['K_sweep'][K] = {'germany_macro': round(de, 3), 'atoms_per_dx': round(apd, 1), 'dx_per_atom': round(dpa, 1),
                         'georgia_mapped': ga}
    nets[K] = net
    print(f'  K={K}: DE-macro {de:.3f} | atoms/dx {apd:.1f} | dx/atom {dpa:.1f} | GA {ga}', flush=True)

# random-atom control at K=16 (freeze random enc/dec, train only readout)
Kc = 16
netr = Atomic(Kc).to(DEV)
for p in list(netr.enc.parameters()) + list(netr.dec.parameters()): p.requires_grad_(False)
opt = torch.optim.Adam(netr.read.parameters(), lr=3e-3)
for st in range(4000):
    bi = torch.randint(0, len(Xtr), (512,), device=DEV)
    _, lg, _ = netr(Xtr[bi]); loss = F.binary_cross_entropy_with_logits(lg, Ytr[bi])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    _, lte, _ = netr(Xte); rand_de = float(np.mean([auc(lte[:, c], Yte[:, c]) for c in capable]))
res['random_atoms_K16_macro'] = round(rand_de, 3)

# template-match baseline (§42) + full model georgia, for comparison
tmt = json.load(open(f'{QK}/ecg_refwave2.json'))['by_category'] if False else None
rw = json.load(open(f'{QK}/ecg_refwave2_data.json'))['rows']
res['baseline_georgia'] = {code: {'template_match': rw[code]['us_template_match_auc'], 'model': rw[code]['us_model_auc']}
                           for code in rw}

# render atoms for the best small K (K=12): decoder columns = waveform primitives + top diagnoses
Kpick = 12; net = nets[Kpick]
Wd = net.dec.weight.detach().cpu().numpy()             # (DIM,K)
Wr = net.read.weight.detach().cpu().numpy()            # (NCLS,K)
atoms = {}
for k in range(Kpick):
    wave = Wd[:, k].reshape(NLEAD, WIN)
    peak = {LEADS[L]: round(float(np.abs(wave[L]).max()), 2) for L in range(NLEAD)}
    topleads = sorted(peak, key=peak.get, reverse=True)[:3]
    dxw = [(CODES[c], round(float(Wr[c, k]), 2)) for c in capable]
    dxw = sorted(dxw, key=lambda t: -abs(t[1]))[:4]
    atoms[k] = {'top_leads': topleads, 'top_diagnoses': dxw,
                'wave': [[round(float(v), 3) for v in wave[L]] for L in range(NLEAD)]}
json.dump({'leads': LEADS, 'K': Kpick, 'atoms': atoms}, open(f'{QK}/ecg_atomic_data.json', 'w'))
json.dump(res, open(f'{QK}/ecg_atomic_basis.json', 'w'), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != 'baseline_georgia'}, indent=1), flush=True)
print('ECG ATOMIC BASIS DONE', flush=True)
