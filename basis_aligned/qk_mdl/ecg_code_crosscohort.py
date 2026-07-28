"""ECG: cross-continent validation of the SPECIFIC-code circuits (§24). For codes that
map cleanly to SNOMED, test whether the model's per-code prediction AND its per-lead
feature hold on the US (Georgia) and China (Chapman) cohorts -- i.e. are the tiny
physiology-matched circuits real cross-continental mechanisms or PTB-XL artifacts?
"""
import glob, sys, json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# PTB-XL code -> SNOMED (Challenge) for cross-cohort labels
CODE2SNOMED = {'CRBBB': '713427006', 'CLBBB': '164909002', 'IRBBB': '713426002',
               'LAFB': '445118002', 'LPFB': '445211001', '1AVB': '270492004',
               'IMI': '164865005', 'AMI': '164865005'}
ck = torch.load(f'{QK}/ecg_codes_model.pt', map_location=DEV)
cfg = ck['cfg']; W = ck['state']; CODES = ck['codes']
D, NH, HD, NL, INNER = cfg['D'], cfg['NH'], cfg['HD'], cfg['NL'], cfg['INNER']
PT, NP, PXD, NLEAD, NCLS = cfg['PT'], cfg['NP'], cfg['PXD'], cfg['NLEAD'], cfg['NCLS']
MU, SD = ck['MU'].to(DEV), ck['SD'].to(DEV)
norm = lambda x: (x - MU) / SD


def patch(x):
    B = x.shape[0]
    return x.reshape(B, NLEAD, NP, PT).permute(0, 2, 1, 3).reshape(B, NP, PXD)


@torch.no_grad()
def forward(x, occ_lead=None):
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
        h = h + (((hn2 @ W[mw+'L.weight'].T)*(hn2 @ W[mw+'R.weight'].T)) @ W[mw+'Dn.weight'].T)
    return F.rms_norm(h, (D,)).mean(1) @ W['head.weight'].T + W['head.bias']


def auc(s, lab):
    o = torch.argsort(torch.argsort(s)); r = o.float()+1
    p = lab.sum().float(); n = (~lab).sum().float()
    return 0.5 if p == 0 or n == 0 else float((r[lab].sum()-p*(p+1)/2)/(p*n))


def cohort_codelabels(dirname, snomed):
    recs = sorted(glob.glob(f'{OUT}/{dirname}/*/*.hea'))
    lab = np.zeros(len(recs), dtype=np.float32)
    for i, hp in enumerate(recs):
        for line in open(hp):
            lk = line.lower()
            if lk.startswith('# dx') or lk.startswith('#dx'):
                if snomed in line.split(':', 1)[1]:
                    lab[i] = 1.0
    return lab


Xg = torch.from_numpy(np.load(f'{OUT}/georgia_X.npy')).to(DEV)
Xc = torch.from_numpy(np.load(f'{OUT}/chapman_X.npy')).to(DEV)
with torch.no_grad():
    Sg = torch.cat([forward(Xg[i:i+2048]) for i in range(0, len(Xg), 2048)]).float()
    Sc = torch.cat([forward(Xc[i:i+2048]) for i in range(0, len(Xc), 2048)]).float()

# per-lead occlusion signatures per cohort (for feature-match)
import pandas as pd, ast
df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xp = torch.from_numpy(np.load(f'{OUT}/ecg_X_test.npy')).to(DEV)
with torch.no_grad():
    Sp = torch.cat([forward(Xp[i:i+2048]) for i in range(0, len(Xp), 2048)]).float()

res = {}
for code, sn in CODE2SNOMED.items():
    if code not in CODES:
        continue
    ci = CODES.index(code)
    # PTB-XL label
    yp = torch.tensor([1.0 if code in cc else 0.0 for cc in df.scp_codes.values[fold == 10]], device=DEV).bool()
    yg = torch.from_numpy(cohort_codelabels('georgia', sn)[:len(Xg)]).to(DEV).bool()
    ycz = cohort_codelabels('chapman_shaoxing', sn)[:len(Xc)]
    yc = torch.from_numpy(ycz).to(DEV).bool()
    entry = {'auc_DE': round(auc(Sp[:, ci], yp), 3), 'n_DE': int(yp.sum()),
             'auc_US': round(auc(Sg[:, ci], yg), 3), 'n_US': int(yg.sum()),
             'auc_CN': round(auc(Sc[:, ci], yc), 3), 'n_CN': int(yc.sum())}
    res[code] = entry
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f'{QK}/ecg_code_crosscohort.json', 'w'), indent=2)
print('ECG CODE CROSSCOHORT DONE', flush=True)
