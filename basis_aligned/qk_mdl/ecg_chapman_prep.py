"""ECG STAGE 2 prep: load the Chapman-Shaoxing (China) cohort as an INDEPENDENT cross-country
validation set. Challenge WFDB format (.mat + .hea, 500 Hz), resample to 100 Hz to
match PTB-XL and our model. Label conduction disturbance (CD, = bundle branch block
family) vs the rest, from SNOMED Dx codes, so we can test whether the V1/QRS BBB
feature found on German PTB-XL (Stage 1) generalizes to a Chinese cohort.
"""
import os, sys, glob
import numpy as np
import wfdb
from scipy.signal import resample_poly

G = '/workspace/tensor_language/ecg_data/chapman_shaoxing'
OUT = '/workspace/tensor_language/ecg_data'
# CD / bundle-branch-block family SNOMED CT codes (Challenge 2021 scored set)
CD_SNOMED = {'713427006', '59118001',        # complete / incomplete RBBB
             '733534002', '164909002',        # complete / (incomplete) LBBB
             '445118002', '445211001',        # LAFB / LPFB
             '698252002',                       # IVCD (nonspecific intraventricular block)
             '270492004', '54016002', '28189009'}  # 1st/2nd deg AV block
NORM_SNOMED = {'426783006'}                    # sinus rhythm / normal


def dx_of(hea_path):
    with open(hea_path) as f:
        for line in f:
            if line.startswith('# Dx:') or line.startswith('#Dx:'):
                return set(line.split(':', 1)[1].strip().split(','))
    return set()


recs = sorted(glob.glob(f'{G}/*/*.hea'))
print(f'{len(recs)} Chapman records', flush=True)
X = np.zeros((len(recs), 12, 1000), dtype=np.float32)
yCD = np.zeros(len(recs), dtype=np.float32)
yNORM = np.zeros(len(recs), dtype=np.float32)
keep = np.ones(len(recs), dtype=bool)
for i, hp in enumerate(recs):
    base = hp[:-4]
    try:
        sig, meta = wfdb.rdsamp(base)          # (N,12) at fs
    except Exception:
        keep[i] = False; continue
    fs = meta['fs']
    s = sig.T.astype(np.float32)               # (12, N)
    if s.shape[1] != 1000:                      # resample to 100 Hz, 10 s
        s = resample_poly(s, 100, fs, axis=1)
        if s.shape[1] >= 1000:
            s = s[:, :1000]
        else:
            s = np.pad(s, ((0, 0), (0, 1000 - s.shape[1])))
    X[i] = np.nan_to_num(s)
    dx = dx_of(hp)
    yCD[i] = 1.0 if dx & CD_SNOMED else 0.0
    yNORM[i] = 1.0 if dx & NORM_SNOMED else 0.0
    if i % 2000 == 0:
        print(f'{i}/{len(recs)}', flush=True)
X, yCD, yNORM = X[keep], yCD[keep], yNORM[keep]
np.save(f'{OUT}/chapman_X.npy', X)
np.save(f'{OUT}/chapman_yCD.npy', yCD)
np.save(f'{OUT}/chapman_yNORM.npy', yNORM)
print(f'saved {len(X)} records | CD prevalence {yCD.mean():.3f} | NORM prevalence {yNORM.mean():.3f}', flush=True)
print('CHAPMAN PREP DONE', flush=True)
