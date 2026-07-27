"""ECG STAGE 1 prep: load PTB-XL (100 Hz), build the standard 5-superclass
diagnostic task with the canonical patient-stratified fold split (folds 1-8 train,
9 val, 10 test), cache to npy. Signals: (N, 12 leads, 1000 samples)."""
import ast, sys, os
import numpy as np
import pandas as pd
import wfdb

OUT = '/workspace/tensor_language/ecg_data'
ROOT = OUT   # targeted download places CSVs + records100/ directly under ecg_data
print('dataset root:', ROOT, flush=True)

df = pd.read_csv(f'{ROOT}/ptbxl_database.csv', index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
agg = pd.read_csv(f'{ROOT}/scp_statements.csv', index_col=0)
agg = agg[agg.diagnostic == 1]
SUP = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def superclasses(codes):
    s = set()
    for c in codes:
        if c in agg.index:
            s.add(agg.loc[c].diagnostic_class)
    return s


df['sup'] = df.scp_codes.apply(superclasses)
Y = np.zeros((len(df), len(SUP)), dtype=np.float32)
for i, s in enumerate(df['sup'].values):
    for k, name in enumerate(SUP):
        if name in s:
            Y[i, k] = 1.0

# load 100 Hz signals
paths = df.filename_lr.values
X = np.zeros((len(df), 12, 1000), dtype=np.float32)
for i, p in enumerate(paths):
    sig, _ = wfdb.rdsamp(f'{ROOT}/{p}')
    X[i] = sig.T.astype(np.float32)
    if i % 3000 == 0:
        print(f'{i}/{len(df)} loaded', flush=True)

fold = df.strat_fold.values
for name, mask in (('train', fold <= 8), ('val', fold == 9), ('test', fold == 10)):
    np.save(f'{OUT}/ecg_X_{name}.npy', X[mask])
    np.save(f'{OUT}/ecg_Y_{name}.npy', Y[mask])
    print(f'{name}: {mask.sum()} records, label prevalence {Y[mask].mean(0).round(3)}', flush=True)
print('ECG PREP DONE', flush=True)
