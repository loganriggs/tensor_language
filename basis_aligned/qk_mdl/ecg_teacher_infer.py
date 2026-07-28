"""Path 2 step 1: run the SOTA teacher (Ribeiro CODE ResNet) on PTB-XL to generate soft
labels for distillation. Teacher: input (4096,12) @ 400Hz, units 1e-4V, outputs 6 classes
[1dAVb, RBBB, LBBB, SB, AF, ST]. We validate preprocessing by checking teacher outputs
against PTB-XL's own labels (CLBBB/CRBBB/1AVB) across a scale sweep BEFORE trusting it.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ast, json
import numpy as np
import pandas as pd
from scipy.signal import resample
import tensorflow as tf

TEACH = '/workspace/tensor_language/code_teacher/model/model.hdf5'
OUT = '/workspace/tensor_language/ecg_data'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
TCLASSES = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
model = tf.keras.models.load_model(TEACH, compile=False)
print('teacher loaded; params', model.count_params(), 'out', model.output_shape, flush=True)


def preprocess(X, scale):
    # X: (N,12,1000) @100Hz -> (N,4096,12) @400Hz, teacher units
    Xr = resample(X, 4000, axis=2)                  # 1000 -> 4000 samples (100->400Hz)
    N = Xr.shape[0]
    out = np.zeros((N, 4096, 12), np.float32)
    out[:, 48:4048, :] = np.transpose(Xr, (0, 2, 1)) * scale   # center-pad, (N,4096,12)
    return out


df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
Xte = np.load(f'{OUT}/ecg_X_test.npy')
def ptlabel(code, mask): return np.array([1.0 if code in cc else 0.0 for cc in df.scp_codes.values[mask]])
te = fold == 10
lab = {'1dAVb': ptlabel('1AVB', te), 'RBBB': ptlabel('CRBBB', te), 'LBBB': ptlabel('CLBBB', te)}


def auc(score, y):
    y = y.astype(bool); p = y.sum(); n = (~y).sum()
    if p == 0 or n == 0: return 0.5
    o = np.argsort(np.argsort(score)) + 1
    return float((o[y].sum() - p*(p+1)/2) / (p*n))


# scale sweep on TEST (validate preprocessing)
print('scale sweep (teacher AUC vs PTB-XL labels):', flush=True)
best = None
for scale in [1.0, 5.0, 10.0, 20.0]:
    Xp = preprocess(Xte, scale)
    pr = model.predict(Xp, batch_size=64, verbose=0)   # (N,6)
    aucs = {c: round(auc(pr[:, TCLASSES.index(c)], lab[c]), 3) for c in lab}
    m = np.mean(list(aucs.values()))
    print(f'  scale {scale}: {aucs} mean {m:.3f}', flush=True)
    if best is None or m > best[0]:
        best = (m, scale, pr)
SCALE = best[1]
print(f'chosen scale {SCALE} (mean teacher-vs-PTBXL AUC {best[0]:.3f})', flush=True)

# generate soft labels for train/val/test
soft = {}
for split, mask in [('train', fold <= 8), ('val', fold == 9), ('test', fold == 10)]:
    X = np.load(f'{OUT}/ecg_X_{split}.npy')
    prs = []
    for i in range(0, len(X), 512):
        prs.append(model.predict(preprocess(X[i:i+512], SCALE), batch_size=64, verbose=0))
    soft[split] = np.concatenate(prs).astype(np.float32)
    np.save(f'{QK}/teacher_soft_{split}.npy', soft[split])
    print(f'{split}: soft labels {soft[split].shape}, class means {np.round(soft[split].mean(0),3)}', flush=True)

# final teacher AUC on test (report card)
prte = soft['test']
card = {'scale': SCALE, 'teacher_classes': TCLASSES,
        'teacher_test_auc_vs_ptbxl': {c: round(auc(prte[:, TCLASSES.index(c)], lab[c]), 3) for c in lab},
        'class_prevalence_soft>0.5': {TCLASSES[j]: int((prte[:, j] > 0.5).sum()) for j in range(6)}}
json.dump(card, open(f'{QK}/ecg_teacher_infer.json', 'w'), indent=2)
print(json.dumps(card, indent=1), flush=True)
print('ECG TEACHER INFER DONE', flush=True)
