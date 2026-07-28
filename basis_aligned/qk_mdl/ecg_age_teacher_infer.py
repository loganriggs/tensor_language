"""Tier-2 teacher: ECG-AGE model (Lima et al. 2021, age-gap predicts MORTALITY). PyTorch ResNet1d.
Run on PTB-XL to (a) VALIDATE: predicted ECG-age vs true PTB-XL age (corr + MAE, scale sweep);
(b) generate age soft labels for distillation into a foldable student. This is the impactful
Tier-2 target: interpret WHAT makes an ECG look older (the mortality-linked biomarker).
"""
import sys, ast, json
sys.path.insert(0, '/workspace/tensor_language/code_age_teacher')
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from resnet import ResNet1d

DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = '/workspace/tensor_language/ecg_data'
AGE = '/workspace/tensor_language/code_age_teacher'
cfg = json.load(open(f'{AGE}/model/config.json'))
blocks_dim = list(zip(cfg['net_filter_size'], cfg['net_seq_lengh']))
model = ResNet1d(input_dim=(12, cfg['seq_length']), blocks_dim=blocks_dim, n_classes=1,
                 kernel_size=cfg['kernel_size'], dropout_rate=cfg['dropout_rate']).to(DEV)
sd = torch.load(f'{AGE}/model/model.pth', map_location=DEV, weights_only=False)
state = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
model.load_state_dict(state); model.eval()
print('age teacher loaded; params', sum(p.numel() for p in model.parameters()), flush=True)


def preprocess(X, scale):
    # X (N,12,1000)@100Hz -> (N,12,4096)@400Hz, scaled
    Xr = resample(X, 4000, axis=2)
    out = np.zeros((len(X), 12, 4096), np.float32)
    out[:, :, 48:4048] = Xr * scale
    return torch.from_numpy(out).to(DEV)


@torch.no_grad()
def predict(X, scale):
    out = []
    for i in range(0, len(X), 256):
        out.append(model(preprocess(X[i:i+256], scale)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


df = pd.read_csv(f'{OUT}/ptbxl_database.csv', index_col='ecg_id'); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
fold = df.strat_fold.values
age = df['age'].values.astype(np.float32)
Xte = np.load(f'{OUT}/ecg_X_test.npy')
te = fold == 10
true_te = age[te]
valid = (true_te >= 18) & (true_te <= 89)   # exclude anonymized age>=90 (set to 300) and peds

# scale sweep: which scale makes predicted age track true age
print('scale sweep (predicted ECG-age vs true PTB-XL age):', flush=True)
best = None
for scale in [1.0, 10.0, 100.0]:
    pred = predict(Xte, scale)
    r = float(np.corrcoef(pred[valid], true_te[valid])[0, 1])
    mae = float(np.mean(np.abs(pred[valid] - true_te[valid])))
    print(f'  scale {scale}: corr {r:.3f} MAE {mae:.1f}y (pred range {pred.min():.0f}-{pred.max():.0f})', flush=True)
    if best is None or r > best[0]:
        best = (r, scale, pred)
SCALE = best[1]
print(f'chosen scale {SCALE}: corr {best[0]:.3f}', flush=True)

# generate age soft labels for train/val/test
soft = {}
for split, mask in [('train', fold <= 8), ('val', fold == 9), ('test', fold == 10)]:
    X = np.load(f'{OUT}/ecg_X_{split}.npy')
    soft[split] = predict(X, SCALE).astype(np.float32)
    np.save(f'{QK}/age_soft_{split}.npy', soft[split])
    print(f'{split}: ECG-age labels {soft[split].shape}, mean {soft[split].mean():.1f} std {soft[split].std():.1f}', flush=True)

# age-gap (predicted - true) sanity: pathology should read OLDER (the mortality signal, §22)
pred_te = soft['test']
gap = pred_te[valid] - true_te[valid]
norm_mask = np.array(['NORM' in cc for cc in df.scp_codes.values[te]])[valid]
card = {'scale': SCALE, 'corr_pred_vs_true': round(best[0], 3),
        'mae_years': round(float(np.mean(np.abs(pred_te[valid]-true_te[valid]))), 2),
        'mean_age_gap_normal': round(float(gap[norm_mask].mean()), 2),
        'mean_age_gap_pathology': round(float(gap[~norm_mask].mean()), 2)}
json.dump(card, open(f'{QK}/ecg_age_teacher_infer.json', 'w'), indent=2)
print(json.dumps(card, indent=1), flush=True)
print('ECG AGE TEACHER INFER DONE', flush=True)
