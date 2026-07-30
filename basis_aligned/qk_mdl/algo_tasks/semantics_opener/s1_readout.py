"""Step 1: channel activation vs TRUE coded opener-state on natural text.

Activation a1 = q1^T x, a4 = Q4^T x at layer-13-entry residual, every position.
Coded state from raw tokens (byte tracker): paren/square/curly depth, ASCII
double-quote parity, curly-quote depth (independent knowledge, no model).

Fit: cooc rows 0:240. Exploration held-out: cooc rows 240:300.
FINAL numbers: audit slice fineweb rows 448:600 (held-back).
Functional forms: univariate + multivariate linear; R^2 on held-out; effect
sizes (Cohen's d) for the binary forms; position-only control.
Saves readout.json + calib.json (coefficients for the gate).
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_opener')
from common import (OUT, get_model, collect_activations, coded_states, derived,
                    cooc, fineweb_audit, FEATS)

m, cfg = get_model()
q1 = torch.load(f'{OUT}/Q_r1.pt').cuda()
Q4 = torch.load(f'{OUT}/Q_r4.pt').cuda()
Qall = torch.cat([q1, Q4], dim=1)  # (D,5): col0 = a1, cols1:5 = a4

rows_fit = cooc((0, 240))
rows_expl = cooc((240, 300))
rows_audit = fineweb_audit()

acts, states = {}, {}
for name, rows in [('fit', rows_fit), ('expl', rows_expl), ('audit', rows_audit)]:
    a = collect_activations(m, cfg, rows, Qall)   # (N,T,5)
    acts[name] = a
    states[name] = coded_states(rows)
    print(f'{name}: acts {a.shape}, state coverage: '
          f'{derived(states[name])["any_open"].mean():.3f} frac any-open', flush=True)

D = {name: derived(states[name]) for name in acts}


def design(name, cols):
    X = np.stack([D[name][c].ravel() for c in cols], 1).astype(np.float64)
    return np.concatenate([X, np.ones((X.shape[0], 1))], 1)


def fit_eval(cols, target_col=0):
    """OLS fit on 'fit', R^2 on all three splits. target: activation column."""
    y = {n: acts[n][..., target_col].ravel().astype(np.float64) for n in acts}
    Xf = design('fit', cols)
    beta, *_ = np.linalg.lstsq(Xf, y['fit'], rcond=None)
    out = {}
    for n in acts:
        Xn = design(n, cols)
        pred = Xn @ beta
        ss_res = ((y[n] - pred) ** 2).sum()
        ss_tot = ((y[n] - y[n].mean()) ** 2).sum()
        out[f'R2_{n}'] = round(1 - ss_res / ss_tot, 4)
    out['beta'] = [round(float(b), 4) for b in beta]
    return out, beta


FORMS = {
    'p_depth': ['p_depth'],
    'tot_depth': ['tot_depth'],
    'any_open': ['any_open'],
    'q_any': ['q_any'],
    'q_par': ['q_par'],
    'p_open_binary': None,   # handled below (binarized depth)
    'raw5': FEATS,
    'tot+qany': ['tot_depth', 'q_any'],
}

res = {'n_positions': {n: int(acts[n].shape[0] * acts[n].shape[1]) for n in acts},
       'frac_any_open': {n: round(float(D[n]['any_open'].mean()), 4) for n in acts}}

# add binarized depth as derived feature
for n in D:
    D[n]['p_open_binary'] = (D[n]['p_depth'] > 0).astype(np.int64)
FORMS['p_open_binary'] = ['p_open_binary']

r1 = {}
betas = {}
for fname, cols in FORMS.items():
    out, beta = fit_eval(cols, target_col=0)
    r1[fname] = out
    betas[fname] = beta
    print(f'a1 ~ {fname}: {out}', flush=True)
res['a1_forms'] = r1

# position-only control
for n in D:
    T = acts[n].shape[1]
    D[n]['pos'] = np.broadcast_to(np.arange(T), acts[n].shape[:2]).copy()
out, _ = fit_eval(['pos'])
res['a1_position_control'] = out
out, _ = fit_eval(['pos', 'tot_depth', 'q_any'])
res['a1_pos_plus_state'] = out
print('position control:', res['a1_position_control'], flush=True)

# effect sizes on audit (binary forms), a1
a1_aud = acts['audit'][..., 0].ravel()
eff = {}
for feat in ['any_open', 'q_any', 'p_open_binary']:
    f = D['audit'][feat].ravel().astype(bool)
    x1, x0 = a1_aud[f], a1_aud[~f]
    sp = np.sqrt(((len(x1) - 1) * x1.var(ddof=1) + (len(x0) - 1) * x0.var(ddof=1))
                 / (len(x1) + len(x0) - 2))
    eff[feat] = {'mean_open': round(float(x1.mean()), 3),
                 'mean_closed': round(float(x0.mean()), 3),
                 'cohens_d': round(float((x1.mean() - x0.mean()) / sp), 3),
                 'n_open': int(f.sum()), 'n_closed': int((~f).sum())}
res['a1_effect_sizes_audit'] = eff
print('effect sizes:', eff, flush=True)

# does a1 track depth beyond binary? conditional means on audit by paren depth
cond = {}
pd_aud = D['audit']['p_depth'].ravel()
for d in range(0, 5):
    sel = pd_aud == d
    if sel.sum() >= 50:
        cond[str(d)] = {'mean_a1': round(float(a1_aud[sel].mean()), 3),
                        'se': round(float(a1_aud[sel].std(ddof=1) / np.sqrt(sel.sum())), 4),
                        'n': int(sel.sum())}
res['a1_by_paren_depth_audit'] = cond
print('a1 by paren depth (audit):', cond, flush=True)
# same conditioned on quotes closed (isolate paren effect)
qa = D['audit']['q_any'].ravel() == 0
cond2 = {}
for d in range(0, 5):
    sel = (pd_aud == d) & qa
    if sel.sum() >= 50:
        cond2[str(d)] = {'mean_a1': round(float(a1_aud[sel].mean()), 3),
                         'se': round(float(a1_aud[sel].std(ddof=1) / np.sqrt(sel.sum())), 4),
                         'n': int(sel.sum())}
res['a1_by_paren_depth_no_quote_audit'] = cond2

# ---- 4-dim channel: forward R^2 per dim, and linear DECODING of state ----
r4_fwd = {}
for j in range(4):
    out, beta = fit_eval(FEATS, target_col=1 + j)
    r4_fwd[f'dim{j}'] = {k: v for k, v in out.items() if k.startswith('R2')}
res['a4_forward_R2_raw5'] = r4_fwd
print('a4 forward:', r4_fwd, flush=True)

# decoding: predict state features from a4 (+a1) with OLS / logistic-style AUC
def decode(target_feat, use_cols):
    Xf = np.concatenate([acts['fit'][..., use_cols].reshape(-1, len(use_cols)),
                         np.ones((acts['fit'].shape[0] * acts['fit'].shape[1], 1))], 1)
    yf = D['fit'][target_feat].ravel().astype(np.float64)
    beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
    out = {}
    for n in acts:
        Xn = np.concatenate([acts[n][..., use_cols].reshape(-1, len(use_cols)),
                             np.ones((acts[n].shape[0] * acts[n].shape[1], 1))], 1)
        pred = Xn @ beta
        yv = D[n][target_feat].ravel().astype(np.float64)
        ss = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        out[f'R2_{n}'] = round(float(ss), 4)
        if set(np.unique(yv)) <= {0.0, 1.0} and 0 < yv.mean() < 1:
            # AUC by rank statistic
            order = np.argsort(pred)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(pred))
            n1 = yv.sum(); n0 = len(yv) - n1
            auc = (ranks[yv == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
            out[f'AUC_{n}'] = round(float(auc), 4)
    return out

dec = {}
for feat in ['any_open', 'p_open_binary', 'q_any', 'tot_depth']:
    dec[f'{feat}_from_a1'] = decode(feat, [0])
    dec[f'{feat}_from_a4'] = decode(feat, [1, 2, 3, 4])
    print(f'decode {feat}: a1 {dec[f"{feat}_from_a1"]} | a4 {dec[f"{feat}_from_a4"]}',
          flush=True)
res['decoding'] = dec

# ---- calibration for the gate (saved): a1_hat and a4_hat from raw5 ----
calib = {'feats': FEATS}
_, b1 = fit_eval(FEATS, target_col=0)
calib['a1_beta'] = [float(v) for v in b1]
W4 = []
for j in range(4):
    _, bj = fit_eval(FEATS, target_col=1 + j)
    W4.append([float(v) for v in bj])
calib['a4_beta'] = W4
# binary calibration (any_open) for the simplest code: a1 = alpha*any_open+beta
_, bb = fit_eval(['any_open'], target_col=0)
calib['a1_anyopen_alpha_beta'] = [float(bb[0]), float(bb[1])]
json.dump(calib, open(f'{OUT}/calib.json', 'w'), indent=1)
json.dump(res, open(f'{OUT}/readout.json', 'w'), indent=1)
print('S1 DONE', flush=True)
