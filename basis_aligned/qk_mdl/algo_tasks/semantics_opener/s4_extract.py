"""Step 4: EXTRACTION — a standalone bracket-closure predictor as runnable code.

Predictor (no model access at prediction time): given raw tokens, compute the
coded opener-state; predict the channel-mediated closer-boost
    delta_hat_paren[t] = c0p + c1p * (p_depth[t] > 0)
    delta_hat_quote[t] = c0q + c1q * (q_any[t] > 0)
where the four constants are calibrated ONCE on cooc rows 300:360 against the
model's actual channel-mediated boost delta[t] = lp_base(closer) - lp_zero(closer)
(zero = r1 channel zeroed at all positions, layer 13).

Final evaluation on the HELD-BACK audit slice:
  (a) circuit-mediated: Pearson r(delta_hat, delta); AUC of the binary open
      flag classifying delta > 1 nat; accuracy.
  (b) behavioral: AUC of the open flag classifying elevated lp_base(closer)
      (top-5% positions).
Also prints a per-position demo trace on two battery prompts.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_opener')
from common import (OUT, get_model, forward_hooked, sub_hook, coded_state,
                    coded_states, derived, cooc, fineweb_audit, safe, get_tok, BATCH)

DEV = 'cuda'
m, cfg = get_model()
q1 = torch.load(f'{OUT}/Q_r1.pt').to(DEV)
CLOSER_IDS = [8, 1]   # ')' , '"'


@torch.no_grad()
def lp_pass(rows, zero=False):
    lps = []
    hook = sub_hook(q1, None) if zero else None
    for i in range(0, len(rows), BATCH):
        idx = torch.from_numpy(np.ascontiguousarray(rows[i:i + BATCH]).astype(np.int64)).to(DEV)
        lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
        lps.append(F.log_softmax(lg, -1)[:, :, CLOSER_IDS].cpu().numpy())
        del lg
    return np.concatenate(lps)


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(len(score))
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[label].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


# ---- calibration on cooc rows 300:360 ----
cal_rows = cooc((300, 360))
st_cal = derived(coded_states(cal_rows))
lp_b = lp_pass(cal_rows)
lp_z = lp_pass(cal_rows, zero=True)
delta_cal = lp_b - lp_z                       # (N,T,2) channel-mediated boost
coef = {}
for j, (cl, feat) in enumerate([('paren', 'p_depth'), ('quote', 'q_any')]):
    f = (st_cal[feat] > 0).ravel().astype(np.float64)
    d = delta_cal[..., j].ravel()
    c1 = d[f > 0].mean() - d[f == 0].mean()
    c0 = d[f == 0].mean()
    coef[cl] = {'c0': float(c0), 'c1': float(c1)}
print('calibrated coefficients:', coef, flush=True)


def predict_closer_boost(token_ids):
    """THE STANDALONE PREDICTOR: raw gpt2 token ids -> per-position predicted
    channel-mediated boost for ')' and '"'. Pure python, no model."""
    st = derived(coded_state(np.asarray(token_ids))[None])
    p_open = (st['p_depth'][0] > 0).astype(np.float64)
    q_open = (st['q_any'][0] > 0).astype(np.float64)
    return (coef['paren']['c0'] + coef['paren']['c1'] * p_open,
            coef['quote']['c0'] + coef['quote']['c1'] * q_open)


# ---- final evaluation on audit ----
audit = fineweb_audit()
st_aud = derived(coded_states(audit))
lp_base = np.load(f'{OUT}/audit_lp_base.npy')
lp_zero = np.load(f'{OUT}/audit_lp_r1_zero.npy')
delta = lp_base - lp_zero

res = {'coef': coef}
for j, (cl, feat) in enumerate([('paren', 'p_depth'), ('quote', 'q_any')]):
    f = (st_aud[feat] > 0).ravel()
    dh = np.where(f, coef[cl]['c0'] + coef[cl]['c1'], coef[cl]['c0'])
    d = delta[..., j].ravel()
    r = float(np.corrcoef(dh, d)[0, 1])
    lab = d > 1.0
    out = {
        'pearson_r_deltahat_vs_delta': round(r, 4),
        'auc_openflag_vs_delta_gt_1nat': round(auc(f.astype(float), lab), 4),
        'frac_delta_gt_1nat': round(float(lab.mean()), 4),
        'mean_delta_open': round(float(d[f].mean()), 3),
        'mean_delta_closed': round(float(d[~f].mean()), 3),
        'acc_sign': round(float(((dh > coef[cl]['c0'] + 0.5 * coef[cl]['c1'])
                                 == (d > np.median(d))).mean()), 4),
        'behavioral_auc_openflag_vs_top5pct_lp': round(
            auc(f.astype(float), lp_base[..., j].ravel()
                >= np.quantile(lp_base[..., j].ravel(), 0.95)), 4)}
    res[cl] = out
    print(f'{cl}: {out}', flush=True)

# also: correlation restricted to positions where prediction says open OR a
# random matched sample of closed (balanced), to avoid base-rate deflation
bal = {}
rng = np.random.RandomState(0)
for j, (cl, feat) in enumerate([('paren', 'p_depth'), ('quote', 'q_any')]):
    f = (st_aud[feat] > 0).ravel()
    d = delta[..., j].ravel()
    idx_open = np.where(f)[0]
    idx_closed = rng.choice(np.where(~f)[0], size=len(idx_open), replace=False)
    sel = np.concatenate([idx_open, idx_closed])
    dh = f[sel].astype(float)
    bal[cl] = {'balanced_pearson_r': round(float(np.corrcoef(dh, d[sel])[0, 1]), 4),
               'balanced_auc': round(auc(dh, d[sel] > 1.0), 4)}
    print(f'{cl} balanced: {bal[cl]}', flush=True)
res['balanced'] = bal

# ---- demo trace on two battery prompts ----
tok = get_tok()
demos = []
for text in ['The dogs ( which was near the cats and the birds',
             'She said " bring me the books and the stones']:
    ids = tok(text)['input_ids']
    ph, qh = predict_closer_boost(ids)
    idx = torch.tensor([ids], device=DEV)
    lg_b = safe(forward_hooked, m, cfg, idx).float()
    lg_z = safe(forward_hooked, m, cfg, idx, hook=sub_hook(q1, None)).float()
    db = (F.log_softmax(lg_b, -1)[0, :, CLOSER_IDS]
          - F.log_softmax(lg_z, -1)[0, :, CLOSER_IDS]).cpu().numpy()
    rows = []
    for t, tid in enumerate(ids):
        rows.append({'tok': tok.decode([tid]), 'pred_paren': round(float(ph[t]), 2),
                     'actual_paren': round(float(db[t, 0]), 2),
                     'pred_quote': round(float(qh[t]), 2),
                     'actual_quote': round(float(db[t, 1]), 2)})
    demos.append({'text': text, 'trace': rows})
    print(f'\ndemo: {text}')
    for r in rows:
        print(f"  {r['tok']!r:12s} paren pred {r['pred_paren']:7.2f} act {r['actual_paren']:7.2f}"
              f" | quote pred {r['pred_quote']:7.2f} act {r['actual_quote']:7.2f}", flush=True)
res['demos'] = demos

json.dump(res, open(f'{OUT}/extraction.json', 'w'), indent=1)
print('S4 DONE', flush=True)
