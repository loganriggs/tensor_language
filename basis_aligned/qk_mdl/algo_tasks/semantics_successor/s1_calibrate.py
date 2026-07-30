"""s1: calibration of the payload CODE + early test of BOTH candidate sites.

1. Build calibration prompts (comma lists length 3 and 5 for weekday / month /
   alphabet / digit; numbered lists), record baseline behavior.
2. Collect the real payload Delta = yh4[pred_pos, h] for h in (3,7) at layer 8.
3. Fit CODE  payload(e) = W emb(e) + b  (ridge; W_full on all elements,
   W_holdout excluding red-team elements {Thursday, October, m, 7}).
4. SITE TEST with real-activation swaps on clean/corr pairs (length-4 eval
   prompts, last element replaced): follow-rate of the donor's successor for
   site A (L8 head outputs at pred pos: H3 alone / H3+H7 / all 9 heads) and
   site B (v1 cache slice at last pos, layer-8 read, H3+H7).
"""
import random
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_successor')
from semlib import (HERE, DEV, L_PAY, HEADS, FAMILIES, CYCLIC, NOUNS, get_model,
                    comma_prompt, numlist_prompt, follow_ans_tok, batched_run,
                    fit_ridge, save_json, free_gb)

torch.manual_seed(0)
random.seed(0)
print(f'free GPU mem: {free_gb():.1f} GB', flush=True)
m, cfg = get_model()

HOLDOUT_ELEMS = {'Thursday', 'October', 'm', '7'}

# ---------------------------------------------------------- calibration set --
cal = []
for fam, lst in FAMILIES.items():
    n = len(lst)
    for length in (3, 5):
        starts = range(n) if CYCLIC[fam] else range(n - length + 1)
        for s in starts:
            cal.append(comma_prompt(fam, s, length))
for k in range(1, 8):
    for w1, w2 in [('dogs', 'cats'), ('books', 'trees'), ('cars', 'ships' if 'ships' in NOUNS else 'fish')]:
        cal.append(numlist_prompt(k, w1, w2))
print(f'calibration prompts: {len(cal)}', flush=True)

# ------------------------------------------- collect payloads + baseline -----
by_len = {}
for i, p in enumerate(cal):
    by_len.setdefault(len(p['tokens']), []).append(i)

pay = torch.zeros(len(cal), 2 * 128)          # h3 | h7 head-space payloads
patw = torch.zeros(len(cal), 2)               # pattern weight pred->last per head
base_pred = [None] * len(cal)
for L, ids in sorted(by_len.items()):
    idx = torch.tensor([cal[i]['tokens'] for i in ids], device=DEV)
    lg, caches = batched_run(m, cfg, idx, bs=6, want_head=(L_PAY,), want_pat=(L_PAY,))
    off = 0
    for c in caches:
        yh = c[('h', L_PAY)]; pt = c[('pat', L_PAY)]
        for b in range(yh.shape[0]):
            i = ids[off + b]
            pp, lp = cal[i]['pred_pos'], cal[i]['last_pos']
            pay[i] = torch.cat([yh[b, pp, h, :] for h in HEADS]).cpu()
            patw[i] = torch.tensor([pt[b, h, pp, lp].item() for h in HEADS])
        off += yh.shape[0]
    for j, i in enumerate(ids):
        base_pred[i] = lg[j, cal[i]['pred_pos']].argmax().item()

from semlib import get_tok
tok = get_tok()
acc = {}
for fam in list(FAMILIES) + ['numlist']:
    rows = [(i, p) for i, p in enumerate(cal) if p['family'] == fam and p['ans_tok'] is not None]
    ok = sum(base_pred[i] == p['ans_tok'] for i, p in rows)
    acc[fam] = {'n': len(rows), 'acc': ok / len(rows)}
print('baseline (calibration prompts, non-wrap):', acc, flush=True)

# pattern-weight variability (context dependence of the broadcast strength)
pw = {}
for fam in list(FAMILIES) + ['numlist']:
    ii = [i for i, p in enumerate(cal) if p['family'] == fam]
    pw[fam] = {f'h{h}': {'mean': patw[ii][:, j].mean().item(),
                         'std': patw[ii][:, j].std().item()}
               for j, h in enumerate(HEADS)}

# ------------------------------------------------------------------ fit W ----
emb = m.transformer.wte.weight.detach().double().cpu()
Phi = emb[torch.tensor([p['last_tok'] for p in cal])]
Y = pay.double()

W_full, lam_f, r2_f = fit_ridge(Phi, Y)
keep = [i for i, p in enumerate(cal) if p['last_elem'] not in HOLDOUT_ELEMS]
W_hold, lam_h, r2_h = fit_ridge(Phi[keep], Y[keep])
torch.save({'W_full': W_full, 'W_hold': W_hold, 'holdout_elems': sorted(HOLDOUT_ELEMS)},
           f'{HERE}/code_W.pt')
torch.save({'pay': pay, 'patw': patw, 'cal_meta': [
    {k: p[k] for k in ('family', 'last_elem', 'last_tok', 'text')} for p in cal]},
    f'{HERE}/calibration.pt')
print(f'W_full: lam={lam_f} split-R2={r2_f:.3f} | W_hold: lam={lam_h} R2={r2_h:.3f}', flush=True)

# in-sample fit quality per family (descriptive only, not a gate)
X1 = torch.cat([Phi, torch.ones(len(cal), 1, dtype=torch.float64)], 1)
pred = X1 @ W_full
fitq = {}
for fam in list(FAMILIES) + ['numlist']:
    ii = torch.tensor([i for i, p in enumerate(cal) if p['family'] == fam])
    e = pred[ii] - Y[ii]
    fitq[fam] = {'r2': (1 - (e ** 2).sum() / ((Y[ii] - Y[ii].mean(0)) ** 2).sum()).item(),
                 'cos_mean': torch.nn.functional.cosine_similarity(pred[ii], Y[ii], dim=1).mean().item()}
print('fit quality per family:', fitq, flush=True)

# ------------------------------------------------------------- site test -----
# clean/corr pairs, length-4 comma prompts + numlist; donor = corr (different
# last element, same context). Swap donor activation into clean at each site.
rng = random.Random(1)
pairs = []
for fam, lst in FAMILIES.items():
    n = len(lst)
    starts = range(n) if CYCLIC[fam] else range(n - 4 + 1)
    for s in starts:
        p = comma_prompt(fam, s, 4)
        if p['ans_tok'] is None or (CYCLIC[fam] and (s + 3) % n == n - 1):
            continue   # skip prompts whose own answer is a wrap
        cands = [e for e in lst if e != p['last_elem'] and follow_ans_tok(fam, e) is not None]
        e2 = rng.choice(cands)
        q = dict(p)
        q['tokens'] = list(p['tokens'])
        from semlib import tok1
        q['tokens'][p['last_pos']] = tok1(' ' + e2)
        q['last_elem'], q['last_tok'] = e2, tok1(' ' + e2)
        q['follow_tok'] = follow_ans_tok(fam, e2)
        pairs.append((p, q))
for k in range(1, 7):
    p = numlist_prompt(k, *rng.sample(NOUNS, 2))
    k2 = rng.choice([j for j in range(1, 8) if j != k])
    q = numlist_prompt(k2, *[tok.decode([p['tokens'][2]]).strip(), tok.decode([p['tokens'][6]]).strip()])
    q['follow_tok'] = q['ans_tok']
    pairs.append((p, q))
print(f'site-test pairs: {len(pairs)}', flush=True)

ci = torch.tensor([p['tokens'] for p, q in pairs], device=DEV)
xi = torch.tensor([q['tokens'] for p, q in pairs], device=DEV)
pred_pos = torch.tensor([p['pred_pos'] for p, q in pairs], device=DEV)
last_pos = torch.tensor([p['last_pos'] for p, q in pairs], device=DEV)
clean_ans = torch.tensor([p['ans_tok'] for p, q in pairs])
follow = torch.tensor([q['follow_tok'] for p, q in pairs])

# donor caches: head outputs at pred pos, and v1 slices at last pos
lgx, cx = batched_run(m, cfg, xi, bs=6, want_head=(L_PAY,))
donor_head = {h: [] for h in range(9)}
for bi, c in enumerate(cx):
    yh = c[('h', L_PAY)]
    for b in range(yh.shape[0]):
        i = bi * 6 + b
        for h in range(9):
            donor_head[h].append(yh[b, pairs[i][0]['pred_pos'], h, :])
donor_head = {h: torch.stack(v).to(DEV) for h, v in donor_head.items()}

# donor v1 slice = layer-0 c_v of the donor's LAST token (v1 is token-determined)
import torch.nn.functional as Fn
D = cfg['n_embd']
with torch.no_grad():
    e0 = Fn.rms_norm(m.transformer.wte(xi[torch.arange(len(pairs)), last_pos]), (D,))
    blk0 = m.transformer.h[0]
    x_in = (blk0.lambdas[0] + blk0.lambdas[1]) * e0
    v1_all = blk0.attn.c_v(Fn.rms_norm(x_in, (D,))).view(len(pairs), 9, -1)

def eval_cond(name, **kw):
    lg, _ = batched_run(m, cfg, ci, bs=6, **kw)
    pr = lg[torch.arange(len(pairs)), pred_pos.cpu()].argmax(-1)
    res = {'follow_rate': (pr == follow).float().mean().item(),
           'stay_rate': (pr == clean_ans).float().mean().item()}
    perfam = {}
    for fam in list(FAMILIES) + ['numlist']:
        ii = torch.tensor([i for i, (p, q) in enumerate(pairs) if p['family'] == fam])
        perfam[fam] = {'n': len(ii), 'follow': (pr[ii] == follow[ii]).float().mean().item(),
                       'stay': (pr[ii] == clean_ans[ii]).float().mean().item()}
    res['per_family'] = perfam
    print(f'{name}: follow={res["follow_rate"]:.3f} stay={res["stay_rate"]:.3f} '
          + ' '.join(f'{f}:{v["follow"]:.2f}' for f, v in perfam.items()), flush=True)
    return res

site = {}
site['baseline'] = eval_cond('baseline (no swap)')
site['A_h3'] = eval_cond('site A, H3 only',
                         head_sub={(L_PAY, 3): (donor_head[3], pred_pos)})
site['A_h3h7'] = eval_cond('site A, H3+H7',
                           head_sub={(L_PAY, h): (donor_head[h], pred_pos) for h in HEADS})
site['A_all9'] = eval_cond('site A, all 9 heads',
                           head_sub={(L_PAY, h): (donor_head[h], pred_pos) for h in range(9)})
site['B_v1_h3h7'] = eval_cond('site B, v1@last L8 H3+H7',
                              v1_sub={(L_PAY, h): (v1_all[:, h], last_pos) for h in HEADS})
site['B_v1_all9'] = eval_cond('site B, v1@last L8 all9',
                              v1_sub={(L_PAY, h): (v1_all[:, h], last_pos) for h in range(9)})

save_json('s1_calibrate.json', {
    'n_cal': len(cal), 'baseline_acc': acc, 'pattern_weight': pw,
    'ridge': {'full': {'lam': lam_f, 'split_r2': r2_f},
              'holdout': {'lam': lam_h, 'split_r2': r2_h},
              'per_family_fit': fitq},
    'holdout_elems': sorted(HOLDOUT_ELEMS),
    'site_test': site})
print('done', flush=True)
