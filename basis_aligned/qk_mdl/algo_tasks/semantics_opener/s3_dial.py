"""Step 3: DIAL. Scale the r=1 channel activation by s in {0,0.5,1,1.5,2}
(x <- x + (s-1) q q^T x at ALL positions, layer-13 entry).
Dose-response of: battery closer-boost (held-20 final), natural-text closer
logprob at open vs closed positions, and natural dCE (audit, paired, SE).
Note channel polarity: LOW activation = opener pending, so s>1 amplifies
whatever state is present (more negative when open, more positive when closed).
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_opener')
from common import (OUT, BRACKET, get_model, forward_hooked, scale_hook,
                    coded_states, derived, fineweb_audit, paired_dce, safe, BATCH)

DEV = 'cuda'
m, cfg = get_model()
q1 = torch.load(f'{OUT}/Q_r1.pt').to(DEV)
S = json.load(open(f'{BRACKET}/stimuli.json'))
CLOSER = {t: S['summary'][t]['closer_id'] for t in ['paren', 'quote']}
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]
CLOSER_IDS = [8, 1]


def pad_batch(ids_list):
    T = max(len(c) for c in ids_list)
    idx = torch.full((len(ids_list), T), 50256, dtype=torch.long)
    for j, c in enumerate(ids_list):
        idx[j, :len(c)] = torch.tensor(c)
    return idx.to(DEV)


@torch.no_grad()
def lp_final(idx, fins, cid, hook=None):
    lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
    lp = F.log_softmax(lg, -1)
    ar = torch.arange(idx.shape[0], device=DEV)
    return lp[ar, fins, cid].cpu().numpy()


res = {'scales': SCALES}

# ---- battery dose-response ----
batt = {}
for s in SCALES:
    hook = None if s == 1.0 else scale_hook(q1, s)
    boosts = []
    for task in ['paren', 'quote']:
        pairs = S['pairs'][task]
        cid = CLOSER[task]
        for i0 in range(0, len(pairs), BATCH):
            chunk = pairs[i0:i0 + BATCH]
            vals = {}
            for kind in ['clean', 'corr']:
                ids_l = [p[f'{kind}_ids'] for p in chunk]
                idx = pad_batch(ids_l)
                fins = torch.tensor([len(c) - 1 for c in ids_l], device=DEV)
                vals[kind] = lp_final(idx, fins, cid, hook=hook)
            boosts.extend((vals['clean'] - vals['corr']).tolist())
    bs = np.array(boosts)
    held = np.array([i % 40 >= 30 for i in range(80)])
    batt[str(s)] = {'held_boost': round(float(bs[held].mean()), 3),
                    'held_se': round(float(bs[held].std(ddof=1) / np.sqrt(held.sum())), 3),
                    'held_paren': round(float(bs[held][:10].mean()), 3),
                    'held_quote': round(float(bs[held][10:].mean()), 3)}
    print(f'battery s={s}: {batt[str(s)]}', flush=True)
res['battery'] = batt

# ---- audit: dCE + natural closer-logprob separation ----
audit = fineweb_audit()
st = derived(coded_states(audit))
p_open = st['p_depth'][:, :] > 0
q_open = st['q_any'][:, :] > 0
ce_base = np.load(f'{OUT}/audit_ce_base.npy')
lp_base = np.load(f'{OUT}/audit_lp_base.npy')


@torch.no_grad()
def audit_pass(hook):
    ces, lps = [], []
    for i in range(0, len(audit), BATCH):
        idx = torch.from_numpy(audit[i:i + BATCH].astype(np.int64)).to(DEV)
        lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
        ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                             idx[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.view(idx.shape[0], -1).cpu().numpy())
        lps.append(F.log_softmax(lg, -1)[:, :, CLOSER_IDS].cpu().numpy())
        del lg
    return np.concatenate(ces), np.concatenate(lps)


def closer_sep(lp):
    """natural-text closer-anticipation: mean lp(')') at paren-open minus
    paren-closed positions; same for '"' with quote state."""
    return {
        'lp_paren_open': round(float(lp[..., 0][p_open].mean()), 3),
        'lp_paren_closed': round(float(lp[..., 0][~p_open].mean()), 3),
        'sep_paren': round(float(lp[..., 0][p_open].mean() - lp[..., 0][~p_open].mean()), 3),
        'lp_quote_open': round(float(lp[..., 1][q_open].mean()), 3),
        'lp_quote_closed': round(float(lp[..., 1][~q_open].mean()), 3),
        'sep_quote': round(float(lp[..., 1][q_open].mean() - lp[..., 1][~q_open].mean()), 3)}


aud = {}
for s in SCALES:
    if s == 1.0:
        ce, lp = ce_base, lp_base
    elif s == 0.0:
        ce = np.load(f'{OUT}/audit_ce_r1_zero.npy')
        lp = np.load(f'{OUT}/audit_lp_r1_zero.npy')
    else:
        ce, lp = audit_pass(scale_hook(q1, s))
    d = paired_dce(ce, ce_base) if s != 1.0 else {'dce': 0.0, 'se_token': 0.0,
                                                  'se_seq': 0.0, 'n_tokens': int(ce.size)}
    d = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}
    d.update(closer_sep(lp))
    aud[str(s)] = d
    print(f'audit s={s}: {d}', flush=True)
res['audit'] = aud

json.dump(res, open(f'{OUT}/dial.json', 'w'), indent=1)
print('S3 DONE', flush=True)
