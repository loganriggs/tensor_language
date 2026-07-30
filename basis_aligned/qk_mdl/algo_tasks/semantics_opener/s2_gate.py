"""Step 2: MEANING GATE by substitution.

A. Interchange-style gate on the bracket battery (corrupted run, patch the
   channel at the FINAL position only):
     exact   a <- q^T x_clean_final          (reference; s0: 0.56 / 0.87)
     coded   a <- beta . feats(CLEAN token stream at final pos)   [pure code]
     codedX  a <- beta . feats(CORR stream)  (wrong-state placebo)
     zero    a <- 0                          (deletion control)
     mean    a <- cooc mean activation       (neutral deletion)
     shuf    a <- beta . feats(random cooc position)  (shuffled placebo, 10 draws)
   Metric: recovery of the clean-corr closer logit gap; held-20 is final.

B. Full-stream substitution on natural text (audit rows 448:600, HELD-BACK):
   a <- coded / zero / mean / within-seq-shuffled coded, at ALL positions.
   Metrics: paired per-token dCE with SE (token + seq-clustered), top-1
   agreement. Also the battery boost under full-stream substitution.
   Saves per-token CE + closer logprobs (base, zero) for steps 3-4.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_opener')
from common import (OUT, BRACKET, get_model, forward_hooked, coded_state,
                    coded_states, cooc, fineweb_audit, paired_dce, safe,
                    collect_activations, FEATS, BATCH)

DEV = 'cuda'
m, cfg = get_model()
q1 = torch.load(f'{OUT}/Q_r1.pt').to(DEV)   # (D,1)
Q4 = torch.load(f'{OUT}/Q_r4.pt').to(DEV)   # (D,4)
calib = json.load(open(f'{OUT}/calib.json'))
b1 = np.array(calib['a1_beta'])             # (6,)
W4 = np.array(calib['a4_beta'])             # (4,6)
S = json.load(open(f'{BRACKET}/stimuli.json'))
CLOSER = {t: S['summary'][t]['closer_id'] for t in ['paren', 'quote']}
rng = np.random.RandomState(0)

# cooc reference stats for mean / shuffled controls (fit data only)
cooc_fit = cooc((0, 240))
acts_ref = collect_activations(m, cfg, cooc_fit[:60], torch.cat([q1, Q4], 1))
mean_a1 = float(acts_ref[..., 0].mean())
mean_a4 = acts_ref[..., 1:].reshape(-1, 4).mean(0)
states_ref = coded_states(cooc_fit[:60]).reshape(-1, 5)
print(f'cooc ref: mean_a1 {mean_a1:.1f}, mean_a4 {np.round(mean_a4,1)}', flush=True)


def feats_aug(st5):
    return np.concatenate([st5.astype(np.float64), [1.0]])


def ahat1(st5):
    return float(b1 @ feats_aug(st5))


def ahat4(st5):
    return W4 @ feats_aug(st5)


def pad_batch(ids_list):
    T = max(len(c) for c in ids_list)
    idx = torch.full((len(ids_list), T), 50256, dtype=torch.long)
    for j, c in enumerate(ids_list):
        idx[j, :len(c)] = torch.tensor(c)
    return idx.to(DEV)


def final_patch_hook(Q, fins, values):
    """set channel at each row's final position to values (B,r) tensor."""
    def h(x):
        ar = torch.arange(x.shape[0], device=x.device)
        cur = x[ar, fins]                       # (B,D)
        a = cur @ Q                             # (B,r)
        x = x.clone()
        x[ar, fins] = cur + (values.to(x.dtype) - a) @ Q.T
        return x
    return h


@torch.no_grad()
def closer_lp_final(idx, fins, cid, hook=None, raw=False):
    """raw=True: closer LOGIT (the prior program's recovery metric);
    raw=False: log-softmax logprob (the boost metric of s1_stimuli)."""
    lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
    lp = lg if raw else F.log_softmax(lg, -1)
    ar = torch.arange(idx.shape[0], device=DEV)
    return lp[ar, fins, cid].cpu().numpy()


# =============================== A: interchange gate =========================
gateA = {}
for rname, Q, ah, meanv in [('r1', q1, lambda s: np.array([ahat1(s)]), np.array([mean_a1])),
                            ('r4', Q4, ahat4, mean_a4)]:
    per_pair = {c: [] for c in ['exact', 'coded', 'codedX', 'zero', 'mean', 'shuf']}
    tags = []   # (task, is_held)
    diag = {t: {'exact_a': [], 'coded_a': [], 'corr_a': []} for t in ['paren', 'quote']}
    for task in ['paren', 'quote']:
        pairs = S['pairs'][task]
        cid = CLOSER[task]
        for i0 in range(0, len(pairs), BATCH):
            chunk = pairs[i0:i0 + BATCH]
            B = len(chunk)
            idx_c = pad_batch([p['clean_ids'] for p in chunk])
            idx_x = pad_batch([p['corr_ids'] for p in chunk])
            fin_c = torch.tensor([len(p['clean_ids']) - 1 for p in chunk], device=DEV)
            fin_x = torch.tensor([len(p['corr_ids']) - 1 for p in chunk], device=DEV)
            lc = closer_lp_final(idx_c, fin_c, cid, raw=True)
            lx = closer_lp_final(idx_x, fin_x, cid, raw=True)
            # exact clean channel value at clean final pos
            x13c = safe(forward_hooked, m, cfg, idx_c, stop_at_L=True)
            ar = torch.arange(B, device=DEV)
            exact_v = (x13c[ar, fin_c] @ Q).float()          # (B,r)
            del x13c
            x13x = safe(forward_hooked, m, cfg, idx_x, stop_at_L=True)
            corr_v = (x13x[ar, fin_x] @ Q).float()
            del x13x
            st_clean = np.stack([coded_state(np.array(p['clean_ids']))[-1] for p in chunk])
            st_corr = np.stack([coded_state(np.array(p['corr_ids']))[-1] for p in chunk])
            vals = {
                'exact': exact_v,
                'coded': torch.tensor(np.stack([ah(s) for s in st_clean]), device=DEV).float(),
                'codedX': torch.tensor(np.stack([ah(s) for s in st_corr]), device=DEV).float(),
                'zero': torch.zeros(B, Q.shape[1], device=DEV),
                'mean': torch.tensor(np.tile(meanv, (B, 1)), device=DEV).float(),
            }
            diag[task]['exact_a'].extend(exact_v[:, 0].cpu().tolist())
            diag[task]['coded_a'].extend(vals['coded'][:, 0].cpu().tolist())
            diag[task]['corr_a'].extend(corr_v[:, 0].cpu().tolist())
            for cname, v in vals.items():
                lp = closer_lp_final(idx_x, fin_x, cid, raw=True,
                                     hook=final_patch_hook(Q, fin_x, v))
                per_pair[cname].extend(((lp - lx) / (lc - lx)).tolist())
            # shuffled: 10 random cooc states per pair, mean recovery
            recs = np.zeros(B)
            ND = 10
            for d in range(ND):
                sts = states_ref[rng.randint(0, len(states_ref), B)]
                v = torch.tensor(np.stack([ah(s) for s in sts]), device=DEV).float()
                lp = closer_lp_final(idx_x, fin_x, cid, raw=True,
                                     hook=final_patch_hook(Q, fin_x, v))
                recs += (lp - lx) / (lc - lx)
            per_pair['shuf'].extend((recs / ND).tolist())
            tags.extend([(task, i0 + j >= 30) for j in range(B)])
        print(f'gate A {rname} {task} done', flush=True)
    gateA[f'{rname}_value_diag'] = {
        t: {k: round(float(np.mean(v)), 1) for k, v in diag[t].items()}
        for t in diag}
    print(f'  {rname} value diag: {gateA[f"{rname}_value_diag"]}', flush=True)
    out = {}
    for cname, rs in per_pair.items():
        rs = np.array(rs)
        held = np.array([h for _, h in tags])
        tk = np.array([t for t, _ in tags])
        out[cname] = {
            'held_recovery': round(float(rs[held].mean()), 4),
            'held_se': round(float(rs[held].std(ddof=1) / np.sqrt(held.sum())), 4),
            'held_paren': round(float(rs[held & (tk == 'paren')].mean()), 4),
            'held_quote': round(float(rs[held & (tk == 'quote')].mean()), 4),
            'analysis_recovery': round(float(rs[~held].mean()), 4)}
        print(f'  {rname} {cname}: {out[cname]}', flush=True)
    gateA[rname] = out

# =============================== B: full-stream substitution =================
import os
if os.environ.get('GATEA_ONLY'):
    old = json.load(open(f'{OUT}/gate.json')) if os.path.exists(f'{OUT}/gate.json') else {}
    old['gateA_interchange'] = gateA
    old['gateA_metric'] = 'raw closer logit recovery (matches bracket/das.json)'
    json.dump(old, open(f'{OUT}/gate.json', 'w'), indent=1)
    print('S2 GATEA-ONLY DONE', flush=True)
    sys.exit(0)

audit = fineweb_audit()
st_audit = coded_states(audit)                                   # (N,T,5)
aug = np.concatenate([st_audit.astype(np.float64),
                      np.ones(st_audit.shape[:2] + (1,))], -1)   # (N,T,6)
coded1 = (aug @ b1)[..., None]                                   # (N,T,1)
coded4 = aug @ W4.T                                              # (N,T,4)
# within-sequence shuffle of the coded values (fixed seed)
perm_rng = np.random.RandomState(1)
shuf1 = np.stack([c[perm_rng.permutation(c.shape[0])] for c in coded1])
shuf4 = np.stack([c[perm_rng.permutation(c.shape[0])] for c in coded4])

CLOSER_IDS = [8, 1]   # ')' and '"'


@torch.no_grad()
def eval_pass(rows, Q=None, coded=None, batch=BATCH, base_argmax=None):
    """one pass: per-token CE (N,T-1), closer logprobs (N,T,2), argmax (N,T)
    or (if base_argmax given) agreement fraction."""
    ces, lps, ams = [], [], []
    for i in range(0, len(rows), batch):
        idx = torch.from_numpy(np.ascontiguousarray(rows[i:i + batch]).astype(np.int64)).to(DEV)
        hook = None
        if Q is not None:
            cv = torch.from_numpy(coded[i:i + batch]).to(DEV)
            def hook(x, cv=cv, Q=Q):
                a = torch.einsum('btd,dr->btr', x, Q)
                return x + torch.einsum('btr,dr->btd', cv.to(x.dtype) - a, Q)
        lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
        ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                             idx[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.view(idx.shape[0], -1).cpu().numpy())
        lp = F.log_softmax(lg, -1)
        lps.append(lp[:, :, CLOSER_IDS].cpu().numpy())
        ams.append(lg.argmax(-1).cpu().numpy())
        del lg, lp
    am = np.concatenate(ams)
    agree = None if base_argmax is None else float((am == base_argmax).mean())
    return np.concatenate(ces), np.concatenate(lps), am, agree


gateB = {}
N, T = audit.shape
ce_base, lp_base, base_argmax, _ = eval_pass(audit)
np.save(f'{OUT}/audit_ce_base.npy', ce_base)
np.save(f'{OUT}/audit_lp_base.npy', lp_base)

conds = {
    'r1_coded': (q1, coded1),
    'r1_zero': (q1, np.zeros_like(coded1)),
    'r1_mean': (q1, np.full_like(coded1, mean_a1)),
    'r1_shuf': (q1, shuf1),
    'r4_coded': (Q4, coded4),
    'r4_zero': (Q4, np.zeros_like(coded4)),
    'r4_mean': (Q4, np.tile(mean_a4, (N, T, 1))),
    'r4_shuf': (Q4, shuf4),
}
for cname, (Q, cv) in conds.items():
    ce, lp, _, agree = eval_pass(audit, Q, cv, base_argmax=base_argmax)
    d = paired_dce(ce, ce_base)
    d['top1_agree'] = round(agree, 4)
    gateB[cname] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}
    if cname.endswith('zero'):
        np.save(f'{OUT}/audit_ce_{cname}.npy', ce)
        np.save(f'{OUT}/audit_lp_{cname}.npy', lp)
    print(f'gate B {cname}: {gateB[cname]}', flush=True)

# battery boost under FULL-STREAM substitution (does the code preserve the task?)
gateB_batt = {}
for rname, Q, ah, meanv in [('r1', q1, lambda s: np.array([ahat1(s)]), np.array([mean_a1])),
                            ('r4', Q4, ahat4, mean_a4)]:
    boosts = {c: [] for c in ['base', 'coded', 'zero', 'mean', 'shuf']}
    for task in ['paren', 'quote']:
        pairs = S['pairs'][task]
        cid = CLOSER[task]
        for i0 in range(0, len(pairs), BATCH):
            chunk = pairs[i0:i0 + BATCH]
            for cname in boosts:
                lps = {}
                for kind in ['clean', 'corr']:
                    ids_l = [p[f'{kind}_ids'] for p in chunk]
                    idx = pad_batch(ids_l)
                    fins = torch.tensor([len(c) - 1 for c in ids_l], device=DEV)
                    if cname == 'base':
                        hook = None
                    else:
                        Tb = idx.shape[1]
                        sts = np.stack([np.concatenate([
                            coded_state(np.array(c)),
                            np.tile(coded_state(np.array(c))[-1], (Tb - len(c), 1))])
                            for c in ids_l])
                        augb = np.concatenate([sts.astype(np.float64),
                                               np.ones(sts.shape[:2] + (1,))], -1)
                        if cname == 'coded':
                            cv = augb @ (np.array([b1]).T if Q is q1 else W4.T)
                        elif cname == 'zero':
                            cv = np.zeros(sts.shape[:2] + (Q.shape[1],))
                        elif cname == 'mean':
                            cv = np.tile(meanv, sts.shape[:2] + (1,))
                        elif cname == 'shuf':
                            pr = np.random.RandomState(2)
                            cvv = augb @ (np.array([b1]).T if Q is q1 else W4.T)
                            cv = np.stack([c[pr.permutation(c.shape[0])] for c in cvv])
                        cvt = torch.from_numpy(cv).to(DEV)
                        def hook(x, cvt=cvt, Q=Q):
                            a = torch.einsum('btd,dr->btr', x, Q)
                            return x + torch.einsum('btr,dr->btd', cvt.to(x.dtype) - a, Q)
                    lps[kind] = closer_lp_final(idx, fins, cid, hook=hook)
                boosts[cname].extend((lps['clean'] - lps['corr']).tolist())
    out = {}
    for cname, bs in boosts.items():
        bs = np.array(bs)
        held = np.array([i % 40 >= 30 for i in range(80)])
        out[cname] = {'held_boost': round(float(bs[held].mean()), 3),
                      'held_se': round(float(bs[held].std(ddof=1) / np.sqrt(held.sum())), 3),
                      'held_paren': round(float(bs[held][:10].mean()), 3),
                      'held_quote': round(float(bs[held][10:].mean()), 3),
                      'all_boost': round(float(bs.mean()), 3)}
        print(f'battery {rname} {cname}: {out[cname]}', flush=True)
    gateB_batt[rname] = out

json.dump({'gateA_interchange': gateA, 'gateB_natural': gateB,
           'gateB_battery_fullstream': gateB_batt,
           'cooc_mean_a1': mean_a1, 'cooc_mean_a4': mean_a4.tolist()},
          open(f'{OUT}/gate.json', 'w'), indent=1)
print('S2 DONE', flush=True)
