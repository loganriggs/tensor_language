"""Step 5: RED-TEAM the name "channel activation = f(number of unclosed
openers)".

 A. Depth series: 0-3 stacked '( ' openers — is a1 linear in depth, saturating,
    or binary? Does the model's ')' boost track depth?
 B. Opener-type series: ( [ { " — does the SAME 1-dim channel move for all
    types? Is the channel-mediated share of each type's closer boost similar?
 C. Distance stress: opener..final distance 3-128 tokens — activation and
    boost decay.
 D. Closed-long-ago: '( ... )' + k continuation tokens vs never-opened — does
    the channel reset?
 E. Cross-substitution: inject quote-coded value into parenless context and
    vice versa; does the channel discriminate WHICH closer?
 F. Natural-text attenuation: a1 | open by distance-since-opener on cooc;
    distance-aware recalibrated injection (coded_dist) on the battery.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/semantics_opener')
from common import (OUT, BRACKET, get_model, forward_hooked, coded_state,
                    coded_states, derived, cooc, safe, get_tok, BATCH)

DEV = 'cuda'
m, cfg = get_model()
q1 = torch.load(f'{OUT}/Q_r1.pt').to(DEV)
Q4 = torch.load(f'{OUT}/Q_r4.pt').to(DEV)
calib = json.load(open(f'{OUT}/calib.json'))
b1 = np.array(calib['a1_beta'])
tok = get_tok()
S = json.load(open(f'{BRACKET}/stimuli.json'))
rng = np.random.RandomState(11)
NOUNS = ['dogs', 'cats', 'birds', 'cars', 'trees', 'books', 'stars', 'rivers',
         'houses', 'chairs', 'apples', 'stones', 'clouds', 'roads', 'lamps',
         'boats', 'fields', 'doors', 'horses', 'flowers', 'windows', 'tables']
CID = {')': tok(')')['input_ids'][0], ']': tok(']')['input_ids'][0],
       '}': tok('}')['input_ids'][0], '"': tok('"')['input_ids'][0]}
print('closer ids:', CID, flush=True)
res = {}


def pad_batch(ids_list):
    T = max(len(c) for c in ids_list)
    idx = torch.full((len(ids_list), T), 50256, dtype=torch.long)
    for j, c in enumerate(ids_list):
        idx[j, :len(c)] = torch.tensor(c)
    return idx.to(DEV)


@torch.no_grad()
def probe(ids_list, cids, hook=None):
    """returns (a1_final (n,), lp[cid]_final (n,len(cids)))"""
    a1s, lps = [], []
    for i in range(0, len(ids_list), BATCH):
        chunk = ids_list[i:i + BATCH]
        idx = pad_batch(chunk)
        fins = torch.tensor([len(c) - 1 for c in chunk], device=DEV)
        ar = torch.arange(len(chunk), device=DEV)
        x13 = safe(forward_hooked, m, cfg, idx, stop_at_L=True)
        a1s.append((x13[ar, fins] @ q1)[:, 0].float().cpu().numpy())
        del x13
        lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
        lp = F.log_softmax(lg, -1)
        lps.append(lp[ar, fins][:, cids].cpu().numpy())
        del lg
    return np.concatenate(a1s), np.concatenate(lps)


def mk(pre, opener, body):
    return tok(f'{pre} {opener}{body}' if opener else f'{pre} {body}')['input_ids']


def carriers(n=12):
    outs = []
    for i in range(n):
        w = rng.choice(NOUNS, 4, replace=False)
        pre = ['The', 'Yesterday the', 'My friend saw the'][i % 3] + f' {w[0]}'
        body = f'which was near the {w[1]} and the {w[2]}'
        outs.append((pre, body))
    return outs


# ============================ A: depth series ================================
cars = carriers(12)
depthA = {}
a1_by_d, lp_by_d = {}, {}
for d in range(4):
    ids_list = [mk(pre, '( ' * d, body) for pre, body in cars]
    a1, lp = probe(ids_list, [CID[')']])
    a1_by_d[d], lp_by_d[d] = a1, lp[:, 0]
for d in range(4):
    depthA[str(d)] = {
        'a1_mean': round(float(a1_by_d[d].mean()), 1),
        'a1_se': round(float(a1_by_d[d].std(ddof=1) / np.sqrt(12)), 1),
        'lp_closer': round(float(lp_by_d[d].mean()), 3),
        'boost_vs_d0': round(float((lp_by_d[d] - lp_by_d[0]).mean()), 3),
        'boost_se': round(float((lp_by_d[d] - lp_by_d[0]).std(ddof=1) / np.sqrt(12)), 3)}
    print(f'depth {d}: {depthA[str(d)]}', flush=True)
res['A_depth'] = depthA

# ============================ B: opener types ================================
typeB = {}
ctrl_ids = [mk(pre, '', body) for pre, body in cars]
for opener, closer in [('( ', ')'), ('[ ', ']'), ('{ ', '}'), ('" ', '"')]:
    ids_list = [mk(pre, opener, body) for pre, body in cars]
    a1o, lpo = probe(ids_list, [CID[closer]])
    a1c, lpc = probe(ctrl_ids, [CID[closer]])
    # channel-zeroed boost (channel-mediated share)
    from common import sub_hook
    _, lpo_z = probe(ids_list, [CID[closer]], hook=sub_hook(q1, None))
    _, lpc_z = probe(ctrl_ids, [CID[closer]], hook=sub_hook(q1, None))
    boost = (lpo[:, 0] - lpc[:, 0])
    boost_z = (lpo_z[:, 0] - lpc_z[:, 0])
    typeB[opener.strip()] = {
        'a1_open': round(float(a1o.mean()), 1), 'a1_ctrl': round(float(a1c.mean()), 1),
        'a1_drop': round(float((a1c - a1o).mean()), 1),
        'boost': round(float(boost.mean()), 3),
        'boost_se': round(float(boost.std(ddof=1) / np.sqrt(12)), 3),
        'boost_channel_zeroed': round(float(boost_z.mean()), 3),
        'channel_share': round(float(1 - boost_z.mean() / boost.mean()), 3) if boost.mean() else None}
    print(f'type {opener.strip()}: {typeB[opener.strip()]}', flush=True)
res['B_types'] = typeB

# ============================ C: distance stress =============================
distC = {}
for nf in [3, 8, 16, 32, 64, 128]:
    ids_c, ids_x = [], []
    for i in range(10):
        w = rng.choice(NOUNS, len(NOUNS), replace=False)
        pre = f'The {w[0]}'
        # build filler clause of ~nf tokens: 'which was near the X and the Y ...'
        body = 'which was near the ' + w[1]
        j = 2
        while len(tok(body)['input_ids']) < nf:
            body += f' and the {w[j % len(w)]}'
            j += 1
        ids_c.append(tok(f'{pre} ( {body}')['input_ids'])
        ids_x.append(tok(f'{pre} {body}')['input_ids'])
    a1c, lpc = probe(ids_c, [CID[')']])
    a1x, lpx = probe(ids_x, [CID[')']])
    boost = lpc[:, 0] - lpx[:, 0]
    dist = [len(c) - 1 - coded_state(np.array(c))[:, 0].argmax() for c in ids_c]
    distC[str(nf)] = {
        'mean_opener_to_final_tokens': round(float(np.mean(dist)), 1),
        'a1_open': round(float(a1c.mean()), 1), 'a1_ctrl': round(float(a1x.mean()), 1),
        'a1_drop': round(float((a1x - a1c).mean()), 1),
        'boost': round(float(boost.mean()), 3),
        'boost_se': round(float(boost.std(ddof=1) / np.sqrt(10)), 3)}
    print(f'dist nf={nf}: {distC[str(nf)]}', flush=True)
res['C_distance'] = distC

# ============================ D: closed long ago =============================
closedD = {}
for k in [2, 8, 24]:
    ids_cl, ids_nv = [], []
    for i in range(10):
        w = rng.choice(NOUNS, 8, replace=False)
        pre = f'The {w[0]}'
        clause = f'which was near the {w[1]}'
        cont = 'ran to the ' + w[2]
        j = 3
        while len(tok(cont)['input_ids']) < k:
            cont += f' and the {w[j % 8]}'
            j += 1
        ids_cl.append(tok(f'{pre} ( {clause} ) {cont}')['input_ids'])
        ids_nv.append(tok(f'{pre} {clause} {cont}')['input_ids'])
    a1cl, lpcl = probe(ids_cl, [CID[')']])
    a1nv, lpnv = probe(ids_nv, [CID[')']])
    closedD[str(k)] = {
        'a1_closed_long_ago': round(float(a1cl.mean()), 1),
        'a1_never_opened': round(float(a1nv.mean()), 1),
        'a1_residual': round(float((a1nv - a1cl).mean()), 1),
        'lp_residual_boost': round(float((lpcl[:, 0] - lpnv[:, 0]).mean()), 3),
        'lp_residual_se': round(float((lpcl[:, 0] - lpnv[:, 0]).std(ddof=1) / np.sqrt(10)), 3)}
    print(f'closed k={k}: {closedD[str(k)]}', flush=True)
res['D_closed_long_ago'] = closedD

# ============================ E: cross-substitution ==========================
# corrupted battery pairs (held 10 each); inject SAME-type vs CROSS-type coded
# value at the final position; measure recovery of ')' and effect on '"'.
def feats_aug(st5):
    return np.concatenate([st5.astype(np.float64), [1.0]])


A_PAREN = float(b1 @ feats_aug(np.array([1, 0, 0, 0, 0])))
A_QUOTE = float(b1 @ feats_aug(np.array([0, 0, 0, 1, 0])))
A_CLOSED = float(b1 @ feats_aug(np.zeros(5)))
print(f'coded values: paren-open {A_PAREN:.0f}, quote-open {A_QUOTE:.0f}, '
      f'closed {A_CLOSED:.0f}', flush=True)

crossE = {}
for task, own_closer, other_closer in [('paren', ')', '"'), ('quote', '"', ')')]:
    pairs = S['pairs'][task][30:]
    cid_own, cid_oth = CID[own_closer], CID[other_closer]
    ids_c = [p['clean_ids'] for p in pairs]
    ids_x = [p['corr_ids'] for p in pairs]
    _, lp_c = probe(ids_c, [cid_own, cid_oth])
    _, lp_x = probe(ids_x, [cid_own, cid_oth])
    out = {}
    for vname, aval in [('same_coded', A_PAREN if task == 'paren' else A_QUOTE),
                        ('cross_coded', A_QUOTE if task == 'paren' else A_PAREN)]:
        lps = []
        for i in range(0, len(ids_x), BATCH):
            chunk = ids_x[i:i + BATCH]
            idx = pad_batch(chunk)
            fins = torch.tensor([len(c) - 1 for c in chunk], device=DEV)
            v = torch.full((len(chunk), 1), aval, device=DEV)
            def hook(x, fins=fins, v=v):
                ar = torch.arange(x.shape[0], device=x.device)
                cur = x[ar, fins]
                a = cur @ q1
                x = x.clone()
                x[ar, fins] = cur + (v.to(x.dtype) - a) @ q1.T
                return x
            lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
            lp = F.log_softmax(lg, -1)
            ar = torch.arange(len(chunk), device=DEV)
            lps.append(lp[ar, fins][:, [cid_own, cid_oth]].cpu().numpy())
        lps = np.concatenate(lps)
        out[vname] = {
            'own_closer_recovery': round(float(((lps[:, 0] - lp_x[:, 0])
                                                / (lp_c[:, 0] - lp_x[:, 0])).mean()), 4),
            'own_closer_dlp': round(float((lps[:, 0] - lp_x[:, 0]).mean()), 3),
            'other_closer_dlp': round(float((lps[:, 1] - lp_x[:, 1]).mean()), 3)}
        print(f'cross {task} {vname}: {out[vname]}', flush=True)
    crossE[task] = out
res['E_cross_substitution'] = crossE

# ============================ F: natural attenuation + coded_dist ============
rows = cooc((0, 120))
sts = coded_states(rows)
a1_nat = []
for i in range(0, len(rows), BATCH):
    idx = torch.from_numpy(rows[i:i + BATCH].astype(np.int64)).to(DEV)
    x13 = safe(forward_hooked, m, cfg, idx, stop_at_L=True)
    a1_nat.append((x13 @ q1)[..., 0].float().cpu().numpy())
    del x13
a1_nat = np.concatenate(a1_nat)
# distance since most recent paren-opener for p_depth>0 positions
p = sts[..., 0]
attF = {}
dists = np.full(p.shape, -1)
for n in range(p.shape[0]):
    last_open = -1
    for t in range(p.shape[1]):
        if p[n, t] > 0 and (t == 0 or p[n, t] > p[n, t - 1]):
            last_open = t
        if p[n, t] > 0 and last_open >= 0:
            dists[n, t] = t - last_open
bins = [(0, 2), (3, 6), (7, 12), (13, 24), (25, 60), (61, 512)]
for lo, hi in bins:
    sel = (dists >= lo) & (dists <= hi)
    if sel.sum() >= 30:
        attF[f'{lo}-{hi}'] = {'mean_a1': round(float(a1_nat[sel].mean()), 1),
                              'se': round(float(a1_nat[sel].std(ddof=1) / np.sqrt(sel.sum())), 1),
                              'n': int(sel.sum())}
attF['closed_baseline'] = {'mean_a1': round(float(a1_nat[p == 0].mean()), 1),
                           'n': int((p == 0).sum())}
res['F_natural_attenuation'] = attF
print('natural attenuation:', attF, flush=True)

# coded_dist: recalibrate the injected "open" value on cooc positions with
# distance <= 12 (battery-like recency), pure code+cooc; rerun gate injection
sel = (dists >= 0) & (dists <= 12)
A_OPEN_RECENT = float(a1_nat[sel].mean())
res['coded_dist_value'] = round(A_OPEN_RECENT, 1)
print(f'coded_dist open value: {A_OPEN_RECENT:.0f}', flush=True)
gate_dist = {}
for task in ['paren', 'quote']:
    pairs = S['pairs'][task][30:]
    cid = S['summary'][task]['closer_id']
    ids_c = [p['clean_ids'] for p in pairs]
    ids_x = [p['corr_ids'] for p in pairs]


    @torch.no_grad()
    def logit_final(ids_list, hook=None):
        outs = []
        for i in range(0, len(ids_list), BATCH):
            chunk = ids_list[i:i + BATCH]
            idx = pad_batch(chunk)
            fins = torch.tensor([len(c) - 1 for c in chunk], device=DEV)
            lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
            ar = torch.arange(len(chunk), device=DEV)
            outs.append(lg[ar, fins, cid].cpu().numpy())
        return np.concatenate(outs)

    lc = logit_final(ids_c)
    lx = logit_final(ids_x)
    recs = {}
    for vname, aval in [('coded_dist', A_OPEN_RECENT)]:
        lps = []
        for i in range(0, len(ids_x), BATCH):
            chunk = ids_x[i:i + BATCH]
            idx = pad_batch(chunk)
            fins = torch.tensor([len(c) - 1 for c in chunk], device=DEV)
            v = torch.full((len(chunk), 1), aval, device=DEV)
            def hook(x, fins=fins, v=v):
                ar = torch.arange(x.shape[0], device=x.device)
                cur = x[ar, fins]
                a = cur @ q1
                x = x.clone()
                x[ar, fins] = cur + (v.to(x.dtype) - a) @ q1.T
                return x
            lg = safe(forward_hooked, m, cfg, idx, hook=hook).float()
            ar = torch.arange(len(chunk), device=DEV)
            lps.append(lg[ar, fins, cid].cpu().numpy())
        lps = np.concatenate(lps)
        recs[vname] = round(float(((lps - lx) / (lc - lx)).mean()), 4)
    gate_dist[task] = recs
    print(f'coded_dist gate {task}: {recs}', flush=True)
res['F_coded_dist_gate'] = gate_dist

json.dump(res, open(f'{OUT}/redteam.json', 'w'), indent=1)
print('S5 DONE', flush=True)
