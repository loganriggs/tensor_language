# THE REALISED-ATTENTION ROUTE: does what a prev head ACTUALLY ATTENDED TO predict leaf membership?
#
# BACKLOG rung 8, the design decision §2095 recorded rather than started. §2094 refuted §332's
# composition in its literal form (previous-token IDENTITY -> membership: held-out AUC 0.5086 on the 31
# leaves whose both top-2 heads are prev) and §2095 refuted the directional form (previous-token
# EMBEDDING ridge: 0.5052, below the identity lookup, with a shuffled-label control at 0.5006 showing
# the ridge had nothing to find). Both sections closed with the same scope line: every feature tested
# was a LOCAL TOKEN feature, and a mechanism over the head's REALISED ATTENTION PATTERN -- which
# positions it attended, how much, and how concentrated -- was untested. §1108 is the reason that is
# not a rephrasing: bilin18's attention is unnormalised, so per-query row MASS is a real degree of
# freedom (it carried +0.25 of the gatherer band's recovery there), and a prev head's pattern at a
# position is not determined by which token sits at t-1.
#
# METHOD. Same 31 treatment leaves, same held-out row split (rows < 500 fit, >= 500 score), same AUC as
# §2094/§2095. For each leaf's two top-2 heads (both prev-class, at the leaf's own attention layer) the
# realised pattern row p(t, .) = (q.k/D)(q2.k2/D) over j <= t is captured from a forward pass of the
# census rows and summarised per position by 12 features per head: signed mass at offsets 0, 1, 2, 3,
# 4-7, 8-15, 16-31, 32-63, >=64; total signed mass; total absolute mass; entropy of |p|/sum|p|. A ridge
# from the 24 standardised features to membership is fitted on the fit half and scored on the other.
# The capture is verified against the model's own attention output on the first batch (max |diff|
# recorded; the run refuses to score if it exceeds 1e-3) so the features are the pattern the model used.
#
# REGISTERED PREDICTIONS (bar inherited from §2095: §2094's measured 0.5086 + 0.05 = 0.5586):
#   (a) THE PATTERN CLEARS THE BAR: median held-out AUC of the top-2-head pattern ridge over the 31
#       leaves >= 0.5586. If FALSE, the third and last reading §332's wording supports -- the head's
#       realised routing -- fails on the population where prev heads dominate, and rung 8's proposed
#       mechanism language is refuted in every form; what remains is the census's own two-signed
#       activation-space definition of these leaves (§348), with no head-grain condition found.
#   (b) AND IT IS THESE HEADS' PATTERNS: the same 24 features taken from two CONTROL heads at the same
#       layer (heads outside the leaf's top-2, diffuse-class where available) score lower than the
#       top-2 heads' features on >= 60% of the 31 leaves. Without this, any success is "attention
#       patterns at layer L are informative", a fact about the layer and not about the identified heads.
#   (c) CAPACITY CONTROL, and (a) may not be read without it: with membership labels SHUFFLED within the
#       leaf's slice, the same ridge scores median AUC <= 0.52. Twenty-four features against a few
#       hundred members has less room than §2095's 1152, but the control is registered, not assumed.
#
# Descriptive, NOT registered: the AUC of two sub-models -- offset-1 signed mass alone (both heads,
# 2 features) and total row mass alone (2 features) -- so a pass or fail can be located within the
# feature set rather than asserted about "the pattern" as a whole.
#
# Writes realised_attention_composition_results.json.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

NEED = ['prev_token_composition_results.json', 'value_read_composition_results.json',
        'motif_vocabulary_results.json', 'attn_motifs3_results.json', 'census_state_diverse.pt']
if os.environ.get('BQLIB_DRYRUN') == '1':
    miss = [f for f in NEED if not os.path.exists(os.path.join(BQ, f))]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    v = json.load(open(os.path.join(BQ, 'prev_token_composition_results.json')))
    w = json.load(open(os.path.join(BQ, 'value_read_composition_results.json')))
    print(f"DRYRUN OK: S2094 treatment {v['n_treatment']} leaves, token-id AUC "
          f"{v['treatment_median_auc_prev']}; S2095 embedding AUC {w['median_auc_prev_embedding']} "
          f"vs bar {w['bar']}; this run's bar {v['treatment_median_auc_prev'] + 0.05:.4f}")
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import census_lib as C                                                    # noqa: E402

T0 = time.time()
BASE = json.load(open('prev_token_composition_results.json'))
PRIOR = json.load(open('value_read_composition_results.json'))
BAR = BASE['treatment_median_auc_prev'] + 0.05
TAGS = [r['tag'] for r in BASE['treatment']]
LEAFMETA = {lf['tag']: lf for lf in json.load(open('motif_vocabulary_results.json'))['leaves']}
MOTIF = json.load(open('attn_motifs3_results.json'))['motif_table']      # [L, h, class, score]
ST = torch.load('census_state_diverse.pt', map_location='cpu', weights_only=False)
ROWS = ST['rows']
BY = {lf['tag']: lf for lf in ST['leaves']}
NR, NP = ROWS.shape[0], 256
HALF = NR // 2
NH, HD, DM = 9, 128, 1152
BATCH = 16
NF = 12
FEATNAMES = ['off0', 'off1', 'off2', 'off3', 'off4_7', 'off8_15', 'off16_31', 'off32_63', 'off64p',
             'mass', 'absmass', 'entropy']
M = C.m
DEV = next(M.parameters()).device
H = M.transformer.h
LAYERS = sorted({int(LEAFMETA[t]['comp'][1:]) for t in TAGS})
print(f'{len(TAGS)} treatment leaves at layers {LAYERS} | bar {BAR:.4f} (S2094 '
      f'{BASE["treatment_median_auc_prev"]} + 0.05) | S2095 embedding ridge scored '
      f'{PRIOR["median_auc_prev_embedding"]}', flush=True)

OFFS = torch.arange(NP).view(-1, 1) - torch.arange(NP).view(1, -1)          # d = t - j
CAUSAL = (OFFS >= 0)
EDGES = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 6)]
BUCKETS = torch.stack([(OFFS >= lo) & (OFFS <= hi) & CAUSAL for lo, hi in EDGES]).float().to(DEV)
CAUSAL_DEV = CAUSAL.to(DEV)
FEATS = {L: torch.zeros(NR, NH, NP, NF) for L in LAYERS}
CUR = {'rows': None}
CHECK = {'maxdiff': 0.0, 'done': False}


def pattern_features(pat):
    """(B, H, T, T) unnormalised pattern -> (B, H, T, NF) per-query summary."""
    B, Hn, T, _ = pat.shape
    bucket = torch.einsum('bhqk,nqk->bhqn', pat, BUCKETS[:, :T, :T])       # signed mass per offset bin
    mass = pat.sum(-1, keepdim=True)
    ab = pat.abs()
    absmass = ab.sum(-1, keepdim=True)
    p = ab / absmass.clamp_min(1e-12)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1, keepdim=True)
    return torch.cat([bucket, mass, absmass, ent], dim=-1)


def make_capture(L):
    attn = H[L].attn
    original = attn.squared_attention

    def captured(q, k, v, q2, k2):
        B, T, Hn, Dh = q.shape
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
        pat = (s1 / Dh) * (s2 / Dh)
        pat = pat.masked_fill(~CAUSAL_DEV[:T, :T], 0.0)
        FEATS[L][CUR['rows']] = pattern_features(pat.float()).cpu()
        z = torch.einsum('bhqk,bkhd->bhqd', pat, v)
        if not CHECK['done']:
            ref = original(q, k, v, q2, k2)
            CHECK['maxdiff'] = max(CHECK['maxdiff'], float((ref - z).abs().max()))
        return z
    return captured


def run_rows():
    for L in LAYERS:
        H[L].attn.squared_attention = make_capture(L)
    with torch.no_grad():
        for s in range(0, NR, BATCH):
            idx = ROWS[s:s + BATCH, :NP].to(DEV)
            CUR['rows'] = slice(s, s + idx.shape[0])
            x = F.rms_norm(M.transformer.wte(idx), (DM,))
            x0 = x
            v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            CHECK['done'] = True
            if s % (BATCH * 16) == 0:
                print(f'  rows {s + idx.shape[0]}/{NR}  {time.time() - T0:.0f}s', flush=True)


def auc(score, lab):
    o = score.argsort()
    rk = torch.empty(len(score)); rk[o] = torch.arange(len(score)).float()
    p = lab.bool(); npos = int(p.sum()); nneg = len(lab) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((rk[p].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def ridge_auc(X, y, fit, ev, shuffle=False):
    if fit.sum() < 50 or ev.sum() < 50:
        return float('nan')
    y = y.float()
    if shuffle:
        g = torch.Generator().manual_seed(4242)
        idxf = fit.nonzero().squeeze(1)
        y = y.clone()
        y[idxf] = y[idxf][torch.randperm(len(idxf), generator=g)]
    Xf = X[fit]; Xe = X[ev]
    mu = Xf.mean(0, keepdim=True); sd = Xf.std(0, keepdim=True).clamp_min(1e-6)
    Xf = (Xf - mu) / sd; Xe = (Xe - mu) / sd
    yf = y[fit]
    lam = 1e-2 * len(Xf)
    A = Xf.T @ Xf + lam * torch.eye(X.shape[1])
    b = Xf.T @ (yf - yf.mean())
    w = torch.linalg.solve(A, b)
    return auc(Xe @ w, y[ev])


def control_heads(L, top2):
    cands = [(h, cls, sc) for (l, h, cls, sc) in MOTIF if l == L and h not in top2]
    diffuse = sorted([c for c in cands if c[1] == 'diffuse'], key=lambda c: -c[2])
    rest = sorted([c for c in cands if c[1] != 'diffuse'], key=lambda c: -c[2])
    pick = (diffuse + rest)[:2]
    return [p[0] for p in pick], [p[1] for p in pick]


def leaf_features(L, heads, idx):
    row = idx // NP
    col = idx % NP
    return torch.cat([FEATS[L][row, h, col] for h in heads], dim=1)


def score_leaf(tag):
    lf = BY.get(tag)
    meta = LEAFMETA.get(tag)
    if lf is None or meta is None:
        return None
    L = int(meta['comp'][1:])
    top2 = [int(h) for h in meta['heads']]
    ctl, ctl_cls = control_heads(L, top2)
    sl = torch.zeros(NR * NP, dtype=torch.bool); sl[lf['slice']] = True
    idx = sl.nonzero().squeeze(1)
    mm = torch.zeros(NR * NP, dtype=torch.bool); mm[lf['member']] = True
    y = mm[idx]
    fit = (idx // NP) < HALF
    ev = ~fit
    Xt = leaf_features(L, top2, idx)
    Xc = leaf_features(L, ctl, idx)
    out = {'tag': tag, 'comp': meta['comp'], 'heads': top2, 'classes': meta['classes'],
           'control_heads': ctl, 'control_classes': ctl_cls,
           'n_fit': int(fit.sum()), 'n_eval': int(ev.sum()), 'members_eval': int(y[ev].sum())}
    a_top = ridge_auc(Xt, y, fit, ev)
    a_ctl = ridge_auc(Xc, y, fit, ev)
    a_sh = ridge_auc(Xt, y, fit, ev, shuffle=True)
    a_off1 = ridge_auc(Xt[:, [1, NF + 1]], y, fit, ev)
    a_mass = ridge_auc(Xt[:, [9, NF + 9]], y, fit, ev)
    if any(v != v for v in (a_top, a_ctl, a_sh)):
        return None
    out.update({'auc_top2_pattern': round(a_top, 4), 'auc_control_pattern': round(a_ctl, 4),
                'auc_shuffled': round(a_sh, 4), 'auc_offset1_only': round(a_off1, 4),
                'auc_mass_only': round(a_mass, 4)})
    return out


def med(v):
    return sorted(v)[len(v) // 2] if v else float('nan')


run_rows()
print(f'capture check: max |captured - model attention| on first batch = {CHECK["maxdiff"]:.2e}',
      flush=True)
if CHECK['maxdiff'] > 1e-3:
    print('CAPTURE MISMATCH -- the features are not the pattern the model used; refusing to score.')
    json.dump({'capture_maxdiff': CHECK['maxdiff'], 'status': 'refused'},
              open('realised_attention_composition_results.json', 'w'), indent=1)
    raise SystemExit(2)

LEAVES = []
for tag in TAGS:
    r = score_leaf(tag)
    if r is not None:
        LEAVES.append(r)
print(f'scored {len(LEAVES)}/{len(TAGS)} leaves', flush=True)

mt = med([r['auc_top2_pattern'] for r in LEAVES])
mc = med([r['auc_control_pattern'] for r in LEAVES])
ms = med([r['auc_shuffled'] for r in LEAVES])
m1 = med([r['auc_offset1_only'] for r in LEAVES])
mm_ = med([r['auc_mass_only'] for r in LEAVES])
frac = sum(1 for r in LEAVES if r['auc_top2_pattern'] > r['auc_control_pattern']) / max(len(LEAVES), 1)
n60 = sum(1 for r in LEAVES if r['auc_top2_pattern'] >= 0.60)
pc = ms <= 0.52
pa = mt >= BAR
pb = frac >= 0.60
out = {'n_leaves': len(LEAVES), 'bar': round(BAR, 4),
       'S2094_token_id_auc': BASE['treatment_median_auc_prev'],
       'S2095_embedding_auc': PRIOR['median_auc_prev_embedding'],
       'capture_maxdiff': CHECK['maxdiff'], 'features_per_head': FEATNAMES,
       'median_auc_top2_pattern': round(mt, 4), 'median_auc_control_pattern': round(mc, 4),
       'median_auc_shuffled': round(ms, 4), 'median_auc_offset1_only': round(m1, 4),
       'median_auc_mass_only': round(mm_, 4), 'frac_top2_beats_control': round(frac, 4),
       'leaves_at_or_above_0.60': n60, 'leaves': LEAVES,
       'pred_a_pattern_clears_bar': bool(pa), 'pred_b_specific_to_top2_heads': bool(pb),
       'pred_c_capacity_control': bool(pc), 'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open('realised_attention_composition_results.json', 'w'), indent=1)
print(f'\nmedian AUC: top-2 pattern {mt:.4f} | control heads {mc:.4f} | shuffled {ms:.4f} | '
      f'offset-1 only {m1:.4f} | mass only {mm_:.4f} | leaves >= 0.60: {n60}/{len(LEAVES)}')
print(f'(c) CAPACITY CONTROL shuffled {ms:.4f} <= 0.52: {"HELD" if pc else "FAILED"}')
if not pc:
    print('    CONTROL FAILED -- the ridge fits shuffled labels; (a) and (b) measure capacity, not signal.')
else:
    print(f'(a) top-2 pattern {mt:.4f} >= bar {BAR:.4f}: {"HELD" if pa else "FAILED"}')
    print(f'(b) top-2 beats control heads for {frac:.1%} (bar 60%): {"HELD" if pb else "FAILED"}')
    if not pa:
        print('    READING: the realised routing of the identified prev heads does not predict membership '
              'either; S332\'s composition proposal is refuted in every form its wording supports.')
print(f'wrote realised_attention_composition_results.json ({time.time() - T0:.0f}s)')
