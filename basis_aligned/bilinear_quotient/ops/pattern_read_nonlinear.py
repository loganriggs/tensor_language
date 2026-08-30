# THE ONLY UNTESTED BRANCH OF §2096: a NONLINEAR and a CONTEXT-AGGREGATING read of the realised pattern.
#
# NEW RUNG (opened 2026-08-30 after the backlog audit in §2096 found every listed rung closed or
# blocked). §2096 gave the census's a3/a4 prev-head leaves their first non-null head-grain feature: the
# two top-2 heads' realised attention pattern, summarised by 24 per-query numbers, scores median held-out
# AUC 0.5409 on the 31 both-prev leaves -- above every local-token feature (0.5086 identity, 0.5052
# embedding), specific to the identified heads (21/31 over same-layer controls), and STILL a third of
# the way to the 0.5586 bar, with 0/31 leaves at 0.60. It closed rung 8 and named its own two scope
# limits: the ridge is LINEAR in 24 summaries of ONE query position. This tests both limits, and
# nothing else, on the same 31 leaves, the same split, the same bar.
#
# WHY THIS IS NOT A REPHRASING OF §332. §332 proposed conditions on what a head READS (motif composed
# with a value read). §2096's finding is that the informative quantity is how the head ROUTES -- mostly
# total mass and offset-1 mass. If membership depends on routing in a way a linear-in-summaries read
# cannot express (a threshold on mass, an interaction between the two heads' masses, or a run of
# positions rather than one), a nonlinear or aggregated read would show it. If it does not, the
# head-grain description of these leaves is exhausted at 0.54 and the bands stay §348's two-signed
# activation-space objects.
#
# ARMS (all from the identical 24 features per position of §2096, captured the same way):
#   L1   linear ridge on the 24 features at t                       -- §2096 replicate, must reproduce
#   NL   random-Fourier-feature ridge (512 cosine features, fixed seed, bandwidth = median pairwise
#        distance on the fit half) on the 24 features at t          -- the nonlinear read
#   CTX  linear ridge on the 24 features at t, t-1, t-2, t-3 (96)   -- the context-aggregating read
#   NLC  RFF ridge on the 96 context features                       -- both at once
# Each arm has a shuffled-label control at the same capacity; the winning arm additionally gets the
# same-layer control-head comparison §2096 used.
#
# REGISTERED PREDICTIONS (bar inherited: §2094's measured 0.5086 + 0.05 = 0.5586):
#   (a) A RICHER READ CLEARS THE BAR: the best of {NL, CTX, NLC} by median held-out AUC over the 31 leaves
#       reaches >= 0.5586. If FALSE, the realised-routing signal is exhausted by a linear read of one
#       position at ~0.54; no head-grain read of any tested kind predicts membership, and this rung
#       closes with the a3/a4 leaves described at head grain only as "weakly mass-dependent".
#   (b) AND IT IS NOT CAPACITY: every arm's shuffled-label control scores median AUC <= 0.52. RFF on 512
#       features against a few hundred members is exactly the shape §2088 warned about; the control is
#       per arm, not pooled, so the winning arm's own capacity is what gets checked.
#   (c) REPLICATION GATE, and nothing is read without it: L1 reproduces §2096's 0.5409 to within 0.005
#       (median over the same 31 leaves, same features, same solver). Cross-run per LESSON 42: if the
#       replicate moves, the capture changed and the arms above are not comparable to §2096.
#
# Descriptive, NOT registered: which of NL / CTX / NLC is best, and the winning arm's advantage over its
# control-head counterpart (§2096's specificity check, +0.0062 there).
#
# Writes pattern_read_nonlinear_results.json.
import json
import math
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

NEED = ['prev_token_composition_results.json', 'realised_attention_composition_results.json',
        'motif_vocabulary_results.json', 'attn_motifs3_results.json', 'census_state_diverse.pt']
if os.environ.get('BQLIB_DRYRUN') == '1':
    miss = [f for f in NEED if not os.path.exists(os.path.join(BQ, f))]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    v = json.load(open(os.path.join(BQ, 'prev_token_composition_results.json')))
    p = json.load(open(os.path.join(BQ, 'realised_attention_composition_results.json')))
    if p.get('status') == 'refused' or 'median_auc_top2_pattern' not in p:
        print('DRYRUN FAIL: S2096 artifact has no scored result'); raise SystemExit(1)
    print(f"DRYRUN OK: S2096 linear pattern AUC {p['median_auc_top2_pattern']} on {p['n_leaves']} leaves; "
          f"bar {v['treatment_median_auc_prev'] + 0.05:.4f}; replication tolerance 0.005")
    raise SystemExit(0)

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import census_lib as C                                                    # noqa: E402

T0 = time.time()
BASE = json.load(open('prev_token_composition_results.json'))
PRIOR = json.load(open('realised_attention_composition_results.json'))
BAR = BASE['treatment_median_auc_prev'] + 0.05
REPL_TOL = 0.005
TAGS = [r['tag'] for r in BASE['treatment']]
LEAFMETA = {lf['tag']: lf for lf in json.load(open('motif_vocabulary_results.json'))['leaves']}
MOTIF = json.load(open('attn_motifs3_results.json'))['motif_table']
ST = torch.load('census_state_diverse.pt', map_location='cpu', weights_only=False)
ROWS = ST['rows']
BY = {lf['tag']: lf for lf in ST['leaves']}
NR, NP = ROWS.shape[0], 256
HALF = NR // 2
NH, HD, DM = 9, 128, 1152
BATCH = 16
NF = 12
NCTX = 4                      # positions t, t-1, t-2, t-3
NRFF = 512
RFF_SEED = 2096
M = C.m
DEV = next(M.parameters()).device
H = M.transformer.h
LAYERS = sorted({int(LEAFMETA[t]['comp'][1:]) for t in TAGS})
print(f'{len(TAGS)} leaves at layers {LAYERS} | bar {BAR:.4f} | S2096 linear {PRIOR["median_auc_top2_pattern"]} '
      f'(replication tol {REPL_TOL})', flush=True)

OFFS = torch.arange(NP).view(-1, 1) - torch.arange(NP).view(1, -1)
CAUSAL = (OFFS >= 0)
EDGES = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10 ** 6)]
BUCKETS = torch.stack([(OFFS >= lo) & (OFFS <= hi) & CAUSAL for lo, hi in EDGES]).float().to(DEV)
CAUSAL_DEV = CAUSAL.to(DEV)
FEATS = {L: torch.zeros(NR, NH, NP, NF) for L in LAYERS}
CUR = {'rows': None}
CHECK = {'maxdiff': 0.0, 'done': False}


def pattern_features(pat):
    B, Hn, T, _ = pat.shape
    bucket = torch.einsum('bhqk,nqk->bhqn', pat, BUCKETS[:, :T, :T])
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


def auc(score, lab):
    o = score.argsort()
    rk = torch.empty(len(score)); rk[o] = torch.arange(len(score)).float()
    p = lab.bool(); npos = int(p.sum()); nneg = len(lab) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((rk[p].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def standardise(Xf, Xe):
    mu = Xf.mean(0, keepdim=True); sd = Xf.std(0, keepdim=True).clamp_min(1e-6)
    return (Xf - mu) / sd, (Xe - mu) / sd


def rff(Xf, Xe):
    """Random Fourier features with the bandwidth set from the fit half only."""
    g = torch.Generator().manual_seed(RFF_SEED)
    sub = Xf[torch.randperm(len(Xf), generator=g)[:512]]
    d2 = torch.cdist(sub, sub)
    band = float(d2[d2 > 0].median()) if bool((d2 > 0).any()) else 1.0
    Wr = torch.randn(Xf.shape[1], NRFF, generator=g) / band
    br = torch.rand(NRFF, generator=g) * 2 * math.pi
    scale = math.sqrt(2.0 / NRFF)
    return scale * torch.cos(Xf @ Wr + br), scale * torch.cos(Xe @ Wr + br)


def ridge_auc(X, y, fit, ev, nonlinear=False, shuffle=False):
    if fit.sum() < 50 or ev.sum() < 50:
        return float('nan')
    y = y.float()
    if shuffle:
        g = torch.Generator().manual_seed(4242)
        idxf = fit.nonzero().squeeze(1)
        y = y.clone()
        y[idxf] = y[idxf][torch.randperm(len(idxf), generator=g)]
    Xf, Xe = standardise(X[fit], X[ev])
    if nonlinear:
        Xf, Xe = rff(Xf, Xe)
    yf = y[fit]
    lam = 1e-2 * len(Xf)
    A = Xf.T @ Xf + lam * torch.eye(Xf.shape[1])
    b = Xf.T @ (yf - yf.mean())
    w = torch.linalg.solve(A, b)
    return auc(Xe @ w, y[ev])


def control_heads(L, top2):
    cands = [(h, cls, sc) for (l, h, cls, sc) in MOTIF if l == L and h not in top2]
    diffuse = sorted([c for c in cands if c[1] == 'diffuse'], key=lambda c: -c[2])
    rest = sorted([c for c in cands if c[1] != 'diffuse'], key=lambda c: -c[2])
    pick = (diffuse + rest)[:2]
    return [p[0] for p in pick]


def leaf_features(L, heads, idx, context):
    row = idx // NP
    col = idx % NP
    parts = []
    for back in range(NCTX if context else 1):
        c = (col - back).clamp(min=0)
        valid = (col - back >= 0).float().unsqueeze(1)
        for h in heads:
            parts.append(FEATS[L][row, h, c] * valid)
    return torch.cat(parts, dim=1)


ARMS = (('L1', False, False), ('NL', True, False), ('CTX', False, True), ('NLC', True, True))


def score_leaf(tag):
    lf = BY.get(tag)
    meta = LEAFMETA.get(tag)
    if lf is None or meta is None:
        return None
    L = int(meta['comp'][1:])
    top2 = [int(h) for h in meta['heads']]
    ctl = control_heads(L, top2)
    sl = torch.zeros(NR * NP, dtype=torch.bool); sl[lf['slice']] = True
    idx = sl.nonzero().squeeze(1)
    mm = torch.zeros(NR * NP, dtype=torch.bool); mm[lf['member']] = True
    y = mm[idx]
    fit = (idx // NP) < HALF
    ev = ~fit
    out = {'tag': tag, 'comp': meta['comp'], 'heads': top2, 'control_heads': ctl,
           'n_fit': int(fit.sum()), 'n_eval': int(ev.sum()), 'members_eval': int(y[ev].sum())}
    for name, nonlin, context in ARMS:
        Xt = leaf_features(L, top2, idx, context)
        Xc = leaf_features(L, ctl, idx, context)
        a = ridge_auc(Xt, y, fit, ev, nonlinear=nonlin)
        s = ridge_auc(Xt, y, fit, ev, nonlinear=nonlin, shuffle=True)
        c = ridge_auc(Xc, y, fit, ev, nonlinear=nonlin)
        if a != a or s != s or c != c:
            return None
        out[f'auc_{name}'] = round(a, 4)
        out[f'auc_{name}_shuffled'] = round(s, 4)
        out[f'auc_{name}_control'] = round(c, 4)
    return out


def med(v):
    return sorted(v)[len(v) // 2] if v else float('nan')


run_rows()
print(f'capture check: max |captured - model attention| = {CHECK["maxdiff"]:.2e}', flush=True)
if CHECK['maxdiff'] > 1e-3:
    json.dump({'capture_maxdiff': CHECK['maxdiff'], 'status': 'refused'},
              open('pattern_read_nonlinear_results.json', 'w'), indent=1)
    raise SystemExit(2)

LEAVES = []
for tag in TAGS:
    r = score_leaf(tag)
    if r is not None:
        LEAVES.append(r)
    print(f'  {len(LEAVES)} leaves scored  {time.time() - T0:.0f}s', flush=True)

medians = {name: med([r[f'auc_{name}'] for r in LEAVES]) for name, _n, _c in ARMS}
shuffled = {name: med([r[f'auc_{name}_shuffled'] for r in LEAVES]) for name, _n, _c in ARMS}
controls = {name: med([r[f'auc_{name}_control'] for r in LEAVES]) for name, _n, _c in ARMS}
best = max(('NL', 'CTX', 'NLC'), key=lambda n: medians[n])
frac_best = sum(1 for r in LEAVES if r[f'auc_{best}'] > r[f'auc_{best}_control']) / max(len(LEAVES), 1)
n60 = {name: sum(1 for r in LEAVES if r[f'auc_{name}'] >= 0.60) for name, _n, _c in ARMS}
pc = abs(medians['L1'] - PRIOR['median_auc_top2_pattern']) <= REPL_TOL
pb = all(v <= 0.52 for v in shuffled.values())
pa = medians[best] >= BAR
out = {'n_leaves': len(LEAVES), 'bar': round(BAR, 4), 'S2096_linear_auc': PRIOR['median_auc_top2_pattern'],
       'capture_maxdiff': CHECK['maxdiff'], 'arms': [a[0] for a in ARMS], 'rff_features': NRFF,
       'context_positions': NCTX, 'median_auc': {k: round(v, 4) for k, v in medians.items()},
       'median_auc_shuffled': {k: round(v, 4) for k, v in shuffled.items()},
       'median_auc_control_heads': {k: round(v, 4) for k, v in controls.items()},
       'leaves_at_or_above_0.60': n60, 'best_richer_arm': best,
       'frac_best_beats_control_heads': round(frac_best, 4), 'leaves': LEAVES,
       'pred_a_richer_read_clears_bar': bool(pa), 'pred_b_not_capacity': bool(pb),
       'pred_c_L1_replicates_S2096': bool(pc), 'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open('pattern_read_nonlinear_results.json', 'w'), indent=1)
print('\nmedian AUC   ' + ' | '.join(f'{k} {v:.4f} (shuf {shuffled[k]:.4f}, ctl {controls[k]:.4f}, '
                                   f'>=0.60: {n60[k]})' for k, v in medians.items()))
print(f'(c) REPLICATION L1 {medians["L1"]:.4f} vs S2096 {PRIOR["median_auc_top2_pattern"]} '
      f'(tol {REPL_TOL}): {"HELD" if pc else "FAILED"}')
print(f'(b) CAPACITY every shuffled control <= 0.52: {"HELD" if pb else "FAILED"}')
if not pc:
    print('    REPLICATION FAILED -- the capture or solver changed; (a) is not comparable to S2096.')
elif not pb:
    print('    A CONTROL FAILED -- at least one arm fits shuffled labels; (a) measures capacity there.')
else:
    print(f'(a) best richer arm {best} {medians[best]:.4f} >= bar {BAR:.4f}: {"HELD" if pa else "FAILED"}'
          f'   (beats control heads on {frac_best:.1%})')
    if not pa:
        print('    READING: neither a nonlinear nor a context-aggregating read of the realised pattern '
              'clears the bar; the head-grain signal at these leaves is exhausted at ~0.54.')
print(f'wrote pattern_read_nonlinear_results.json ({time.time() - T0:.0f}s)')
