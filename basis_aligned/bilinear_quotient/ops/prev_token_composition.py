# DOES THE PREVIOUS TOKEN PREDICT MEMBERSHIP? §332's composition claim, first direct test.
#
# BACKLOG rung 8. §2092 established the expressibility precondition (census probe bands are
# head-concentrated, 208/208 leaves). §2093 tested the vocabulary: motif-named heads are over-represented
# at 1.1449x -- failing its registered 1.20x effect-size bar while clearing the permutation null at
# z = 4.46, p = 0.00005 -- and found PREV carries the whole effect (215 of 416 leaf top-2 head slots).
#
# §332's proposal is motifs COMPOSED WITH VALUE READS: "prev-motif head at L, values carrying X -> fires
# when previous token writes X". Only the motif half has been tested. The composition half makes a sharp,
# cheap prediction: if a leaf is driven by prev-motif heads reading values, its members should be
# characterised by the PREVIOUS token, and by the previous token MORE than by the current one.
#
# POPULATIONS, counted before designing (the control I first sketched does not exist):
#   * treatment -- BOTH top-2 heads prev: 31 leaves (a4 26, a3 4, a2 1)
#   * control   -- NEITHER top-2 head prev: 24 leaves
#   * "both diffuse" was the intended control and has exactly ONE leaf, so it is not usable.
#
# METHOD. No model forward is needed: membership is a set of grid positions and the tokens are in the row
# cache. For each leaf, a per-token-id empirical membership rate is fitted on HALF the rows and scored by
# AUC on the other half -- non-parametric, and held out because a lookup table fitted and scored on the
# same rows is a memorisation check, not a prediction.
#
# REGISTERED PREDICTIONS:
#   (a) THE PREVIOUS TOKEN PREDICTS: for the 31 both-prev leaves, the median held-out AUC of the
#       previous-token predictor is >= 0.65. If FALSE the composition claim fails at its first step --
#       leaves driven by prev heads are not characterised by the previous token at all.
#   (b) AND IT BEATS THE CURRENT TOKEN: within those 31, prev-token AUC exceeds current-token AUC for
#       >= 60% of leaves. This is the part that makes it a PREV claim rather than a generic
#       "tokens predict membership" claim, which would be unsurprising.
#   (c) AND IT IS SPECIFIC TO PREV HEADS: the median (prev AUC - current AUC) advantage is LARGER for the
#       31 both-prev leaves than for the 24 no-prev control leaves. Without this, (b) could just say that
#       the previous token predicts membership everywhere in the census, which would tell us nothing
#       about the heads.
#
# Writes prev_token_composition_results.json. CPU only; no GPU, no model.
import json
import os
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
os.chdir(BQ)

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['motif_vocabulary_results.json', 'census_state_diverse.pt']
    miss = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    v = json.load(open(os.path.join(BQ, 'motif_vocabulary_results.json')))['leaves']
    nt = sum(1 for r in v if r['classes'].count('prev') == 2)
    nc = sum(1 for r in v if 'prev' not in r['classes'])
    if nt < 10 or nc < 10:
        print(f'DRYRUN FAIL: populations too small (treatment {nt}, control {nc})')
        raise SystemExit(1)
    print(f'DRYRUN OK: treatment {nt} both-prev leaves, control {nc} no-prev leaves')
    raise SystemExit(0)

import torch                                                              # noqa: E402

ST = torch.load('census_state_diverse.pt', map_location='cpu', weights_only=False)
ROWS = ST['rows']
BY = {lf['tag']: lf for lf in ST['leaves']}
NP = 256
NR = ROWS.shape[0]
HALF = NR // 2
V = json.load(open('motif_vocabulary_results.json'))['leaves']
TREAT = [r for r in V if r['classes'].count('prev') == 2]
CTRL = [r for r in V if 'prev' not in r['classes']]
print(f'treatment {len(TREAT)} both-prev | control {len(CTRL)} no-prev', flush=True)

pos = torch.arange(NR * NP)
ROW = pos // NP
COL = pos % NP
CUR = ROWS[ROW, COL].long()
PRV = torch.where(COL > 0, ROWS[ROW, (COL - 1).clamp(min=0)], torch.zeros_like(CUR)).long()
FITM = ROW < HALF
EVLM = ~FITM


def auc(score, lab):
    o = score.argsort()
    rk = torch.empty(len(score)); rk[o] = torch.arange(len(score)).float()
    p = lab.bool(); npos = int(p.sum()); nneg = len(lab) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((rk[p].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def score_leaf(tag, feat):
    lf = BY.get(tag)
    if lf is None:
        return float('nan')
    mm = torch.zeros(NR * NP, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(NR * NP, dtype=torch.bool); sl[lf['slice']] = True
    fit = FITM & sl; ev = EVLM & sl
    if fit.sum() == 0 or ev.sum() == 0:
        return float('nan')
    nv = int(feat.max()) + 1
    cnt = torch.zeros(nv); hit = torch.zeros(nv)
    cnt.index_add_(0, feat[fit], torch.ones(int(fit.sum())))
    hit.index_add_(0, feat[fit], mm[fit].float())
    prior = float(mm[fit].float().mean())
    rate = (hit + 5.0 * prior) / (cnt + 5.0)          # smoothed toward the fit-window prior
    return auc(rate[feat[ev]], mm[ev].float())


def run(group, name):
    out = []
    for r in group:
        ap = score_leaf(r['tag'], PRV)
        ac = score_leaf(r['tag'], CUR)
        if ap == ap and ac == ac:
            out.append({'tag': r['tag'], 'comp': r['comp'],
                        'auc_prev': round(ap, 4), 'auc_cur': round(ac, 4),
                        'adv': round(ap - ac, 4)})
    print(f'  {name}: scored {len(out)}/{len(group)}', flush=True)
    return out


T = run(TREAT, 'treatment'); C_ = run(CTRL, 'control')
med = lambda v: sorted(v)[len(v) // 2] if v else float('nan')
tp = med([r['auc_prev'] for r in T]); tc = med([r['auc_cur'] for r in T])
tadv = med([r['adv'] for r in T]); cadv = med([r['adv'] for r in C_])
frac = sum(1 for r in T if r['auc_prev'] > r['auc_cur']) / max(len(T), 1)
pa = tp >= 0.65
pb = frac >= 0.60
pc = tadv > cadv
out = {'n_treatment': len(T), 'n_control': len(C_),
       'treatment_median_auc_prev': round(tp, 4),
       'treatment_median_auc_cur': round(tc, 4),
       'treatment_median_advantage': round(tadv, 4),
       'control_median_advantage': round(cadv, 4),
       'treatment_frac_prev_beats_cur': round(frac, 4),
       'treatment': T, 'control': C_,
       'pred_a_prev_predicts': bool(pa),
       'pred_b_prev_beats_current': bool(pb),
       'pred_c_specific_to_prev_heads': bool(pc)}
json.dump(out, open('prev_token_composition_results.json', 'w'), indent=1)
print(f'\ntreatment: median AUC prev {tp:.4f} | cur {tc:.4f} | advantage {tadv:+.4f}')
print(f'control  : median advantage {cadv:+.4f}')
print(f"(a) treatment median prev-AUC {tp:.4f} >= 0.65: {'HELD' if pa else 'FAILED'}")
print(f"(b) prev beats current for {frac:.1%} of treatment (bar 60%): "
      f"{'HELD' if pb else 'FAILED'}")
print(f"(c) treatment advantage {tadv:+.4f} > control {cadv:+.4f}: "
      f"{'HELD' if pc else 'FAILED'}")
if not pc:
    print('    READING: the previous token predicts membership no better at '
          'prev-head leaves than elsewhere -- the composition claim gets no '
          'support from the head identities.')
print('wrote prev_token_composition_results.json')
