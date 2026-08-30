# "VALUES CARRYING X" WITH X A DIRECTION, NOT A TOKEN. Rung 8's richer composition test.
#
# BACKLOG rung 8. §2094 refuted §332's literal form -- on the 31 leaves whose both top-2 heads are prev,
# a per-token-id membership predictor scored held-out AUC 0.5086 (previous token) and 0.5130 (current),
# with 0 of 31 leaves reaching 0.60 on either. It stated the limit of that refutation plainly: "the
# predictor is a per-token-id lookup, the simplest literal form ... 'values carrying X' may have meant a
# richer read", and set the bar for any richer attempt at beating 0.5086 on the same 31 leaves.
#
# This is that attempt, with the feature chosen from what §332's phrase actually denotes rather than from
# convenience. A prev-motif head attends to position t-1 and writes its VALUE read of that position. The
# value read is a linear function of the source token's representation -- so "values carrying X" is X as a
# DIRECTION in embedding space, not X as a token identity. §2094 tested identity; this tests direction.
#
# METHOD. Same 31 treatment leaves, same held-out row split, same AUC. The predictor is a ridge from the
# PREVIOUS token's embedding (1152-dim) to membership, fitted on half the rows and scored on the other.
# The current-token embedding is the paired comparison, and §2094's 0.5086 is the number to beat.
#
# REGISTERED PREDICTIONS:
#   (a) THE RICHER READ CLEARS §2094's BAR: median held-out AUC of the prev-embedding ridge over the 31
#       leaves is >= 0.5586, i.e. §2094's 0.5086 plus 0.05. Registered against a measured number rather
#       than a round one. If FALSE, both the literal and the directional readings of "values carrying X"
#       fail on the population where prev heads dominate, and §332's composition proposal is refuted in
#       the two forms its own wording supports.
#   (b) AND IT STAYS A *PREV* CLAIM: the prev-embedding AUC exceeds the current-embedding AUC for >= 60%
#       of the 31. Without this, any success is "membership is linearly readable from nearby token
#       embeddings", which would be a fact about embeddings and not about prev heads.
#   (c) CAPACITY CONTROL, and (a) may not be read without it: with membership labels SHUFFLED within the
#       leaf's slice, the same ridge scores median AUC <= 0.52. A 1152-dimensional ridge against a few
#       hundred members has ample room to fit noise, and §2088's negative-R^2 result is the reminder that
#       held-out scoring is what separates capacity from signal.
#
# Writes value_read_composition_results.json.
import json
import os
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['prev_token_composition_results.json', 'census_state_diverse.pt']
    miss = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if miss:
        print(f'DRYRUN FAIL: missing {miss}'); raise SystemExit(1)
    v = json.load(open(os.path.join(BQ, 'prev_token_composition_results.json')))
    print(f"DRYRUN OK: S2094 present (treatment {v['n_treatment']} leaves, median "
          f"prev-token AUC {v['treatment_median_auc_prev']}); bar to beat "
          f"{v['treatment_median_auc_prev']} + 0.05")
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

BASE = json.load(open('prev_token_composition_results.json'))
BAR = BASE['treatment_median_auc_prev'] + 0.05
TAGS = [r['tag'] for r in BASE['treatment']]
ST = torch.load('census_state_diverse.pt', map_location='cpu', weights_only=False)
ROWS = ST['rows']
BY = {lf['tag']: lf for lf in ST['leaves']}
NR, NP = ROWS.shape[0], 256
HALF = NR // 2
print(f'{len(TAGS)} treatment leaves | bar {BAR:.4f} (S2094 {BASE["treatment_median_auc_prev"]} + 0.05)',
      flush=True)

W = C.m.transformer.wte.weight.detach().float().cpu()
pos = torch.arange(NR * NP)
ROW = pos // NP
COL = pos % NP
CUR = ROWS[ROW, COL].long()
PRV = torch.where(COL > 0, ROWS[ROW, (COL - 1).clamp(min=0)], torch.zeros_like(CUR)).long()
FITM = ROW < HALF
D = W.shape[1]


def auc(score, lab):
    o = score.argsort()
    rk = torch.empty(len(score)); rk[o] = torch.arange(len(score)).float()
    p = lab.bool(); npos = int(p.sum()); nneg = len(lab) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((rk[p].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def ridge_auc(tag, ids, shuffle=False):
    lf = BY.get(tag)
    if lf is None:
        return float('nan')
    mm = torch.zeros(NR * NP, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(NR * NP, dtype=torch.bool); sl[lf['slice']] = True
    fit = FITM & sl; ev = (~FITM) & sl
    if fit.sum() < 50 or ev.sum() < 50:
        return float('nan')
    y = mm.float()
    if shuffle:
        g = torch.Generator().manual_seed(4242)
        idxf = fit.nonzero().squeeze(1)
        y = y.clone()
        y[idxf] = y[idxf][torch.randperm(len(idxf), generator=g)]
    Xf = W[ids[fit]]; Xe = W[ids[ev]]
    yf = y[fit]
    lam = 1e-2 * len(Xf)
    A = Xf.T @ Xf + lam * torch.eye(D)
    b = Xf.T @ (yf - yf.mean())
    w = torch.linalg.solve(A, b)
    return auc(Xe @ w, mm[ev].float())


ROWS_OUT = []
for tag in TAGS:
    ap = ridge_auc(tag, PRV)
    ac = ridge_auc(tag, CUR)
    ash = ridge_auc(tag, PRV, shuffle=True)
    if ap == ap and ac == ac and ash == ash:
        ROWS_OUT.append({'tag': tag, 'auc_prev_emb': round(ap, 4),
                         'auc_cur_emb': round(ac, 4), 'auc_shuffled': round(ash, 4)})
print(f'scored {len(ROWS_OUT)}/{len(TAGS)}', flush=True)
med = lambda v: sorted(v)[len(v) // 2] if v else float('nan')
mp = med([r['auc_prev_emb'] for r in ROWS_OUT])
mc = med([r['auc_cur_emb'] for r in ROWS_OUT])
ms = med([r['auc_shuffled'] for r in ROWS_OUT])
frac = sum(1 for r in ROWS_OUT if r['auc_prev_emb'] > r['auc_cur_emb']) / max(len(ROWS_OUT), 1)
pc = ms <= 0.52
pa = mp >= BAR
pb = frac >= 0.60
out = {'n_leaves': len(ROWS_OUT), 'bar': round(BAR, 4),
       'S2094_token_id_auc': BASE['treatment_median_auc_prev'],
       'median_auc_prev_embedding': round(mp, 4),
       'median_auc_cur_embedding': round(mc, 4),
       'median_auc_shuffled': round(ms, 4),
       'frac_prev_beats_cur': round(frac, 4), 'leaves': ROWS_OUT,
       'pred_a_clears_S2094_bar': bool(pa),
       'pred_b_stays_a_prev_claim': bool(pb),
       'pred_c_capacity_control': bool(pc)}
json.dump(out, open('value_read_composition_results.json', 'w'), indent=1)
print(f'\nmedian AUC: prev-emb {mp:.4f} | cur-emb {mc:.4f} | shuffled {ms:.4f}')
print(f'(c) CAPACITY CONTROL shuffled {ms:.4f} <= 0.52: {"HELD" if pc else "FAILED"}')
if not pc:
    print('    CONTROL FAILED -- the ridge fits shuffled labels, so (a) and (b) '
          'measure capacity, not signal.')
else:
    print(f'(a) prev-emb {mp:.4f} >= bar {BAR:.4f} (S2094 + 0.05): '
          f'{"HELD" if pa else "FAILED"}')
    print(f'(b) prev beats current for {frac:.1%} (bar 60%): '
          f'{"HELD" if pb else "FAILED"}')
    if not pa:
        print('    READING: both the literal (S2094) and the directional (here) '
              'readings of "values carrying X" fail on the prev-dominated '
              'leaves -- S332\'s composition proposal is refuted in the two '
              'forms its own wording supports.')
print('wrote value_read_composition_results.json')
