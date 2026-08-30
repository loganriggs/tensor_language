# ARE §2056/§2058'S PER-PAIR a8 CLAIMS REPRESENTATIVE OF a8? THE LAST UNAUDITED CORNER.
#
# RUNG 3: the open question named at the end of §2073.
#
# §2056 and §2058 read a8's cosine matrix PAIR BY PAIR on five circuits -- claims like "r.11.1.1's
# direction is best explained by r.11.1.2's" come from individual entries, not aggregates. §2073 showed
# that for LEARNED directions the mean over fifteen pairs is stable at three seeds while individual pairs
# swing (max pairwise cosine 0.5605 / 0.9343 / 0.8120), so per-pair claims are the fragile kind. Those a8
# directions are CLOSED-FORM and have no seed, so §2073 does not impugn them directly.
#
# The exposure is different and it is one §2065 and §2066 already documented twice: those five circuits
# are an unrepresentative subset of the SIXTEEN that §2059 localises at a8. §2065 found the reversal is
# 5/16 -> 9/16 on all sixteen against 1/5 -> 4/5 on the five; §2066 found the geometry-causality
# correlation is +0.4212 on sixteen against +0.6611 on five. Every re-measurement on the full set has
# shrunk the effect. Whether the PER-PAIR structure survives the same substitution has never been checked,
# and it is what the surviving mechanism readings in §2058 rest on.
#
# No fitting and no seeds: closed-form directions over both circuit sets, one capture.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  The five-circuit subset is BIASED HIGH on pairwise cosine, consistent with every other
#           re-measurement: the mean pairwise |cos| among the five exceeds the mean over all 120 pairs of
#           the sixteen. Registered in the direction the record establishes; if FALSE the five are not
#           systematically unusual in geometry and §2058's per-pair reading is on firmer ground than I
#           expect.
#   pred_b  §2058's specific claim survives: among all sixteen, r.11.1.1's closed-form direction is still
#           closest to r.11.1.2's -- that is, r.11.1.2 remains its argmax neighbour when eleven circuits
#           it was never compared against are added. If FALSE, the neighbour was an artefact of the subset
#           and §2058's "r.11.1.1 may be a shadow of r.11.1.2" -- which I put to Codex at 05:02 and again
#           at 05:52 -- has to be withdrawn.
#   pred_c  The five circuits do NOT form a distinguishable cluster within the sixteen: at least one of
#           the eleven others has a mean cosine to the five that exceeds the five's own internal mean. If
#           TRUE the subset is arbitrary rather than a natural group, which is the simplest explanation
#           for why every aggregate measured on it has been optimistic.
#
# Writes circuits/A8_PAIRWISE.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

COMPONENT = 'a8'
FIVE = ['r.11.1.1', 'r.11.1.2', 'r.11.3.1', 'r.23.2.1', 'r.23.2.3']

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    b = json.load(open(os.path.join(BQ, 'circuits/BATTERY.json')))
    n = [t for t, v in b['by_tag'].items() if v['best_mean'] == COMPONENT]
    if not set(FIVE) <= set(n):
        print(f'DRYRUN FAIL: S2056 five not all at {COMPONENT}: {sorted(set(FIVE) - set(n))}')
        raise SystemExit(1)
    print(f'DRYRUN OK: {len(n)} circuits at {COMPONENT}, all five of S2056 among them')
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

BAT = json.load(open('circuits/BATTERY.json'))
C.use_state('census_state_diverse.pt')
nflat = C.nflat()
CAND = [t for t, v in BAT['by_tag'].items() if v['best_mean'] == COMPONENT]

t0 = time.time()
R = C.rows()
cap = []
h = C.MODS[COMPONENT].register_forward_hook(
    lambda mo, i_, o_: cap.append(((o_[0] if isinstance(o_, tuple) else o_)
                                   .detach().float().reshape(-1, C.D).cpu()))) 
with torch.no_grad():
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        C.m(bb[:, :-1].contiguous(), bb[:, 1:].contiguous())
h.remove()
acts = torch.cat(cap)
print(f'captured {COMPONENT} ({time.time()-t0:.0f}s)', flush=True)

TAGS, DIRS = [], {}
for t in CAND:
    lf = C.leaf(t)
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    if mm.sum() == 0 or (~sl).sum() == 0:
        continue
    u = acts[mm].mean(0) - acts[~sl].mean(0)
    if not torch.isfinite(u).all() or u.norm() == 0:
        continue
    TAGS.append(t); DIRS[t] = u / u.norm()
print(f'{len(TAGS)} usable circuits at {COMPONENT}; the S2056 five are '
      f'{[t for t in FIVE if t in TAGS]}', flush=True)


def mean_cos(group):
    p = [float(abs(DIRS[a] @ DIRS[b])) for i, a in enumerate(group) for b in group[i + 1:]]
    return sum(p) / len(p), len(p)


five = [t for t in FIVE if t in TAGS]
m5, n5 = mean_cos(five)
m16, n16 = mean_cos(TAGS)
others = [t for t in TAGS if t not in five]
to_five = {t: sum(float(abs(DIRS[t] @ DIRS[f])) for f in five) / len(five) for t in others}
best_out = max(to_five, key=to_five.get) if to_five else None

nbr = {}
for t in five:
    cand = {b: float(abs(DIRS[t] @ DIRS[b])) for b in TAGS if b != t}
    nbr[t] = {'argmax_over_all': max(cand, key=cand.get),
              'argmax_value': round(max(cand.values()), 4),
              'argmax_within_five': max((b for b in five if b != t),
                                        key=lambda b: float(abs(DIRS[t] @ DIRS[b])))}
    print(f'  {t:10s} nearest among all {len(TAGS)}: {nbr[t]["argmax_over_all"]:10s} '
          f'({nbr[t]["argmax_value"]:.4f})   nearest within the five: '
          f'{nbr[t]["argmax_within_five"]}', flush=True)

r111_ok = nbr.get('r.11.1.1', {}).get('argmax_over_all') == 'r.11.1.2'
cluster_broken = bool(best_out is not None and to_five[best_out] > m5)
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude', 'component': COMPONENT,
       'checks': "S2056/S2058 per-pair a8 claims, made on five of the sixteen circuits at a8",
       'circuits_all': TAGS, 'circuits_S2056_five': five,
       'mean_pairwise_abs_cos_five': round(m5, 4), 'n_pairs_five': n5,
       'mean_pairwise_abs_cos_all': round(m16, 4), 'n_pairs_all': n16,
       'nearest_neighbour_of_each_of_the_five': nbr,
       'outsider_closest_to_the_five': best_out,
       'outsider_mean_cos_to_five': round(to_five[best_out], 4) if best_out else None,
       'pred_a_five_biased_high': bool(m5 > m16),
       'pred_b_r111_nearest_is_r112_among_all': bool(r111_ok),
       'pred_c_five_are_not_a_cluster': cluster_broken,
       'note': 'read-only artifact; no circuit file was modified'}
json.dump(rep, open('circuits/A8_PAIRWISE.json', 'w'), indent=1)
print(f'\nwrote circuits/A8_PAIRWISE.json ({time.time()-t0:.0f}s)')
print(f'pred_a  mean |cos| five {m5:.4f} ({n5} pairs) vs all {len(TAGS)} {m16:.4f} ({n16} pairs) '
      f'-- five biased high : {rep["pred_a_five_biased_high"]}')
print(f'pred_b  r.11.1.1 nearest among all is r.11.1.2 : {rep["pred_b_r111_nearest_is_r112_among_all"]}'
      f'   (actual: {nbr.get("r.11.1.1", {}).get("argmax_over_all")})')
print(f'pred_c  an outsider is closer to the five than they are to each other : {cluster_broken}'
      f'   ({best_out} at {rep["outsider_mean_cos_to_five"]} vs internal {m5:.4f})')
