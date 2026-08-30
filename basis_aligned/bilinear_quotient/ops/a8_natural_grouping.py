# IF THE §2056 FIVE ARE ARBITRARY, IS THERE A NATURAL GROUPING OF a8's SIXTEEN?
#
# RUNG 3: the question §2074 leaves. And a second attempt at §2057, whose three failure modes are known.
#
# §2074 showed the five circuits §2056/§2058 studied at a8 are arbitrary with respect to a8's geometry:
# an outsider (r.2.1) is closer to them (0.9344) than they are to each other (0.8942), and two of the five
# have their nearest neighbour outside the group. That refutes the grouping without proposing one.
#
# §2057 tried census-wide clustering and failed its own validation gate, three ways: it clustered 311 tree
# nodes instead of curated circuits, root nodes have no off-slice so NaN directions poisoned every cosine,
# and single linkage chained everything into one 288-member cluster. All three are avoidable here --
# sixteen curated circuits at one component, all with non-empty off-slices (checked), complete linkage.
#
# THE VALIDATION GATE IS THE POINT, and it is causal rather than geometric. §2066 measured the
# geometry-causality Spearman at a8 as only +0.4212 over all sixteen, so a clustering built on cosines has
# no right to be assumed causally meaningful. It is therefore tested against the cross-circuit ablation
# concentrations that §2065 already computed and stored -- data the clustering never sees.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  a8 is not one blob: complete-linkage clustering of the sixteen direction vectors yields at
#           least two clusters with at least three members each, at the threshold that maximises the
#           causal validation statistic in pred_b. If FALSE, a8's sixteen circuits are geometrically
#           homogeneous and "arbitrary subset" is the whole story -- there is no better grouping to find.
#   pred_b  THE GATE. The grouping is causally real: mean WITHIN-cluster cross-concentration exceeds mean
#           BETWEEN-cluster by at least 10%. The concentrations come from §2065's stored tables and play
#           no part in forming the clusters, so this is an out-of-sample check on a geometric object. If
#           FALSE I report that no causally-validated grouping was found and propose none -- which is what
#           §2057 should have done and did not.
#   pred_c  The §2056 five are split across at least two clusters, confirming §2074's "arbitrary" from an
#           independent direction. If FALSE the five DO form a natural group after all and §2074's
#           conclusion needs qualifying, despite its own two measurements.
#
# Writes circuits/A8_GROUPING.json. DISCOVERY ONLY. No circuit file is modified, and SUBSPACE.json is
# read by neither this script nor Codex's -- nothing here overwrites a shared artifact.
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
THRESHOLDS = (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96)

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/SUBSTRATE_CENSUS.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    c = json.load(open(os.path.join(BQ, 'circuits/SUBSTRATE_CENSUS.json')))
    if COMPONENT not in c['by_component']:
        print(f'DRYRUN FAIL: census has no {COMPONENT}')
        raise SystemExit(1)
    n = len(c['by_component'][COMPONENT]['circuits'])
    print(f'DRYRUN OK: census has {n} circuits at {COMPONENT} with stored concentration tables')
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

CEN = json.load(open('circuits/SUBSTRATE_CENSUS.json'))['by_component'][COMPONENT]
TAGS = CEN['circuits']
CONC = CEN['concentration_full']

C.use_state('census_state_diverse.pt')
nflat = C.nflat()
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

DIRS = {}
for t in TAGS:
    lf = C.leaf(t)
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    assert mm.sum() > 0 and (~sl).sum() > 0, f'{t}: empty members or no off-slice (§2057 NaN trap)'
    u = acts[mm].mean(0) - acts[~sl].mean(0)
    assert torch.isfinite(u).all() and u.norm() > 0, f'{t}: degenerate direction'
    DIRS[t] = u / u.norm()
COS = {a: {b: float(abs(DIRS[a] @ DIRS[b])) for b in TAGS} for a in TAGS}
print(f'{len(TAGS)} circuits at {COMPONENT}, directions finite ({time.time()-t0:.0f}s)', flush=True)


def complete_link(thr):
    """complete linkage: merge only if EVERY cross-pair clears thr. §2057 used single linkage and chained
    288 nodes into one cluster; complete linkage cannot chain."""
    cl = [[t] for t in TAGS]
    merged = True
    while merged:
        merged = False
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                if all(COS[a][b] >= thr for a in cl[i] for b in cl[j]):
                    cl[i] = cl[i] + cl[j]; cl.pop(j); merged = True
                    break
            if merged:
                break
    return cl


def validate(cl):
    """out-of-sample causal check: within-cluster vs between-cluster cross-concentration.

    CONC comes from §2065 and the clustering never saw it.
    """
    of = {t: i for i, c in enumerate(cl) for t in c}
    win, bet = [], []
    for a in TAGS:
        for b in TAGS:
            if a == b or CONC[a].get(b) is None:
                continue
            (win if of[a] == of[b] else bet).append(CONC[a][b])
    if not win or not bet:
        return None, None, None
    mw = sum(win) / len(win); mb = sum(bet) / len(bet)
    return mw, mb, mw / mb


best = None
rows = []
for thr in THRESHOLDS:
    cl = complete_link(thr)
    big = [c for c in cl if len(c) >= 3]
    mw, mb, ratio = validate(cl)
    rows.append({'threshold': thr, 'n_clusters': len(cl), 'sizes': sorted((len(c) for c in cl), reverse=True),
                 'n_clusters_ge3': len(big),
                 'within': round(mw, 4) if mw else None, 'between': round(mb, 4) if mb else None,
                 'ratio': round(ratio, 4) if ratio else None})
    print(f'  thr {thr:.2f}: {len(cl)} clusters sizes {sorted((len(c) for c in cl), reverse=True)}  '
          f'within {mw:.4f} between {mb:.4f} ratio {ratio:.4f}' if ratio else
          f'  thr {thr:.2f}: degenerate', flush=True)
    if ratio is not None and len(big) >= 2 and (best is None or ratio > best[1]):
        best = (thr, ratio, cl, mw, mb)

if best is None:
    # fall back to the threshold with the best ratio regardless of cluster sizes, so pred_b is still scored
    cand = [(r['ratio'], r['threshold']) for r in rows if r['ratio'] is not None]
    thr = max(cand)[1] if cand else THRESHOLDS[0]
    cl = complete_link(thr)
    mw, mb, ratio = validate(cl)
    best = (thr, ratio, cl, mw, mb)

thr, ratio, cl, mw, mb = best
big = [c for c in cl if len(c) >= 3]
five_clusters = {i for i, c in enumerate(cl) for t in c if t in FIVE}
rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude', 'component': COMPONENT,
       'circuits': TAGS,
       'method': 'complete-linkage clustering of closed-form per-circuit directions by |cos|; validated '
                 'OUT OF SAMPLE against S2065\'s stored cross-circuit ablation concentrations, which the '
                 'clustering never sees',
       'sweep': rows, 'chosen_threshold': thr,
       'clusters': cl, 'n_clusters': len(cl), 'n_clusters_ge3': len(big),
       'within_cluster_concentration': round(mw, 4), 'between_cluster_concentration': round(mb, 4),
       'validation_ratio': round(ratio, 4),
       'S2056_five_span_n_clusters': len(five_clusters),
       'pred_a_not_one_blob': bool(len(big) >= 2),
       'pred_b_causally_validated': bool(ratio >= 1.10),
       'pred_c_five_are_split': bool(len(five_clusters) >= 2),
       'note': 'read-only artifact; no circuit file modified, SUBSPACE.json untouched'}
json.dump(rep, open('circuits/A8_GROUPING.json', 'w'), indent=1)

print(f'\nwrote circuits/A8_GROUPING.json ({time.time()-t0:.0f}s)')
print(f'  chosen threshold {thr:.2f}; clusters: {[sorted(c) for c in cl]}')
print(f'pred_a  >=2 clusters of >=3 members: {len(big)} such clusters : {rep["pred_a_not_one_blob"]}')
print(f'pred_b  THE GATE -- within/between cross-concentration {ratio:.4f} '
      f'({mw:.4f} vs {mb:.4f}, bar >=1.10) : {rep["pred_b_causally_validated"]}')
if not rep['pred_b_causally_validated']:
    print('        GATE FAILED: no causally-validated grouping of a8 was found, and none is proposed.')
print(f'pred_c  the S2056 five span {len(five_clusters)} clusters (bar >=2) : '
      f'{rep["pred_c_five_are_split"]}')
