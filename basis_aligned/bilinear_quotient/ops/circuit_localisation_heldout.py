# DOES §2059'S LOCALISATION SURVIVE BEING SELECTED AND SCORED ON DIFFERENT ROWS?
#
# RUNG 2 (replicate / second-class confirm a just-certified result), house pattern §1595/§1598/§1603.
#
# §2059 reported that all 62 curated circuits localise to a single component at concentration 2.61 to
# 12.28. Every one of those numbers has the same structural weakness: the best component was chosen as the
# argmax over 36 candidates, and its concentration was then reported ON THE SAME 256,000 positions the
# argmax was taken over. That is the exact shape LESSON 106 was written about -- a quantity selected on
# the rows it is measured on is not yet a result -- and §2059 does not carry the check.
#
# This is that check, and it is the cheapest possible form of it: the 36 mean-ablation sweeps are re-run
# once, but every dCE vector is scored SEPARATELY on two disjoint halves of the census rows. The best
# component is selected on half A alone and its concentration is reported on half B alone, so selection
# and measurement never touch the same position.
#
# The row halves are split by ROW, not by position, because positions within a row share a context and
# would leak. Rows are interleaved (even/odd) rather than split at the midpoint so that any drift across
# the corpus order affects both halves equally.
#
# REGISTERED PREDICTIONS (written before running):
#   pred_a  For at least 90% of the 62 circuits, the component selected on half A is ALSO the argmax on
#           half B. If FALSE, §2059's per-circuit component assignments are partly selection noise and the
#           dossier's "best component" column has to be reported with a stability caveat.
#   pred_b  The median across circuits of (held-out concentration / in-sample concentration) for the
#           A-selected component is >= 0.80. This is the size of the selection inflation. If FALSE, the
#           concentrations in §2059 and in circuits/DOSSIER.md are inflated by more than 20% and every
#           number in both must be restated.
#   pred_c  All 62 circuits still clear concentration >= 2.0 on half B alone, using the half-A-selected
#           component -- §2059's headline claim, with selection and measurement fully separated. This is
#           the one that decides whether "62 of 62 localise" stands as written.
#
# Writes circuits/HELDOUT.json. DISCOVERY ONLY. No circuit file is modified.
import json
import os
import sys
import time

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)                                          # circuits/ and the census state are BQ-relative

# PLAN PRE-FLIGHT. ops/enqueue.sh gates every queued script by executing it under BQLIB_DRYRUN=1, which
# works for bqlib scripts because B.run() has a no-op path. A census_lib script has none, and census_lib
# builds MODS from the live model AT IMPORT -- so without this guard the gate loads the model and runs the
# whole experiment, onto whatever the GPU is already doing. The guard therefore has to come BEFORE the
# census_lib import, and it checks exactly what the run needs to exist.
if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    ntags = len([f for f in os.listdir(os.path.join(BQ, 'circuits'))
                 if f.endswith('.json') and not f.split('.')[0].isupper()])
    print(f'DRYRUN OK: state and circuits/ present, {ntags} curated circuit files, 36 components')
    raise SystemExit(0)

import torch                                                              # noqa: E402

import census_lib as C                                                    # noqa: E402

KEYS = [f'{k}{L}' for k in ('a', 'm') for L in range(18)]

CUR = []
for fn in sorted(os.listdir('circuits')):
    if not fn.endswith('.json') or fn.split('.')[0].isupper():
        continue
    try:
        d = json.load(open('circuits/' + fn))
    except Exception:
        continue
    if isinstance(d, dict) and 'tag' in d:
        CUR.append(d['tag'])

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()
NR = len(C.rows())
NP = C.T
assert NR * NP == nflat, f'grid {nflat} != rows {NR} x positions {NP}'

rowhalf = torch.zeros(NR, dtype=torch.bool)
rowhalf[::2] = True                                   # half A = even rows, half B = odd rows
A = rowhalf.view(NR, 1).expand(NR, NP).reshape(-1)
B = ~A

TAGS, masks = [], {}
for t in CUR:
    try:
        lf = C.leaf(t)
    except Exception:
        continue
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    if mm.sum() == 0 or sl.sum() <= mm.sum():
        continue
    if (mm & A).sum() == 0 or (mm & B).sum() == 0:
        continue                                      # circuit must be present in both halves
    TAGS.append(t); masks[t] = (mm, sl)
print(f'grid {nflat} = {NR} rows x {NP}; half A {int(A.sum())}, half B {int(B.sum())}', flush=True)
print(f'{len(TAGS)} circuits present in both halves', flush=True)

CA, CB = {}, {}
t0 = time.time()
for i, key in enumerate(KEYS):
    d = C.ce_sweep(C.mean_hooks([key])) - base
    for t in TAGS:
        mm, sl = masks[t]
        for H, out in ((A, CA), (B, CB)):
            am = float(d[mm & H].abs().mean())
            ag = float(d[(~sl) & H].abs().mean())
            out.setdefault(t, {})[key] = round(am / ag, 4) if ag > 0 else None
    print(f'  [{i+1:2d}/36] {key}  ({time.time()-t0:.0f}s)', flush=True)

rows = {}
for t in TAGS:
    a_best = max(KEYS, key=lambda k: CA[t][k] if CA[t][k] is not None else -1)
    b_best = max(KEYS, key=lambda k: CB[t][k] if CB[t][k] is not None else -1)
    rows[t] = {'selected_on_A': a_best, 'argmax_on_B': b_best, 'stable': a_best == b_best,
               'conc_A_of_A_selected': CA[t][a_best], 'conc_B_of_A_selected': CB[t][a_best],
               'ratio_heldout_over_insample': (round(CB[t][a_best] / CA[t][a_best], 4)
                                               if CA[t][a_best] else None)}

stable = [t for t in TAGS if rows[t]['stable']]
ratios = sorted(r['ratio_heldout_over_insample'] for r in rows.values()
                if r['ratio_heldout_over_insample'] is not None)
med = ratios[len(ratios) // 2] if ratios else float('nan')
clear = [t for t in TAGS if (rows[t]['conc_B_of_A_selected'] or 0) >= 2.0]

rep = {'schema_version': 1, 'generated': '2026-08-30 by Claude',
       'method': 'mean-ablation of each of the 36 components; every dCE vector scored separately on two '
                 'disjoint halves of the census ROWS (even/odd interleaved). Best component selected on '
                 'half A, concentration reported on half B, so selection and measurement never share a '
                 'position.',
       'state': 'census_state_diverse.pt', 'rows': NR, 'positions_per_row': NP,
       'pred_a_stable': f'{len(stable)}/{len(TAGS)}', 'pred_b_median_ratio': med,
       'pred_c_clear_two_heldout': f'{len(clear)}/{len(TAGS)}', 'by_tag': rows}
json.dump(rep, open('circuits/HELDOUT.json', 'w'), indent=1)

print(f'\nwrote circuits/HELDOUT.json ({time.time()-t0:.0f}s)')
print(f'pred_a  same component chosen on both halves: {len(stable)}/{len(TAGS)} = '
      f'{100*len(stable)/len(TAGS):.0f}%  (bar >=90%) : {len(stable) >= 0.90*len(TAGS)}')
print(f'pred_b  median held-out/in-sample concentration ratio: {med:.4f}  (bar >=0.80) : {med >= 0.80}')
print(f'        ratio spread: min {ratios[0]:.3f}  p25 {ratios[len(ratios)//4]:.3f}  '
      f'p75 {ratios[3*len(ratios)//4]:.3f}  max {ratios[-1]:.3f}')
print(f'pred_c  still >=2.0 on held-out half alone: {len(clear)}/{len(TAGS)} : {len(clear) == len(TAGS)}')
uns = [(t, rows[t]['selected_on_A'], rows[t]['argmax_on_B']) for t in TAGS if not rows[t]['stable']]
if uns:
    print(f'  unstable ({len(uns)}): ' + ', '.join(f'{t}:{a}->{b}' for t, a, b in uns[:12]))
low = sorted((rows[t]['conc_B_of_A_selected'] or 0, t) for t in TAGS)[:5]
print('  lowest held-out concentrations: ' + ', '.join(f'{t} {c:.2f}' for c, t in low))
