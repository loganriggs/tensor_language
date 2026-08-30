# LOCALISE THE UNLOCALISED CIRCUITS BY DIRECT MEAN-ABLATION.
#
# TASK CONTEXT (Logan, 2026-08-30): produce as much useful information on as many circuits as possible,
# using ablation / interchange / DAS to isolate where each one lives. Codex works the same folder.
#
# WHY THESE TWELVE. circuits/ holds 70 circuits. Twelve carry no `components` field, and the reason is not
# that nobody looked: every one of them is selectivity-HELD with causal concentration 4.83 to 6.84, and
# every one FAILED the pipeline's ATTRIBUTION tests -- input_decomp_enrichment, mechanism_enrichment,
# surface_program, behavior_story. They are real, causally concentrated circuits whose mechanism the
# writer-attribution machinery could not name.
#
# WHAT IS NEW HERE. Attribution asks "which upstream writer explains this component's input?". This asks
# the direct causal question instead: mean-ablate each of the 36 components in turn and measure whether
# the damage lands ON the circuit's members or off its slice. That is the one method the pipeline has not
# applied to these twelve, and it is what Logan asked for.
#
# EFFICIENCY. The 36 ablation dCE vectors do not depend on which circuit is being scored, so they are
# computed ONCE over the census grid and all twelve circuits are scored against the same 36 vectors.
# Thirty-six sweeps, not four hundred and thirty-two.
#
# REGISTERED PREDICTIONS (before running):
#   pred_a  Every one of the twelve has at least one component whose ablation concentrates on its members
#           at >= 2.0x the off-slice rate. If FALSE for some circuit, that circuit is not localisable by
#           single-component ablation either, which is a stronger negative than the attribution failure.
#   pred_b  The best component for each circuit beats the SECOND best by >= 20% in concentration, i.e. the
#           localisation is to one component rather than a diffuse band. Registered against the writer
#           tests' finding of no single enriched writer -- if ablation also finds no single site, the two
#           methods agree and these circuits are genuinely distributed.
#   pred_c  The circuits' own recorded concentration (4.83-6.84, from ablating their OWN probes) exceeds
#           the best single-component concentration found here -- a circuit should be more concentrated on
#           itself than any whole component is.
#
# Writes circuits/LOCALISATION.json. Does NOT modify any circuit file: Codex is working the same folder
# and a shared read-only artifact cannot collide with them.
import json
import time

import torch

import census_lib as C

TAGS = ['r.11.1.1', 'r.11.1.2', 'r.11.3.1', 'r.13.2.1', 'r.18.2.0', 'r.1.3.1',
        'r.23.2.1', 'r.23.2.3', 'r.3.0', 'r.3.0.2', 'r.4.1.1', 'r.7.1.1']
KEYS = [f'{k}{L}' for k in ('a', 'm') for L in range(18)]

C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()
print(f'grid {nflat} positions, base CE {base.mean():.4f}', flush=True)

avail, missing = [], []
for t in TAGS:
    try:
        C.leaf(t)
        avail.append(t)
    except Exception:
        missing.append(t)
print(f'tags present in this state: {len(avail)}  missing: {missing}', flush=True)

masks = {}
for t in avail:
    lf = C.leaf(t)
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    sl = torch.zeros(nflat, dtype=torch.bool); sl[lf['slice']] = True
    masks[t] = (mm, sl)

out = {}
t0 = time.time()
for i, key in enumerate(KEYS):
    d = C.ce_sweep(C.mean_hooks([key])) - base
    for t in avail:
        mm, sl = masks[t]
        am = float(d[mm].abs().mean())
        ag = float(d[~sl].abs().mean())
        out.setdefault(t, {})[key] = {
            'abs_dce_members': round(am, 4),
            'abs_dce_offslice': round(ag, 4),
            'concentration': round(am / ag, 3) if ag > 0 else None,
            'signed_dce_members': round(float(d[mm].mean()), 4)}
    print(f'  [{i+1:2d}/36] {key}  ({time.time()-t0:.0f}s)', flush=True)

report = {'schema_version': 1,
          'generated': '2026-08-30 by Claude, circuit task',
          'method': 'mean-ablation of each of the 36 components over the census grid; concentration is '
                    'mean|dCE| on circuit members divided by mean|dCE| off the circuit slice',
          'state': 'census_state_diverse.pt',
          'note': 'read-only artifact; no circuit file was modified',
          'tags_missing_from_state': missing,
          'by_tag': {}}
for t, per in out.items():
    rank = sorted((v['concentration'], k) for k, v in per.items() if v['concentration'] is not None)
    rank.reverse()
    report['by_tag'][t] = {
        'recorded_own_concentration': (C.leaf(t).get('conc') if isinstance(C.leaf(t), dict) else None),
        'top': [{'component': k, 'concentration': c, **per[k]} for c, k in rank[:6]],
        'all': per}
json.dump(report, open('circuits/LOCALISATION.json', 'w'), indent=1)
print(f'\nwrote circuits/LOCALISATION.json for {len(out)} circuits ({time.time()-t0:.0f}s)', flush=True)
for t, r in report['by_tag'].items():
    top = r['top'][0] if r['top'] else None
    print(f"  {t:10s} best {top['component'] if top else '-':4s} conc {top['concentration'] if top else '-'}")
