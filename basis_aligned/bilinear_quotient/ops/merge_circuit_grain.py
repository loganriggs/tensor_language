# CIRCUIT-GRAIN VALIDATION OF THE SCALAR MERGE (rung 96; user-suggested cross-view).
#
# CONVENTION (S2135): per-position dCE = CE(intervened) - CE(base) on the census grid; positive = damage.
# The 62-circuit battery (circuits/BATTERY.json, census_state_diverse.pt) gives member/slice masks and
# mean-ablation reference damages per circuit. S2184's merge (zero attn3 + 1.45 x attn2) recovered 55% of
# AGGREGATE CE on the assembly base. Question: does it repair the a3-localized CIRCUITS, or is it a CE hack
# that leaves them broken? Interventions here are applied to the REAL model (the battery's frame).
#
# REGISTERED PREDICTIONS (arm-named):
#   (a) CIRCUIT-GRAIN REPAIR: for the a3-localized circuits, median over circuits of
#       [member mean|dCE| under MERGE] <= 0.6 x [member mean|dCE| under DROP-alone].
#   (b) NO COLLATERAL BREAKAGE: no circuit localized outside {a2,a3} has member mean|dCE| under MERGE
#       >= 0.5 x its BATTERY mean-ablation member reference (the 1.45x amplification must not break others).
#   (c) BATTERY CONSISTENCY: for a3-circuits, member mean|dCE| under DROP-alone >= 0.8 x their BATTERY
#       mean-ablation reference (zeroing at least as sharp as mean-ablation, anchoring this run to the battery).
# NULL: aggregate recovery without circuit repair (the a3 circuits stay broken under the merge).
# PRICE: none (validation). Self-reviewed.
import json
import os
import sys

BQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BQ)
sys.path.insert(0, '/workspace/rspd')
os.chdir(BQ)

if os.environ.get('BQLIB_DRYRUN') == '1':
    need = ['census_state_diverse.pt', 'circuits/BATTERY.json', 'merge_scalar_results.json']
    missing = [f for f in need if not os.path.exists(os.path.join(BQ, f))]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    b = json.load(open(os.path.join(BQ, 'circuits/BATTERY.json')))
    n3 = sum(1 for v in b['by_tag'].values() if v['best_mean'] == 'a3')
    print(f'DRYRUN OK: battery present; {n3} circuits localise to a3; merge alpha from S2184')
    raise SystemExit(0)

import time                                                               # noqa: E402
import statistics as stt                                                  # noqa: E402

import torch                                                              # noqa: E402
import torch.nn.functional as F                                           # noqa: E402

import census_lib as C                                                    # noqa: E402

T0 = time.time()
ALPHA = json.load(open('merge_scalar_results.json'))['alpha_star']
BAT = json.load(open('circuits/BATTERY.json'))['by_tag']
C.use_state('census_state_diverse.pt')
base = C.base_ce()
nflat = C.nflat()
masks = {}
ref = {}
loc = {}
for t, v in BAT.items():
    try:
        lf = C.leaf(t)
    except Exception:
        continue
    mm = torch.zeros(nflat, dtype=torch.bool); mm[lf['member']] = True
    if mm.sum() == 0:
        continue
    masks[t] = mm
    loc[t] = v['best_mean']
    ref[t] = v['mean_ablation']['top'][0]['abs_dce_members']
print(f'{len(masks)} circuits usable; a3-localized: {sorted(t for t in masks if loc[t]=="a3")}', flush=True)


@torch.no_grad()
def dce_with(hooks_spec):
    hs = []
    for key, mode in hooks_spec:
        if mode == 'zero':
            hs.append(C.MODS[key].register_forward_hook(
                lambda mo, i_, o_: (torch.zeros_like(o_[0]), o_[1]) if isinstance(o_, tuple)
                else torch.zeros_like(o_)))
        else:
            al = mode
            hs.append(C.MODS[key].register_forward_hook(
                lambda mo, i_, o_, al=al: (o_[0] * al, o_[1]) if isinstance(o_, tuple) else o_ * al))
    R = C.rows(); ces = []
    for i in range(0, R.shape[0], 4):
        bb = R[i:i + 4, :257].to(C.DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1)
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,)); x0 = x; v1 = None
        for blkm in C.m.transformer.h:
            x, v1 = blkm(x, v1, x0)
        lg = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
        ces.append(F.cross_entropy(lg.view(-1, lg.size(-1)), tg, reduction='none').cpu())
    for h in hs:
        h.remove()
    return torch.cat(ces).float() - base


print('pass 1/2: DROP (zero a3)', flush=True)
d_drop = dce_with([('a3', 'zero')])
print('pass 2/2: MERGE (zero a3 + %.2f x a2)' % ALPHA, flush=True)
d_merge = dce_with([('a3', 'zero'), ('a2', ALPHA)])
if float(d_merge.abs().sum()) < 1e-3 or float((d_merge - d_drop).abs().sum()) < 1e-3:
    raise SystemExit('INSTRUMENT FAIL: arms inert or identical')

rows = {}
for t, mm in masks.items():
    rows[t] = {'loc': loc[t], 'ref': round(ref[t], 3),
               'drop': round(float(d_drop[mm].abs().mean()), 4),
               'merge': round(float(d_merge[mm].abs().mean()), 4)}
a3tags = [t for t in rows if rows[t]['loc'] == 'a3']
repair = stt.median([rows[t]['merge'] for t in a3tags])
dropm = stt.median([rows[t]['drop'] for t in a3tags])
others = [t for t in rows if rows[t]['loc'] not in ('a2', 'a3')]
broken = [t for t in others if rows[t]['merge'] >= 0.5 * rows[t]['ref']]
pa = repair <= 0.6 * dropm
pb = len(broken) == 0
pc = all(rows[t]['drop'] >= 0.8 * rows[t]['ref'] for t in a3tags)
out = {'alpha': ALPHA, 'a3_circuits': {t: rows[t] for t in a3tags},
       'median_member_absdce': {'drop': round(dropm, 4), 'merge': round(repair, 4)},
       'collateral_broken': {t: rows[t] for t in broken},
       'n_circuits_checked': len(rows),
       'convention': 'per-position dCE vs the real model on the census grid; positive = damage',
       'pred_a_circuit_repair': bool(pa), 'pred_b_no_collateral': bool(pb),
       'pred_c_battery_consistent': bool(pc), 'self_reviewed': True,
       'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open('merge_circuit_grain_results.json', 'w'), indent=1)
for t in a3tags:
    print(f"  {t}: ref {rows[t]['ref']} | drop {rows[t]['drop']} | merge {rows[t]['merge']}")
print(f"(a) a3-circuit median merge {repair:.4f} <= 0.6 x drop {dropm:.4f}: {'HELD' if pa else 'FAILED'}")
print(f"(b) collateral broken: {len(broken)} ({broken[:5]}): {'HELD' if pb else 'FAILED'}")
print(f"(c) drop >= 0.8 x battery ref on all a3 circuits: {'HELD' if pc else 'FAILED'}")
print(f'wrote merge_circuit_grain_results.json ({time.time()-T0:.0f}s)')
