"""THE ROUTE-USE TEST FOR THE NEWLY-OPENED ATTENTION-TO-ATTENTION PATH.

`tf_interp3.induction_route_split` asks the right question but only of the
depth-2 route (layer-0 attention into layer 1's read), because that is the only
attention-to-attention route a two-layer model has.  At depth 3 the causal
composition budget shows a route that does NOT exist at depth 2: layer-1
attention into layer 2's Q/K/V read carries 0.08-0.18 nats, five orders of
magnitude above layer-0 attention's 1e-6 into anything.

An open route that carries no algorithm is a weaker result than an open route
the algorithm runs on (FINDING 11 §2 made exactly this distinction).  So for
every attention-to-attention route in the model, this deletes that write from
that read ONLY -- residual untouched, everything downstream recomputed -- and
re-measures the induction score with the same battery and the same planted-
oracle power floor.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

import tf_interp as I1
import tf_interp2 as I2
import tf_interp3 as I3

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def route_use(stem, seeds=5):
    D = I3.VariantFold(stem)
    if D.L < 2:
        return None
    out = {'stem': stem, 'depth': D.L, 'width': D.cfg.width, 'arms': {}}

    def mk(li, drop):
        def fn(P_):
            v = P_['rem'][li]
            for j in range(li):
                if drop != f'A{j}':
                    v = v + P_['A'][j]
                if drop != f'M{j}':
                    v = v + P_['M'][j]
            return D._pre(2 * li, v, {})
        return fn

    arms = {'baseline': None}
    for li in range(1, D.L):
        for j in range(li):
            arms[f'A{j}_out_of_layer{li}_read'] = (li, f'A{j}')
        arms[f'M{li-1}_out_of_layer{li}_read'] = (li, f'M{li-1}')
    for nm, spec in arms.items():
        if spec is None:
            fwd = lambda z: D.readout(D.run(z)['r'])
        else:
            li, drop = spec
            fwd = (lambda li_, dr: lambda z: D.readout(
                D.run(z, reads={li_: mk(li_, dr)})['r']))(li, drop)
        r = [I1.induction_battery(D, seed=s, model=fwd) for s in range(seeds)]
        out['arms'][nm] = {
            'induction_score_mean': float(np.mean(
                [x['induction_score'] for x in r])),
            'induction_score_sd': float(np.std(
                [x['induction_score'] for x in r], ddof=1)),
            'bag_score_mean': float(np.mean([x['bag_score'] for x in r]))}
    b = out['arms']['baseline']['induction_score_mean']
    out['baseline_induction'] = b
    out['fraction_of_induction_removed'] = {
        k: (b - v['induction_score_mean']) / b if b else None
        for k, v in out['arms'].items() if k != 'baseline'}
    out['note'] = (
        'An attention-to-attention route that carries a large causal KL but '
        'removes no induction is OPEN BUT NOT USED BY THIS ALGORITHM; a route '
        'whose deletion removes a large fraction of the induction score is one '
        'the algorithm runs on.  Both are reported because the distinction is '
        'exactly the one FINDING 11 sec 2 had to make.')
    json.dump(out, open(f'{HERE}/{stem}_routeuse.json', 'w'), indent=2)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem')
    ap.add_argument('--all-deep', action='store_true')
    a = ap.parse_args()
    stems = ([a.stem] if a.stem else
             sorted(s[:-3] for s in
                    [os.path.basename(p) for p in
                     glob.glob(f'{HERE}/tf_vanilla_d[34]_w*_b8192_s*.pt')]))
    for s in stems:
        if os.path.exists(f'{HERE}/{s}_routeuse.json'):
            print('skip', s, flush=True)
            continue
        t = time.time()
        r = route_use(s)
        if r:
            print(f'{s}  baseline {r["baseline_induction"]:+.4f}  '
                  + '  '.join(f'{k}={v:+.3f}' for k, v in
                              r['fraction_of_induction_removed'].items())
                  + f'  ({time.time()-t:.0f}s)', flush=True)
