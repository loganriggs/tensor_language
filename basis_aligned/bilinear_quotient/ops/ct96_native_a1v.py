"""RUNG 288: ct96 with the block-1 value map restored to its smaller native linear primitive.

Rungs 281/282 found +0.0520 CE in the nominal full-rank path. Rung 285 isolates the path's hidden `a1v`
token table: despite the "values real" label, block-1 c_v was still replaced by a token-only table even
though its input is context-dependent. This arm applies that correction to the registered ct96 program:
front motif QK maps exact, tail QK rank 96, current values native, exact-width MLP factors, and no `a1v`.
Nothing is fit to the census.

REGISTERED PREDICTIONS (CE added above native; LOWER IS BETTER):
  (a) FLOOR BREAK: held-out census damage <= 0.010 and >=50/62 certificates.
  (b) FRESH: every one of the eight disjoint fresh-window damages is <= 0.020.
  (c) PRE-OUTCOME VECTOR PREDICTION: the new signed damage matches `cev_ct96-cev_pathfull`, with cosine
      >=0.95 and relative error <=0.25.
NULL: census >=0.040, so removing a1v does not transport from full-rank path to compressed ct96.
PRICE: replace V*D=57,896,064 table values by D^2=1,327,104 native weight values, a literal delta of
-56,568,960 values (about 154.4M total from the 211M ct96 anchor). This is identification; adoption still
requires corrected-config intervention transfer. Self-reviewed. GPU only through bqrunner.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'ct96_native_a1v_results.json'
CEV = ROOT / 'cev_ct96_native_a1v.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [
        ROOT / 'frontier_tail_traj_results.json', ROOT / 'circuits/BATTERY.json',
        ROOT / 'census_state_diverse.pt', ROOT / 'cev_ct96.pt', ROOT / 'cev_pathfull.pt',
        ROOT / 'ops/cevdump_ct96.py',
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: rung 288 ct96 with native a1v')
    raise SystemExit(0)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, '/workspace/rspd')

import torch
import census_lib as CN
import cevdump_ct96 as C


@torch.no_grad()
def main():
    started = time.time()
    CN.use_state('census_state_diverse.pt')
    rows = CN.rows().cpu()
    base_ce = CN.base_ce().float().cpu()
    nflat = CN.nflat()
    C.CROWS = rows
    C.CBASE = base_ce
    C.NFLAT = nflat
    C.ANCH = json.load(open(ROOT / 'frontier_tail_traj_results.json'))
    C.SEL.update({
        'mode': 'norm', 'K': 4608, 'K69': 4608, 'K69MAP': {},
        'skipset': tuple(range(10, 18)), 'motif_off': (), 'clsdmg': True,
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 96,
        'qk_rmap': {li: 128 for li in range(2, 10)}, 'qk_tail': True,
        'drop_tailE': True, 'drop_a1v': True,
    })
    print('ARM: ct96 with native block-1 values (a1v omitted)', flush=True)
    run = C.main()
    cev = C.SEL['cev'].float().cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    d = cev - base_ce

    old = torch.load(ROOT / 'cev_ct96.pt', map_location='cpu').float().reshape(-1)
    path = torch.load(ROOT / 'cev_pathfull.pt', map_location='cpu').float().reshape(-1)
    predicted = old - path
    cosine = float(torch.dot(d, predicted) / (d.norm() * predicted.norm()).clamp_min(1e-12))
    relative_error = float((d - predicted).norm() / predicted.norm().clamp_min(1e-12))

    battery = json.load(open(ROOT / 'circuits/BATTERY.json'))['by_tag']
    member_abs = {}
    valid = 0
    for tag, receipt in battery.items():
        try:
            member = CN.leaf(tag)['member'].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        value = float(d[member].abs().mean())
        member_abs[tag] = round(value, 6)
        reference = receipt['mean_ablation']['top'][0]['abs_dce_members']
        valid += int(value < 0.5 * reference)

    damage = float(d.mean())
    fresh = [float(x) for x in run['fresh8']]
    pred_a = damage <= 0.010 and valid >= 50
    pred_b = max(fresh) <= 0.020
    pred_c = cosine >= 0.95 and relative_error <= 0.25
    result = {
        'convention': 'CE added above native model; lower is better',
        'census_damage': round(damage, 7),
        'certificates_valid': valid,
        'fresh8': fresh,
        'max_fresh_damage': round(max(fresh), 7),
        'predicted_vector_cosine': round(cosine, 7),
        'predicted_vector_relative_error': round(relative_error, 7),
        'price_old_a1v_table_values': C.V * C.D,
        'price_native_c_v1_values': C.D * C.D,
        'price_delta_values': C.D * C.D - C.V * C.D,
        'approx_total_values_from_211m_anchor': 211_000_000 + C.D * C.D - C.V * C.D,
        'member_abs_dce': member_abs,
        'pred_a_floor_and_certificates': bool(pred_a),
        'pred_b_fresh': bool(pred_b),
        'pred_c_vector_prediction': bool(pred_c),
        'null_triggered': bool(damage >= 0.040),
        'decision_level': 'predictive/certificate/price identification; intervention adoption pending',
        'runtime_s': round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'member_abs_dce'}, indent=2), flush=True)
    print(f'wrote {OUT} and {CEV}', flush=True)


if __name__ == '__main__':
    main()
