"""RUNG 290: corrected mixed-spectrum program (top-96 QK everywhere, native block-1 values).

The registered 180M mixed point used rank-96 score-map factors at every replaced head but also carried
the context-blind ``a1v`` token table now identified as a +0.0520 instrument.  This arm changes only
that primitive to the smaller native block-1 c_v matrix.  Nothing is fit to census or fresh rows.

REGISTERED PREDICTIONS (CE added above native; LOWER IS BETTER):
  (a) CORRECTED FRONTIER: census damage <= 0.012 and >=50/62 certificates.
  (b) FRESH: every one of the eight disjoint fresh-window damages is <= 0.020.
  (c) AGGREGATE TRANSPORT: damage is within 0.008 of 0.0573 - 0.0520 = +0.0053.
NULL: census >=0.040, so a genuine pattern-compression floor remains after removing a1v.
PRICE: replace V*D=57,896,064 table values by D^2=1,327,104 native values, delta -56,568,960;
approximately 123.4M total from the registered ~180M mixed anchor.  Exact component pricing follows
before adoption.  Live tripwire: every installed QK factor at blocks 2--17 must physically have rank
96 (unlike rung 288's exact-front rank 128).  Self-reviewed.  GPU only through bqrunner.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'mixed_native_a1v_results.json'
CEV = ROOT / 'cev_mixed_native_a1v.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [
        ROOT / 'frontier_claim_mixed_results.json', ROOT / 'path_full_results.json',
        ROOT / 'circuits/BATTERY.json', ROOT / 'census_state_diverse.pt',
        ROOT / 'ops/cevdump_ct96.py',
    ]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: rung 290 corrected mixed-spectrum point')
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
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 96, 'qk_rmap': {},
        'qk_tail': True, 'drop_tailE': True, 'drop_a1v': True,
    })
    print('ARM: rank-96 QK at all replaced heads, native block-1 values', flush=True)
    run = C.main()

    qk = C.SEL.get('_QKR', {})
    expected_layers = set(range(2, 18))
    if set(qk) != expected_layers:
        raise SystemExit(f'INSTRUMENT FAIL: QK layers {sorted(qk)} != 2..17')
    factor_ranks = {
        int(factor[0].shape[1])
        for heads in qk.values() for factors in heads.values() for factor in factors
    }
    if factor_ranks != {96}:
        raise SystemExit(f'INSTRUMENT FAIL: physical QK factor ranks {factor_ranks} != {{96}}')

    cev = C.SEL['cev'].float().cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce

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
        value = float(damage_vector[member].abs().mean())
        member_abs[tag] = round(value, 7)
        reference = receipt['mean_ablation']['top'][0]['abs_dce_members']
        valid += int(value < 0.5 * reference)

    damage = float(damage_vector.mean())
    fresh = [float(value) for value in run['fresh8']]
    mixed_anchor = json.load(open(ROOT / 'frontier_claim_mixed_results.json'))['census_agg']
    path_anchor = json.load(open(ROOT / 'path_full_results.json'))['census_agg']
    transported = float(mixed_anchor - path_anchor)
    pred_a = damage <= 0.012 and valid >= 50
    pred_b = max(fresh) <= 0.020
    pred_c = abs(damage - transported) <= 0.008
    result = {
        'convention': 'CE added above native model; lower is better',
        'physical_qk_factor_ranks': sorted(factor_ranks),
        'census_damage': round(damage, 8),
        'certificates_valid': valid,
        'fresh8': fresh,
        'max_fresh_damage': round(max(fresh), 7),
        'transport_prediction': round(transported, 7),
        'transport_absolute_error': round(abs(damage - transported), 7),
        'price_old_a1v_table_values': C.V * C.D,
        'price_native_c_v1_values': C.D * C.D,
        'price_delta_values': C.D * C.D - C.V * C.D,
        'approx_total_values_from_180m_anchor': 180_000_000 + C.D * C.D - C.V * C.D,
        'member_abs_dce': member_abs,
        'pred_a_frontier_and_certificates': bool(pred_a),
        'pred_b_fresh': bool(pred_b),
        'pred_c_aggregate_transport': bool(pred_c),
        'null_triggered': bool(damage >= 0.040),
        'decision_level': 'prediction/certificate/price identification; exact price, composition, and intervention adoption pending',
        'runtime_s': round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'member_abs_dce'}, indent=2), flush=True)
    print(f'wrote {OUT} and {CEV}', flush=True)


if __name__ == '__main__':
    main()
