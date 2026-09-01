"""RUNG 292: compose corrected mixed-spectrum with per-token top-1152 CP execution.

Rung 290 is the faithful rank-96/native-a1v base at +0.00853845 census and 52/62 certificates.  Four old
anchors measured a nearly constant +0.0161--0.0168 surcharge from selecting one quarter of each live CP
MLP's units per token.  This run tests that composition on the corrected base, not by receipt addition.

REGISTERED PREDICTIONS (CE added above native; LOWER IS BETTER):
  (a) ADDITIVE COMPUTE PRICE: surcharge in [0.010,0.024] and total census <=0.035.
  (b) CERTIFICATES: >=40/62 remain valid.
  (c) FRESH + LIVE TRIPWIRE: all eight fresh damages <=0.040 and observed selected width is exactly 1152.
NULL: surcharge >=0.040 or certificates <=20; sparsity no longer composes after the path correction.
PRICE: storage unchanged from corrected mixed (~123.4M pending exact bill); executed units in each replaced
4608-wide CP MLP are 1152 per token, a literal 4x unit-compute reduction.  Self-reviewed; bqrunner only.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'mixed_topk_native_a1v_results.json'
CEV = ROOT / 'cev_mixed_topk_native_a1v.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [
        ROOT / 'mixed_native_a1v_results.json', ROOT / 'circuits/BATTERY.json',
        ROOT / 'census_state_diverse.pt', ROOT / 'ops/cevdump_ct96.py',
    ]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        print(f'DRYRUN WAIT: prerequisite missing {missing}')
        raise SystemExit(1)
    baseline = json.load(open(ROOT / 'mixed_native_a1v_results.json'))
    if (baseline.get('null_triggered') or baseline.get('census_damage', 1.0) > 0.012 or
            baseline.get('certificates_valid', 0) < 50):
        print('DRYRUN FAIL: corrected mixed prerequisite did not hold')
        raise SystemExit(1)
    print('DRYRUN OK: rung 292 corrected mixed plus top-1152 CP')
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
    baseline = json.load(open(ROOT / 'mixed_native_a1v_results.json'))
    C.CROWS = rows
    C.CBASE = base_ce
    C.NFLAT = nflat
    C.ANCH = json.load(open(ROOT / 'frontier_tail_traj_results.json'))
    C.SEL.update({
        'mode': 'norm', 'K': 4608, 'K69': 4608, 'K69MAP': {},
        'skipset': tuple(range(10, 18)), 'motif_off': (), 'clsdmg': True,
        'ext_rows': rows, 'cp_swap': 4608, 'cp_topk': 1152,
        'qk_r': 96, 'qk_rmap': {}, 'qk_tail': True,
        'drop_tailE': True, 'drop_a1v': True,
    })
    print('ARM: corrected mixed with per-token CP top-1152', flush=True)
    run = C.main()
    if C.SEL.get('_cp_topk_observed') != 1152:
        raise SystemExit(f"INSTRUMENT FAIL: selected width {C.SEL.get('_cp_topk_observed')} != 1152")
    qk = C.SEL.get('_QKR', {})
    factor_ranks = {
        int(factor[0].shape[1])
        for heads in qk.values() for factors in heads.values() for factor in factors
    }
    if set(qk) != set(range(2, 18)) or factor_ranks != {96}:
        raise SystemExit(f'INSTRUMENT FAIL: QK layers/ranks {sorted(qk)}/{factor_ranks}')

    cev = C.SEL['cev'].float().cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    damage = float(damage_vector.mean())
    surcharge = damage - float(baseline['census_damage'])

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

    fresh = [float(value) for value in run['fresh8']]
    pred_a = 0.010 <= surcharge <= 0.024 and damage <= 0.035
    pred_b = valid >= 40
    pred_c = max(fresh) <= 0.040 and C.SEL['_cp_topk_observed'] == 1152
    result = {
        'convention': 'CE added above native model; lower is better',
        'census_damage': round(damage, 8),
        'baseline_census_damage': baseline['census_damage'],
        'topk_surcharge': round(surcharge, 8),
        'certificates_valid': valid,
        'fresh8': fresh,
        'max_fresh_damage': round(max(fresh), 7),
        'selected_cp_width': C.SEL['_cp_topk_observed'],
        'full_cp_width': 4608,
        'cp_unit_compute_reduction': 4.0,
        'physical_qk_factor_ranks': sorted(factor_ranks),
        'price_status': 'exact physical top96 bill pending; former mixed-anchor estimate withdrawn',
        'member_abs_dce': member_abs,
        'pred_a_additive_compute_price': bool(pred_a),
        'pred_b_certificates': bool(pred_b),
        'pred_c_fresh_and_live': bool(pred_c),
        'null_triggered': bool(surcharge >= 0.040 or valid <= 20),
        'decision_level': 'compute-composition candidate; exact price and shifted-corpus OOD pending',
        'runtime_s': round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'member_abs_dce'},
                     indent=2), flush=True)
    print(f'wrote {OUT} and {CEV}', flush=True)


if __name__ == '__main__':
    main()
