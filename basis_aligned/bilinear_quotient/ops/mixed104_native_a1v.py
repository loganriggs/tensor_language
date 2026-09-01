"""RUNG 293: true corrected mixed spectrum, top-96 plus smallest eight QK directions.

Rung 290 was valid contiguous top-96 science but was mislabeled as the historical mixed spectrum.  This
companion physically selects singular indices {0..95,120..127} at every replaced QK map and restores native
block-1 values.  No historical damage vector is subtracted and nothing is fit to evaluation rows.

REGISTERED PREDICTIONS (CE added above native; LOWER IS BETTER):
  (a) LAST-8 BENEFIT: census <=0.0065 and >=54/62 certificates.
  (b) FRESH: every one of eight disjoint fresh-window damages <=0.020.
  (c) LIVE INDEX SET: every installed QK map uses exactly {0..95,120..127}, factor width 104, layers 2..17.
NULL/NO BENEFIT: census >=0.0080 and certificates <=52, so the last eight directions add no useful behavior.
PRICE: exact bill pending; no rounded historical anchor is used for adoption.  Self-reviewed; bqrunner only.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'mixed104_native_a1v_results.json'
CEV = ROOT / 'cev_mixed104_native_a1v.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [ROOT / 'circuits/BATTERY.json', ROOT / 'census_state_diverse.pt',
              ROOT / 'ops/cevdump_ct96.py']
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: rung 293 true corrected mixed104')
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
        'skipset': tuple(range(10,18)), 'motif_off': (), 'clsdmg': True,
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 96,
        'qk_rmap': {}, 'qk_extra_tail': 8, 'qk_tail': True,
        'drop_tailE': True, 'drop_a1v': True,
    })
    print('ARM: true mixed QK indices top96+last8, native block-1 values', flush=True)
    run = C.main()

    wanted = tuple(list(range(96))+list(range(120,128)))
    index_sets = C.SEL.get('_QK_INDEX_SETS', {})
    if set(index_sets) != set(range(2,18)) or any(value != wanted for value in index_sets.values()):
        raise SystemExit('INSTRUMENT FAIL: physical QK index sets differ from top96+last8')
    qk = C.SEL.get('_QKR', {})
    factor_widths = {
        int(factor[0].shape[1])
        for heads in qk.values() for factors in heads.values() for factor in factors
    }
    if factor_widths != {104}:
        raise SystemExit(f'INSTRUMENT FAIL: factor widths {factor_widths} != {{104}}')

    cev = C.SEL['cev'].float().cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev-base_ce
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
        member_abs[tag] = round(value,7)
        reference = receipt['mean_ablation']['top'][0]['abs_dce_members']
        valid += int(value < 0.5*reference)

    damage = float(damage_vector.mean())
    fresh = [float(value) for value in run['fresh8']]
    pred_a = damage <= 0.0065 and valid >= 54
    pred_b = max(fresh) <= 0.020
    pred_c = (set(index_sets) == set(range(2,18)) and
              all(value == wanted for value in index_sets.values()) and factor_widths == {104})
    result = {
        'convention': 'CE added above native model; lower is better',
        'qk_singular_indices': list(wanted),
        'physical_qk_factor_widths': sorted(factor_widths),
        'census_damage': round(damage,8),
        'certificates_valid': valid,
        'fresh8': fresh,
        'max_fresh_damage': round(max(fresh),7),
        'member_abs_dce': member_abs,
        'pred_a_last8_benefit': bool(pred_a),
        'pred_b_fresh': bool(pred_b),
        'pred_c_live_index_set': bool(pred_c),
        'null_no_benefit': bool(damage >= 0.0080 and valid <= 52),
        'price_status': 'exact component bill pending; no rounded anchor adopted',
        'decision_level': 'true mixed identification; causal and exact-price adoption pending',
        'runtime_s': round(time.time()-started,1),
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='member_abs_dce'},indent=2),flush=True)
    print(f'wrote {OUT} and {CEV}',flush=True)


if __name__ == '__main__':
    main()
