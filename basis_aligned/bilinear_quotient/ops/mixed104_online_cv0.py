"""RUNG 296: storage-minimal corrected mixed104 with online native block-0 c_v.

The exact CPU bill rejected the tested artifact's old price: its fp16 a0 token
table makes the standalone object 596,164,022 scalars, larger than native.  The
only storage-winning semantic realization drops that table and executes the
native 1152x1152 c_v0 matrix from the already-required normalized embedding.
Everything else is the rung-293 mixed104 configuration.

REGISTERED PREDICTIONS (CE above native; lower is better):
  (a) LITERAL CANDIDATE: census <= 0.0065 and >=54/62 certificates.
  (b) TABLE EQUIVALENCE: versus the saved physical-table CE vector, mean
      absolute per-position difference <=0.002 and |mean difference| <=0.0015.
  (c) TRANSFER/IDENTITY: every fresh8 damage <=0.020; exact QK indices are
      {0..95,120..127}; factor width 104; final active set excludes a0/a1v/tailE.
NULL: census >=0.010 or certificates <=50, so online c_v0 is not a faithful
drop-in for the tested table path.  PRICE: exact standalone bill 539,595,062
scalars / 2,042,438,252 raw tensor bytes.  Self-reviewed; bqrunner only.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'mixed104_online_cv0_results.json'
CEV = ROOT / 'cev_mixed104_online_cv0.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [ROOT / 'circuits/BATTERY.json', ROOT / 'census_state_diverse.pt',
              ROOT / 'cev_mixed104_native_a1v.pt',
              ROOT / 'mixed104_exact_bill_results.json',
              ROOT / 'ops/cevdump_ct96.py']
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    bill = json.load(open(ROOT / 'mixed104_exact_bill_results.json'))
    assert bill['storage_minimal_semantic_candidate']['scalars'] == 539_595_062
    print('DRYRUN OK: rung 296 mixed104 online-c_v0 literal candidate')
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
        'drop_tailE': True, 'drop_a1v': True, 'drop_a0': True,
    })
    print('ARM: mixed104 with online native block-0 c_v; no a0/a1v tables', flush=True)
    run = C.main()

    wanted = tuple(list(range(96)) + list(range(120,128)))
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
    active = tuple(C.SEL.get('_ORDER2', ()))
    if any(name in active for name in ('a0', 'a1v', 'tailE')):
        raise SystemExit(f'INSTRUMENT FAIL: forbidden table/dictionary active in {active}')
    if not all(name in active for name in
               ('m0E','m1','m2E','m3E','c4','c5','c6','c7','c8','c9')):
        raise SystemExit(f'INSTRUMENT FAIL: full CP stack missing from {active}')

    cev = C.SEL['cev'].float().cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    damage_vector = cev - base_ce
    table_cev = torch.load(ROOT / 'cev_mixed104_native_a1v.pt', map_location='cpu',
                           weights_only=True).float().reshape(-1)
    assert table_cev.numel() == nflat
    table_delta = cev - table_cev

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
    mad_table = float(table_delta.abs().mean())
    mean_table = float(table_delta.mean())
    fresh = [float(value) for value in run['fresh8']]
    pred_a = damage <= 0.0065 and valid >= 54
    pred_b = mad_table <= 0.002 and abs(mean_table) <= 0.0015
    pred_c = (max(fresh) <= 0.020 and factor_widths == {104} and
              set(index_sets) == set(range(2,18)) and
              all(value == wanted for value in index_sets.values()) and
              not any(name in active for name in ('a0','a1v','tailE')))
    null = damage >= 0.010 or valid <= 50
    bill = json.load(open(ROOT / 'mixed104_exact_bill_results.json'))
    literal = bill['storage_minimal_semantic_candidate']

    result = {
        'convention': 'CE added above native model; lower is better',
        'implementation': 'online native c_v0; no a0/a1v/tailE table hooks',
        'active_replacements': list(active),
        'qk_singular_indices': list(wanted),
        'physical_qk_factor_widths': sorted(factor_widths),
        'census_damage': round(damage, 8),
        'certificates_valid': valid,
        'fresh8': fresh,
        'max_fresh_damage': round(max(fresh), 7),
        'table_ce_vector_mean_abs_difference': round(mad_table, 9),
        'table_ce_vector_mean_signed_difference': round(mean_table, 9),
        'member_abs_dce': member_abs,
        'literal_standalone_scalars': literal['scalars'],
        'literal_raw_tensor_bytes': literal['raw_bytes'],
        'pred_a_literal_candidate': bool(pred_a),
        'pred_b_table_equivalence': bool(pred_b),
        'pred_c_transfer_and_identity': bool(pred_c),
        'null_triggered': bool(null),
        'decision_level': 'literal-price identification; shifted-corpus OOD still required',
        'runtime_s': round(time.time()-started, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'member_abs_dce'}, indent=2), flush=True)
    print(f'wrote {OUT} and {CEV}', flush=True)


if __name__ == '__main__':
    main()
