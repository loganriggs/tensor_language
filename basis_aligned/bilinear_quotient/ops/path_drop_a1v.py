"""RUNG 285: remove the hidden block-1 value-table approximation from the full-rank path.

CONFOUND: the registered ct96/t120 CE vectors were nearly collinear, but the first full-rank path control
itself costs +0.0520 CE and its vector has cosine 0.971/0.995 with ct96/t120. After path subtraction the
rank deficiencies cost only +0.00327/+0.00057. Code inspection finds that "patterns full rank, values
real" still installs `a1v`: a token-indexed table replacing block-1 c_v even though block-1 input is
context-dependent. This experiment changes that one hidden path component.

ARM: rebuild the same full-rank ct lineage (rank-128 QK at every replaced head, native current values,
exact-width MLP factors, tail table removed), but omit `a1v` from the installed program. No learned
replacement is added, so block-1 c_v executes natively. Save the signed census CE vector.

REGISTERED PREDICTIONS (CE added above native; LOWER IS BETTER):
  (a) LIVE PATH: the already-landed unchanged path receipt is in [0.040, 0.070].
  (b) CULPRIT: this arm has census damage <= 0.010 and >=55/62 certificates.
  (c) VECTOR: ||d_no_a1v|| / ||d_pathfull|| <= 0.25.
NULL: census damage >=0.040, so a different replacement primitive dominates the path mode.
PRICE: identification probe; removing a table restores native values and does not claim compression.
Self-reviewed. GPU only through bqrunner.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'path_drop_a1v_results.json'
CEV = ROOT / 'cev_path_drop_a1v.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [
        ROOT / 'frontier_tail_traj_results.json',
        ROOT / 'circuits/BATTERY.json',
        ROOT / 'census_state_diverse.pt',
        ROOT / 'cev_pathfull.pt',
        ROOT / 'ops/cevdump_ct96.py',
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: rung 285 full QK path with a1v restored to native')
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
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 128,
        'qk_rmap': {li: 128 for li in range(2, 10)}, 'qk_tail': True,
        'drop_tailE': True, 'drop_a1v': True,
    })
    print('ARM: full-rank QK path, native block-1 values (a1v omitted)', flush=True)
    C.main()
    cev = C.SEL['cev'].float().cpu()
    assert cev.numel() == nflat
    torch.save(cev, CEV)
    d = cev - base_ce
    path = torch.load(ROOT / 'cev_pathfull.pt', map_location='cpu').float().reshape(-1) - base_ce
    path_result = json.load(open(ROOT / 'path_full_results.json'))
    path_damage = float(path_result['census_agg'])
    norm_ratio = float(d.norm() / path.norm())
    cosine = float(torch.dot(d, path) / (d.norm() * path.norm()).clamp_min(1e-12))

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
    pred_a = 0.040 <= path_damage <= 0.070
    pred_b = damage <= 0.010 and valid >= 55
    pred_c = norm_ratio <= 0.25
    result = {
        'convention': 'CE added above native model; lower is better',
        'pathfull_anchor_damage': round(path_damage, 7),
        'census_damage': round(damage, 7),
        'certificates_valid': valid,
        'vector_norm_ratio_vs_pathfull': round(norm_ratio, 7),
        'cosine_vs_pathfull': round(cosine, 7),
        'member_abs_dce': member_abs,
        'pred_a_live_path': bool(pred_a),
        'pred_b_a1v_culprit': bool(pred_b),
        'pred_c_vector_removed': bool(pred_c),
        'null_triggered': bool(damage >= 0.040),
        'runtime_s': round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'member_abs_dce'}, indent=2), flush=True)
    print(f'wrote {OUT} and {CEV}', flush=True)
    if not pred_a:
        raise SystemExit('INSTRUMENT FAIL: path-full anchor outside registered band')


if __name__ == '__main__':
    main()
