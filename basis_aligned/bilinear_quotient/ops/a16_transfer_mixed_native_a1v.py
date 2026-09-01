"""RUNG 291: signed a16 intervention falsifier on corrected mixed-spectrum.

The earlier mixed program's a16 collateral transfer was the lone six-component outlier, but that program
carried the now-identified context-blind ``a1v`` error.  This run applies the same native a16 mean ablation
to corrected mixed and native, then compares direct signed causal-effect vectors:
    e_comp = CE(corrected_mixed + KO) - CE(corrected_mixed)
    e_real = CE(native + KO) - CE(native).
No target, circuit label, census row, or intervention outcome enters the compiled execution.

REGISTERED PREDICTIONS (lower base CE damage is better):
  (a) LIVE CONFIG: saved census damage <=0.012 and max fresh-window damage <=0.020.
  (b) SIGNED EFFECT: cosine(e_comp,e_real) >=0.90 and normalized vector error <=0.60.
  (c) CIRCUITS: non-own collateral Spearman >=0.90; a16-own median absolute-effect ratio in [0.60,1.40].
NULL: effect cosine <0.70 or collateral Spearman <0.75; the a16 anomaly is genuine and tangent-aware
compilation becomes the live causal route.  PRICE: diagnostic only; deployed candidate unchanged.
Live tripwire: all physical QK factors have rank 96.  Self-reviewed; bqrunner only.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'a16_transfer_mixed_native_a1v_results.json'
COMP_KO = ROOT / 'cev_a16ko_mixed_native_a1v.pt'
NATIVE_KO = ROOT / 'cev_a16ko_native.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [
        ROOT / 'frontier_tail_traj_results.json', ROOT / 'circuits/BATTERY.json',
        ROOT / 'census_state_diverse.pt', ROOT / 'mixed_native_a1v_results.json',
        ROOT / 'cev_mixed_native_a1v.pt', ROOT / 'ops/cevdump_ct96.py',
    ]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        print(f'DRYRUN WAIT: prerequisite rung 290 missing {missing}')
        raise SystemExit(1)
    baseline = json.load(open(ROOT / 'mixed_native_a1v_results.json'))
    if (baseline.get('null_triggered') or baseline.get('census_damage', 1.0) > 0.012 or
            baseline.get('certificates_valid', 0) < 50 or
            baseline.get('max_fresh_damage', 1.0) > 0.020):
        print('DRYRUN FAIL: rung 290 did not pass its frozen bars')
        raise SystemExit(1)
    print('DRYRUN OK: rung 291 signed a16 transfer on corrected mixed')
    raise SystemExit(0)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, '/workspace/rspd')

import torch
import torch.nn.functional as F
import census_lib as CN
import cevdump_ct96 as C


def spearman(x, y):
    xr = torch.tensor(x).argsort().argsort().float()
    yr = torch.tensor(y).argsort().argsort().float()
    xr -= xr.mean()
    yr -= yr.mean()
    return float((xr * yr).sum() / (xr.norm() * yr.norm()).clamp_min(1e-12))


@torch.no_grad()
def direct_native_cev(rows, ablation_hook):
    ces = []
    C.SEL['qk_tail_on'] = False
    C.SEL['abl_on'] = True
    handle = C.m.transformer.h[16].attn.register_forward_hook(ablation_hook)
    try:
        for start in range(0, rows.shape[0], 4):
            batch = rows[start:start + 4, :257].to(C.DEV)
            idx = batch[:, :256]
            targets = batch[:, 1:257].reshape(-1)
            x = F.rms_norm(C.m.transformer.wte(idx), (C.D,))
            x0 = x
            v1 = None
            for block in C.m.transformer.h:
                x, v1 = block(x, v1, x0)
            logits = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
            ces.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets,
                                       reduction='none').cpu())
    finally:
        handle.remove()
        C.SEL['abl_on'] = False
    return torch.cat(ces)


@torch.no_grad()
def main():
    started = time.time()
    CN.use_state('census_state_diverse.pt')
    rows = CN.rows().cpu()
    base_ce = CN.base_ce().float().cpu()
    nflat = CN.nflat()
    baseline = torch.load(ROOT / 'cev_mixed_native_a1v.pt', map_location='cpu').float().reshape(-1)
    baseline_result = json.load(open(ROOT / 'mixed_native_a1v_results.json'))
    assert baseline.numel() == nflat

    C.CROWS = rows
    C.CBASE = base_ce
    C.NFLAT = nflat
    C.ANCH = json.load(open(ROOT / 'frontier_tail_traj_results.json'))
    C.SEL.update({
        'mode': 'norm', 'K': 4608, 'K69': 4608, 'K69MAP': {},
        'skipset': tuple(range(10, 18)), 'motif_off': (), 'clsdmg': True,
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 96, 'qk_rmap': {},
        'qk_tail': True, 'drop_tailE': True, 'drop_a1v': True,
        'ablate_on_census': True,
    })

    capture = {'sum': torch.zeros(C.D, device=C.DEV), 'n': 0}

    def capture_mean(module, inputs, output):
        values = output[0].detach().float().reshape(-1, C.D)
        capture['sum'] += values.sum(0)
        capture['n'] += values.shape[0]

    handle = C.m.transformer.h[16].attn.register_forward_hook(capture_mean)
    for start in range(0, 128, 4):
        idx = C.FW[start:start + 4, :256].to(C.DEV)
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,))
        x0 = x
        v1 = None
        for block in C.m.transformer.h:
            x, v1 = block(x, v1, x0)
    handle.remove()
    mean_value = (capture['sum'] / capture['n']).clone()

    def ablate(module, inputs, output):
        if not C.SEL.get('abl_on'):
            return None
        values, v1 = output
        return mean_value.expand_as(values).to(values.dtype), v1

    C.SEL['_ablh'] = ablate
    print('ARM: corrected mixed with a16 mean ablated only at census', flush=True)
    run = C.main()
    qk = C.SEL.get('_QKR', {})
    factor_ranks = {
        int(factor[0].shape[1])
        for heads in qk.values() for factors in heads.values() for factor in factors
    }
    if set(qk) != set(range(2, 18)) or factor_ranks != {96}:
        raise SystemExit(f'INSTRUMENT FAIL: QK layers/ranks {sorted(qk)}/{factor_ranks}')

    compiled_ko = C.SEL['cev'].float().cpu()
    assert compiled_ko.numel() == nflat
    native_ko = direct_native_cev(rows, ablate)
    torch.save(compiled_ko, COMP_KO)
    torch.save(native_ko, NATIVE_KO)

    effect_comp = compiled_ko - baseline
    effect_real = native_ko - base_ce
    cosine = float(torch.dot(effect_comp, effect_real) /
                   (effect_comp.norm() * effect_real.norm()).clamp_min(1e-12))
    normalized_error = float((effect_comp - effect_real).norm() /
                             effect_real.norm().clamp_min(1e-12))
    norm_ratio = float(effect_comp.norm() / effect_real.norm().clamp_min(1e-12))

    battery = json.load(open(ROOT / 'circuits/BATTERY.json'))['by_tag']
    top_component = {tag: receipt['mean_ablation']['top'][0]['component']
                     for tag, receipt in battery.items()}
    leaf_rows = []
    collateral_real = []
    collateral_comp = []
    own_ratios = []
    for tag in battery:
        try:
            member = CN.leaf(tag)['member'].long()
        except Exception:
            continue
        if member.numel() == 0:
            continue
        real_abs = float(effect_real[member].abs().mean())
        comp_abs = float(effect_comp[member].abs().mean())
        own = top_component[tag] == 'a16'
        leaf_rows.append({'tag': tag, 'native_abs_effect': round(real_abs, 7),
                          'compiled_abs_effect': round(comp_abs, 7), 'own': own})
        if own:
            own_ratios.append(comp_abs / max(real_abs, 1e-12))
        else:
            collateral_real.append(real_abs)
            collateral_comp.append(comp_abs)
    own_ratios.sort()
    if not own_ratios:
        raise SystemExit('INSTRUMENT FAIL: no a16-own circuit leaves')
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = spearman(collateral_real, collateral_comp)

    pred_a = (baseline_result['census_damage'] <= 0.012 and
              baseline_result['max_fresh_damage'] <= 0.020 and run['L2_F'] <= 0.020)
    pred_b = cosine >= 0.90 and normalized_error <= 0.60
    pred_c = collateral_rho >= 0.90 and 0.60 <= own_median <= 1.40
    result = {
        'convention': 'signed effect = KO CE minus unablated CE within each model',
        'physical_qk_factor_ranks': sorted(factor_ranks),
        'unablated_fresh_damage': run['L2_F'],
        'unablated_census_damage': baseline_result['census_damage'],
        'compiled_ko_census_damage': round(float((compiled_ko - base_ce).mean()), 7),
        'native_ko_census_damage': round(float(effect_real.mean()), 7),
        'effect_cosine': round(cosine, 7),
        'effect_normalized_error': round(normalized_error, 7),
        'effect_norm_ratio': round(norm_ratio, 7),
        'collateral_spearman': round(collateral_rho, 7),
        'own_effect_median_ratio': round(own_median, 7),
        'own_effect_ratios': [round(value, 7) for value in own_ratios],
        'circuits': leaf_rows,
        'pred_a_live_config': bool(pred_a),
        'pred_b_signed_effect': bool(pred_b),
        'pred_c_circuits': bool(pred_c),
        'null_triggered': bool(cosine < 0.70 or collateral_rho < 0.75),
        'decision_level': 'a16 causal falsifier for corrected mixed adoption',
        'runtime_s': round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'circuits'},
                     indent=2), flush=True)
    print(f'wrote {OUT}, {COMP_KO}, and {NATIVE_KO}', flush=True)


if __name__ == '__main__':
    main()
