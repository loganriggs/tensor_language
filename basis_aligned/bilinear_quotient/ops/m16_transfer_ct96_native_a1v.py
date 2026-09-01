"""RUNG 289: signed m16 intervention transfer on corrected ct96 (adoption sentinel).

The corrected ct96 program restores block-1 c_v to its native linear map and removes the false +0.052
path floor. This experiment tests whether that predictive/certificate win preserves manipulation. It fits
nothing to census targets. A native m16 mean is estimated on the frozen FIT rows, then the identical mean
ablation is applied to (1) corrected ct96 and (2) the native model. The primary objects are signed causal
effect vectors:
    e_comp = CE(corrected_ct96 + KO) - CE(corrected_ct96)
    e_real = CE(native + KO) - CE(native).
This avoids the earlier ambiguity of subtracting two unsigned member-damage summaries.

REGISTERED PREDICTIONS (lower base CE damage is better):
  (a) LIVE CONFIG: the unablated corrected-ct96 fresh damage is <=0.020 and its saved census damage <=0.010.
  (b) SIGNED EFFECT: cosine(e_comp,e_real) >=0.90 and ||e_comp-e_real||/||e_real|| <=0.60.
  (c) CIRCUITS: collateral Spearman >=0.90 across non-own leaves; among leaves whose registered top
      component is m16, median mean-absolute-effect ratio is in [0.60,1.20].
NULL: effect cosine <0.70 or collateral rho <0.80; predictive repair then fails causal adoption.
PRICE: intervention probe only; deployed corrected ct96 price is unchanged. Self-reviewed; bqrunner only.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'm16_transfer_ct96_native_a1v_results.json'
COMP_KO = ROOT / 'cev_m16ko_ct96_native_a1v.pt'
NATIVE_KO = ROOT / 'cev_m16ko_native.pt'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [
        ROOT / 'frontier_tail_traj_results.json', ROOT / 'circuits/BATTERY.json',
        ROOT / 'census_state_diverse.pt', ROOT / 'ct96_native_a1v_results.json',
        ROOT / 'cev_ct96_native_a1v.pt', ROOT / 'ops/cevdump_ct96.py',
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(f'DRYRUN WAIT: prerequisite rung 288 missing {missing}')
        raise SystemExit(1)
    print('DRYRUN OK: rung 289 signed m16 transfer on corrected ct96')
    raise SystemExit(0)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, '/workspace/rspd')

import torch
import torch.nn.functional as F
import census_lib as CN
import cevdump_ct96 as C


def spearman(x, y):
    x = torch.tensor(x).argsort().argsort().float()
    y = torch.tensor(y).argsort().argsort().float()
    x -= x.mean()
    y -= y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


@torch.no_grad()
def direct_native_cev(rows):
    ces = []
    C.SEL['qk_tail_on'] = False
    C.SEL['abl_on'] = True
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4, :257].to(C.DEV)
        idx = bb[:, :256]
        targets = bb[:, 1:257].reshape(-1)
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,))
        x0 = x
        v1 = None
        for block in C.m.transformer.h:
            x, v1 = block(x, v1, x0)
        logits = (30 * torch.tanh(C.m.lm_head(F.rms_norm(x, (C.D,))) / 30)).float()
        ces.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, reduction='none').cpu())
    C.SEL['abl_on'] = False
    return torch.cat(ces)


@torch.no_grad()
def main():
    started = time.time()
    CN.use_state('census_state_diverse.pt')
    rows = CN.rows().cpu()
    base_ce = CN.base_ce().float().cpu()
    nflat = CN.nflat()
    baseline = torch.load(ROOT / 'cev_ct96_native_a1v.pt', map_location='cpu').float().reshape(-1)
    baseline_result = json.load(open(ROOT / 'ct96_native_a1v_results.json'))
    assert baseline.numel() == nflat

    C.CROWS = rows
    C.CBASE = base_ce
    C.NFLAT = nflat
    C.ANCH = json.load(open(ROOT / 'frontier_tail_traj_results.json'))
    C.SEL.update({
        'mode': 'norm', 'K': 4608, 'K69': 4608, 'K69MAP': {},
        'skipset': tuple(range(10, 18)), 'motif_off': (), 'clsdmg': True,
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 96,
        'qk_rmap': {li: 128 for li in range(2, 10)}, 'qk_tail': True,
        'drop_tailE': True, 'drop_a1v': True, 'ablate_on_census': True,
    })

    mucap = {'sum': torch.zeros(C.D, device=C.DEV), 'n': 0}
    def capture_mean(module, inputs, output):
        flat = output.detach().float().reshape(-1, C.D)
        mucap['sum'] += flat.sum(0)
        mucap['n'] += flat.shape[0]
    handle = C.m.transformer.h[16].mlp.register_forward_hook(capture_mean)
    for i in range(0, 128, 4):
        idx = C.FW[i:i + 4, :256].to(C.DEV)
        x = F.rms_norm(C.m.transformer.wte(idx), (C.D,))
        x0 = x
        v1 = None
        for block in C.m.transformer.h:
            x, v1 = block(x, v1, x0)
    handle.remove()
    mean_value = (mucap['sum'] / mucap['n']).clone()

    def ablate(module, inputs, output):
        if not C.SEL.get('abl_on'):
            return None
        return mean_value.expand_as(output).to(output.dtype)
    C.m.transformer.h[16].mlp.register_forward_hook(ablate)
    print('ARM: corrected ct96 with m16 mean ablated only at census', flush=True)
    run = C.main()
    compiled_ko = C.SEL['cev'].float().cpu()
    assert compiled_ko.numel() == nflat
    native_ko = direct_native_cev(rows)
    torch.save(compiled_ko, COMP_KO)
    torch.save(native_ko, NATIVE_KO)

    effect_comp = compiled_ko - baseline
    effect_real = native_ko - base_ce
    cosine = float(torch.dot(effect_comp, effect_real) /
                   (effect_comp.norm() * effect_real.norm()).clamp_min(1e-12))
    normalized_error = float((effect_comp - effect_real).norm() / effect_real.norm())
    norm_ratio = float(effect_comp.norm() / effect_real.norm())

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
        own = top_component[tag] == 'm16'
        leaf_rows.append({'tag': tag, 'native_abs_effect': round(real_abs, 7),
                          'compiled_abs_effect': round(comp_abs, 7), 'own': own})
        if own:
            own_ratios.append(comp_abs / max(real_abs, 1e-12))
        else:
            collateral_real.append(real_abs)
            collateral_comp.append(comp_abs)
    own_ratios.sort()
    own_median = own_ratios[len(own_ratios) // 2]
    collateral_rho = spearman(collateral_real, collateral_comp)

    pred_a = run['L2_F'] <= 0.020 and baseline_result['census_damage'] <= 0.010
    pred_b = cosine >= 0.90 and normalized_error <= 0.60
    pred_c = collateral_rho >= 0.90 and 0.60 <= own_median <= 1.20
    result = {
        'convention': 'signed effect = KO CE minus unablated CE within each model',
        'unablated_fresh_damage': run['L2_F'],
        'unablated_census_damage': baseline_result['census_damage'],
        'compiled_ko_census_damage': round(float((compiled_ko - base_ce).mean()), 7),
        'native_ko_census_damage': round(float(effect_real.mean()), 7),
        'effect_cosine': round(cosine, 7),
        'effect_normalized_error': round(normalized_error, 7),
        'effect_norm_ratio': round(norm_ratio, 7),
        'collateral_spearman': round(collateral_rho, 7),
        'own_effect_median_ratio': round(own_median, 7),
        'own_effect_ratios': [round(x, 7) for x in own_ratios],
        'circuits': leaf_rows,
        'pred_a_live_config': bool(pred_a),
        'pred_b_signed_effect': bool(pred_b),
        'pred_c_circuits': bool(pred_c),
        'null_triggered': bool(cosine < 0.70 or collateral_rho < 0.80),
        'decision_level': 'adoption sentinel for corrected ct96',
        'runtime_s': round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'circuits'}, indent=2), flush=True)
    print(f'wrote {OUT}, {COMP_KO}, and {NATIVE_KO}', flush=True)


if __name__ == '__main__':
    main()
