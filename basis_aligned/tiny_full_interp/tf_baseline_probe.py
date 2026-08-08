"""THE INDUCTION PROBE, RUN VERBATIM ON THE CONVENTIONAL MODEL.

The interesting half of the conventional-baseline comparison is not the loss.
It is this: FINDING 16 says induction appears at width 256 at depth 2 and width
128 at depths 3-4, and every foldable architecture except the one with the
capability hand-installed is null at depth 2 width 128.  If a CONVENTIONAL
transformer inducts there, that surface is a statement about OUR architecture
family and not about transformers, and the finding has to be relabelled.

For that question to mean anything the probe must be THE SAME PROBE.  So this
file imports and calls, without modification:

  tf_interp.induction_battery      the three-condition synthetic battery
                                   (repeat / shuffled / control on the same
                                   token multiset); induction = shuffled -
                                   repeat, bag = control - shuffled
  tf_interp.induction_sensitivity  the planted-oracle calibration
  tf_interp2.induction_power       the detectable-effect floor read off it
  tf_interp2.natural_induction     the order-only SWAP patch on held text

Those functions take a container `D` with `.V`, `.cfg`, `.dev`, `.model` --
tf_interp.induction_for_stem already builds exactly such a shim for foldable
checkpoints.  This file builds the same shim for a conventional checkpoint, and
CONTROL C4 requires the shim to reproduce a PUBLISHED foldable number
(tf_vanilla_d2_w256_b8192_s0's rung3_induction mean) so that a family
difference cannot be a difference of probe code.

THRESHOLD DISCIPLINE (the programme's own standing failure mode).  A floor
defined over PROBE seeds shrinks as 1/sqrt(n) and is not a detection threshold.
The planted-oracle floor is recorded because it calibrates what the battery
could have seen, but the verdict is decided over MODEL seeds in
tf_baseline_report.py.

Usage
  python tf_baseline_probe.py --stem tfb_std4_d2_w128_b8192_s0
  python tf_baseline_probe.py --control          # C4 only
"""
import argparse
import json
import os

import numpy as np
import torch

import tf_fold
import tf_interp as I1
import tf_interp2 as I2

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_SEEDS = 5
C4_STEM = 'tf_vanilla_d2_w256_b8192_s0'


class ProbeShim:
    """The container tf_interp's battery expects.  Deliberately the same shape
    as tf_interp.induction_for_stem._Shim."""

    def __init__(self, model, cfg, dev):
        self.V, self.cfg, self.dev, self.model = cfg.vocab, cfg, dev, model


@torch.no_grad()
def probe(model, cfg, dev, seeds=PROBE_SEEDS, sensitivity=True, natural=True):
    """The identical measurement made on every cell of both families."""
    model.eval()
    D = ProbeShim(model, cfg, dev)
    runs = [I1.induction_battery(D, seed=s, model=model) for s in range(seeds)]
    out = {
        'probe_seeds': seeds,
        'per_probe_seed': runs,
        'induction_score_mean': float(np.mean([r['induction_score']
                                               for r in runs])),
        'induction_score_sd': float(np.std([r['induction_score'] for r in runs],
                                           ddof=1)),
        'bag_score_mean': float(np.mean([r['bag_score'] for r in runs])),
        'bag_score_sd': float(np.std([r['bag_score'] for r in runs], ddof=1))}
    if sensitivity:
        out['induction_power'] = I2.induction_power(D, seeds=seeds)
        out['detectable_effect_floor_nats_3se'] = \
            out['induction_power']['detectable_effect_floor_nats_3se']
        out['clears_own_probe_floor'] = bool(
            out['induction_score_mean'] > out['detectable_effect_floor_nats_3se'])
    if natural:
        out['natural_induction'] = I2.natural_induction(D, n_seq=512)
    out['threshold_note'] = (
        'the probe floor is 3 standard errors over PROBE seeds and shrinks as '
        '1/sqrt(n); it calibrates the battery, it is NOT the detection '
        'threshold. The verdict is taken over MODEL seeds in '
        'tf_baseline_report.py.')
    return out


def probe_std_stem(stem, seeds=PROBE_SEEDS, device=None):
    import tf_baseline_std as B
    dev = device or B.DEV
    jp = f'{HERE}/{stem}_induction.json'
    if os.path.exists(jp):
        print(f'{stem}: induction already probed -- skip', flush=True)
        return json.load(open(jp))
    model, cfg, _ = B.load_std(stem, dev)
    out = probe(model, cfg, dev, seeds=seeds)
    out['stem'] = stem
    out['depth'], out['width'], out['exp'] = cfg.depth, cfg.width, cfg.exp
    out['seed'] = cfg.seed
    json.dump(out, open(jp, 'w'), indent=2)
    print(f'{stem}: induction {out["induction_score_mean"]:+.4f} '
          f'+- {out["induction_score_sd"]:.4f} '
          f'(probe floor {out.get("detectable_effect_floor_nats_3se", 0):.4f}), '
          f'bag {out["bag_score_mean"]:+.4f}, natural swap '
          f'{out["natural_induction"]["ORDER_ONLY_patch_swap"]["mean"]:+.5f}',
          flush=True)
    return out


def control_c4(device=None, stem=C4_STEM):
    """C4.  The shim, on a PUBLISHED foldable checkpoint, must reproduce that
    cell's published induction score.  If it does not, every conventional-vs-
    foldable induction difference is a probe-code difference."""
    import tf_baseline_std as B
    dev = device or B.DEV
    jp = f'{HERE}/{stem}_interp3.json'
    published = None
    if os.path.exists(jp):
        d = json.load(open(jp))
        published = d.get('rung3_induction', {}).get('induction_score_mean')
    model, cfg, _ = tf_fold.load_checkpoint(stem, dev)
    got = probe(model, cfg, dev, seeds=PROBE_SEEDS, sensitivity=False,
                natural=False)
    mine = got['induction_score_mean']
    dd = None if published is None else abs(mine - published)
    res = {'control': 'C4_probe_shim_reproduces_a_published_foldable_number',
           'stem': stem, 'published_induction_score_mean': published,
           'shim_induction_score_mean': mine, 'abs_diff': dd,
           'pass': bool(published is not None and dd < 1e-6),
           'claim': 'the conventional models are probed through this shim; it '
                    'reproduces the published foldable number exactly, so a '
                    'family induction difference is not a probe difference.'}
    json.dump(res, open(f'{HERE}/tf_baseline_probe_control.json', 'w'),
              indent=2)
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default=None)
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--seeds', type=int, default=PROBE_SEEDS)
    a = ap.parse_args()
    if a.control:
        control_c4()
    else:
        assert a.stem, 'need --stem or --control'
        probe_std_stem(a.stem, seeds=a.seeds)
