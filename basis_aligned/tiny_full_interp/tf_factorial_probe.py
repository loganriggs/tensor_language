"""The induction battery on a factorial arm.

Deliberately a two-function file.  It calls tf_baseline_probe.probe UNCHANGED
-- the same function, not a copy of it -- so a factorial-versus-family
induction difference cannot be a probe-code difference.  That probe's own C4
control already establishes it reproduces a published foldable number exactly.

  python tf_factorial_probe.py --stem tff_softmax_bilin_d2_w128_b8192_s0
  python tf_factorial_probe.py --control     # the corner check, see below

THE CORNER CONTROL is the probe-side twin of gates G1/G2: probing the
factorial's (bilin, bilin) code path on a TRANSPLANTED published foldable
checkpoint must return that checkpoint's published induction score.  G1 shows
the forwards agree; this shows the battery reads them the same way too.
"""
import argparse
import json
import os

import torch

import tf_factorial as FA
import tf_baseline_probe as P
import tf_fold

HERE = os.path.dirname(os.path.abspath(__file__))


def probe_fac_stem(stem, seeds=P.PROBE_SEEDS, device=None):
    dev = device or FA.DEV
    jp = f'{HERE}/{stem}_induction.json'
    if os.path.exists(jp):
        print(f'{stem}: induction already probed -- skip', flush=True)
        return json.load(open(jp))
    model, cfg, _ = FA.load_fac(stem, dev)
    out = P.probe(model, cfg, dev, seeds=seeds)
    out['stem'] = stem
    out['attn'], out['mlp'] = cfg.attn, cfg.mlp
    out['depth'], out['width'], out['hidden'] = cfg.depth, cfg.width, cfg.hidden
    out['seed'] = cfg.seed
    json.dump(out, open(jp, 'w'), indent=2)
    print(f'{stem}: induction {out["induction_score_mean"]:+.4f} '
          f'+- {out["induction_score_sd"]:.4f} (probe floor '
          f'{out.get("detectable_effect_floor_nats_3se", 0):.4f}), '
          f'bag {out["bag_score_mean"]:+.4f}', flush=True)
    return out


def corner_control(stem='tf_vanilla_d2_w256_b8192_s0', device=None):
    """Transplant a published foldable checkpoint into the factorial's
    (bilin, bilin) path and re-probe.  Must reproduce the published score."""
    dev = device or FA.DEV
    ref, rcfg, _ = tf_fold.load_checkpoint(stem, dev)
    fcfg = FA.FacConfig(attn='bilin', mlp='bilin', depth=rcfg.depth,
                        width=rcfg.width, vocab=rcfg.vocab, tok=rcfg.tok,
                        seed=rcfg.seed, T=rcfg.T)
    mine = FA.make_fac(fcfg, dev)
    src = dict(ref.named_parameters())
    dst = dict(mine.named_parameters())
    assert set(src) == set(dst), sorted(set(src) ^ set(dst))
    with torch.no_grad():
        for n, p in dst.items():
            p.copy_(src[n])
    mine.eval()
    got = P.probe(mine, fcfg, dev, seeds=P.PROBE_SEEDS, sensitivity=False,
                  natural=False)['induction_score_mean']
    published = None
    jp = f'{HERE}/{stem}_interp3.json'
    if os.path.exists(jp):
        published = json.load(open(jp)).get('rung3_induction', {}).get(
            'induction_score_mean')
    dd = None if published is None else abs(got - published)
    res = {'control': 'factorial_bilin_bilin_corner_reproduces_a_published_'
                      'foldable_induction_score',
           'stem': stem, 'published': published, 'factorial_path': got,
           'abs_diff': dd, 'pass': bool(published is not None and dd < 1e-6),
           'claim': 'the factorial arms are probed through the same battery '
                    'that produced every published family number, and its '
                    'foldable corner returns the published value, so an arm-to-'
                    'arm induction difference is an architecture difference.'}
    json.dump(res, open(f'{HERE}/tf_factorial_probe_control.json', 'w'),
              indent=2)
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default=None)
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--seeds', type=int, default=P.PROBE_SEEDS)
    a = ap.parse_args()
    if a.control:
        corner_control()
    else:
        assert a.stem, 'need --stem or --control'
        probe_fac_stem(a.stem, seeds=a.seeds)
