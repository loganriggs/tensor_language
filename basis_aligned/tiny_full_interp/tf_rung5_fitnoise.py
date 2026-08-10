"""FIT-NOISE CALIBRATION for the rung-5 program ladder.

WHY.  `tf_rung5_leaveout.py` reports differences of a few thousandths of a nat
between variants (removing the rotary from the gate came out at -0.002, i.e.
apparently HELPFUL).  A difference is not a measurement until the spread of the
estimator is known -- "uncalibrated nulls" is one of the programme's standing
failure modes.  This file refits the SAME top-stage configuration several times,
varying only the table initialisation and the batch order, and reports the
spread of the held KL.  Any leave-one-out cost inside that band is noise.

Usage
    python tf_rung5_fitnoise.py --repeats 4 --merge-into tf_rung5_program.json
"""
import argparse
import json
import os
import time

import torch

import tf_fold
import tf_rung5_program as R
from tf_rung5_leaveout import TOP


def reseed(prog, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        for n, p in prog.named_parameters():
            if n == 'U':
                continue
            if n == 'g':
                p.fill_(1.0 / p.shape[-1])
            elif n in ('cscale',):
                p.fill_(1.0)
            elif n in ('lam0', 'lam_tok'):
                p.zero_()
            else:
                std = 0.002 if n == 'W2' else 0.02
                p.copy_(torch.randn(p.shape, generator=g).to(p.device) * std)


def run_cell(stem, args):
    model, cfg, ck = tf_fold.load_checkpoint(stem, R.DEV)
    V, T = cfg.vocab, args.T
    r = args.rank or {32: 16, 64: 32, 128: 64, 256: 128}.get(cfg.width, 32)
    xs_est = R.load_x(V, 'est', args.n_est, T, cfg.tok)
    xs_held = R.load_x(V, 'held', args.n_held, T, cfg.tok)
    kls = []
    for s in range(args.repeats):
        p = R.Program(R.PCfg(V=V, T=T, r=r, a=args.a, nh=args.nh, **TOP), R.DEV)
        reseed(p, 90000 + 137 * s)
        R.init_unigram(p, model, xs_est, T, args.batch)
        R.fit(p, model, xs_est, T, args.steps, args.lr, args.batch, seed=s)
        kls.append(R.score(p, model, xs_held, T, args.batch)['kl_from_model'])
        print(f'  repeat {s}: KL {kls[-1]:.5f}', flush=True)
        del p
        torch.cuda.empty_cache()
    m = sum(kls) / len(kls)
    sd = (sum((k - m) ** 2 for k in kls) / max(1, len(kls) - 1)) ** 0.5
    return {'stem': stem, 'rank_r': r, 'repeats': args.repeats,
            'kls': kls, 'mean': m, 'sd': sd, 'range': max(kls) - min(kls),
            'what': 'same TOP configuration, different table init and batch '
                    'order; any leave-one-out cost inside this band is noise'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='tf_vanilla_d1_w32_b8192_s0,'
                                       'tf_vanilla_d2_w128_b8192_s0,'
                                       'tf_vanilla_d4_w256_b8192_s0')
    ap.add_argument('--out', default='tf_rung5_fitnoise.json')
    ap.add_argument('--merge-into', default='')
    ap.add_argument('--T', type=int, default=256)
    ap.add_argument('--n-est', type=int, default=24576)
    ap.add_argument('--n-held', type=int, default=512)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--a', type=int, default=8)
    ap.add_argument('--nh', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=4)
    args = ap.parse_args()
    rep = {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'what': 'refit spread of the top-stage program (fit noise floor)',
           'args': vars(args), 'cells': {}}
    HERE = os.path.dirname(os.path.abspath(__file__))
    for stem in args.cells.split(','):
        stem = stem.strip()
        if not stem:
            continue
        print(f'=== {stem}', flush=True)
        rep['cells'][stem] = run_cell(stem, args)
        json.dump(rep, open(f'{HERE}/{args.out}', 'w'), indent=1)
    if args.merge_into:
        path = f'{HERE}/{args.merge_into}'
        d = json.load(open(path))
        d['fit_noise'] = rep
        json.dump(d, open(path, 'w'), indent=1)
        print('merged into', args.merge_into, flush=True)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
