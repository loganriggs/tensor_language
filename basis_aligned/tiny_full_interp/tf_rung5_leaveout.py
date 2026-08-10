"""LEAVE-ONE-OUT companion to `tf_rung5_program.py`.

WHY.  The staged ladder is CUMULATIVE, so every increment it reports is an
"added last given everything before it" number.  The programme's own standing
failure mode -- *quoting a component's value without its ladder position* --
says that is only half the story: at depth 2 attention was worth 8.17 nats added
first and 0.61 added last.  This file measures the other end.  It refits the TOP
stage of the ladder with exactly one ingredient removed, so each ingredient also
gets a "removed last" value, and the two bracket its worth.

Nothing here is fitted on held.  Same grammar, same bit rules, same splits.

Usage
    python tf_rung5_leaveout.py --cells a,b,c --merge-into tf_rung5_program.json
"""
import argparse
import json
import os
import time

import torch

import tf_fold
import tf_rung5_program as R

HERE = os.path.dirname(os.path.abspath(__file__))

# top stage = bigram + two-branch rotary gate + heads + induction + squared
TOP = dict(bigram=True, ctx='twobranch', induction=True, square=True)

DROPS = {
    'no_current_token': dict(bigram=False),
    'no_context': dict(ctx=''),
    'no_heads': dict(nh=1),
    'no_induction': dict(induction=False),
    'no_squared_content': dict(square=False),
    'no_rotary_in_gate': dict(ctx='pair'),
    'no_second_gate_branch': dict(ctx='pair_rot'),
}


def run_cell(stem, args):
    t0 = time.time()
    model, cfg, ck = tf_fold.load_checkpoint(stem, R.DEV)
    V, T = cfg.vocab, args.T
    r = args.rank or {32: 16, 64: 32, 128: 64, 256: 128}.get(cfg.width, 32)
    xs_est = R.load_x(V, 'est', args.n_est, T, cfg.tok)
    xs_held = R.load_x(V, 'held', args.n_held, T, cfg.tok)
    out = {'stem': stem, 'rank_r': r, 'gate_rank_a': args.a,
           'n_gates': args.nh, 'context_length': T,
           'est_rows_fitted_on': int(xs_est.shape[0]),
           'held_tokens_scored': int(xs_held.shape[0] * T),
           'model_bits_at_32': int(model.n_params() * 32),
           'variants': {}}
    base = dict(V=V, T=T, r=r, a=args.a, nh=args.nh)
    for name, drop in [('TOP', {})] + list(DROPS.items()):
        pc = R.PCfg(**{**base, **TOP, **drop})
        p = R.Program(pc, R.DEV)
        R.init_unigram(p, model, xs_est, T, args.batch)
        R.fit(p, model, xs_est, T, args.steps, args.lr, args.batch)
        s = R.score(p, model, xs_held, T, args.batch)
        b = p.bits()
        out['variants'][name] = {
            'ingredient_removed': name,
            'kl_from_model_held': s['kl_from_model'],
            'total_entries': b['total_entries'],
            'total_bits': b['total_bits'],
            'bits_over_model': b['total_bits'] / (model.n_params() * 32)}
        print(f'  {name:<24s} KL {s["kl_from_model"]:.4f}  '
              f'{b["total_entries"]:>10,} entries  {time.time() - t0:.0f}s',
              flush=True)
        del p
        torch.cuda.empty_cache()
    top = out['variants']['TOP']['kl_from_model_held']
    for name in DROPS:
        v = out['variants'][name]
        v['kl_cost_of_removing'] = v['kl_from_model_held'] - top
        v['bits_saved_by_removing'] = (out['variants']['TOP']['total_bits']
                                       - v['total_bits'])
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='tf_vanilla_d1_w32_b8192_s0,'
                                       'tf_vanilla_d2_w128_b8192_s0,'
                                       'tf_vanilla_d4_w256_b8192_s0')
    ap.add_argument('--out', default='tf_rung5_leaveout.json')
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
    args = ap.parse_args()
    rep = {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'what': 'leave-one-out around the top stage of the rung-5 program '
                   'ladder: each ingredient also gets a REMOVED-LAST value, '
                   'because the cumulative ladder only gives ADDED-LAST',
           'args': vars(args), 'cells': {}}
    for stem in args.cells.split(','):
        stem = stem.strip()
        if not stem:
            continue
        print(f'=== {stem}', flush=True)
        rep['cells'][stem] = run_cell(stem, args)
        json.dump(rep, open(f'{HERE}/{args.out}', 'w'), indent=1)
    json.dump(rep, open(f'{HERE}/{args.out}', 'w'), indent=1)
    if args.merge_into:
        path = f'{HERE}/{args.merge_into}'
        d = json.load(open(path))
        d['leave_one_out'] = rep
        json.dump(d, open(path, 'w'), indent=1)
        print('merged into', args.merge_into, flush=True)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
