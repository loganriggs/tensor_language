"""KNOWN-ANSWER CONTROL for the depth-2 machinery.

`tf_interp2.DeepFold` is a NEW implementation of the folded pipeline.  Before
any depth-2 number is believed it must reproduce, on a DEPTH-1 cell, the
numbers the already-reviewed depth-1 code (`tf_interp.Depth1Fold` +
`tf_interp.ladder`) produced for the same stages on the same text.  Anything
that does not agree to ~1e-3 nats of KL is a bug in the new code, not a
finding about depth.

Also runs the ladder-ORDER test on the depth-1 cells so the depth-1 and
depth-2 order comparisons are made with one implementation.
"""
import json
import os
import sys

import numpy as np
import torch

import tf_interp as I1
import tf_interp2 as I2

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = ['embed_only', 'plus_self_attn', 'model_bigram', 'full_exact',
          'no_mlp', 'mlp_write_only', 'no_attention_at_all',
          'past_attn_mean_ablated', 'past_attn_direct_route_only',
          'past_attn_mlp_route_only', 'trunc_delta1_only', 'trunc_delta_le4',
          'trunc_delta_le16', 'trunc_delta_le64', 'positional_only_pattern',
          'no_rotary_pattern']


def main(stems, n_seq=48, T=256):
    rep = {'note': 'depth-1 cells scored by BOTH implementations on the same '
                   'held text; the depth-2 code is only trusted where it '
                   'reproduces the reviewed depth-1 code'}
    worst = 0.0
    for stem in stems:
        D1 = I1.Depth1Fold(stem)
        L1 = I1.ladder(D1, n_seq, T, extra=True)
        del D1
        torch.cuda.empty_cache()
        D2 = I2.DeepFold(stem)
        L2 = I2.ladder2(D2, n_seq, T, extra=True)
        order = I2.ladder_order(D2, min(n_seq, 64), T)
        rows = {}
        for k in SHARED:
            if k in L1 and k in L2:
                a, b = L1[k]['kl_from_model'], L2[k]['kl_from_model']
                rows[k] = {'depth1_code': a, 'depth2_code': b, 'abs_diff': abs(a - b)}
                worst = max(worst, abs(a - b))
        for h in range(D2.H):
            k1, k2 = f'drop_head{h}', f'drop_l0_head{h}'
            if k1 in L1 and k2 in L2:
                a, b = L1[k1]['kl_from_model'], L2[k2]['kl_from_model']
                rows[k2] = {'depth1_code': a, 'depth2_code': b,
                            'abs_diff': abs(a - b)}
                worst = max(worst, abs(a - b))
        rep[stem] = {'stages': rows, 'ladder_order': order,
                     'max_abs_diff': max(r['abs_diff'] for r in rows.values())}
        print(stem, 'max abs KL diff', rep[stem]['max_abs_diff'], flush=True)
        json.dump(order, open(f'{HERE}/{stem}_order.json', 'w'), indent=2)
        del D2
        torch.cuda.empty_cache()
    rep['worst_abs_diff'] = worst
    rep['pass'] = bool(worst < 2e-3)
    json.dump(rep, open(f'{HERE}/tf_interp2_control.json', 'w'), indent=2)
    print('WORST', worst, 'PASS', rep['pass'])
    return rep


if __name__ == '__main__':
    main(sys.argv[1:] or ['tf_vanilla_d1_w32_b8192_s0',
                          'tf_vanilla_d1_w64_b8192_s0',
                          'tf_vanilla_d1_w128_b8192_s0'])
