"""O8c -- THE MONEY EXPERIMENT for Logan's second redirection.

O8b's confound control fired: at lr 5e-5 a full-precision description refitted
to the DATA cross-entropy on fresh `est` text reaches held CE 4.70413 against
the model's 4.71140.  The model is improvable, so the test has power, and the
question becomes the interesting one:

    what is the SHORTEST description whose held cross-entropy is below the
    model's own 4.7114 nats?

A description that is both smaller than the model and better at predicting the
text is the strongest result this rung can produce, and KL-from-the-model was
structurally blind to it.  Every arm is fitted only on `est` (iterate selected
on a disjoint `est` slice) and scored on `held`; the bit bill is the same
q_scalar_entropy bill every other point pays.
"""
import json
import os

import tf_compress as CC
import tf_reviewer_r3 as R3

HERE = os.path.dirname(os.path.abspath(__file__))
P = f'{HERE}/tf_reviewer_round_3_compression.json'


def main():
    D = CC.D1Desc('tf_vanilla_d1_w128_b8192_s0')
    mce = D.score()['ce']
    rows = []
    for emb_b, body_b in ((8, 8), (6, 8), (5, 8), (4, 8), (4, 6), (3, 6)):
        for lr in (5e-5, 2e-5):
            Pd, bits, tl = R3.distill_ce(D, emb_b, body_b, lr=lr)
            s = D.score(Pd)
            rows.append({'emb_bits': emb_b, 'body_bits': body_b, 'lr': lr,
                         'bits': bits.total, 'held_ce': s['ce'],
                         'delta_ce_vs_model': s['ce'] - mce,
                         'kl_from_model': s['kl'],
                         'best_step': tl['best_step'],
                         'beats_the_model_on_data': bool(s['ce'] < mce),
                         'x_smaller_than_fp32': 32 * D.n_params_model / bits.total,
                         'x_smaller_than_fp16': 16 * D.n_params_model / bits.total})
            print(f'{emb_b}/{body_b} lr={lr:g}: {bits.total/1e6:6.3f} Mbit  '
                  f'held CE {s["ce"]:.5f} (delta {s["ce"]-mce:+.5f}) '
                  f'step {tl["best_step"]}  '
                  f'{"BEATS THE MODEL" if s["ce"] < mce else ""}', flush=True)
    win = [r for r in rows if r['beats_the_model_on_data']]
    o = json.load(open(P))
    o['O8c_shortest_description_that_beats_the_model_on_data'] = {
        'model_held_ce': mce, 'rows': rows,
        'n_arms_beating_the_model': len(win),
        'shortest_beating_description': (min(win, key=lambda r: r['bits'])
                                         if win else None),
        'note': ('These descriptions are fitted to fresh `est` text, which the '
                 'model never saw, so they carry information the model does '
                 'not. Under the finding\'s declared convention the corpus is '
                 'free to the decoder, so this is inside the rules — but the '
                 'honest reading is a 2-D one: the full-precision arm gains '
                 '0.0073 nats from the same data (O8b), so a compressed arm '
                 'that beats the model is beating it on DATA, not on '
                 'PARSIMONY alone.')}
    json.dump(o, open(P, 'w'), indent=1)


if __name__ == '__main__':
    main()
