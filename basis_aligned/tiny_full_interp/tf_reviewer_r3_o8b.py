"""O8b -- POWER CHECK on the "does any description predict the data better
than the model?" test.

The O8 confound control (full precision, Adam lr 6e-4, cross-entropy on fresh
est text) selected step 0, i.e. it never improved on the trained model at all.
A test whose control has no power cannot license a negative, so the same arm is
re-run across learning rates, and a shorter description is given the same
sweep.  If even the full-precision arm cannot be improved by fresh data, the
negative is about the OPTIMISER's reach, not about the descriptions -- and that
has to be said, not hidden.
"""
import json
import os

import torch

import tf_compress as CC
import tf_reviewer_r3 as R3

HERE = os.path.dirname(os.path.abspath(__file__))
P = f'{HERE}/tf_reviewer_round_3_compression.json'


def main():
    D = CC.D1Desc('tf_vanilla_d1_w128_b8192_s0')
    mce = D.score()['ce']
    rows = []
    for emb_b, body_b in ((32, 32), (6, 8)):
        for lr in (2e-4, 5e-5, 1e-5):
            Pd, bits, tl = R3.distill_ce(D, emb_b, body_b, lr=lr)
            s = D.score(Pd)
            rows.append({'emb_bits': emb_b, 'body_bits': body_b, 'lr': lr,
                         'bits': bits.total, 'held_ce': s['ce'],
                         'delta_ce_vs_model': s['ce'] - mce,
                         'kl_from_model': s['kl'],
                         'best_step': tl['best_step'],
                         'est_val_ce': tl['est_val_ce'],
                         'beats_the_model_on_data': bool(s['ce'] < mce)})
            print(f'{emb_b}/{body_b} lr={lr:g}: held CE {s["ce"]:.5f} '
                  f'(model {mce:.5f}, delta {s["ce"]-mce:+.5f}) best_step '
                  f'{tl["best_step"]}', flush=True)
    o = json.load(open(P))
    o['O8b_power_check_on_the_data_fit_test'] = {
        'model_held_ce': mce, 'rows': rows,
        'any_arm_beats_the_model': any(r['beats_the_model_on_data']
                                       for r in rows),
        'best_full_precision_delta_ce': min(
            r['delta_ce_vs_model'] for r in rows if r['emb_bits'] == 32)}
    json.dump(o, open(P, 'w'), indent=1)


if __name__ == '__main__':
    main()
