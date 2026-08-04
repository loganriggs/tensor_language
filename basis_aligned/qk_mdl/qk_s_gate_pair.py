"""Paired per-token CE stats for the width-1152 scale gate (qk_s_gate_run.py
arms). Pairs every trained arm against vanilla (and the slots arms against
each other) on BOTH held sets, with sequence-clustered SEs. Idempotent; safe
to run any time -- it uses whatever heldloss files exist. Writes
qk_s_w1152_gate.json.
"""
import json
import math
import os

import numpy as np

QK = os.path.dirname(os.path.abspath(__file__))
T = 512
PATH = f'{QK}/qk_s_w1152_gate.json'
ARMS = ('vanilla', 'slots', 'gc3e5', 'gc1e4', 'muonbase')
PAIRS = [('slots', 'vanilla'), ('gc3e5', 'vanilla'), ('gc3e5', 'slots'),
         ('gc1e4', 'vanilla'), ('gc1e4', 'slots'),
         ('muonbase', 'gc1e4'),          # THE optimizer gate (same arch/data)
         ('muonbase', 'vanilla')]


def paired(fa, fb):
    la, lb = np.load(fa), np.load(fb)
    assert la.shape == lb.shape and len(la) % T == 0
    d = la - lb
    ds = d.reshape(-1, T).mean(1)
    return {'arm_ce': round(float(la.mean()), 5),
            'ctl_ce': round(float(lb.mean()), 5),
            'delta': round(float(d.mean()), 5),
            'se_token': round(float(d.std(ddof=1) / math.sqrt(len(d))), 6),
            'se_seq': round(float(ds.std(ddof=1) / math.sqrt(len(ds))), 6),
            'n_seq': len(ds)}


def main():
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    for suffix, label in (('heldloss', 'scale_held'), ('f34kloss', 'f34k')):
        for a, b in PAIRS:
            fa = f'{QK}/qk_s_w1152_{a}_{suffix}.npy'
            fb = f'{QK}/qk_s_w1152_{b}_{suffix}.npy'
            if os.path.exists(fa) and os.path.exists(fb):
                key = f'{a}_minus_{b}_{label}'
                out[key] = paired(fa, fb)
                print(key, json.dumps(out[key]), flush=True)
    for arm in ARMS:
        jp = f'{QK}/qk_s_w1152_{arm}.json'
        if os.path.exists(jp):
            d = json.load(open(jp))
            if 'run' in d:
                out[f'{arm}_run'] = {k: d['run'].get(k) for k in
                                     ('lr', 'held_ce_scale_bf16',
                                      'held_ce_f34k_bf16', 'spikes',
                                      'diverged', 'final_penalty',
                                      'sec_per_step', 'peak_mem_mib')}
    json.dump(out, open(PATH, 'w'), indent=2)
    print(f"wrote {PATH}", flush=True)


if __name__ == '__main__':
    main()
