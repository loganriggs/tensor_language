"""V9 (Logan's queue item 1, follow-up to RESULTS section 109): subspace-partitioned
write slots + group-lasso reads + N=6 LOOKBACK WINDOW. All three ingredients were
individually free or near-free (slots+lasso: tie with vanilla; the window-6 model of
qk_window_train was the free windowed config) -- is the COMBINATION free too?

Architecture: V8Route with variant 'V9' (sum kernel, V5 slot projections, nonzero
write init std 0.02/sqrt(24) from generator 888) and visibility overridden to the
qk_window_train N=6 rule: block l reads only the writes of blocks l-6..l-1, the
embedding only if l < 6; the readout (consumer 12) reads blocks 6..11, no embedding.
Group-lasso coefficient FIXED at 1e-4 (the section-109 winner, per directive); lr
swept {0.001, 0.002, 0.003} x 400 steps WITH the penalty active; full 6-epoch run
(4122 steps, batch 8) at the winner. Held CE paired vs the section-108 re-swept
vanilla control V1 (qk_deeproute_heldloss_V1.npy, 5.7105).

Positive controls: (a) V9 arch with full-window visibility (N=13), identity
projections, zero writes == variant-A MiniBilin at init; (b) the window visibility
table matches the spec at hand-checked rows.

Saves qk_v9.pt / qk_v9_heldloss.npy; sweep + CE -> qk_v9.json. Probe in qk_v9_probe.py.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import qk_tokenline_train as Q
import qk_v8_train as V8T
import qk_v9_common as C

QK = C.QK
VIS = lambda li: C.window_vis(li, N=6)
FACTORY = lambda: C.make_variant('V9', VIS)


def controls():
    # (b) spec spot-checks of the window table
    v = [VIS(li) for li in range(13)]
    assert v[0] == [0]
    assert v[3] == [0, 1, 2, 3, 4, 5, 6], v[3]                    # emb + blocks 0-2
    assert v[6] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], v[6]  # blocks 0-5, no emb
    assert v[9] == sorted(sum([[1 + 2*j, 2 + 2*j] for j in range(3, 9)], [])), v[9]
    assert v[12] == sorted(sum([[1 + 2*j, 2 + 2*j] for j in range(6, 12)], [])), v[12]
    print("window visibility table matches spec (rows 0,3,6,9,12 checked)", flush=True)
    # (a) identity control through the same construction path
    C.identity_control('V9', lambda li: C.window_vis(li, N=13))


if __name__ == '__main__':
    print(f"V9 run: slots + group-lasso {C.GROUP_COEFF} + window-6; "
          f"lr sweep {C.LRS} x 400, full {V8T.STEPS} steps", flush=True)
    Q.gpu_guard(min_free=7000)
    controls()

    path = f'{QK}/qk_v9.json'
    out = json.load(open(path)) if os.path.exists(path) else {}

    # ---- lr sweep (penalty active; coefficient fixed by directive) ----
    if 'lrsweep' not in out:
        res = {}
        for lr in C.LRS:
            print(f"-- V9 lr sweep {lr} (gc {C.GROUP_COEFF})", flush=True)
            log = V8T.train_v8(lr, C.GROUP_COEFF, 400, log_every=100, save=False,
                               factory=FACTORY)
            res[str(lr)] = {'held100_ce': (None if log['diverged'] else
                                           round(log['final_held_ce'], 4)),
                            'diverged': log['diverged'], 'spikes': log['spikes']}
        ok = {k: v for k, v in res.items() if not v['diverged']}
        out['lrsweep'] = {'results': res,
                          'chosen': float(min(ok, key=lambda k: ok[k]['held100_ce'])),
                          'note': 'swept with the fixed 1e-4 group penalty active'}
        json.dump(out, open(path, 'w'), indent=2)
    LR = out['lrsweep']['chosen']
    print(f"V9 lr chosen: {LR}", flush=True)

    # ---- full run ----
    if not os.path.exists(f'{QK}/qk_v9.pt'):
        print(f"==== training V9 full run (lr {LR}, gc {C.GROUP_COEFF}) ====", flush=True)
        log = V8T.train_v8(LR, C.GROUP_COEFF, V8T.STEPS, factory=FACTORY,
                           save_stem='qk_v9')
        out['full_run'] = {'lr': LR, 'group_coeff': C.GROUP_COEFF,
                           'held_ce_bf16': log['final_held_ce'],
                           'spikes': log['spikes'],
                           'final_penalty': log.get('final_penalty')}
        json.dump(out, open(path, 'w'), indent=2)
    else:
        print("qk_v9.pt exists -- skip training", flush=True)

    # ---- paired CE vs the width-384 vanilla control V1 ----
    out['ce'] = C.paired_ce('qk_v9_heldloss.npy', 'qk_deeproute_heldloss_V1.npy',
                            label='V1')
    out['ce'].update({'lr': LR, 'group_coeff': C.GROUP_COEFF})
    json.dump(out, open(path, 'w'), indent=2)
    print(json.dumps(out['ce'], indent=2), flush=True)
