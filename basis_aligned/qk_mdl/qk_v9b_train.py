"""V9b (Logan's queue item 2): slots + group-lasso + LOOSER DILATION mask,
visibility distances {1,2,3,4,8} instead of section 109's {1,2,4,8} --
interpolating toward the sharp-edge masked regime at (hopefully) lower CE cost.
Section 109: the full dilation mask cost +0.115 nats; the extra distance-3 edge
adds 2 more visible source blocks per mid-stack consumer.

Same conventions as qk_v9_train.py (V8Route, slot projections, nonzero write init,
group coeff fixed 1e-4, lr swept {0.001,0.002,0.003} with penalty active, 4122
steps, paired vs V1). Visibility: consumer l reads block j iff l-j in {1,2,3,4,8};
embedding iff l in {1,2,3,4,8} plus base case l=0; readout (12) sees blocks
{4,8,9,10,11}. Saves qk_v9b.pt / qk_v9b_heldloss.npy / qk_v9b.json.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import qk_tokenline_train as Q
import qk_v8_train as V8T
import qk_v9_common as C

QK = C.QK
VIS = lambda li: C.dil_vis(li, dil=C.DIL_LOOSE)
FACTORY = lambda: C.make_variant('V9b', VIS)


def controls():
    v = [VIS(li) for li in range(13)]
    assert v[0] == [0]
    # block 5: distances {1,2,3,4} -> blocks 1,2,3,4; l=5 not in set -> no emb...
    # careful: 5 not in {1,2,3,4,8} -> no emb; blocks j with 5-j in set: j in {1,2,3,4}
    assert v[5] == sorted(sum([[1 + 2*j, 2 + 2*j] for j in (1, 2, 3, 4)], [])), v[5]
    # block 9: blocks {1,5,6,7,8}; 9 not in set -> no emb
    assert v[9] == sorted(sum([[1 + 2*j, 2 + 2*j] for j in (1, 5, 6, 7, 8)], [])), v[9]
    # readout: blocks {4,8,9,10,11}; 12 not in set -> no emb
    assert v[12] == sorted(sum([[1 + 2*j, 2 + 2*j] for j in (4, 8, 9, 10, 11)], [])), v[12]
    # block 3 IS in the set -> emb visible; blocks 0,1,2 (distances 3,2,1) - wait 3-0=3 yes
    assert v[3] == [0] + sorted(sum([[1 + 2*j, 2 + 2*j] for j in (0, 1, 2)], [])), v[3]
    print("loose-dilation visibility table matches spec (rows 0,3,5,9,12 checked)",
          flush=True)
    C.identity_control('V9b', lambda li: C.dil_vis(li, dil=tuple(range(1, 13))))


if __name__ == '__main__':
    print(f"V9b run: slots + group-lasso {C.GROUP_COEFF} + dilation {C.DIL_LOOSE}; "
          f"lr sweep {C.LRS} x 400, full {V8T.STEPS} steps", flush=True)
    Q.gpu_guard(min_free=7000)
    controls()

    path = f'{QK}/qk_v9b.json'
    out = json.load(open(path)) if os.path.exists(path) else {}

    if 'lrsweep' not in out:
        res = {}
        for lr in C.LRS:
            print(f"-- V9b lr sweep {lr} (gc {C.GROUP_COEFF})", flush=True)
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
    print(f"V9b lr chosen: {LR}", flush=True)

    if not os.path.exists(f'{QK}/qk_v9b.pt'):
        print(f"==== training V9b full run (lr {LR}, gc {C.GROUP_COEFF}) ====", flush=True)
        log = V8T.train_v8(LR, C.GROUP_COEFF, V8T.STEPS, factory=FACTORY,
                           save_stem='qk_v9b')
        out['full_run'] = {'lr': LR, 'group_coeff': C.GROUP_COEFF,
                           'held_ce_bf16': log['final_held_ce'],
                           'spikes': log['spikes'],
                           'final_penalty': log.get('final_penalty')}
        json.dump(out, open(path, 'w'), indent=2)
    else:
        print("qk_v9b.pt exists -- skip training", flush=True)

    out['ce'] = C.paired_ce('qk_v9b_heldloss.npy', 'qk_deeproute_heldloss_V1.npy',
                            label='V1')
    out['ce'].update({'lr': LR, 'group_coeff': C.GROUP_COEFF})
    json.dump(out, open(path, 'w'), indent=2)
    print(json.dumps(out['ce'], indent=2), flush=True)
