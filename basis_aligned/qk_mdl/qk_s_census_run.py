"""E14a slot-utilization census on the width-1152 scale checkpoints (the
11-vs-48-dim comparison the local session requested): write-covariance
effective rank / slot dim per module, using the local session's exact
census implementation (qk_e14_slotcap_run.census, checkpoint-agnostic).

Local verdict at 11-dim slots (qk_e9_a): SATURATION -- 10/24 modules above
0.8 utilization, mid-stack MLPs at 0.91-0.94. Question: does utilization
drop at 48-dim slots (saturation eases with width) or stay high (the model
fills whatever it gets)?

Census set: combo3e5loss (the readable recipe candidate), muonbase, gc3e5,
slots (unpenalized baseline), combo (proximal). Census data = scale held
rows (never trained). Output qk_s_w1152_census.json. Safe to run alongside
a training arm (needs ~4 GB; run on the GPU with headroom).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json

import torch

import qk_s_gate_run as G
import qk_s_e1_run as E1R
import qk_e_common as E
import qk_v9_common as C
import qk_w1152_train as W2
from qk_e14_slotcap_run import census

OUT = os.path.join(G.OUT_DIR, 'qk_s_w1152_census.json')
ARMS = [
    ('qk_s_w1152_combo3e5loss', E1R.make_e1),
    ('qk_s_w1152_muonbase', lambda: C.make_variant('W1152', None)),
    ('qk_s_w1152_gc3e5', lambda: C.make_variant('W1152', None)),
    ('qk_s_w1152_slots', lambda: C.make_variant('W1152', None)),
    ('qk_s_w1152_combo', E1R.make_e1),
]


def main():
    W2.patch_width(G.WIDTH)
    G.setup_data()          # Q.HELD = scale held (never trained), width 1152
    out = G.loadj(OUT)
    out['note'] = ('E14a census at slot dim 48 (local ref: 11-dim qk_e9_a '
                   'SATURATED, 10/24 > 0.8). census data = scale held rows.')
    for stem, factory in ARMS:
        if stem in out:
            print(f"{stem}: done -- skip", flush=True)
            continue
        if not os.path.exists(os.path.join(G.OUT_DIR, f'{stem}.pt')):
            continue
        ck = torch.load(os.path.join(G.OUT_DIR, f'{stem}.pt'),
                        map_location='cuda', weights_only=False)
        m = factory()
        m.load_state_dict(ck['state_dict'])
        m.eval().float()
        rec = census(m, stem)
        out[stem] = rec
        G.savej(OUT, out)
        print(f"{stem}: {rec['verdict']} (sat {rec['n_saturated']} / "
              f"mod {rec['n_moderate']} / slack {rec['n_slack']})",
              flush=True)
        del m
        torch.cuda.empty_cache()
    print("census done", flush=True)


if __name__ == '__main__':
    main()
