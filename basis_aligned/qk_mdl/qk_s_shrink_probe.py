"""Remnant-aware probes for the w1152 shrink3e5 checkpoint (E16b transfer).

Two probes from the local session's qk_e16_shrinkemb_run, reused verbatim
(both width-generic): remnant_probe (per-consumer token recovery off
wte + W_i, no forward needed) and e16_light_probe (the wiring probe whose
stream-0 ablation mean is the PER-CONSUMER remnant mean -- plain
E.light_probe would ablate with the wrong token-channel mean). Same width
patching as qk_s_probe_run; probe eval rows = the substitute cooc corpus
R2.HELD, same caveat as every other w1152 wiring probe on this box.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch

import qk_s_gate_run as G
import qk_w1152_train as W2
import qk_e_common as E
import qk_s_e16_run as S16
import qk_e16_shrinkemb_run as E16

STEM = 'qk_s_w1152_shrink3e5'


def main():
    W2.patch_width(G.WIDTH)
    G.setup_data()
    E.WIDTH, E.SUB = G.WIDTH, G.WIDTH // E.NGROUP
    jp = os.path.join(G.OUT_DIR, f'{STEM}.json')
    out = G.loadj(jp)
    ck = torch.load(os.path.join(G.OUT_DIR, f'{STEM}.pt'),
                    map_location='cuda', weights_only=False)
    m = S16.make_shrink1152()
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    if 'remnant_probe' not in out:
        out['remnant_probe'] = E16.remnant_probe(m)
        G.savej(jp, out)
        print('remnant probe done', flush=True)
    if 'light_probe' not in out:
        out['light_probe'] = E16.e16_light_probe(m)
        out['light_probe']['caveat'] = ('probe eval rows = substitute cooc '
                                        'corpus (fresh34k[5500:6000])')
        G.savej(jp, out)
    print('shrink probes done', flush=True)


if __name__ == '__main__':
    main()
