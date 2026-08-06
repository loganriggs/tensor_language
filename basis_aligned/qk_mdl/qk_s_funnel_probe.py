"""Wiring/neck probes for the two overnight scale-funnel checkpoints
(qk_s_w1152_funnelsv / qk_s_w1152_funnel). Uses the funnel family's own
probe suite (funnel_light_probe consumption + weight-support Spearman,
neck_spectra, neck_info_probe) on the scale held rows -- the same eval rows
as every other w1152 probe on this box. argv[1] in {funnelsv, funnel}.
Results merge into the arm's qk_s_ JSON. ~1.5-2h fp32 at these widths.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys

import torch

import qk_s_gate_run as G
import qk_e_common as E
import qk_s_funnel1152_run as FS

M12 = FS.M12

ARM = sys.argv[1] if len(sys.argv) > 1 else 'funnelsv'
STEM = f'qk_s_w1152_{ARM}'
FACTORY = FS.make_funnel_sv if ARM == 'funnelsv' else FS.make_funnel_plain


def main():
    import qk_w1152_train as W2
    W2.patch_width(G.WIDTH)
    G.setup_data()
    jp = os.path.join(G.OUT_DIR, f'{STEM}.json')
    out = G.loadj(jp)
    ck = torch.load(os.path.join(G.OUT_DIR, f'{STEM}.pt'),
                    map_location='cuda', weights_only=False)
    m = FACTORY()
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    if 'neck_spectra' not in out:
        out['neck_spectra'] = M12.neck_spectra(m)
        G.savej(jp, out)
        print('neck_spectra done', flush=True)
    if 'neck_info_probe' not in out:
        out['neck_info_probe'] = M12.neck_info_probe(m)
        G.savej(jp, out)
        print('neck_info done', flush=True)
    if 'light_probe' not in out:
        out['light_probe'] = M12.funnel_light_probe(m)
        out['light_probe']['caveat'] = ('scale held rows (never trained); '
                                        'same eval rows as all w1152 probes')
        G.savej(jp, out)
    print(f'{ARM} probes done', flush=True)


if __name__ == '__main__':
    main()
