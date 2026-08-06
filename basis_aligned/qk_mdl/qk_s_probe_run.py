"""Light wiring probes (weight-vs-causal Spearman) on the width-1152 scale
checkpoints. THE question after the anneal negative: did proximal 1e-4 buy
ANY wiring readability at scale, or was it free because non-binding?

Small-scale in-loss AdamW reference (w264, qk_e5): Spearman 0.07 (no lasso)
-> 0.42 (1e-5) -> 0.62 (3e-5) -> 0.78 (1e-4).

CAVEAT recorded in every output: the probe eval data on this box is the
SUBSTITUTE corpus (fresh34k rows [5500:6000] bound as R2.HELD at import) --
never trained on by any scale model, so the causal graph is valid, but the
absolute numbers are not on the same eval rows as the small-scale probes.

Probes (in order): combo, gc3e5, slots, gc1e4, muonprox, e1 -- each ~30-40
min fp32 at width 1152; results merge into each arm's qk_s_ JSON under
'light_probe' as they land, so partial runs still pay off. Skips arms whose
JSON already has the key (idempotent) or whose checkpoint is missing.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time

import torch

import qk_s_gate_run as G          # neuters gpu_guard BEFORE the probe chain
import qk_s_e1_run as E1R
import qk_e_common as E
import qk_v9_common as C
import qk_w1152_train as W2

CAVEAT = ('probe eval data = substitute corpus (fresh34k[5500:6000]); '
          'valid causal graph, rows differ from small-scale probes')

ARMS = [
    ('qk_s_w1152_combo', E1R.make_e1),
    ('qk_s_w1152_gc3e5', lambda: C.make_variant('W1152', None)),
    ('qk_s_w1152_slots', lambda: C.make_variant('W1152', None)),
    ('qk_s_w1152_gc1e4', lambda: C.make_variant('W1152', None)),
    ('qk_s_w1152_muonprox', lambda: C.make_variant('W1152', None)),
    ('qk_s_w1152_e1', E1R.make_e1),
    # in-loss 1e-4 under Muon: if the lasso keeps its readability under Muon
    # (which already wins CE), the recipe candidate becomes
    # in-loss-3e-5 + per-slot norm + Muon
    ('qk_s_w1152_muonbase', lambda: C.make_variant('W1152', None)),
]


def _add_night_arms():
    """Shared-values arms from the overnight queue (import deferred so the
    original probe chain's imports stay unchanged). shrink3e5 is NOT here:
    its stream-0 ablation needs the per-consumer remnant mean
    (qk_e16_shrinkemb_run.e16_light_probe), run attended."""
    import qk_s_e1sv_run as SVR
    ARMS.append(('qk_s_w1152_combo3e5sv', SVR.make_e1sv))
    ARMS.append(('qk_s_w1152_combo3e5svpb', SVR.make_e1svpb))


_add_night_arms()


def main():
    W2.patch_width(G.WIDTH)
    G.setup_data()                    # width/batch globals; R2.HELD unaffected
    E.WIDTH, E.SUB = G.WIDTH, G.WIDTH // E.NGROUP
    for stem, factory in ARMS:
        jp = os.path.join(G.OUT_DIR, f'{stem}.json')
        ckp = os.path.join(G.OUT_DIR, f'{stem}.pt')
        out = G.loadj(jp)
        if 'light_probe' in out:
            print(f"{stem}: probed already -- skip", flush=True)
            continue
        if not os.path.exists(ckp):
            print(f"{stem}: no checkpoint -- skip", flush=True)
            continue
        t0 = time.time()
        print(f"==== probing {stem} ====", flush=True)
        ck = torch.load(ckp, map_location='cuda', weights_only=False)
        m = factory()
        m.load_state_dict(ck['state_dict'])
        m.eval().float()
        rec = E.light_probe(m)
        rec['caveat'] = CAVEAT
        out = G.loadj(jp)
        out['light_probe'] = rec
        G.savej(jp, out)
        print(f"{stem}: Spearman all {rec['wiring_spearman_all']} "
              f"effectual {rec['wiring_spearman_effectual']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
