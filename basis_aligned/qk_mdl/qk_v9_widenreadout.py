"""Eval-time readout-visibility test on the EXISTING V9 checkpoint (Logan's queue,
2026-08-03; the V9-N3 / V9-N3r trainings were SKIPPED by Logan's same-day update --
this eval-only piece is what survives of queue item 1).

Question: how much does the readout's window matter? V9 (slots + group-lasso +
N=6 window) trains its readout on the writes of blocks 6..11 only. Here we widen
the readout's visibility AT EVAL TIME (no retrain) to
  (a) all 24 module writes (blocks 0..11), and
  (b) all 24 module writes PLUS the normed embedding,
and report the paired per-token CE delta vs the V9 baseline (and vs the vanilla
control V1 = 5.7105). bf16 eval convention matching qk_v9_heldloss.npy.

Saves qk_v9n3.json (the file reserved for queue item 1).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math
import numpy as np

import qk_tokenline_train as Q
import qk_v9_common as C
from qk_deeproute_train import DEPTH

QK = C.QK


def paired(pt_a, pt_b):
    """mean(a-b) with per-token and per-sequence standard errors."""
    dd = pt_a - pt_b
    ds = dd.reshape(len(Q.HELD), Q.T).mean(1)
    return {'delta': float(dd.mean()),
            'se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
            'se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}


if __name__ == '__main__':
    Q.gpu_guard(min_free=2500)
    model, ck = C.load_variant('qk_v9', 'V9', lambda li: C.window_vis(li, N=6))

    # sanity gate: reproduce the stored V9 held CE under the same convention
    base_ce, base_pt = Q.eval_held(model, per_token=True)
    saved = np.load(f'{QK}/qk_v9_heldloss.npy')
    drift = float(np.abs(base_pt - saved).mean())
    print(f"V9 base held CE {base_ce:.5f} (stored 5.75192), "
          f"mean |per-token drift| {drift:.2e}", flush=True)
    assert abs(base_ce - 5.75192) < 2e-3, "V9 baseline does not reproduce"

    v1 = np.load(f'{QK}/qk_deeproute_heldloss_V1.npy')
    out = {'note': ('V9-N3 and V9-N3r trainings SKIPPED per Logan update '
                    '2026-08-03; this file holds the surviving eval-time '
                    'readout-visibility test on the existing V9 (N=6) model. '
                    'No retraining: the readout visibility list is widened at '
                    'eval only.'),
           'v9_base': {'held_ce': float(base_ce),
                       'readout_vis': 'writes of blocks 6..11 (12 streams)',
                       'vs_V1': paired(base_pt, v1)}}

    # (a) readout reads ALL 24 module writes
    model.vis[DEPTH] = list(range(1, 2 * DEPTH + 1))
    ce_a, pt_a = Q.eval_held(model, per_token=True)
    out['widen_all_writes'] = {
        'held_ce': float(ce_a),
        'readout_vis': 'all 24 module writes (blocks 0..11), no embedding',
        'vs_v9': paired(pt_a, base_pt), 'vs_V1': paired(pt_a, v1)}
    print(f"widened (all writes)      held CE {ce_a:.5f} "
          f"delta vs V9 {out['widen_all_writes']['vs_v9']['delta']:+.4f}",
          flush=True)

    # (b) all 24 writes + the normed embedding
    model.vis[DEPTH] = list(range(2 * DEPTH + 1))
    ce_b, pt_b = Q.eval_held(model, per_token=True)
    out['widen_all_writes_plus_embedding'] = {
        'held_ce': float(ce_b),
        'readout_vis': 'embedding + all 24 module writes',
        'vs_v9': paired(pt_b, base_pt), 'vs_V1': paired(pt_b, v1)}
    print(f"widened (writes + emb)    held CE {ce_b:.5f} "
          f"delta vs V9 {out['widen_all_writes_plus_embedding']['vs_v9']['delta']:+.4f}",
          flush=True)

    json.dump(out, open(f'{QK}/qk_v9n3.json', 'w'), indent=2)
    print('saved qk_v9n3.json', flush=True)
