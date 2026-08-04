"""E1 PER-SLOT RMSNORM at width 1152 (scale session; round 3).

Same experiment as qk_e1_slotnorm_run.py (per-48-dim-slot RMSNorm at the two
module inputs, global norms kept at the embedding and pre-readout) on the
scale protocol of qk_s_gate_run. At width 264 E1 was the ONLY arm that beat
the base (−0.026 vs E0b); question: does the win hold at slot dim 48, and
against WHICH control (paired vs vanilla, slots-only, and gc1e4 when those
exist)?

Penalty: group-lasso 1e-4 as in small-scale E1 (it trained under the E0b
family convention -- the arm pairs against gc1e4 for the like-for-like
comparison). AdamW, lr swept as in the gate arms.

Positive controls at width 1152 before training (ported from the small
runner): (i) E1 with ONE norm group == plain V8Route at init; (ii)
vectorized per-slot norm == naive loop at slot dim 48; plus the shared
identity/penalty/accum controls of the gate runner.

Imports E1Route from qk_e1_slotnorm_run (which pulls qk_e_common -- fine on
this box, the substitute cooc file exists). Outputs qk_s_w1152_e1.{json,pt},
_heldloss.npy, _f34kloss.npy. TEST mode: QK_S_TEST=1.
"""
import os
import sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import qk_s_gate_run as G
import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_v9_common as C
import qk_w1152_train as W2
from qk_deeproute_train import DEPTH
from qk_e1_slotnorm_run import E1Route

COEFF = 1e-4
NGROUP = 2 * DEPTH
STEM = 'qk_s_w1152_e1'
JP = os.path.join(G.OUT_DIR, f'{STEM}.json')

# splice the arm into the gate runner's tables so its sweep/train/preflight
# machinery applies unchanged
ARM = 'e1'
G.COEFF[ARM] = COEFF


def make_e1(norm_groups=None):
    C.register('E1')
    torch.manual_seed(Q.SEED)
    m = E1Route('E1', DEPTH).to('cuda')
    m.norm_groups = NGROUP if norm_groups is None else norm_groups
    return m


_orig_factory_for = G.factory_for


def factory_for(arm):
    if arm == ARM:
        return make_e1
    return _orig_factory_for(arm)


G.factory_for = factory_for


@torch.no_grad()
def e1_controls(out):
    if out.get('e1_controls_ok'):
        return
    idx = Q.HELD[:2, :Q.T]
    sub = Q.D // NGROUP
    base = C.make_variant('E1ctl').eval().float()
    m1 = make_e1(norm_groups=1).eval().float()
    d = (m1(idx) - base(idx)).abs().max().item()
    print(f"control E1(1 group)==V8Route at w1152 init: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-4
    m24 = make_e1().eval().float()
    d2 = (m24(idx) - base(idx)).abs().max().item()
    print(f"sanity E1({NGROUP} groups) differs from base: {d2:.2e}", flush=True)
    assert d2 > 1e-6
    x = torch.randn(3, 5, Q.D, device='cuda')
    fast = m24.slot_norm(x)
    naive = torch.empty_like(x)
    for k in range(NGROUP):
        sl = slice(sub * k, sub * (k + 1))
        naive[..., sl] = F.rms_norm(x[..., sl], (sub,))
    rel = (fast - naive).abs().max().item()
    print(f"control slot-norm fast vs naive (slot {sub}): {rel:.2e}", flush=True)
    assert rel < 1e-6
    del base, m1, m24
    torch.cuda.empty_cache()
    out['e1_controls_ok'] = True
    G.savej(JP, out)


def main():
    out = G.loadj(JP)
    W2.patch_width(G.WIDTH)
    total_steps, spec, f34k_held = G.setup_data()
    out['env'] = {'gpu': torch.cuda.get_device_name(0),
                  'torch': torch.__version__, 'cooc_substitute': True}
    out['data'] = spec
    out['arm'] = {'name': 'e1_slotnorm_w1152', 'group_coeff': COEFF,
                  'norm_groups': NGROUP, 'slot_dim': Q.D // NGROUP,
                  'write_init_std': R.WRITE_INIT_STD}
    G.savej(JP, out)
    e1_controls(out)
    # reuse the gate runner's preflight/sweep/train on the spliced arm, but
    # with this file's JSON path
    G.jp_of = lambda arm: JP
    micro = G.preflight_micro(ARM, out)
    print(f"e1: micro {micro} (accum {G.EFF_BATCH // micro})", flush=True)
    chosen, ranking = G.lr_sweep(ARM, micro, out)
    if os.path.exists(os.path.join(G.OUT_DIR, f'{STEM}.pt')) and 'run' in out:
        print(f"{STEM}.pt exists -- done", flush=True)
        return
    for pick, lr in enumerate(ranking):
        print(f"==== training e1 w1152 (lr {lr}"
              + (", fallback" if pick else "") + ") ====", flush=True)
        log = G.train_run(ARM, lr, total_steps, micro, save_stem=STEM,
                          f34k_held=f34k_held)
        if not log['diverged']:
            break
    out = G.loadj(JP)
    out['run'] = {'lr': lr, 'lr_fallback': pick > 0,
                  'held_ce_scale_bf16': log.get('final_held_ce'),
                  'held_ce_f34k_bf16': log.get('final_f34k_ce'),
                  'spikes': log['spikes'], 'diverged': log['diverged'],
                  'final_penalty': log.get('final_penalty'),
                  'peak_mem_mib': log.get('peak_mem_mib'),
                  'sec_per_step': log.get('sec_per_step_measured'),
                  'train_curve_every200': log['train_loss'],
                  'held100_scale_curve': log['held_ce']}
    G.savej(JP, out)
    print(json.dumps({k: out['run'][k] for k in
                      ('lr', 'held_ce_scale_bf16', 'held_ce_f34k_bf16',
                       'spikes', 'diverged')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
