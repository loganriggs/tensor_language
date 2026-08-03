"""E0 fresh-stream controls (Logan spec update 2026-08-03): the existing
width-264 checkpoints are 6-epoch cooc models, so the fresh single-epoch E
family needs its own controls trained on the identical fresh data stream:

  E0a  vanilla width 264 (no slots, no lasso, zero-init writes)
  E0b  V8 base at 264 (partitioned write slots + group-lasso 1e-4 on reads +
       nonzero write init, full visibility)

Both train exactly one pass over fineweb_fresh34k rows [0:33000] (8250 steps x
batch 4, fixed shuffle, same order as every other E arm). Deltas for E1-E4 are
paired vs both. E0b also gets the wiring-readability light probe and the
token-determined standard probes (baseline for E1's and E4's questions), and
this run replicates the slots+lasso CE cost on fresh data (E0b minus E0a).

Positive controls (before training): E0a at init == variant-A MiniBilin at the
same width (max |logit diff| < 1e-3); vectorized group penalty on E0b == naive
per-slot loop (rel err < 1e-6). Everything -> qk_e0.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

import qk_e_common as E
from qk_e_common import Q, V8T, W, DEPTH

JP = E.jpath('qk_e0.json')


@torch.no_grad()
def controls():
    idx = Q.HELD[:2, :Q.T]
    Q.NL = DEPTH
    ma = Q.make_model('A').eval().float()
    ref = ma(idx)
    del ma
    torch.cuda.empty_cache()
    m = E.make_e0a().eval().float()
    d = (m(idx) - ref).abs().max().item()
    print(f"control E0a(init)==vanilla-A@{Q.D}: max |logit diff| {d:.2e}",
          flush=True)
    assert d < 1e-3
    del m, ref
    torch.cuda.empty_cache()

    m = E.make_e0b().eval().float()
    p_fast = float(V8T._orig_group_penalty(m) if hasattr(V8T, '_orig_group_penalty')
                   else V8T.group_penalty(m))
    p_naive = 0.0
    for blk in m.h:
        for M in V8T.READ_MATS(blk):
            for k in range(E.NGROUP):
                p_naive += float(
                    M[:, E.SUB * k:E.SUB * (k + 1)].pow(2).sum()) ** 0.5
    rel = abs(p_fast - p_naive) / p_naive
    print(f"control E0b penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}", flush=True)
    assert rel < 1e-6
    del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()

    # family AdamW lr sweep on the E0b base arm (batch changed to 16, so the
    # old single-lr convention is re-validated; winner used by EVERY AdamW arm)
    chosen = E.lr_sweep(JP, 'lrsweep', E.LRS,
                        lambda lr, steps: V8T.train_v8(
                            lr, E.GC, steps, log_every=100, save=False,
                            factory=E.make_e0b))

    for stem, key, factory, gc in ((E.E0A_STEM, 'E0a', E.make_e0a, 0.0),
                                   (E.E0B_STEM, 'E0b', E.make_e0b, E.GC)):
        m = factory()
        E.merge(JP, f'param_counts_{key}', W.param_counts(m))
        del m
        torch.cuda.empty_cache()
        E.train_arm(stem, JP, key, factory, gc)
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')

    if not E.SMOKE:
        f_a, f_b = f'{E.E0A_STEM}_heldloss.npy', f'{E.E0B_STEM}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_a}') and os.path.exists(f'{E.QK}/{f_b}'):
            E.merge(JP, 'E0b_minus_E0a_fresh',
                    E.paired(f_b, f_a, len(Q.HELD), 'e0a'))
        E.probe_arm(E.E0B_STEM, E.make_e0b, JP,
                    'light_probe_E0b', tok_key='tok_probe_E0b')
    print('e0 controls run done', flush=True)
