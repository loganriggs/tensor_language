"""E5: decompose the fresh-data cost of the interpretable base recipe.

The overnight fresh single-epoch chain re-priced the slots+lasso base (E0b) at
+0.3417 nats vs vanilla (E0a), SE 0.0022 -- versus ~+0.08 under the old 6-epoch
protocol. The old number was memorization-subsidized (Logan's concern,
confirmed). This run splits the fresh-data cost into its two components and
checks whether a lighter penalty keeps the wiring readability that motivated it:

  E5slots   slots + nonzero write init, NO group-lasso (gc 0)   -> slots-only cost
  E5gc3e5   slots + lasso coeff 3e-5                            -> penalty dial
  E5gc1e5   slots + lasso coeff 1e-5                            -> penalty dial
  E5van8    vanilla at lr 0.008 (family sweep chose 0.004 at the GRID EDGE on
            the E0b arm; this checks whether the vanilla control is understated,
            which would WIDEN the true recipe cost)

All arms: fresh single-epoch protocol (identical data order, batch 16, 8250
steps), paired vs E0a and E0b, full loss curves saved. Wiring light probe +
token probe on the slots arms (the question: wiring Spearman vs penalty
strength; E0b reference at gc 1e-4 was all 0.778 / effectual 0.578).
Positive control: E5slots at init == E0b at init (same factory, penalty is
train-time only). Results -> qk_e5.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

import qk_e_common as E
from qk_e_common import Q, W

JP = E.jpath('qk_e5.json')

ARMS = (
    ('qk_e5_slots264',  'E5slots', E.make_e0b, 0.0,   None),
    ('qk_e5_gc3e5_264', 'E5gc3e5', E.make_e0b, 3e-5,  None),
    ('qk_e5_gc1e5_264', 'E5gc1e5', E.make_e0b, 1e-5,  None),
    ('qk_e5_van008_264', 'E5van8', E.make_e0a, 0.0,   0.008),
)


@torch.no_grad()
def controls():
    # E5slots and E0b share a factory; the lasso only exists in the loss, so
    # their init forwards must agree exactly (known-answer identity).
    idx = Q.HELD[:2, :Q.T]
    torch.manual_seed(0)
    m1 = E.make_e0b().eval().float()
    torch.manual_seed(0)
    m2 = E.make_e0b().eval().float()
    d = (m1(idx) - m2(idx)).abs().max().item()
    print(f"control E5slots(init)==E0b(init): max |logit diff| {d:.2e}",
          flush=True)
    assert d == 0.0
    del m1, m2
    torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()
    for stem, key, factory, gc, lr in ARMS:
        m = factory()
        E.merge(JP, f'param_counts_{key}', W.param_counts(m))
        del m
        torch.cuda.empty_cache()
        E.train_arm(stem, JP, key, factory, gc, lr=lr)
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
    if not E.SMOKE:
        for stem, key, factory, gc, lr in ARMS[:3]:      # slots arms only
            E.probe_arm(stem, factory, JP, f'light_probe_{key}',
                        tok_key=f'tok_probe_{key}')
        e0 = E.loadj(E.jpath('qk_e0.json'))
        if 'light_probe_E0b' in e0:
            E.merge(JP, 'wiring_spearman_E0b_reference_gc1e4', {
                k: e0['light_probe_E0b'][k]
                for k in ('wiring_spearman_all', 'wiring_spearman_effectual',
                          'wiring_top10_precision')})
    print('e5 costdecomp run done', flush=True)
