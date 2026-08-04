"""E0-MUON: optimizer control arms (Logan spec update three, fresh
single-epoch batch-16 protocol, see qk_e_common).

Trains the two control ARCHITECTURES with the MUON optimizer instead of AdamW:

  E0a-muon  vanilla width 264
  E0b-muon  V8 base (slots + group-lasso 1e-4 + nonzero write init)

Muon (implemented from scratch in qk_e_common, nanoGPT-speedrun convention):
nesterov momentum 0.95 buffer, then 5-iteration Newton-Schulz
orthogonalization of each 2D update in bfloat16, scaled by
sqrt(max(rows, cols) / min(rows, cols)). Muon applies ONLY to the 2D hidden
matrices (attention projections, Left/Right/Down); the tied
embedding/unembedding stays on AdamW (weight decay 0.1) at the family lr, and
sub-2D parameters (Down_bias) stay on AdamW without decay. Muon lr swept
{0.01, 0.02, 0.04} x 400 steps on E0b-muon (0.02 is the community default at
this scale), winner used for both arms; warmup+cosine schedule scales both
optimizers' learning rates identically.

CAVEAT recorded in the JSON: the group-lasso penalty is kept in the loss
exactly as in the AdamW arms so E0b-muon vs E0b is a pure optimizer
comparison, but Newton-Schulz orthogonalization distorts the penalty's
shrinkage direction (the penalty gradient is orthogonalized together with the
data gradient). A decoupled proximal soft-threshold on the read groups would
be cleaner -- flagged as follow-up, deliberately not implemented now.

Positive controls: (i) Newton-Schulz output has singular values roughly in
[0.7, 1.3] on random matrices of both aspect ratios; (ii) the Muon/AdamW
parameter split covers every parameter exactly once, with the embedding on
AdamW. Results -> qk_e0m.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E
from qk_e_common import Q, torch

JP = E.jpath('qk_e0m.json')
E0AM_STEM = 'qk_e0a_muon264'
E0BM_STEM = 'qk_e0b_muon264'
PENALTY_NOTE = ('group-lasso kept in the loss as in the AdamW arms (pure '
                'optimizer comparison), but Newton-Schulz orthogonalization '
                'distorts the penalty shrinkage direction; decoupled proximal '
                'soft-threshold flagged as follow-up, not implemented')


@torch.no_grad()
def controls():
    # (i) Newton-Schulz spectrum on random matrices, both aspect ratios
    g = torch.Generator().manual_seed(123)
    shapes = [(Q.EXP * Q.D, Q.D), (Q.D, Q.D), (Q.D, Q.EXP * Q.D)]
    spectra = {}
    for sh in shapes:
        X = torch.randn(*sh, generator=g)
        O = E.ns_orth(X).float()
        sv = torch.linalg.svdvals(O)
        smin, smax = float(sv.min()), float(sv.max())
        smed = float(sv.median())
        spectra[str(sh)] = [round(smin, 3), round(smed, 3), round(smax, 3)]
        print(f"control NS spectrum {sh}: singular values min {smin:.3f} "
              f"median {smed:.3f} max {smax:.3f}", flush=True)
        # A random SQUARE matrix has smallest singular values near zero
        # (Marchenko-Pastur) and 5 NS iterations correctly leave near-zero
        # directions near zero, so a hard floor on smin is wrong for square
        # shapes (it aborted the first chain run). Muon's actual guarantee:
        # non-negligible directions are pushed to ~1, nothing overshoots.
        assert 0.5 < smed and smax < 1.5, "Newton-Schulz spectrum off"
    E.merge(JP, 'ns_spectrum_control', spectra)
    # (ii) parameter split covers everything once, embedding on AdamW
    m = E.make_e0b()
    mu, dec, nod = E.muon_params_split(m)
    n_all = sum(p.numel() for p in m.parameters())
    n_split = sum(p.numel() for p in mu + dec + nod)
    assert n_all == n_split
    assert len(dec) == 1 and dec[0] is m.wte.weight
    assert all(p.dim() == 2 for p in mu)
    print(f"control param split: {len(mu)} muon matrices "
          f"({sum(p.numel() for p in mu)} params), embedding + "
          f"{len(nod)} sub-2D on AdamW; total covered {n_split}=={n_all}",
          flush=True)
    del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()

    lr_a = E.get_lr()                            # family AdamW lr (E0 sweep)
    muon_lr = E.lr_sweep(JP, 'muon_lrsweep', E.MUON_LRS,
                         lambda lr, steps: E.train_muon(
                             lr, E.GC, steps, log_every=100, save=False,
                             factory=E.make_e0b, lr_adamw=lr_a))

    trainer = lambda lr, gc, steps, **kw: E.train_muon(
        lr, gc, steps, lr_adamw=lr_a, **kw)
    for stem, key, factory, gc in ((E0AM_STEM, 'E0a_muon', E.make_e0a, 0.0),
                                   (E0BM_STEM, 'E0b_muon', E.make_e0b, E.GC)):
        E.train_arm(stem, JP, key, factory, gc, lr=muon_lr, trainer=trainer,
                    extra={'optimizer': 'muon', 'muon_momentum': 0.95,
                           'ns_steps': 5, 'lr_adamw_for_embedding': lr_a,
                           'penalty_note': PENALTY_NOTE})
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
    print('e0m muon run done', flush=True)
