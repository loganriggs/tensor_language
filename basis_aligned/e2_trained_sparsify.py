"""e2: train the block task from scratch, then run the iterated sparsification
protocol (L1 -> prune bottom frac -> repeat; on degradation revert + finetune
without L1). Arms differ in WHICH matrices get L1+pruned.

Readouts per arm: unfolded vs folded weight sparsity, block-structure of the
invariant interaction form B_c (split into: own-block mass / other-block mass
[probed to be zero by the data] / cross-block mass [NEVER probed - one block
active at a time - so unconstrained junk]), and off-distribution FVU vs the
natural extension y_c = x_{2c}x_{2c+1} on full-support inputs.
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
from common import (forward, fold, interaction, hoyer, near_zero_frac,
                    block_score, block_data, train, eval_fvu, iterated_sparsify,
                    remaining_frac, init_params, random_orthogonal)
from e1_handcoded import handcoded, rotated  # reuse the e1 constructions

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_BLOCKS, D_IN, D_MODEL, D_H = 4, 8, 8, 8
PAIRS = [(2 * c, 2 * c + 1) for c in range(N_BLOCKS)]
SEEDS = [0, 1, 2]

PRETRAIN_STEPS = 6000
PROTO = dict(l1=3e-4, lr=1e-3, steps_per_iter=1000, prune_frac=0.15,
             degrade_fvu=0.01, max_iters=40)


def data_fn():
    return block_data(512, DEV)


def eval_fn(p):
    return eval_fvu(p, lambda: block_data(4096, DEV))


@torch.no_grad()
def ood_fvu(p):
    """Full-support gaussian inputs vs the natural extension y_c=x_{2c}x_{2c+1}."""
    x = torch.randn(65536, D_IN, device=DEV)
    y = torch.stack([x[:, 2 * c] * x[:, 2 * c + 1] for c in range(N_BLOCKS)], 1)
    return float(((forward(p, x) - y) ** 2).mean() / y.var())


def block_mass_split(B):
    """|B| mass fractions: own block / other blocks (probed zero) / cross-block
    (never probed on-distribution)."""
    own = other = 0.0
    total = float(B.abs().sum())
    for c in range(N_BLOCKS):
        for b in range(N_BLOCKS):
            idx = torch.tensor(PAIRS[b], device=B.device)
            m = float(B[c][idx][:, idx].abs().sum())
            if b == c:
                own += m
            else:
                other += m
    return {'own_block': own / total, 'other_block_probed': other / total,
            'cross_block_unprobed': (total - own - other) / total}


def summarize(name, seed, p, masks, hist):
    rep = {k: {'hoyer': round(hoyer(w), 3), 'zero_frac': round(near_zero_frac(w), 3)}
           for k, w in {**p, **fold(p)}.items()}
    B = interaction(p)
    out = {'arm': name, 'seed': seed,
           'fvu': eval_fn(p), 'ood_fvu': ood_fvu(p),
           'frac_weights_remaining': remaining_frac(masks, list(masks)) if masks else 1.0,
           'block_score': block_score(B, PAIRS),
           'B_mass': block_mass_split(B),
           'sparsity': rep, 'hist': hist}
    print(f"[{name} s{seed}] fvu {out['fvu']:.2e}  ood {out['ood_fvu']:.2e}  "
          f"remain {out['frac_weights_remaining']:.1%}  block {out['block_score']:.3f}  "
          f"cross-junk {out['B_mass']['cross_block_unprobed']:.3f}")
    return out


def save_state(name, seed, p, masks):
    torch.save({'p': {k: v.cpu() for k, v in p.items()},
                'masks': {k: v.cpu() for k, v in (masks or {}).items()}},
               f'/workspace/tensor_language/basis_aligned/e2_state_{name}_s{seed}.pt')


results = []
for seed in SEEDS:
    print(f'=== seed {seed}: dense pretrain')
    p0 = init_params(D_IN, D_MODEL, D_H, N_BLOCKS, DEV, seed)
    train(p0, data_fn, PRETRAIN_STEPS, lr=3e-3)
    print(f'  pretrained fvu {eval_fn(p0):.2e}')

    for name, sparse_keys in [('none', None),
                              ('mid_LRD', ('L', 'R', 'D')),
                              ('all_ELRDU', ('E', 'L', 'R', 'D', 'U'))]:
        p = {k: v.clone() for k, v in p0.items()}
        masks, hist = None, []
        if sparse_keys:
            print(f'--- arm {name} seed {seed}')
            p, masks, hist = iterated_sparsify(p, data_fn, eval_fn, sparse_keys, **PROTO)
        results.append(summarize(name, seed, p, masks, hist))
        save_state(name, seed, p, masks)

    # arm: rotated hand-coded init, sparsify everything
    print(f'--- arm rot_handcoded seed {seed}')
    ph = {k: v.to(DEV) for k, v in rotated(handcoded(), seed=seed + 1).items()}
    p, masks, hist = iterated_sparsify(ph, data_fn, eval_fn,
                                       ('E', 'L', 'R', 'D', 'U'), **PROTO)
    results.append(summarize('rot_handcoded', seed, p, masks, hist))
    save_state('rot_handcoded', seed, p, masks)

with open('/workspace/tensor_language/basis_aligned/e2_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print('saved e2_results.json')
