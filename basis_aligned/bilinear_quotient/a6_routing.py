"""A6 — two-hop routing table: path-relative quotients.

DGP. A first bilinear layer computes features A, B and C from its input. Two
downstream bilinear readers consume them: reader 1's target depends only on A,
reader 2's only on B, and C is read by neither (a planted dead feature). The
routing table is therefore known exactly, and the question is whether it can be
read off the composed weights.

The point of the experiment is that "what this layer throws away" is not a
property of the layer. Layer 1 transmits A, B and C; reader 1 is blind to B not
because layer 1 discarded it but because reader 1 does not look. So the kernel
has to be computed **per path**, and the prediction is that the path kernel is
strictly larger than the layer kernel — it equals what the layer discards plus
what the reader ignores.

Predictions, registered before running:
  (i)   for reader 1, the composed path is blind to the planted B directions and
        sensitive to A, and symmetrically for reader 2;
  (ii)  the path kernel is strictly larger than layer 1's own kernel, and the
        difference is exactly the transmitted-but-ignored part;
  (iii) the ledger (feature x reader -> live/dead) recovered from weights matches
        the construction;
  (iv)  a norm test on the composed form agrees with activation patching, and the
        agreement rate is what licenses using norms for discovery.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, interaction, train, lam_relerr,
                       gauge_refactor, row_space_kernel)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

D_IN = 24
D_MID = 18          # layer-1 output width: the bus both readers read from
H1, H2 = 128, 96
STEPS = 15000
BATCH = 512
NFEAT = 3           # planted features A, B, C
FEAT_DIM = 4        # each planted feature occupies this many input directions

REGISTERED = {
    'i': "reader 1's composed path is blind to B and sensitive to A; reader 2 vice versa",
    'ii': 'the path kernel strictly contains the layer kernel; the extra part is the '
          'transmitted-but-ignored feature',
    'iii': 'the feature x reader ledger read from weights matches the construction',
    'iv': 'a norm test on the composed form agrees with activation patching',
}


def make_task(seed=0):
    """Planted features live in disjoint blocks of a random orthonormal frame."""
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(D_IN, D_IN, generator=g))[0].to(DEV)
    feats = {name: U[:, i * FEAT_DIM:(i + 1) * FEAT_DIM]
             for i, name in enumerate(['A', 'B', 'C'])}
    # each feature is a quadratic function of its own block
    forms = {}
    for i, name in enumerate(['A', 'B', 'C']):
        M = torch.randn(FEAT_DIM, FEAT_DIM, generator=g).to(DEV)
        M = 0.5 * (M + M.T)
        forms[name] = feats[name] @ (M / M.norm()) @ feats[name].T
    return {'U': U, 'feats': feats, 'forms': forms}


def sampler(T, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def gen(n):
        x = torch.randn(n, D_IN, generator=g, device=DEV, dtype=torch.get_default_dtype())
        f = {k: torch.einsum('ni,ij,nj->n', x, v, x) for k, v in T['forms'].items()}
        # reader 1 sees only A, reader 2 only B; C is read by nobody
        y = torch.stack([f['A'], f['B']], 1)
        return x, y, f

    return gen


class TwoHop(torch.nn.Module):
    """Layer 1 (bilinear, D_IN -> D_MID) then two bilinear readers, each D_MID -> 1."""

    def __init__(self, seed=0):
        super().__init__()
        p1 = init_params(D_IN, H1, D_MID, seed=seed, device=DEV)
        r1 = init_params(D_MID, H2, 1, seed=seed + 1, device=DEV)
        r2 = init_params(D_MID, H2, 1, seed=seed + 2, device=DEV)
        self.p1 = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.to(torch.get_default_dtype())) for k, v in p1.items()})
        self.r1 = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.to(torch.get_default_dtype())) for k, v in r1.items()})
        self.r2 = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.to(torch.get_default_dtype())) for k, v in r2.items()})

    def mid(self, x):
        return forward({k: v for k, v in self.p1.items()}, x)

    def forward(self, x):
        h = self.mid(x)
        return torch.cat([forward({k: v for k, v in self.r1.items()}, h),
                          forward({k: v for k, v in self.r2.items()}, h)], 1)


def composed_form(model, which, x, n_probe=8192):
    """The path's end-to-end sensitivity in INPUT space.

    The composition of two bilinear layers is quartic, so it has no single
    quadratic form. What the routing question needs is the path's local
    sensitivity, so use the expected outer product of the input gradient:
        S = E_x [ (d out_which / d x)(d out_which / d x)^T ]
    Its kernel is exactly the set of input directions the path cannot be moved
    along, to first order, anywhere on the data.
    """
    xs = x[:n_probe].detach().requires_grad_(True)
    out = model(xs)[:, which].sum()
    g, = torch.autograd.grad(out, xs)
    return (g.T @ g) / xs.shape[0]


def block_energy(S, T):
    """Share of a sensitivity matrix sitting in each planted feature block."""
    tot = float(torch.diagonal(S).sum()) or 1.0
    return {k: float(torch.einsum('ia,ij,ja->', v, S, v)) / tot for k, v in T['feats'].items()}


@torch.no_grad()
def patch_test(model, T, gen, which, feature, n=4096, eps=1.0):
    """Activation patching, the ground truth for the ledger: resample the input
    inside one planted feature's block and measure the output change."""
    x, y, _ = gen(n)
    x2, _, _ = gen(n)
    P = T['feats'][feature]
    xp = x + (x2 - x) @ P @ P.T
    o0, o1 = model(x)[:, which], model(xp)[:, which]
    return float((o1 - o0).pow(2).mean().sqrt() / o0.std().clamp_min(1e-30))


def main():
    t0 = time.time()
    out = {'registered': REGISTERED, 'config': {'d_in': D_IN, 'd_mid': D_MID,
           'h1': H1, 'h2': H2, 'steps': STEPS, 'feat_dim': FEAT_DIM}, 'runs': []}

    for seed in range(2):
        T = make_task(seed)
        gen = sampler(T, seed)
        model = TwoHop(seed).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        for s in range(STEPS):
            x, y, _ = gen(BATCH)
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        xe, ye, fe = gen(16384)
        with torch.no_grad():
            fvu = float(((model(xe) - ye) ** 2).mean() / ye.var())

        # ---- (i)+(iii) the ledger, from weights alone
        ledger, patch = {}, {}
        for w in (0, 1):
            S = composed_form(model, w, xe)
            ledger[w] = block_energy(S, T)
            patch[w] = {k: patch_test(model, T, gen, w, k) for k in T['feats']}

        # ---- (ii) layer kernel vs path kernel
        Q1 = interaction({k: v.detach() for k, v in model.p1.items()})
        rows1, ker1, sv1 = row_space_kernel(Q1, 1e-6)
        # which input directions does layer 1 itself transmit?
        with torch.no_grad():
            xs = xe[:8192].detach().requires_grad_(True)
        xs = xe[:8192].detach().requires_grad_(True)
        hm = model.mid(xs)
        Slayer = torch.zeros(D_IN, D_IN, device=DEV, dtype=xe.dtype)
        for j in range(D_MID):
            g, = torch.autograd.grad(hm[:, j].sum(), xs, retain_graph=True)
            Slayer += (g.T @ g) / xs.shape[0]
        lay = block_energy(Slayer.detach(), T)

        def eff_dim(S, thresh=1e-3):
            ev = torch.linalg.eigvalsh(S).flip(0).clamp_min(0)
            return int((ev > thresh * ev.max()).sum())

        S0, S1 = composed_form(model, 0, xe), composed_form(model, 1, xe)
        rec = {'seed': seed, 'fvu': fvu,
               'layer_block_energy': lay, 'layer_sensitivity_dim': eff_dim(Slayer.detach()),
               'reader0_block_energy': ledger[0], 'reader1_block_energy': ledger[1],
               'reader0_sensitivity_dim': eff_dim(S0.detach()),
               'reader1_sensitivity_dim': eff_dim(S1.detach()),
               'patch_reader0': patch[0], 'patch_reader1': patch[1],
               'layer_lift_rank': int(rows1.shape[0]), 'layer_lift_kernel': int(ker1.shape[0])}
        out['runs'].append(rec)

        print(f'== seed {seed}: fvu {fvu:.2e} ==')
        print(f"  layer 1 transmits (share of its input sensitivity per planted feature): "
              + ' '.join(f'{k} {v:.3f}' for k, v in lay.items())
              + f"  [sensitivity dim {rec['layer_sensitivity_dim']} of {D_IN}]")
        for w in (0, 1):
            print(f"  reader {w} reads: " + ' '.join(f'{k} {v:.4f}' for k, v in ledger[w].items())
                  + f"  [dim {rec[f'reader{w}_sensitivity_dim']}]")
            print(f"    activation patching: "
                  + ' '.join(f'{k} {v:.4f}' for k, v in patch[w].items()))

        # ---- (iv) do the weight-side norms agree with patching?
        pairs = [(ledger[w][k], patch[w][k], w, k) for w in (0, 1) for k in T['feats']]
        agree = sum(1 for e, p, _, _ in pairs if (e > 0.05) == (p > 0.05))
        rec['norm_vs_patch_agreement'] = f'{agree}/{len(pairs)}'
        rec['pairs'] = [{'reader': w, 'feature': k, 'weight_energy': e, 'patch': p}
                        for e, p, w, k in pairs]
        print(f"  norm test vs patching: {agree}/{len(pairs)} agree at a 5% threshold")

        # ---- NULL 2: gauge scramble of layer 1
        if seed == 0:
            pg, resid = gauge_refactor({k: v.detach() for k, v in model.p1.items()}, seed=77)
            import copy
            m2 = copy.deepcopy(model)
            with torch.no_grad():
                m2.p1 = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in pg.items()})
            with torch.no_grad():
                dfn = float((model(xe) - m2(xe)).abs().max())
            l2 = block_energy(composed_form(m2, 0, xe), T)
            out['null_gauge'] = {'refactor_residual': resid, 'max_function_diff': dfn,
                                 'h_before': H1, 'h_after': int(pg['L'].shape[0]),
                                 'reader0_before': ledger[0], 'reader0_after': l2}
            print(f"  NULL gauge: layer-1 refactor residual {resid:.1e}, hidden {H1} -> "
                  f"{pg['L'].shape[0]}, end-to-end function diff {dfn:.1e}; reader-0 ledger "
                  + ' '.join(f'{k} {ledger[0][k]:.3f}->{l2[k]:.3f}' for k in l2))

    # ---- NULL 1: random weights
    print('== NULL 1: random weights ==')
    T = make_task(0)
    gen = sampler(T, 0)
    xe, _, _ = gen(8192)
    mr = TwoHop(999).to(DEV)
    lr0 = block_energy(composed_form(mr, 0, xe), T)
    lr1 = block_energy(composed_form(mr, 1, xe), T)
    out['null_random'] = {'reader0': lr0, 'reader1': lr1}
    print('  reader 0: ' + ' '.join(f'{k} {v:.4f}' for k, v in lr0.items()))
    print('  reader 1: ' + ' '.join(f'{k} {v:.4f}' for k, v in lr1.items()))

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a6_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
