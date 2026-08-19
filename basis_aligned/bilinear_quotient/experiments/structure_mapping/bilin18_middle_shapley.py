"""The middle of the network, attributed fairly: layer-level Shapley over layers 2-15.

§10 found the defining fact about layers 2-15: individually nearly free to delete
(0.024-0.52 nats), jointly catastrophic (5.14 nats, 2.87x the sum). So the solo numbers
say nothing about which middle layers matter, for the same reason §12's solo directions
said nothing -- and the repair is the same as §13's: Shapley values over the fourteen
quadratic parts. Marginal contributions averaged over removal orders, summing exactly
to the joint 5.14 nats, no budget choice.

What this buys beyond a ranking:
  - the participation ratio over LAYERS: is the middle's collective computation spread
    evenly (a true bus pipeline) or carried by a few layers that solo ablation
    misranked?
  - the interference amplification per layer, phi_l / solo_l: which layers' importance
    was most hidden by redundancy. §13 found single directions amplified up to 4.6x;
    the depth version calibrates how misleading one-at-a-time layer ablation is.
  - a check on §10's reading that the two compressible layers (0, 16) are exactly the
    high-individual-cost ones: if some middle layer has large phi but small solo cost,
    it is a compression target §9's table hid.

Mechanics: deleting a layer = replacing its MLP output with the layer's EXACT
intact-model mean (operator C). The first run of this script used §10's Truncated
machinery (operator B: mean minus its top-1-PC component), and cross-checking its gate
against §10.2 exposed that §10.2's 2.87x superadditivity was inflated by two defects:
the numerator and denominator used DIFFERENT deletion operators (B vs §9's
top-R-span-removed A), and operator B's means were computed on a progressively ablated
model (stale means). Under the clean operator C on intact means: joint 2.963, solo sum
2.087, superadditivity 1.42x. The Shapley here is run entirely under C so every number
shares one operator. 14 players x 20 permutations = 280 model evaluations.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, eval_ce
from bilin18_layer17 import Truncated
from bilin18_depth_profile import out_pcs_full

DEV = 'cuda'
MID = tuple(range(2, 16))
N_PERM = 20
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_middle_shapley_C_results.json')


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    d = cfg['n_embd']
    base = eval_ce(model, tokens, batch=4)
    print(f'base CE {base:.4f}')

    # operator C: exact intact-model mean write per layer
    store = {li: [] for li in MID}
    hooks = []

    def mk(li):
        def h(mod, i_, o):
            store[li].append(o.detach().reshape(-1, o.shape[-1]).float())
        return h

    for li in MID:
        hooks.append(model.transformer.h[li].mlp.register_forward_hook(mk(li)))
    for i in range(0, 32, 4):
        b = tokens[i:i + 4].to(DEV)
        model(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    for h in hooks:
        h.remove()
    means = {li: torch.cat(store[li]).mean(0) for li in MID}
    orig = {li: model.transformer.h[li].mlp.forward for li in MID}

    def mk_const(mu):
        def f(x):
            return mu.to(x.dtype).expand(x.shape[:-1] + mu.shape)
        return f

    repl = {li: mk_const(means[li]) for li in MID}
    print('operator-C replacements built for layers 2-15')

    def value(layers):
        if not layers:
            return 0.0
        for li in layers:
            model.transformer.h[li].mlp.forward = repl[li]
        try:
            return eval_ce(model, tokens, batch=4) - base
        finally:
            for li in layers:
                model.transformer.h[li].mlp.forward = orig[li]

    v_all = value(list(MID))
    solo = {li: value([li]) for li in MID}
    print(f'joint deletion of all 14: +{v_all:.4f} | sum of solos '
          f'+{sum(solo.values()):.4f} (gate vs §10: 5.142 / 1.790)')

    g = torch.Generator().manual_seed(0)
    n = len(MID)
    phi = torch.zeros(n, N_PERM, dtype=torch.float64)
    for p in range(N_PERM):
        perm = torch.randperm(n, generator=g).tolist()
        prev = 0.0
        cur = []
        for pos, i in enumerate(perm):
            cur.append(MID[i])
            v = v_all if pos == n - 1 else value(cur)
            phi[i, p] = v - prev
            prev = v
        if (p + 1) % 4 == 0:
            print(f'  permutation {p+1}/{N_PERM}', flush=True)

    est = phi.mean(1)
    se = phi.std(1) / N_PERM ** 0.5
    tot = float(est.sum())
    pr = float(est.sum() ** 2 / (est ** 2).sum())
    out = {'base_ce': base, 'v_all': v_all, 'solo': solo,
           'phi': {li: float(est[i]) for i, li in enumerate(MID)},
           'se': {li: float(se[i]) for i, li in enumerate(MID)},
           'participation_ratio': pr}

    print(f'\n  {"layer":>5} {"solo":>8} {"Shapley":>9} {"amplification":>14} '
          f'{"share":>7}')
    amps = {}
    for i, li in enumerate(MID):
        amp = float(est[i]) / max(solo[li], 1e-6)
        amps[li] = amp
        print(f'  {li:>5} {solo[li]:>+8.4f} {float(est[i]):>+9.4f} '
              f'{amp:>13.1f}x {100*float(est[i])/tot:>6.1f}%')
    out['amplification'] = amps
    print(f'\n  participation ratio over the 14 layers: {pr:.1f} '
          f'(14 = perfectly even, 1 = one layer)')
    top = est.argsort(descending=True)[:4]
    print(f'  top-4 by fair share: layers '
          f'{[MID[int(i)] for i in top]} carrying '
          f'{100*float(est[top].sum())/tot:.0f}%')
    hidden = max(amps, key=amps.get)
    out['most_hidden'] = hidden
    print(f'  most hidden by solo ablation: layer {hidden} '
          f'(amplification {amps[hidden]:.0f}x)')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
