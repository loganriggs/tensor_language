"""B2 — conjunctive retrieval, plus the B1 degeneracy census it feeds.

DGP. Every token carries two independent planted properties in disjoint embedding
subspaces: a TYPE (one of 8) and a TIMING value (one of 8). The query names one of
each. Exactly one key matches both; three keys match the type only and three the
timing only. So any strategy that uses a single property spreads its attention
over four keys and reads the right payload 25% of the time — the ceiling a
single-property head cannot beat.

Arms. Three heads at matched parameter count:
  logit     s = (qᵀW₁k)(qᵀW₂k) then softmax        — product of matching conditions
  postsoft  A = normalise(softmax(qᵀW₁k) ⊙ softmax(qᵀW₂k)) — intersection of two patterns
  standard  s = qᵀWk, one circuit at twice the rank

Tasks. The conjunctive task above, plus two controls where one property alone is
sufficient (no distractors of the other kind), which is where the B1 census should
show degeneracy rather than conjunction.

PREDICTIONS, registered before running (see REGISTERED below).
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import b_common as B

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STEPS = 4000
SEEDS = (0, 1)

REGISTERED = {
    'P1': 'both multiplicative heads beat the 0.25 single-property ceiling on the '
          'conjunctive task; accuracy near 1.',
    'P2': 'each factor QK form concentrates on ONE planted property block, one factor '
          'per property, up to swapping the two factors.',
    'P3': 'ablating either factor drops accuracy to about the single-property ceiling.',
    'P4': 'the B1 census reports (c) genuine conjunction for multiplicative heads on the '
          'conjunctive task, and (a) collapse or (b) alignment on the control tasks.',
    'P5': 'the standard single-circuit head at matched capacity does measurably worse on '
          'the conjunctive task than the multiplicative heads.',
    'P6': 'changing the query type moves attention onto the key carrying the new type.',
}


def make_dgp(seed, control=None, **kw):
    return B.ConjunctiveRetrieval(device=DEV, seed=seed, control=control, **kw)


def gauge_scramble(model, seed=0):
    """NULL 2 for an attention head. Each factor's function depends only on
    W_i = Wq_iᵀWk_i, so Wq → M Wq, Wk → M^{-T} Wk is exactly function preserving
    while every raw weight changes."""
    g = torch.Generator().manual_seed(seed)
    import copy
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for i in range(len(m2.Wq)):
            r = m2.Wq[i].shape[0]
            M = torch.randn(r, r, generator=g).to(m2.Wq[i])
            M = M + torch.eye(r, device=M.device) * 0.5 * r ** 0.5
            m2.Wq[i].copy_(M @ model.Wq[i])
            m2.Wk[i].copy_(torch.linalg.inv(M).T @ model.Wk[i])
    return m2


def run_one(kind, task, seed, steps=STEPS, dgp_kw=None, verbose=True):
    dgp = make_dgp(seed, control=task, **(dgp_kw or {}))
    model = B.Head(dgp.d, dgp.V, kind=kind, rank=8, seed=seed, device=DEV).to(DEV)
    hist = B.train(model, dgp, steps=steps, eval_every=max(1, steps // 20))
    ev = B.evaluate(model, dgp)
    cen = B.census(model, dgp)
    rd = B.factor_readout(model, dgp)
    out = {'kind': kind, 'task': task or 'conjunctive', 'seed': seed,
           'n_params': sum(p.numel() for p in model.parameters()),
           'single_property_ceiling': dgp.single_property_ceiling(),
           **ev, 'census': cen, 'factor_readout': rd,
           'history': hist}
    if kind != 'standard':
        out['ablate_factor0'] = B.ablate_factor(model, dgp, 0)
        out['ablate_factor1'] = B.ablate_factor(model, dgp, 1)
    out['swap_type'] = B.causal_query_swap(model, dgp, 'type')
    out['swap_prop'] = B.causal_query_swap(model, dgp, 'prop')
    if verbose:
        blocks = ' | '.join(f"f{i}: {r['best']} {r['best_frac']:.2f}" for i, r in enumerate(rd))
        print(f"  [{kind:9s} {out['task']:12s} s{seed}] acc {ev['acc']:.4f} "
              f"(ceiling {out['single_property_ceiling']:.2f}) attn→match "
              f"{ev['attn_to_match']:.3f} | {cen['regime']} | {blocks}")
        if kind != 'standard':
            print(f"      ablate f0 -> acc {out['ablate_factor0']['acc']:.4f} | "
                  f"ablate f1 -> acc {out['ablate_factor1']['acc']:.4f} | "
                  f"H(factors) {[round(v,3) for v in cen['H_factor_norm']]} "
                  f"H(product) {cen['H_combined_norm']:.3f} qk-cos {cen['qk_cosine']:+.3f}")
    return out, model, dgp


def main():
    t0 = time.time()
    out = {'registered_predictions': REGISTERED, 'device': DEV, 'steps': STEPS,
           'runs': []}

    print('== main grid: 3 head kinds x 3 tasks x 2 seeds ==')
    for task in (None, 'type', 'prop'):
        for kind in ('logit', 'postsoft', 'standard'):
            for seed in SEEDS:
                r, _, _ = run_one(kind, task, seed)
                out['runs'].append(r)

    print('\n== NULL 1: random weights ==')
    out['null_random'] = []
    for kind in ('logit', 'postsoft'):
        dgp = make_dgp(0)
        model = B.Head(dgp.d, dgp.V, kind=kind, rank=8, seed=99, device=DEV).to(DEV)
        r = {'kind': kind, **B.evaluate(model, dgp), 'census': B.census(model, dgp),
             'factor_readout': B.factor_readout(model, dgp)}
        print(f"  [{kind}] acc {r['acc']:.4f} | {r['census']['regime']} | "
              f"f0 {r['factor_readout'][0]['best']} {r['factor_readout'][0]['best_frac']:.2f}")
        out['null_random'].append(r)

    print('\n== NULL 2: gauge scramble (function-preserving reparameterisation) ==')
    out['null_gauge'] = []
    for kind in ('logit', 'postsoft'):
        r0, model, dgp = run_one(kind, None, 0, verbose=False)
        mg = gauge_scramble(model, seed=7)
        b = dgp.batch(2048)
        with torch.no_grad():
            dfn = float((model(b['x']) - mg(b['x'])).abs().max())
            wq_change = float((model.Wq[0] - mg.Wq[0]).norm() / model.Wq[0].norm())
        eg, cg, rg = B.evaluate(mg, dgp), B.census(mg, dgp), B.factor_readout(mg, dgp)
        rec = {'kind': kind, 'max_function_diff': dfn, 'raw_weight_rel_change': wq_change,
               'acc_before': r0['acc'], 'acc_after': eg['acc'],
               'regime_before': r0['census']['regime'], 'regime_after': cg['regime'],
               'readout_before': [x['best_frac'] for x in r0['factor_readout']],
               'readout_after': [x['best_frac'] for x in rg]}
        print(f"  [{kind}] function unchanged to {dfn:.1e} while raw Wq moved "
              f"{wq_change:.2f} relative | acc {r0['acc']:.4f} -> {eg['acc']:.4f} | "
              f"regime {r0['census']['regime']} -> {cg['regime']}")
        out['null_gauge'].append(rec)

    print('\n== NULL 3: task shuffle (payload labels permuted within the batch) ==')
    out['null_taskshuffle'] = []
    for kind in ('logit', 'postsoft'):
        dgp = make_dgp(0)
        model = B.Head(dgp.d, dgp.V, kind=kind, rank=8, seed=0, device=DEV).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        for s in range(STEPS):
            b = dgp.batch(256)
            y = b['y'][torch.randperm(b['y'].shape[0], device=DEV)]
            loss = torch.nn.functional.cross_entropy(model(b['x']), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        r = {'kind': kind, **B.evaluate(model, dgp), 'census': B.census(model, dgp),
             'factor_readout': B.factor_readout(model, dgp)}
        print(f"  [{kind}] acc {r['acc']:.4f} | {r['census']['regime']} | "
              f"f0 {r['factor_readout'][0]['best']} {r['factor_readout'][0]['best_frac']:.2f}")
        out['null_taskshuffle'].append(r)

    print('\n== NULL 4: factor shuffle (swap which factor is which) ==')
    # the analogue of the reader shuffle: the head's FUNCTION is symmetric under
    # swapping the two factors, so any claim that survives the swap is a claim
    # about the unordered pair, not about "factor 1".
    out['null_factorswap'] = []
    for kind in ('logit', 'postsoft'):
        r0, model, dgp = run_one(kind, None, 0, verbose=False)
        import copy
        ms = copy.deepcopy(model)
        with torch.no_grad():
            ms.Wq[0].copy_(model.Wq[1]); ms.Wq[1].copy_(model.Wq[0])
            ms.Wk[0].copy_(model.Wk[1]); ms.Wk[1].copy_(model.Wk[0])
        b = dgp.batch(2048)
        with torch.no_grad():
            dfn = float((model(b['x']) - ms(b['x'])).abs().max())
        rec = {'kind': kind, 'max_function_diff': dfn,
               'readout_before': [x['best'] for x in r0['factor_readout']],
               'readout_after': [x['best'] for x in B.factor_readout(ms, dgp)]}
        print(f"  [{kind}] function diff {dfn:.1e} (0 => factor identity is a gauge, so "
              f"'factor 1 reads type' is only meaningful as an unordered pairing) | "
              f"{rec['readout_before']} -> {rec['readout_after']}")
        out['null_factorswap'].append(rec)

    print('\n== knob: distractor count (harder conjunctions) ==')
    out['knob_m'] = []
    for m in (1, 3, 5, 7):
        r, _, _ = run_one('logit', None, 0, dgp_kw={'m': m})
        r['m'] = m
        out['knob_m'].append(r)

    print('\n== knob: correlation between the two properties ==')
    out['knob_corr'] = []
    for corr in (0.0, 0.3, 0.6, 0.9):
        r, _, _ = run_one('logit', None, 0, dgp_kw={'prop_corr': corr})
        r['prop_corr'] = corr
        out['knob_corr'].append(r)

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/b2_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
