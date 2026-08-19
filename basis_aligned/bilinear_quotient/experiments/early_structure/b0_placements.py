"""B0 — the three multiplication placements, and a census that works on all of them.

The plan's B0 lists two placements: the product taken before the softmax, and the
product of two softmaxed patterns. bilin18 uses neither. Verified at source
(`jacclust/tt_model.py:134-144`) its attention is

    pattern = (q1.k1 / D) * (q2.k2 / D),  causally masked to zero, NEVER normalised

with no softmax anywhere in the model. Entries are signed and rows do not sum to
one, so every statistic in the plan's B1 taxonomy — all of which are entropies of
attention distributions — is undefined on the model B1 was written for.

This module adds that third placement and replaces the entropy census with
scale-free, sign-tolerant statistics that are defined for all three:

  participation ratio  PR(w) = (Σ w²)² / Σ w⁴, normalised by the number of keys.
      For a flat pattern PR/n → 1; for a pattern concentrated on one key PR/n → 1/n.
      This is the statistic the earlier jacclust program used on bilin18, where it
      reported PR(product) < min(PR(factor)) at 100% of heads.
  negative mass fraction — meaningless for softmax placements, central here.

The question this run answers: **is the "sharpening" signature specific?** B2 showed
the entropy version fires on control tasks with nothing to conjoin. If the PR version
does too, then bilin18's 100%-of-heads result is not evidence of conjunction, and
that stops being an inference in BILIN18_CONNECTION.md and becomes a measurement.
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
KINDS = ('unnorm', 'logit', 'postsoft')
TASKS = ((None, 'conjunctive: needs both'),
         ('type', 'control: timing alone suffices'),
         ('prop', 'control: type alone suffices'))


@torch.no_grad()
def pr_census(model, dgp, n=4096):
    """Scale-free census, defined for signed unnormalised patterns."""
    b = dgp.batch(n)
    L = b['x'].shape[1] - 1
    A, s = model.pattern(b['x'])

    def pr(w):
        return float(((w ** 2).sum(-1) ** 2 / (w ** 4).sum(-1).clamp_min(1e-30)).mean() / L)

    def negfrac(w):
        return float((w.clamp(max=0).abs().sum(-1) / w.abs().sum(-1).clamp_min(1e-30)).mean())

    out = {'PR_combined': pr(A), 'negmass_combined': negfrac(A), 'n_keys': L}
    if len(model.Wq) == 1:
        out['regime'] = 'single circuit'
        return out
    # each factor's own raw score field, in the same (unnormalised) terms
    f = [s[0], s[1]]
    out['PR_factor'] = [pr(v) for v in f]
    out['negmass_factor'] = [negfrac(v) for v in f]
    W1, W2 = model.qk(0), model.qk(1)
    out['qk_cosine'] = float((W1 * W2).sum() / (W1.norm() * W2.norm()).clamp_min(1e-30))
    out['PR_drop'] = min(out['PR_factor']) - out['PR_combined']
    out['sharpening_fires'] = bool(out['PR_combined'] < min(out['PR_factor']))
    return out


def run_one(kind, task, seed, steps=STEPS):
    dgp = B.ConjunctiveRetrieval(device=DEV, seed=seed, control=task)
    model = B.Head(dgp.d, dgp.V, kind=kind, rank=8, seed=seed, device=DEV).to(DEV)
    B.train(model, dgp, steps=steps)
    ev = B.evaluate(model, dgp)
    cen = pr_census(model, dgp)
    out = {'kind': kind, 'task': task or 'conjunctive', 'seed': seed, **ev, 'census': cen,
           'ceiling': dgp.single_property_ceiling(),
           'factor_readout': B.factor_readout(model, dgp)}
    if kind != 'standard':
        out['ablate'] = [B.ablate_factor(model, dgp, i)['acc'] for i in (0, 1)]
    return out, model, dgp


def main():
    t0 = time.time()
    out = {'device': DEV, 'steps': STEPS, 'runs': [],
           'question': 'is the sharpening signature specific to tasks that need a '
                       'conjunction, in the placement bilin18 actually uses?'}

    for kind in KINDS:
        print(f'== {kind} ==')
        for task, label in TASKS:
            for seed in SEEDS:
                r, _, _ = run_one(kind, task, seed)
                out['runs'].append(r)
                c = r['census']
                fires = 'FIRES' if c.get('sharpening_fires') else '  -  '
                print(f"  {label:32s} s{seed} acc {r['acc']:.4f} (ceil {r['ceiling']:.2f}) | "
                      f"PR factors {[round(v,3) for v in c['PR_factor']]} -> product "
                      f"{c['PR_combined']:.3f}  [{fires}] | neg mass {c['negmass_combined']:.3f} "
                      f"| ablate {[round(v,3) for v in r['ablate']]}")

    # the specificity test, stated as one number per placement
    print('\n== specificity of the sharpening signature ==')
    spec = {}
    for kind in KINDS:
        rs = [r for r in out['runs'] if r['kind'] == kind]
        conj = [r for r in rs if r['task'] == 'conjunctive']
        ctrl = [r for r in rs if r['task'] != 'conjunctive']
        spec[kind] = {
            'fires_on_conjunctive': sum(r['census']['sharpening_fires'] for r in conj),
            'n_conjunctive': len(conj),
            'fires_on_controls': sum(r['census']['sharpening_fires'] for r in ctrl),
            'n_controls': len(ctrl),
            'mean_PR_drop_conjunctive': sum(r['census']['PR_drop'] for r in conj) / len(conj),
            'mean_PR_drop_controls': sum(r['census']['PR_drop'] for r in ctrl) / len(ctrl),
            'ablation_gap_conjunctive': sum(max(r['ablate']) - min(r['ablate']) for r in conj) / len(conj),
        }
        s = spec[kind]
        print(f"  {kind:9s}: fires {s['fires_on_conjunctive']}/{s['n_conjunctive']} on the "
              f"conjunctive task and {s['fires_on_controls']}/{s['n_controls']} on controls "
              f"that need no conjunction | mean PR drop {s['mean_PR_drop_conjunctive']:.3f} "
              f"vs {s['mean_PR_drop_controls']:.3f}")
    out['specificity'] = spec

    print('\n== NULL: random weights (does the signature fire with no training at all?) ==')
    out['null_random'] = []
    for kind in KINDS:
        dgp = B.ConjunctiveRetrieval(device=DEV, seed=0)
        model = B.Head(dgp.d, dgp.V, kind=kind, rank=8, seed=555, device=DEV).to(DEV)
        c = pr_census(model, dgp)
        r = {'kind': kind, **B.evaluate(model, dgp), 'census': c}
        print(f"  {kind:9s} acc {r['acc']:.4f} | PR factors "
              f"{[round(v,3) for v in c['PR_factor']]} -> product {c['PR_combined']:.3f} "
              f"[{'FIRES' if c['sharpening_fires'] else '  -  '}]")
        out['null_random'].append(r)

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/b0_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
