"""Repair of the factor ablation that Reviewer 2 broke (REVIEW_RESPONSE R3).

The retracted version replaced a factor by its per-example mean over keys. That is
not a scale: it is negative on 56% of examples, and a negative multiplier inverts
the attention ordering. The tell was that it dropped the head to 0.27 even on
control tasks whose single-property ceiling is 1.000.

Three replacements are implemented and compared, because "ablate a factor" has no
canonical meaning and the choice is doing real work:

  rms      replace the factor's scores by a POSITIVE constant with the same RMS,
           so the surviving factor keeps its scale and its ordering.
  keyshuf  keep the factor's own scores but shuffle them across keys, which
           destroys the key-dependence while preserving the score distribution
           exactly (including its sign structure).
  mean     the broken original, kept so the comparison is visible.

The check that decides whether an ablation is sound is not on the conjunctive task
at all: on a CONTROL task where one property suffices, ablating the factor that
reads the OTHER property must leave accuracy near 1.0. Any ablation that fails
that is measuring itself.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import b_common as B

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STEPS = 4000
MODES = ('rms', 'keyshuf', 'mean')


@torch.no_grad()
def ablate(model, dgp, which, mode, n=4096, seed=0):
    b = dgp.batch(n)
    s = model.scores(b['x'])
    keep, kill = 1 - which, which
    sk = s[kill]
    if mode == 'rms':
        const = sk.pow(2).mean(-1, keepdim=True).sqrt()          # positive, matched RMS
    elif mode == 'mean':
        const = sk.mean(-1, keepdim=True)                        # the broken original
    elif mode == 'keyshuf':
        g = torch.Generator(device=sk.device).manual_seed(seed)
        idx = torch.rand(sk.shape, generator=g, device=sk.device).argsort(-1)
        const = torch.gather(sk, -1, idx)
    else:
        raise ValueError(mode)

    if model.kind == 'logit':
        A = torch.softmax(s[keep] * const, -1)
    elif model.kind == 'unnorm':
        A = s[keep] * const
    else:                                                        # postsoft
        cs = torch.softmax(const, -1) if mode == 'keyshuf' else torch.full_like(s[keep], 1.0)
        a = torch.softmax(s[keep], -1) * (cs / cs.sum(-1, keepdim=True).clamp_min(1e-30)
                                          if mode == 'keyshuf' else 1.0)
        A = a / a.sum(-1, keepdim=True).clamp_min(1e-30)
    vals = b['x'][:, :-1, :] @ model.Wv.T
    out = (A.unsqueeze(-1) * vals).sum(1) @ model.Wo.T
    return float((out.argmax(1) == b['y']).float().mean())


def main():
    t0 = time.time()
    out = {'modes': list(MODES), 'runs': [],
           'soundness_test': 'on a control task where one property suffices, ablating the '
                             'factor reading the other property must leave accuracy near 1.0'}
    for kind in ('logit', 'postsoft', 'unnorm'):
        print(f'== {kind} ==')
        for task, label, ceil in ((None, 'conjunctive', 0.25),
                                  ('type', 'control (timing suffices)', 1.0),
                                  ('prop', 'control (type suffices)', 1.0)):
            for seed in (0, 1):
                dgp = B.ConjunctiveRetrieval(device=DEV, seed=seed, control=task)
                model = B.Head(dgp.d, dgp.V, kind=kind, rank=8, seed=seed, device=DEV).to(DEV)
                B.train(model, dgp, steps=STEPS)
                base = B.evaluate(model, dgp)['acc']
                rec = {'kind': kind, 'task': task or 'conjunctive', 'seed': seed,
                       'ceiling': ceil, 'acc': base,
                       'readout': [r['best'] for r in B.factor_readout(model, dgp)]}
                for mode in MODES:
                    rec[mode] = [ablate(model, dgp, i, mode) for i in (0, 1)]
                out['runs'].append(rec)
                print(f"  {label:28s} s{seed} acc {base:.4f} (ceil {ceil:.2f}) | "
                      + ' | '.join(f"{m} {rec[m][0]:.3f}/{rec[m][1]:.3f}" for m in MODES))

    print('\n== soundness: on a control task, the best ablation should leave >= ceiling ==')
    verdict = {}
    for mode in MODES:
        ctrl = [r for r in out['runs'] if r['task'] != 'conjunctive']
        best_on_ctrl = [max(r[mode]) for r in ctrl]
        conj = [r for r in out['runs'] if r['task'] == 'conjunctive']
        verdict[mode] = {
            'min_best_acc_on_controls': min(best_on_ctrl),
            'mean_best_acc_on_controls': sum(best_on_ctrl) / len(best_on_ctrl),
            'mean_min_acc_on_conjunctive': sum(min(r[mode]) for r in conj) / len(conj),
            'sound': min(best_on_ctrl) > 0.8}
        v = verdict[mode]
        print(f"  {mode:8s}: on controls the better factor retains "
              f"{v['mean_best_acc_on_controls']:.3f} (worst {v['min_best_acc_on_controls']:.3f}) "
              f"-> {'SOUND' if v['sound'] else 'BROKEN'} | on the conjunctive task the worse "
              f"factor falls to {v['mean_min_acc_on_conjunctive']:.3f}")
    out['verdict'] = verdict

    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/fix_ablation_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
