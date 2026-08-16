"""B3 across many seeds: is factor specialisation really unstable, or did three
models just happen to disagree?

B3-2 claims that between-path routing is reliably recoverable while the split of
work between the two factors inside one product head is not. The evidence was
three trained models — one clean split, one weak, one with both factors on the
same property. That is a claim ABOUT VARIABILITY resting on three samples, which
is thin, and Reviewer 2 flagged seed counts generally.

This runs both arms over eight seeds each and reports distributions rather than
anecdotes. Two things get measured per model:

  specialisation  how cleanly the two factors divide the two planted properties,
                  as |read(f1,A) - read(f1,B)| + |read(f2,A) - read(f2,B)|, scored
                  up to the factor-swap gauge (B2's null 4 showed swapping is
                  exactly function preserving, so only the unordered split means
                  anything);
  path isolation  how cleanly the MLP path owns the modifier, as the ratio of the
                  patching effect on the MLP to the larger of the two factors.

REGISTERED PREDICTIONS. Path isolation is high in every seed with small spread.
Specialisation has a wide spread spanning "both factors on one property" to "clean
split". If specialisation turns out consistent, B3-2 is wrong and gets retracted.
"""

import json
import statistics as st
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import b3_fullstack as b3

DEV = b3.DEV
torch.set_default_dtype(torch.float64)
N_SEEDS = 8


def specialisation(led):
    """Up to the factor-swap gauge: how much do the two factors disagree about
    which property they read? 0 = both read the same thing, 1 = a clean split."""
    f0, f1 = led['factor0'], led['factor1']
    a0, b0 = f0['A'], f0['B']
    a1, b1 = f1['A'], f1['B']
    # the split is clean when one factor leans A and the other leans B
    lean0 = (a0 - b0) / max(a0 + b0, 1e-12)
    lean1 = (a1 - b1) / max(a1 + b1, 1e-12)
    return {'lean_factor0': lean0, 'lean_factor1': lean1,
            'split': abs(lean0 - lean1) / 2,          # 1 = opposite leans, 0 = same
            'same_property': bool(lean0 * lean1 > 0)}


def isolation(pat):
    """How cleanly does the MLP path own the modifier C?"""
    return pat['C']['mlp'] / max(pat['C']['factor0'], pat['C']['factor1'], 1e-12)


def main():
    t0 = time.time()
    out = {'n_seeds': N_SEEDS, 'runs': []}
    for norm_bus, arm in ((True, 'rms-normed bus'), (False, 'raw bus')):
        print(f'#### {arm} ####')
        for seed in range(N_SEEDS):
            T = b3.make_tables(seed)
            gen = b3.sampler(T, seed)
            model = b3.Stack(seed, norm_bus=norm_bus).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, b3.STEPS)
            for s in range(b3.STEPS):
                b = gen(b3.BATCH)
                lv, lc = model(b['x'])
                loss = (torch.nn.functional.cross_entropy(lv, b['y'])
                        + torch.nn.functional.cross_entropy(lc, b['yc']))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
            r = b3.report(model, T, gen, f'{arm} s{seed}', verbose=False)
            sp = specialisation(r['ledger'])
            iso = isolation(r['patch'])
            rec = {'arm': arm, 'seed': seed, 'acc_retrieval': r['acc_retrieval'],
                   'acc_modifier': r['acc_modifier'], 'isolation': iso, **sp,
                   'dead_read': max(r['ledger'][p]['dead'] for p in r['ledger'])}
            out['runs'].append(rec)
            print(f"  seed {seed}: retrieval {rec['acc_retrieval']:.4f} | factor leans "
                  f"{sp['lean_factor0']:+.2f} / {sp['lean_factor1']:+.2f} -> split "
                  f"{sp['split']:.2f}"
                  f"{'  (BOTH on the same property)' if sp['same_property'] else ''}"
                  f" | MLP owns C by {iso:5.1f}x | dead {rec['dead_read']:.4f}", flush=True)

    print('\n== distributions ==')
    out['summary'] = {}
    for arm in ('rms-normed bus', 'raw bus'):
        rs = [r for r in out['runs'] if r['arm'] == arm]
        sp = [r['split'] for r in rs]
        iso = [r['isolation'] for r in rs]
        acc = [r['acc_retrieval'] for r in rs]
        same = sum(r['same_property'] for r in rs)
        out['summary'][arm] = {
            'split_mean': st.mean(sp), 'split_sd': st.pstdev(sp),
            'split_min': min(sp), 'split_max': max(sp),
            'isolation_mean': st.mean(iso), 'isolation_sd': st.pstdev(iso),
            'isolation_min': min(iso), 'retrieval_mean': st.mean(acc),
            'n_both_same_property': same, 'n': len(rs)}
        v = out['summary'][arm]
        print(f"  {arm:16s} retrieval {v['retrieval_mean']:.4f} | split "
              f"{v['split_mean']:.2f} +/- {v['split_sd']:.2f} (range {v['split_min']:.2f}"
              f"-{v['split_max']:.2f}) | both factors on one property in "
              f"{v['n_both_same_property']}/{v['n']} seeds | MLP isolation "
              f"{v['isolation_mean']:5.1f} +/- {v['isolation_sd']:4.1f} "
              f"(worst {v['isolation_min']:5.1f}x)")

    allsp = [r['split'] for r in out['runs']]
    alliso = [r['isolation'] for r in out['runs']]
    out['verdict'] = {
        'specialisation_relative_spread': st.pstdev(allsp) / max(st.mean(allsp), 1e-12),
        'isolation_relative_spread': st.pstdev(alliso) / max(st.mean(alliso), 1e-12)}
    print(f"\n  relative spread (sd/mean) across all {len(allsp)*1} models: "
          f"factor split {out['verdict']['specialisation_relative_spread']:.2f} "
          f"vs path isolation {out['verdict']['isolation_relative_spread']:.2f}")

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/b3_seeds_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
