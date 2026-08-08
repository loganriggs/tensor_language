"""The CONSOLIDATED COMPARISON TABLE for the six-architecture slice.

Renders `tf_variant_compare.json` as markdown, per architecture AND per seed,
covering exactly the six things the slice was built to compare:

  held CE and bits/byte | the causal routing measurement as a [zero, resample]
  RANGE | induction with the probe's own power floor beside it | the rung-5
  reconstruction ladder | selection-vs-content effective rank against its null
  | nominal and effective parameter counts.

Every number comes from a `*_interp3.json` produced by ONE revision of
`tf_interp3.py` (the compare file drops any row that did not).

Usage:  python tf_consolidated_table.py   ->  tf_consolidated_table.md
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ['vanilla', 'slots', 'bandwidth', 'predicate', 'codebook', 'shrink']
LADDER = ['embed_only', 'plus_self_attn', 'model_bigram', 'no_attention_at_all',
          'past_attn_mean_ablated', 'no_mlp', 'no_attn_layer0',
          'no_attn_layer1', 'no_mlp_layer0', 'no_mlp_layer1',
          'l1_reads_embedding', 'l1_reads_e_plus_attn0', 'l1_reads_e_plus_mlp0',
          'trunc_delta1_only', 'trunc_delta_le4', 'positional_only_pattern',
          'no_rotary_pattern']


def ms(vals, fmt='{:.4f}'):
    """mean +- sd, or the single value when n == 1."""
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return '-'
    if a.size == 1:
        return fmt.format(a[0])
    return fmt.format(a.mean()) + ' ± ' + fmt.format(a.std(ddof=1))


def main():
    o = json.load(open(f'{HERE}/tf_variant_compare.json'))
    R = o['cells']
    assert not o['dropped_because_produced_by_an_older_analysis_revision'], \
        'a row came from an older analysis revision'
    # the six PRIMARY arms (no suffix), grouped by variant, sorted by seed
    prim = {}
    for k, v in R.items():
        if k.count('_') == 2 and k.split('_')[0] == v['variant']:   # var_d2_sN
            prim.setdefault(v['variant'], []).append((k, v))
    for v in prim:
        prim[v].sort(key=lambda t: t[0])
    vs = [v for v in ORDER if v in prim]
    L = []
    A = L.append

    A('#### Table A — per architecture, per seed (the six primary arms)\n')
    A('| architecture | seed | held CE (T512) | bits/byte | induction '
      '(± probe floor 3 SE) | A0→layer-1 read deleted, KL [zero, resample] | '
      'content/null | selection/null |')
    A('|---|---|---|---|---|---|---|---|')
    for var in vs:
        for k, r in prim[var]:
            z, rs = r['A2A_kl_range_zero_to_resample']
            zf = ('%.2e' % z) if z < 1e-3 else ('%.3f' % z)
            rf = ('%.2e' % rs) if rs < 1e-3 else ('%.3f' % rs)
            A(f"| {var} | {k.rsplit('_s', 1)[1]} | "
              f"{r['held_ce_T512_1500seq']:.4f} | {r['bits_per_byte']:.4f} | "
              f"{r['induction_mean']:+.4f} ± {r['induction_floor_3se']:.4f} "
              f"({r['induction_mean']/r['induction_floor_3se']:+.1f}×) | "
              f"[{zf}, {rf}] | {r['content_over_null']:.3f} | "
              f"{r['selection_over_null']:.3f} |")
    A('')
    A('#### Table B — the same, aggregated over the three seeds (mean ± sd)\n')
    A('| architecture | nominal params (body / embed) | effective params | '
      'stream width | held CE | bits/byte | induction | routing KL zero | '
      'routing KL resample | content/null | selection/null |')
    A('|---|---|---|---|---|---|---|---|---|---|---|')
    for var in vs:
        rs_ = [r for _, r in prim[var]]
        r0 = rs_[0]
        eff = ('%d' % r0['params_effective_total']) + (
            ' *(+%d buffers)*' % (r0['params_effective_total']
                                  - r0['params_total'])
            if r0['params_effective_total'] != r0['params_total'] else '')
        A(f"| **{var}** (n={len(rs_)}) | {r0['params_total']:,} "
          f"({r0['params_body']:,} / {r0['params_embedding']:,}) | {eff} | "
          f"{r0['stream_width']} | "
          f"{ms([r['held_ce_T512_1500seq'] for r in rs_])} | "
          f"{ms([r['bits_per_byte'] for r in rs_])} | "
          f"{ms([r['induction_mean'] for r in rs_], '{:+.4f}')} | "
          f"{ms([r['A2A_kl_range_zero_to_resample'][0] for r in rs_], '{:.2e}')} | "
          f"{ms([r['A2A_kl_range_zero_to_resample'][1] for r in rs_], '{:.2e}')} | "
          f"{ms([r['content_over_null'] for r in rs_], '{:.3f}')} | "
          f"{ms([r['selection_over_null'] for r in rs_], '{:.3f}')} |")
    A('')
    A('#### Table C — the rung-5 reconstruction ladder, KL from the model '
      '(nats), mean ± sd over the three seeds\n')
    A('| ladder stage | ' + ' | '.join(vs) + ' |')
    A('|---' * (len(vs) + 1) + '|')
    for s in LADDER:
        cells = []
        for var in vs:
            cells.append(ms([r['ladder_kl'].get(s) for _, r in prim[var]],
                            '{:.3f}'))
        A(f'| `{s}` | ' + ' | '.join(cells) + ' |')
    A('')
    A('#### Table D — effective rank, selection vs content, against the '
      'same-shape random null\n')
    A('| architecture | content entropy rank | its random-factored null | '
      'ratio | selection entropy rank | its random-table null | ratio |')
    A('|---|---|---|---|---|---|---|')
    for var in vs:
        rs_ = [r for _, r in prim[var]]
        A(f"| {var} | {ms([r['content_mlp_entropy_rank'] for r in rs_], '{:.2f}')} | "
          f"{ms([r['content_random_factored_null'] for r in rs_], '{:.2f}')} | "
          f"{ms([r['content_over_null'] for r in rs_], '{:.3f}')} | "
          f"{ms([r['selection_branch_entropy_rank'] for r in rs_], '{:.2f}')} | "
          f"{ms([r['selection_random_null'] for r in rs_], '{:.2f}')} | "
          f"{ms([r['selection_over_null'] for r in rs_], '{:.3f}')} |")
    A('')
    A('#### Table E — control and robustness arms\n')
    A('| arm | what it controls | held CE | induction (± probe floor 3 SE) | '
      'routing KL [zero, resample] | live slots / read |')
    A('|---|---|---|---|---|---|')
    ctrl = [k for k in sorted(R) if k not in
            {kk for var in prim for kk, _ in prim[var]}]
    WHY = {
        'lr0.01': 'learning-rate falsifier (Muon 0.01)',
        'lr0.04': 'learning-rate falsifier (Muon 0.04)',
        'writeinit_only': 'the nonzero decoder init ALONE (n_slots 1, no lasso)',
        'nolasso': 'partition + per-slot norm without the group lasso',
        'nslots2': 'partition DOSE-RESPONSE: 2 slots between 1 and 4',
        'slot32': 'embedding pinned to vanilla (stream 128, not 160)',
        'gc3e-4': 'group-lasso coefficient x10',
        'gc3e-3': 'group-lasso coefficient x100',
        'gc3e-2': 'group-lasso coefficient x1000',
    }
    for k in ctrl:
        r = R[k]
        z, rr = r['A2A_kl_range_zero_to_resample']
        zf = ('%.2e' % z) if z < 1e-3 else ('%.3f' % z)
        rf = ('%.2e' % rr) if rr < 1e-3 else ('%.3f' % rr)
        why = next((w for t, w in WHY.items() if t in k), '')
        ls = r.get('mean_live_slots_per_read')
        A(f"| `{k}` | {why} | {r['held_ce_T512_1500seq']:.4f} | "
          f"{r['induction_mean']:+.4f} ± {r['induction_floor_3se']:.4f} | "
          f"[{zf}, {rf}] | {'-' if ls is None else f'{ls:.2f}'} |")
    txt = '\n'.join(L) + '\n'
    open(f'{HERE}/tf_consolidated_table.md', 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
