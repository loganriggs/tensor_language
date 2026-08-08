"""THE DEPTH LADDER: pull the quantities that made depths 1-2 interesting out
of every vanilla `_interp3.json` and score them against the predictions
registered in `tf_depth_ladder_predictions.json` BEFORE the first depth-3
training step.

Reported per cell, three seeds each:
  * held CE (training protocol, T=512) and bits/byte, plus the rung-5 ladder CE
  * the induction score with its PLANTED-ORACLE POWER FLOOR (a null is only a
    null down to the floor), and the natural-text swap probe
  * the composition budget measured CAUSALLY -- each upstream write deleted from
    a layer's Q/K/V read only, residual untouched, everything downstream
    recomputed -- in BOTH the zeroing and the resampling flavour.  The norm-share
    version is withdrawn and is not reported.
  * the attention/MLP interaction with its LADDER-ORDER dependence.
"""
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTH_COLOR = {1: '#86b6ef', 2: '#3987e5', 3: '#256abf', 4: '#104281'}
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8c8b85'


def load():
    cells = {}
    for f in sorted(glob.glob(f'{HERE}/tf_vanilla_d*_w*_b8192_s*_interp3.json')):
        m = re.search(r'_d(\d+)_w(\d+)_b8192_s(\d+)_interp3\.json$', f)
        if not m:
            continue                      # lr / control arms
        d = json.load(open(f))
        if 'rung3_induction' not in d or 'read_ablation_causal' not in d:
            continue
        dep, w, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rk = d['read_ablation_causal']['kl_from_model']
        lo = d['ladder_order']
        cells.setdefault((dep, w), []).append({
            'seed': s, 'stem': d['stem'],
            'params': d['params']['total'],
            'held_ce': d['train']['final_held_ce'],
            'bits_per_byte': d['train']['bits_per_byte'],
            'ladder_ce': d['rung5_ladder']['_model_ce'],
            'fold_gate': d['fold_gate']['pass'],
            'pipeline': d['decomposition_control']['pass'],
            'induction': d['rung3_induction']['induction_score_mean'],
            'induction_sd': d['rung3_induction']['induction_score_sd'],
            'induction_floor':
                d['induction_power']['detectable_effect_floor_nats_3se'],
            'natural_swap':
                d['natural_induction']['ORDER_ONLY_patch_swap']['mean'],
            'natural_swap_floor': d['natural_induction']
                ['ORDER_ONLY_patch_swap']['detectable_effect_floor_nats_3se'],
            'attn_first': lo['attention_marginal_first'],
            'attn_last': lo['attention_marginal_last'],
            'attn_order_ratio': lo['order_dependence_ratio_attention'],
            'interaction': lo['interaction_nats'],
            'read_kl': rk})
    return cells


def route_table(rk, L):
    """For each layer l>=1: the dominant read source, and the largest
    attention-to-attention term as a fraction of the dominant MLP term."""
    out = {}
    for l in range(1, L):
        z = {k.split('_')[-1]: v for k, v in rk.items()
             if k.startswith(f'l{l}_read_zero_')}
        r = {k.split('_')[-1]: v for k, v in rk.items()
             if k.startswith(f'l{l}_read_resample_')}
        # the harsher of the two flavours per source (resample is usually
        # harsher; both are quoted so the pair brackets the effect)
        h = {k: max(z.get(k, 0.0), r.get(k, 0.0)) for k in set(z) | set(r)}
        mlp = {k: v for k, v in h.items() if k.startswith('M')}
        att = {k: v for k, v in h.items() if k.startswith('A')}
        dom = max(h, key=h.get)
        dom_mlp = max(mlp, key=mlp.get) if mlp else None
        top_a = max(att, key=att.get) if att else None
        out[l] = {
            'dominant_source': dom, 'dominant_kl': h[dom],
            'dominant_is_immediately_preceding_mlp': dom == f'M{l-1}',
            'dominant_mlp': dom_mlp,
            'dominant_mlp_kl': mlp[dom_mlp] if dom_mlp else None,
            'largest_attention_source': top_a,
            'largest_attention_kl': att[top_a] if top_a else None,
            'attention_over_dominant_mlp':
                (att[top_a] / mlp[dom_mlp]) if (top_a and dom_mlp
                                                and mlp[dom_mlp] > 0) else None,
            'zero': z, 'resample': r}
    return out


def main():
    cells = load()
    pred = json.load(open(f'{HERE}/tf_depth_ladder_predictions.json'))
    rep = {'registered_predictions': pred['task_1_the_depth_ladder'],
           'cells': {}}
    for (dep, w), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r['seed'])
        agg = {'depth': dep, 'width': w, 'n_seeds': len(rs),
               'params': rs[0]['params'],
               'all_gates_pass': all(r['fold_gate'] and r['pipeline']
                                     for r in rs),
               'per_seed': rs}
        for k in ('held_ce', 'bits_per_byte', 'ladder_ce', 'induction',
                  'induction_floor', 'natural_swap', 'attn_first', 'attn_last',
                  'attn_order_ratio', 'interaction'):
            v = [r[k] for r in rs]
            agg[k] = float(np.mean(v))
            agg[k + '_sd'] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        agg['seeds_above_induction_floor'] = int(
            sum(r['induction'] > r['induction_floor'] for r in rs))
        agg['routes'] = [route_table(r['read_kl'], dep) for r in rs]
        if dep > 1:
            aod = [max((x['attention_over_dominant_mlp'] or 0.0)
                       for x in rt.values()) for rt in agg['routes']]
            agg['max_attention_over_dominant_mlp_per_seed'] = aod
            agg['max_attention_over_dominant_mlp'] = float(max(aod))
            agg['dominant_is_preceding_mlp_all_layers_all_seeds'] = all(
                x['dominant_is_immediately_preceding_mlp']
                for rt in agg['routes'] for x in rt.values())
        rep['cells'][f'd{dep}_w{w}'] = agg

    # ------------------------------------------------------- P1 verdict
    C = rep['cells']
    p1 = {'registered': pred['task_1_the_depth_ladder']['P1_induction_width_threshold']['prediction'],
          'per_cell': {k: {'induction': v['induction'],
                           'sd': v['induction_sd'],
                           'floor': v['induction_floor'],
                           'seeds_above_floor':
                               v['seeds_above_induction_floor'],
                           'n_seeds': v['n_seeds']}
                       for k, v in C.items()}}
    thr = {}
    for dep in (1, 2, 3, 4):
        ws = sorted(w for (d_, w) in
                    [(int(k[1]), int(k.split('_w')[1])) for k in C]
                    if d_ == dep)
        got = [w for w in ws
               if C[f'd{dep}_w{w}']['seeds_above_induction_floor']
               >= max(2, C[f'd{dep}_w{w}']['n_seeds'] - 1)]
        thr[dep] = min(got) if got else None
    p1['induction_width_threshold_by_depth'] = thr
    p1['call'] = (
        'CONFIRMED -- depth lowers the threshold an octave'
        if thr.get(3) == 128 and thr.get(2) == 256 else
        ('REFUTED -- the threshold does not move with depth'
         if thr.get(3) == thr.get(2) else
         f'PARTIAL -- thresholds by depth {thr}'))
    rep['P1_verdict'] = p1

    # ------------------------------------------------------- P2 verdict
    bad = {k: v['max_attention_over_dominant_mlp']
           for k, v in C.items()
           if v.get('max_attention_over_dominant_mlp', 0) > 0.01}
    rep['P2_verdict'] = {
        'registered': pred['task_1_the_depth_ladder']
                          ['P2_attention_to_attention_route']['prediction'],
        'max_attention_over_dominant_mlp_by_cell':
            {k: v.get('max_attention_over_dominant_mlp')
             for k, v in C.items() if v['depth'] > 1},
        'cells_above_1pc': bad,
        'named_exception_only':
            set(bad) <= {'d4_w256'},
        'call': ('CONFIRMED -- the feed-forward path dominates every layer at '
                 'every depth and width' if not bad else
                 ('CONFIRMED WITH THE PRE-REGISTERED EXCEPTION (depth 4, '
                  'width 256 only)' if set(bad) <= {'d4_w256'} else
                  f'REFUTED at {sorted(bad)}'))}

    # ------------------------------------------------------- P3/P4
    rep['P3_held_ce'] = {k: {'held_ce': v['held_ce'], 'sd': v['held_ce_sd'],
                             'bits_per_byte': v['bits_per_byte']}
                         for k, v in C.items()}
    rep['P4_ladder_order'] = {
        k: {'attention_first': v['attn_first'], 'attention_last': v['attn_last'],
            'order_ratio': v['attn_order_ratio'],
            'interaction_nats': v['interaction']} for k, v in C.items()}
    json.dump(rep, open(f'{HERE}/tf_depth_ladder.json', 'w'), indent=2)

    # ------------------------------------------------------- markdown
    L = ['# The depth ladder (vanilla, V=8192 trained BPE, three seeds a cell)',
         '',
         'Every number through ONE code path: `tf_interp3.py`, the same '
         'revision that produced the six-architecture slice, gated against '
         '`tf_interp2` on a vanilla checkpoint. Depth-1/2 cells that predated '
         'that path were re-run through it.',
         '',
         '| depth | width | params | held CE (T512) | bits/byte | ladder CE | '
         'induction ± sd (floor) | seeds above floor | natural swap | '
         'attention first / last | order ratio | interaction |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, v in sorted(rep['cells'].items(),
                       key=lambda kv: (kv[1]['depth'], kv[1]['width'])):
        L.append(
            f"| {v['depth']} | {v['width']} | {v['params']:,} | "
            f"{v['held_ce']:.4f} ± {v['held_ce_sd']:.4f} | "
            f"{v['bits_per_byte']:.4f} | {v['ladder_ce']:.4f} | "
            f"{v['induction']:+.4f} ± {v['induction_sd']:.4f} "
            f"({v['induction_floor']:.4f}) | "
            f"**{v['seeds_above_induction_floor']}/{v['n_seeds']}** | "
            f"{v['natural_swap']:+.4f} | "
            f"{v['attn_first']:.2f} / {v['attn_last']:.2f} | "
            f"{v['attn_order_ratio']:.1f}x | {v['interaction']:.2f} |")
    L += ['', '## The composition budget, measured causally',
          '',
          'Each upstream write deleted from layer l\'s Q/K/V read ONLY '
          '(residual untouched, everything downstream recomputed), KL from the '
          'true model in nats/token, [zero, resample]. Seed 0 shown; the '
          'per-seed record is in `tf_depth_ladder.json`.',
          '',
          '| cell | layer | dominant source | its KL | largest '
          'attention-to-attention source | its KL | as a fraction of the '
          'dominant MLP |',
          '|---|---|---|---|---|---|---|']
    for k, v in sorted(rep['cells'].items(),
                       key=lambda kv: (kv[1]['depth'], kv[1]['width'])):
        if v['depth'] < 2:
            continue
        for l, r in v['routes'][0].items():
            fr = r['attention_over_dominant_mlp']
            L.append(
                f"| d{v['depth']} w{v['width']} | {l} | {r['dominant_source']}"
                f" | {r['dominant_kl']:.4g} | {r['largest_attention_source']}"
                f" | {r['largest_attention_kl']:.4g} | "
                f"{'—' if fr is None else f'{fr:.2e}'} |")
    L += ['', '## Verdicts against the registered predictions', '', '```',
          json.dumps({'P1': {kk: vv for kk, vv in rep['P1_verdict'].items()
                             if kk != 'per_cell'},
                      'P2': {kk: vv for kk, vv in rep['P2_verdict'].items()
                             if kk != 'registered'}},
                     indent=2, default=str), '```']
    open(f'{HERE}/tf_depth_ladder_table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))

    # ------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:                                        # pragma: no cover
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    specs = [('held_ce', 'Held cross-entropy (nats/token)',
              'training protocol, T=512', False, None),
             ('induction', 'Induction score (nats)',
              'planted-oracle power floor shaded', False, 'floor'),
             ('attn_order_ratio', 'Ladder-order dependence of attention',
              'attention added first / attention added last', True, None)]
    for ax, (key, title, sub, logy, extra) in zip(axes, specs):
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(MUTED)
            ax.spines[s].set_linewidth(0.8)
        ax.grid(True, color='#e6e5e1', lw=0.8)
        ax.set_axisbelow(True)
        for dep in (1, 2, 3, 4):
            pts = sorted([(v['width'], v[key], v.get(key + '_sd', 0.0),
                           v['induction_floor'])
                          for v in rep['cells'].values() if v['depth'] == dep])
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            e = [p[2] for p in pts]
            c = DEPTH_COLOR[dep]
            ax.errorbar(x, y, yerr=e, fmt='-o', color=c, lw=2, ms=8,
                        capsize=0, ecolor=c, elinewidth=1.4,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(f'depth {dep}', (x[-1], y[-1]), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=9,
                        color=INK2)
            if extra == 'floor':
                ax.fill_between(x, [-p[3] for p in pts], [p[3] for p in pts],
                                color='#e6e5e1', zorder=1, lw=0)
        if extra == 'floor':
            ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xscale('log', base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.set_xticklabels(['32', '64', '128', '256'])
        if logy:
            ax.set_yscale('log')
        ax.set_xlim(28, 420)
        ax.set_xlabel('width', fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc='left', pad=14)
        ax.annotate(sub, (0, 1.015), xycoords='axes fraction', fontsize=8.5,
                    color=MUTED, va='bottom')
        ax.tick_params(colors=INK2, labelsize=8.5)
    fig.suptitle('The depth ladder: what changes as layers are added',
                 fontsize=12.5, color=INK, x=0.006, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f'{HERE}/fig_tf_depth_ladder.png', dpi=150, facecolor=SURFACE)


if __name__ == '__main__':
    main()
