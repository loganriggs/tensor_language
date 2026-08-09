"""Score the attention x feed-forward factorial against the predictions that
were registered before the code was written.

The two known corners are READ FROM THE EXISTING FILES, not retrained:
  (bilin,   bilin)  -> tf_vanilla_d{D}_w{W}_b8192_s{S}      the family model
  (softmax, gelu)   -> tfb_std7_d{D}_w{W}_b8192_s{S}        the conventional one
which is also a free consistency check: gates G1/G2 say the factorial code
reduces to those two models exactly, so if a corner ever disagrees with its
published number the corner claim is wrong.

Writes tf_factorial.json and tf_factorial_table.md.
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ATTNS = ('bilin', 'bilinnorm', 'softmax')
MLPS = ('bilin', 'gelu')
CELLS = [(2, 128), (3, 64), (1, 128), (2, 256), (3, 128)]
SEEDS = (0, 1, 2)
PRETTY_A = {'bilin': 'two-branch product, unnormalised (ours, foldable)',
            'bilinnorm': 'the same product / its row L1 (diagnostic only)',
            'softmax': 'softmax, one branch (conventional)'}
PRETTY_M = {'bilin': 'ungated bilinear product (ours, foldable)',
            'gelu': 'GELU gate (conventional)'}


def _read(path, ce_key=('run', 'final_held_ce')):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    for k in ce_key:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return None
    return d


def _induction(path):
    """Mean over probe seeds and the 3-standard-error probe floor.

    Three shapes are accepted, because the same battery has been written to
    disk three ways over this programme's life: a standalone `_induction.json`
    with `per_probe_seed` (conventional and factorial arms), the same file with
    `per_seed` (older family runs), and the family's `_interp3.json` with the
    battery under `rung3_induction`.  Accepting all three is what keeps the
    family corner from silently reading as blank -- and a blank corner would
    quietly drop the main-effect calculation rather than fail loudly."""
    if not os.path.exists(path):
        alt = path.replace('_induction.json', '_interp3.json')
        if not os.path.exists(alt):
            return None
        j = json.load(open(alt)).get('rung3_induction')
        if not j:
            return None
    else:
        j = json.load(open(path))
    runs = j.get('per_probe_seed') or j.get('per_seed')
    if not runs:
        return None
    s = [r['induction_score'] for r in runs]
    return {'mean': st.mean(s), 'sd': st.stdev(s) if len(s) > 1 else 0.0,
            'probe_floor_3se': (3 * st.stdev(s) / math.sqrt(len(s))
                                if len(s) > 1 else None),
            'n_probe_seeds': len(s)}


def arm_files(attn, mlp, d, w, s):
    """Where this arm lives.  The two corners are published elsewhere."""
    if (attn, mlp) == ('bilin', 'bilin'):
        stem = f'tf_vanilla_d{d}_w{w}_b8192_s{s}'
        return stem, f'{HERE}/{stem}.json', f'{HERE}/{stem}_induction.json', \
            'published family model'
    if (attn, mlp) == ('softmax', 'gelu'):
        stem = f'tfb_std7_d{d}_w{w}_b8192_s{s}'
        return stem, f'{HERE}/{stem}.json', f'{HERE}/{stem}_induction.json', \
            'published conventional baseline, parameter-matched arm'
    stem = f'tff_{attn}_{mlp}_d{d}_w{w}_b8192_s{s}'
    return stem, f'{HERE}/{stem}.json', f'{HERE}/{stem}_induction.json', 'new'


def collect(d, w, s):
    grid = {}
    for a in ATTNS:
        for m in MLPS:
            stem, jp, ip, src = arm_files(a, m, d, w, s)
            grid[(a, m)] = {
                'stem': stem, 'source': src,
                'held_ce': _read(jp), 'induction': _induction(ip)}
    return grid


def effects(grid, field):
    """Main effects and the 2x2 interaction, on whichever field is asked for.

    The interaction is taken on the 2x2 sub-design (ours vs conventional on
    both factors) and NOT on the 3x2, because bilinnorm is not a level of the
    same kind -- it is a diagnostic interpolant, and averaging it into a main
    effect would make that main effect uninterpretable."""
    def val(a, m):
        c = grid[(a, m)]
        v = c[field] if field == 'held_ce' else (
            c['induction']['mean'] if c['induction'] else None)
        return v
    out = {'per_arm': {f'{a}+{m}': val(a, m) for a in ATTNS for m in MLPS}}
    have = all(val(a, m) is not None
               for a in ('bilin', 'softmax') for m in MLPS)
    if not have:
        out['complete'] = False
        return out
    bb, bg = val('bilin', 'bilin'), val('bilin', 'gelu')
    sb, sg = val('softmax', 'bilin'), val('softmax', 'gelu')
    out['complete'] = True
    out['attention_main_effect'] = (sb + sg) / 2 - (bb + bg) / 2
    out['feed_forward_main_effect'] = (bg + sg) / 2 - (bb + sb) / 2
    out['interaction_2x2'] = sg - sb - bg + bb
    out['total_gap_conventional_minus_family'] = sg - bb
    out['attention_share_of_the_total_gap'] = (
        None if abs(sg - bb) < 1e-9
        else out['attention_main_effect'] / (sg - bb))
    nb = val('bilinnorm', 'bilin')
    if nb is not None and abs(sb - bb) > 1e-9:
        out['bilinnorm_recovers_of_the_softmax_move'] = (nb - bb) / (sb - bb)
    return out


def score_predictions(cell_report):
    """F1-F5, scored only where the arms they need are on disk."""
    key = '2x128_s0'
    g = cell_report.get(key)
    res = {'scored_at': 'depth 2 width 128 seed 0', 'available': bool(g)}
    if not g:
        return res
    ind = g['induction_effects']
    ce = g['ce_effects']
    pa = ind.get('per_arm', {})
    sb, bg = pa.get('softmax+bilin'), pa.get('bilin+gelu')
    if sb is not None and bg is not None:
        res['F1_attribution'] = {
            'softmax_with_our_feed_forward': sb,
            'our_attention_with_gelu': bg,
            'predicted': 'softmax arm above +0.08, gelu arm below +0.04',
            'sharp_form_holds': bool(sb > 0.08 and bg < 0.04),
            'direction_holds': bool(abs(sb - pa.get('bilin+bilin', 0))
                                    > abs(bg - pa.get('bilin+bilin', 0)))}
    if 'bilinnorm_recovers_of_the_softmax_move' in ind:
        r = ind['bilinnorm_recovers_of_the_softmax_move']
        res['F2_which_property_of_softmax'] = {
            'fraction_of_the_softmax_move_recovered_by_normalisation': r,
            'predicted': '>= 0.5', 'holds': bool(r >= 0.5)}
    if ce.get('complete') and ind.get('complete'):
        res['F3_ce_main_effects'] = {
            'attention': ce['attention_main_effect'],
            'feed_forward': ce['feed_forward_main_effect'],
            'predicted': 'attention effect larger in magnitude',
            'holds': bool(abs(ce['attention_main_effect'])
                          > abs(ce['feed_forward_main_effect']))}
        res['F4_interaction'] = {
            'ce_interaction_nats': ce['interaction_2x2'],
            'induction_interaction': ind['interaction_2x2'],
            'predicted': 'CE interaction under 0.02 nats; induction '
                         'interaction positive',
            'ce_additive_holds': bool(abs(ce['interaction_2x2']) < 0.02),
            'induction_superadditive_holds': bool(
                ind['interaction_2x2'] > 0)}
    g1 = cell_report.get('1x128_s0')
    if g1:
        rows = [(k, v) for k, v in g1['induction_effects']['per_arm'].items()
                if v is not None]
        res['F5_negative_control'] = {
            'depth_1_scores': dict(rows),
            'predicted': 'no arm inducts at depth 1',
            'holds': bool(rows and all(v < 0.02 for _, v in rows)),
            'note': 'a depth-1 arm above floor means the probe leaks, not that '
                    'the arm is clever'}
    return res


def main():
    out = {'predictions_file': 'tf_factorial_predictions.json',
           'gates_file': 'tf_factorial_controls.json',
           'probe_control_file': 'tf_factorial_probe_control.json'}
    for f in ('tf_factorial_controls.json', 'tf_factorial_probe_control.json'):
        p = f'{HERE}/{f}'
        if os.path.exists(p):
            j = json.load(open(p))
            out.setdefault('gates', {})[f] = j.get('all_pass', j.get('pass'))
    cells = {}
    for (d, w) in CELLS:
        for s in SEEDS:
            grid = collect(d, w, s)
            if not any(v['held_ce'] is not None for v in grid.values()):
                continue
            cells[f'{d}x{w}_s{s}'] = {
                'depth': d, 'width': w, 'seed': s,
                'arms': {f'{a}+{m}': grid[(a, m)] for a in ATTNS for m in MLPS},
                'ce_effects': effects(grid, 'held_ce'),
                'induction_effects': effects(grid, 'induction')}
    out['cells'] = cells
    out['prediction_scores'] = score_predictions(cells)
    json.dump(out, open(f'{HERE}/tf_factorial.json', 'w'), indent=2)

    L = ['# The attention x feed-forward factorial', '',
         'Every arm holds the body parameter count at the family\'s 18 W^2 + W '
         'per block. `bilinnorm` is DIAGNOSTIC ONLY -- its row denominator '
         'depends on every visible key, so it does not fold to a fixed '
         'token-pair table and is never a foldable win.', '']
    for k, c in cells.items():
        L += [f'## depth {c["depth"]} width {c["width"]} seed {c["seed"]}', '',
              '| attention | feed-forward | held CE | induction | probe floor |'
              ' source |', '|---|---|---|---|---|---|']
        for a in ATTNS:
            for m in MLPS:
                r = c['arms'][f'{a}+{m}']
                ce = f'{r["held_ce"]:.5f}' if r['held_ce'] else '--'
                i = r['induction']
                ind = f'{i["mean"]:+.4f}' if i else '--'
                fl = (f'{i["probe_floor_3se"]:.4f}'
                      if i and i['probe_floor_3se'] else '--')
                L.append(f'| {PRETTY_A[a]} | {PRETTY_M[m]} | {ce} | {ind} | '
                         f'{fl} | {r["source"]} |')
        L.append('')
        for name, e in (('held cross-entropy', c['ce_effects']),
                        ('induction score', c['induction_effects'])):
            if e.get('complete'):
                L += [f'**{name}** — attention main effect '
                      f'{e["attention_main_effect"]:+.4f}, feed-forward main '
                      f'effect {e["feed_forward_main_effect"]:+.4f}, '
                      f'interaction {e["interaction_2x2"]:+.4f}, total '
                      f'conventional-minus-family gap '
                      f'{e["total_gap_conventional_minus_family"]:+.4f}.', '']
                if 'bilinnorm_recovers_of_the_softmax_move' in e:
                    L += [f'Row normalisation alone recovers '
                          f'{100 * e["bilinnorm_recovers_of_the_softmax_move"]:.0f}%'
                          f' of the softmax move on {name}.', '']
    L += ['## Predictions, scored', '', '```',
          json.dumps(out['prediction_scores'], indent=2), '```']
    open(f'{HERE}/tf_factorial_table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)
    return out


if __name__ == '__main__':
    main()
