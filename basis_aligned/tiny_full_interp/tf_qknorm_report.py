"""Score the query/key-norm control as a 2x2: {foldable family, conventional}
x {with query/key RMSNorm, without}, at depth 2 width 128.

The conventional half is already on disk from the baseline chain
(tfb_std4_d2_w128_b8192_s*[_noqknorm]); the foldable half comes from the
factorial's (bilin, bilin) path, which gate G1 shows reproduces the family
model bit-for-bit and which is parameter-identical with the norm on or off.

Writes tf_qknorm.json and tf_qknorm_table.md, and scores Q1-Q3 from
tf_qknorm_predictions.json.
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)


def _ind(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    j = j.get('rung3_induction') or j
    r = j.get('per_probe_seed') or j.get('per_seed')
    return st.mean([x['induction_score'] for x in r]) if r else None


def _ce(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path)).get('run', {}).get('final_held_ce')


def _run(path, key):
    if not os.path.exists(path):
        return None
    return json.load(open(path)).get('run', {}).get(key)


def cell(stem_fn):
    ces, inds, spikes, div = [], [], [], []
    for s in SEEDS:
        stem = stem_fn(s)
        c = _ce(f'{HERE}/{stem}.json')
        if c is None:
            continue
        ces.append(c)
        i = _ind(f'{HERE}/{stem}_induction.json')
        if i is None:
            i = _ind(f'{HERE}/{stem}_interp3.json')
        if i is not None:
            inds.append(i)
        spikes.append(_run(f'{HERE}/{stem}.json', 'spikes'))
        div.append(bool(_run(f'{HERE}/{stem}.json', 'diverged')))
    return {'n_ce': len(ces), 'ce': st.mean(ces) if ces else None,
            'ce_sd': st.stdev(ces) if len(ces) > 1 else None,
            'n_ind': len(inds), 'induction': st.mean(inds) if inds else None,
            'induction_sd': st.stdev(inds) if len(inds) > 1 else None,
            'induction_per_seed': inds, 'ce_per_seed': ces,
            'spikes': spikes, 'any_diverged': any(div)}


def main():
    arms = {
        'foldable_with_qknorm': cell(
            lambda s: f'tf_vanilla_d2_w128_b8192_s{s}'),
        'foldable_without_qknorm': cell(
            lambda s: f'tff_bilin_bilin_d2_w128_b8192_s{s}_noqknorm'),
        'conventional_with_qknorm': cell(
            lambda s: f'tfb_std4_d2_w128_b8192_s{s}'),
        'conventional_without_qknorm': cell(
            lambda s: f'tfb_std4_d2_w128_b8192_s{s}_noqknorm'),
    }
    out = {'predictions_file': 'tf_qknorm_predictions.json',
           'cell': 'depth 2 width 128, V=8192 BPE',
           'note': ('the foldable arms are parameter-identical with the norm '
                    'on or off (body 590,080); so are the conventional arms'),
           'arms': arms}

    def d(a, b, f):
        x, y = arms[a][f], arms[b][f]
        return None if (x is None or y is None) else y - x

    out['effect_of_removing_qknorm'] = {
        'foldable_ce_change': d('foldable_with_qknorm',
                                'foldable_without_qknorm', 'ce'),
        'foldable_induction_change': d('foldable_with_qknorm',
                                       'foldable_without_qknorm', 'induction'),
        'conventional_ce_change': d('conventional_with_qknorm',
                                    'conventional_without_qknorm', 'ce'),
        'conventional_induction_change': d('conventional_with_qknorm',
                                           'conventional_without_qknorm',
                                           'induction'),
        'sign_convention': 'negative CE change = removing the norm HELPS'}

    fw, cw = arms['foldable_without_qknorm'], arms['conventional_without_qknorm']
    fwith = arms['foldable_with_qknorm']
    if fw['ce'] is not None and cw['ce'] is not None:
        out['tax_best_against_best'] = {
            'foldable_best_ce': min(x for x in (fwith['ce'], fw['ce'])
                                    if x is not None),
            'conventional_best_ce': min(
                x for x in (arms['conventional_with_qknorm']['ce'], cw['ce'])
                if x is not None),
            'tax_nats': None}
        t = out['tax_best_against_best']
        t['tax_nats'] = round(t['foldable_best_ce']
                              - t['conventional_best_ce'], 5)
        t['claim'] = ('each family at ITS OWN better configuration -- the only '
                      'symmetric way to quote a tax once one arm turns out to '
                      'have been handicapped')

    # ---- score the registered predictions
    sc = {}
    fce = out['effect_of_removing_qknorm']['foldable_ce_change']
    fin = arms['foldable_without_qknorm']['induction']
    if fce is not None:
        sc['Q1_size_of_the_family_gain'] = {
            'family_ce_change': round(fce, 5),
            'conventional_ce_change': round(
                out['effect_of_removing_qknorm']['conventional_ce_change'], 5),
            'predicted': 'family change between -0.05 and +0.10',
            'holds': bool(-0.05 <= fce <= 0.10)}
    if fin is not None:
        sc['Q2_does_the_family_start_inducting'] = {
            'family_induction_without_qknorm': round(fin, 5),
            'predicted': 'stays below +0.05',
            'holds': bool(fin < 0.05),
            'note_if_refuted': ('a large family induction here means the cap '
                                'we imposed, not the missing softmax, was '
                                'blocking induction -- a more actionable '
                                'result than the current story')}
    if fw['n_ce']:
        sc['Q3_training_stability'] = {
            'spikes_without_qknorm': fw['spikes'],
            'spikes_with_qknorm': fwith['spikes'],
            'any_diverged': fw['any_diverged'],
            'predicted': 'trains without divergence, more spikes than with'}
    out['prediction_scores'] = sc
    json.dump(out, open(f'{HERE}/tf_qknorm.json', 'w'), indent=2)

    L = ['# Query/key normalisation: the 2x2', '',
         'Depth 2 width 128. Parameter-identical within each family whether '
         'the norm is on or off.', '',
         '| family | query/key norm | seeds | held CE | induction |',
         '|---|---|---|---|---|']
    label = {'foldable_with_qknorm': ('foldable (ours)', 'on'),
             'foldable_without_qknorm': ('foldable (ours)', 'OFF'),
             'conventional_with_qknorm': ('conventional', 'on'),
             'conventional_without_qknorm': ('conventional', 'OFF')}
    for k, a in arms.items():
        f, n = label[k]
        ce = f'{a["ce"]:.5f}' if a['ce'] is not None else '--'
        i = f'{a["induction"]:+.4f}' if a['induction'] is not None else '--'
        L.append(f'| {f} | {n} | {a["n_ce"]} | {ce} | {i} |')
    L += ['', '```', json.dumps(out['effect_of_removing_qknorm'], indent=2),
          '```', '', '## Predictions, scored', '', '```',
          json.dumps(sc, indent=2), '```']
    open(f'{HERE}/tf_qknorm_table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)
    return out


if __name__ == '__main__':
    main()
