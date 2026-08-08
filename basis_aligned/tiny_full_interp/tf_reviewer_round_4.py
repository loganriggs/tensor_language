"""INDEPENDENT ADVERSARIAL REVIEW (round 4) OF THE DEPTH LADDER (FINDING 14).

The reviewer did not produce FINDING 14.  Three objections were named up front
and each gets a MEASUREMENT, not an argument:

  O1  Is the induction probe's power floor itself DEPTH-DEPENDENT?  A deeper
      model has more machinery that could manufacture a positive score for a
      non-induction reason, and the floor is estimated from only five probe
      seeds.
  O2  Is the route measurement COMPARABLE ACROSS DEPTHS when the number of
      upstream sources differs?  Layer 1 of a depth-2 model has three sources;
      layer 2 of a depth-3 model has five, and "the largest attention source"
      is a maximum over a growing candidate set.
  O3  Does "the threshold moves once per layer" survive if the floor is
      RECOMPUTED PER CELL rather than assumed?

plus what the reviewer found while attacking those three:

  O4  "The attention-to-attention route opens at depth 3" is partly TRUE BY
      CONSTRUCTION -- a depth-2 model's only attention-to-attention route is
      the one from the first attention block, which FINDING 14 itself shows is
      mute at every depth.
  O5  A row of FINDING 14's own route table disagrees with the table its
      generator produces.
  O6  Is seed 0 systematically the flattering seed?

The two GPU controls:

  * CONTENT-FREE ATTENTION.  Replacing every layer's attention pattern with the
    uniform average over the causal past ('meanpast') or with the
    position-only pattern ('pos') leaves the depth, the trained MLPs and the
    trained readout intact but makes induction IMPOSSIBLE -- the pattern no
    longer depends on which tokens are where.  The score the probe returns on
    that object is a depth-matched FALSE-POSITIVE RATE, which is exactly what
    O1 asks for and which no untrained-model control can give (an untrained
    model scores ~0 for the trivial reason that it predicts nothing).
  * A 20-PROBE-SEED FLOOR, with the score recomputed on the same 20 seeds, so
    the "above floor" decision is not resting on a 5-sample standard deviation.
"""
import argparse
import glob
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch

import tf_interp as I1
import tf_interp2 as I2
import tf_interp3 as I3

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------- GPU
@torch.no_grad()
def content_free_and_floor(stem, probe_seeds=20):
    """The depth-matched false-positive control plus a 20-seed floor."""
    D = I3.VariantFold(stem)
    out = {'stem': stem, 'depth': D.L, 'width': D.cfg.width,
           'probe_seeds': probe_seeds}

    def battery(fwd):
        r = [I1.induction_battery(D, seed=s, model=fwd)
             for s in range(probe_seeds)]
        ind = np.array([x['induction_score'] for x in r])
        bag = np.array([x['bag_score'] for x in r])
        se = ind.std(ddof=1) / math.sqrt(len(ind))
        return {'induction_score_mean': float(ind.mean()),
                'induction_score_sd': float(ind.std(ddof=1)),
                'induction_score_se': float(se),
                'floor_3se': float(3 * se),
                'above_own_floor': bool(ind.mean() > 3 * se),
                't': float(ind.mean() / se) if se else None,
                'bag_score_mean': float(bag.mean())}

    out['true_model'] = battery(D.model)
    for mode in ('meanpast', 'pos'):
        fwd = (lambda m: lambda z: D.readout(
            D.run(z, attn={li: m for li in range(D.L)})['r']))(mode)
        out[f'content_free_{mode}'] = battery(fwd)
    return out


# ------------------------------------------------------------------- no GPU
def load_interp(pat='tf_vanilla_d*_w*_b8192_s*_interp3.json'):
    cells = defaultdict(dict)
    for f in sorted(glob.glob(f'{HERE}/{pat}')):
        m = re.search(r'_d(\d+)_w(\d+)_b8192_s(\d+)_interp3\.json$', f)
        if not m:
            continue
        d = json.load(open(f))
        if 'read_ablation_causal' not in d:
            continue
        cells[(int(m.group(1)), int(m.group(2)))][int(m.group(3))] = d
    return cells


def o2_route_comparability(cells):
    """Every normalization of the route number, side by side, plus the
    WITHIN-READ comparison that the source-count objection cannot touch."""
    rows = []
    for (dep, w), by_seed in sorted(cells.items()):
        if dep < 2:
            continue
        for s, d in sorted(by_seed.items()):
            rk = d['read_ablation_causal']['kl_from_model']
            sg = d.get('stream_geometry', {})
            for l in range(1, dep):
                z = {k.split('_')[-1]: v for k, v in rk.items()
                     if k.startswith(f'l{l}_read_zero_')}
                r = {k.split('_')[-1]: v for k, v in rk.items()
                     if k.startswith(f'l{l}_read_resample_')}
                h = {k: max(z.get(k, 0.0), r.get(k, 0.0)) for k in set(z) | set(r)}
                att = {k: v for k, v in h.items() if k.startswith('A')}
                mlp = {k: v for k, v in h.items() if k.startswith('M')}
                if not att or not mlp:
                    continue
                top_a = max(att, key=att.get)
                dom_mlp = max(mlp, key=mlp.get)
                tot = sum(h.values())
                rows.append({
                    'cell': f'd{dep}_w{w}', 'seed': s, 'layer': l,
                    'n_sources': len(h), 'n_attention_candidates': len(att),
                    'attention_sources': {k: h[k] for k in sorted(att)},
                    'largest_attention_source': top_a,
                    'largest_attention_kl': att[top_a],
                    'A0_kl': h.get('A0'),
                    'within_read_A0_vs_largest':
                        (att[top_a] / h['A0']) if h.get('A0') else None,
                    'share_of_dominant_mlp': att[top_a] / mlp[dom_mlp],
                    'share_of_total_read_mass': att[top_a] / tot if tot else None,
                    'rank_of_largest_attention_among_all_sources':
                        1 + sorted(h.values(), reverse=True).index(att[top_a]),
                    'write_norm_A0': sg.get('A0_norm'),
                    'write_norm_largest': sg.get(f'{top_a}_norm')})
    # the source-index hypothesis: is the SECOND attention block's transmission
    # the same at depth 3 and depth 4?  (source-matched, depth-varied)
    a1l2 = defaultdict(list)
    for r in rows:
        if r['layer'] == 2 and r['largest_attention_source'] == 'A1':
            a1l2[r['cell'][:2]].append(r['largest_attention_kl'])
    # does transmission track the WRITE NORM rather than the route?
    xs = [(r['write_norm_A0'], r['A0_kl']) for r in rows
          if r['write_norm_A0'] and r['A0_kl']]
    ys = [(r['write_norm_largest'], r['largest_attention_kl']) for r in rows
          if r['write_norm_largest']]
    pts = [p for p in xs + ys if p[0] and p[0] > 0 and p[1] and p[1] > 0]
    if len(pts) > 3:
        a = np.log10([p[0] for p in pts])
        b = np.log10([p[1] for p in pts])
        rho = float(np.corrcoef(a, b)[0, 1])
        sl = float(np.polyfit(a, b, 1)[0])
    else:
        rho = sl = None
    return {'rows': rows,
            'A1_into_layer2_by_depth': {k: [float(np.mean(v)),
                                            float(np.std(v, ddof=1)), v]
                                        for k, v in a1l2.items()},
            'log_write_norm_vs_log_transmission': {
                'pearson_r': rho, 'slope': sl, 'n': len(pts),
                'what': 'every (attention write norm, read-ablation KL) pair '
                        'in the ladder, both logged: if transmission is '
                        'explained by how big the write is, this is tight'}}


def o2b_transmission_is_magnitude(cells):
    """THE DECISIVE TEST FOR O2 AND O4.  For every (upstream write, downstream
    read) pair in the ladder, regress the read-ablation KL on the write's own
    NORM SHARE of the read it enters.  For a small perturbation the KL is
    locally quadratic in the perturbation, so a slope of 2 is what NO
    direction-specific gating looks like; the RESIDUAL around that line is the
    only place a genuine routing fact can live.  If layer-0 attention were
    gated shut, its residual would be strongly negative."""
    rows = []
    for (dep, w), by_seed in sorted(cells.items()):
        if dep < 2:
            continue
        for s, d in sorted(by_seed.items()):
            rk = d['read_ablation_causal']['kl_from_model']
            sg = d.get('stream_geometry')
            if not sg:
                continue
            for l in range(1, dep):
                srcs = (['e'] + [f'A{j}' for j in range(l)]
                        + [f'M{j}' for j in range(l)])
                den = math.sqrt(sum(sg[f'{k}_norm'] ** 2 for k in srcs))
                for k in srcs:
                    kl = max(rk.get(f'l{l}_read_zero_{k}', 0.0),
                             rk.get(f'l{l}_read_resample_{k}', 0.0))
                    if kl <= 0 or den <= 0:
                        continue
                    rows.append({'cell': f'd{dep}_w{w}', 'seed': s, 'read': l,
                                 'source': k, 'kl': kl,
                                 'write_norm': sg[f'{k}_norm'],
                                 'norm_share_of_read': sg[f'{k}_norm'] / den})
    if len(rows) < 10:
        return {'rows': rows}
    x = np.log10([r['norm_share_of_read'] for r in rows])
    y = np.log10([r['kl'] for r in rows])
    sl, ic = np.polyfit(x, y, 1)
    res = y - (sl * x + ic)
    per_src = defaultdict(list)
    for r_, e_ in zip(rows, res):
        per_src[r_['source']].append(float(e_))
    return {'n_pairs': len(rows),
            'pearson_r': float(np.corrcoef(x, y)[0, 1]),
            'slope_log_kl_per_log_norm_share': float(sl),
            'intercept': float(ic),
            'residual_sd_dex': float(res.std(ddof=2)),
            'residual_by_source': {k: {'mean_dex': float(np.mean(v)),
                                       'sd_dex': float(np.std(v)),
                                       'n': len(v)}
                                   for k, v in sorted(per_src.items())},
            'reading': 'slope 2 with a small residual means the read-ablation '
                       'KL is a QUADRATIC FUNCTION OF HOW BIG THE WRITE IS and '
                       'carries almost no direction-specific information; a '
                       'negative residual for A0 would be evidence of a '
                       'genuinely gated channel.',
            'rows': rows}


def o3_threshold(cells, floors20=None):
    """The threshold table three ways: the published per-cell 5-seed floor, a
    20-probe-seed floor, and a test over MODEL seeds that needs no floor."""
    out = {}
    for (dep, w), by_seed in sorted(cells.items()):
        sc = [d['rung3_induction']['induction_score_mean']
              for d in by_seed.values()]
        fl = [d['induction_power']['detectable_effect_floor_nats_3se']
              for d in by_seed.values()]
        n = len(sc)
        mu, sd = float(np.mean(sc)), (float(np.std(sc, ddof=1)) if n > 1 else 0.0)
        se = sd / math.sqrt(n) if n > 1 else None
        out[f'd{dep}_w{w}'] = {
            'depth': dep, 'width': w, 'n_model_seeds': n,
            'induction_per_seed': sc, 'floor_per_seed': fl,
            'seeds_above_published_floor': int(
                sum(a > b for a, b in zip(sc, fl))),
            'model_seed_mean': mu, 'model_seed_sd': sd,
            'model_seed_t': (mu / se) if se else None,
            'model_seed_test_positive': bool(se and mu / se > 3.182)}  # t.975, 2 df
    if floors20:
        for k, v in out.items():
            f = floors20.get(k)
            if f:
                v['floor_20seed'] = f['floor_3se']
                v['score_20seed'] = f['induction_score_mean']
                v['above_20seed_floor'] = f['above_own_floor']
    thr = {}
    for crit in ('seeds_above_published_floor', 'model_seed_test_positive',
                 'above_20seed_floor'):
        t = {}
        for dep in (1, 2, 3, 4):
            ws = sorted(v['width'] for v in out.values() if v['depth'] == dep)
            got = []
            for w in ws:
                v = out[f'd{dep}_w{w}']
                if crit == 'seeds_above_published_floor':
                    ok = v[crit] >= max(2, 0) and v['n_model_seeds'] >= 2 \
                        and v[crit] >= 2
                elif crit in v:
                    ok = bool(v[crit])
                else:
                    ok = None
                if ok:
                    got.append(w)
            t[dep] = min(got) if got else None
        thr[crit] = t
    return {'per_cell': out, 'threshold_by_criterion': thr}


def o6_seed_extremity(cells):
    """Is seed 0 systematically the flattering seed?  For every quantity the
    depth ladder quoted, where does seed 0 sit in its own 3-seed range?"""
    ranks = []
    for (dep, w), by_seed in sorted(cells.items()):
        if len(by_seed) < 3 or dep < 3:
            continue
        for name, get in (
            ('induction', lambda d: d['rung3_induction']
                ['induction_score_mean']),):
            v = {s: get(d) for s, d in by_seed.items()}
            order = sorted(v, key=v.get)
            ranks.append({'cell': f'd{dep}_w{w}', 'quantity': name,
                          'values': v, 'seed0_rank_low_to_high':
                              1 + order.index(0)})
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true',
                    help='run the content-free control and the 20-seed floor')
    ap.add_argument('--probe-seeds', type=int, default=20)
    a = ap.parse_args()

    gpu_path = f'{HERE}/tf_reviewer_round_4_gpu.json'
    gpu = json.load(open(gpu_path)) if os.path.exists(gpu_path) else {}
    if a.gpu:
        stems = sorted(os.path.basename(p)[:-3] for p in
                       glob.glob(f'{HERE}/tf_vanilla_d*_w*_b8192_s*.pt')
                       if re.search(r'_d\d+_w\d+_b8192_s\d+\.pt$', p))
        for st in stems:
            m = re.search(r'_d(\d+)_w(\d+)_b8192_s(\d+)$', st)
            dep, w, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if w == 32 or (dep <= 2 and s > 0):
                continue          # the control is about depth, not seed noise
            if st in gpu:
                continue
            t = time.time()
            try:
                gpu[st] = content_free_and_floor(st, a.probe_seeds)
            except Exception as e:                       # noqa: BLE001
                print('FAIL', st, e, flush=True)
                continue
            g = gpu[st]
            print(f"{st}  true {g['true_model']['induction_score_mean']:+.4f} "
                  f"(floor {g['true_model']['floor_3se']:.4f})  "
                  f"meanpast {g['content_free_meanpast']['induction_score_mean']:+.4f} "
                  f"(floor {g['content_free_meanpast']['floor_3se']:.4f})  "
                  f"pos {g['content_free_pos']['induction_score_mean']:+.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)
            json.dump(gpu, open(gpu_path, 'w'), indent=2)

    cells = load_interp()
    floors20 = {}
    for st, g in gpu.items():
        m = re.search(r'_d(\d+)_w(\d+)_b8192_s(\d+)$', st)
        if m and int(m.group(3)) == 0:
            floors20[f"d{m.group(1)}_w{m.group(2)}"] = g['true_model']

    rep = {'round': 4, 'reviewer': 'independent (did not produce FINDING 14)',
           'target': 'RESULTS.md FINDING 14, the depth ladder',
           'O1_power_floor_depth_dependence': {},
           'O2_route_comparability': o2_route_comparability(cells),
           'O2b_transmission_is_magnitude': o2b_transmission_is_magnitude(cells),
           'O3_threshold_under_recomputed_floor': o3_threshold(cells, floors20),
           'O6_seed_extremity': o6_seed_extremity(cells)}

    # ---- O1: assemble the false-positive control and the floor-vs-depth fit
    fp = []
    for st, g in sorted(gpu.items()):
        for mode in ('meanpast', 'pos'):
            c = g[f'content_free_{mode}']
            fp.append({'stem': st, 'depth': g['depth'], 'width': g['width'],
                       'mode': mode,
                       'score': c['induction_score_mean'],
                       'floor': c['floor_3se'],
                       'false_positive': bool(c['above_own_floor']),
                       't': c['t'],
                       'true_model_score': g['true_model']
                           ['induction_score_mean'],
                       'true_model_floor': g['true_model']['floor_3se']})
    o1 = {'content_free_control': fp}
    if fp:
        o1['false_positive_rate'] = float(np.mean([x['false_positive']
                                                   for x in fp]))
        o1['max_content_free_score'] = float(max(x['score'] for x in fp))
        by_d = defaultdict(list)
        for x in fp:
            by_d[x['depth']].append(x['score'])
        o1['content_free_score_by_depth'] = {
            k: [float(np.mean(v)), float(np.max(v))] for k, v in
            sorted(by_d.items())}
    # floor vs depth over every cell, from the published 5-seed floors
    dd, ff = [], []
    for (dep, w), by_seed in cells.items():
        for d in by_seed.values():
            dd.append(dep)
            ff.append(d['induction_power']['detectable_effect_floor_nats_3se'])
    if len(dd) > 3:
        sl, ic = np.polyfit(dd, ff, 1)
        r = float(np.corrcoef(dd, ff)[0, 1])
        o1['published_floor_vs_depth'] = {
            'slope_nats_per_layer': float(sl), 'intercept': float(ic),
            'pearson_r': r, 'n_cells': len(dd),
            'floor_by_depth': {k: float(np.mean([f for dpt, f in zip(dd, ff)
                                                 if dpt == k]))
                               for k in sorted(set(dd))}}
    rep['O1_power_floor_depth_dependence'] = o1
    json.dump(rep, open(f'{HERE}/tf_reviewer_round_4_measurements.json', 'w'),
              indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ('O2_route_comparability', 'O2b_transmission_is_magnitude')}, indent=2)[:6000])


if __name__ == '__main__':
    main()
