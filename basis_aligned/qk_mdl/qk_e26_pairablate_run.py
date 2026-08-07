"""E26 EXHAUSTIVE PAIRWISE-ABLATION INTERACTION MAP (checkpoint-only; no
training). Open-problems #2 (BRAINSTORM_STATE 2026-08-07): ablation
non-composition is brute-forceable at width 264 -- replace the first-order
causal ground truth with an exhaustive second-order interaction map.

ON qk_e9_a.pt (the readable-recipe reference arm). All ablations are the
standard light-probe convention EXACTLY: fp32 model, old cooc held rows
[:96] (R2.HELD, bound at import), per-(consumer, source) mean substitution
at the consumer's entry assembly via sub_entry (R2.ce_with), stream means
from R2.base_and_means. Joint ablation = the same machinery with several
(consumer, source) substitutions in one forward; downstream writes are
recomputed, so ablations propagate exactly as in the stored singles.

TWO TIERS:
  (a) MODULE tier: source i mean-ablated JOINTLY AT EVERY consumer that
      sees it (emb + 24 writers = 25 sources; 25 singles + C(25,2) = 300
      pairs, full held-CE each).
  (b) EDGE tier at the TOP consumer (the readout; per-consumer totals
      recorded to confirm it is top): the readout's 24 writer source-edges,
      C(24,2) = 276 pairs; singles are the recomputed light-probe row.
Interaction I(i,j) = dCE(i,j) - dCE(i) - dCE(j).

CONTROLS (hard gates, before the map):
  1. IDENTITY substitution at exactly zero: sub_entry replacing a stream
     with its own true tensor (readout:mlp11, block6:mlp1, block3:emb)
     must reproduce the base logits with max |diff| == 0.0 exactly.
  2. SINGLES REPRODUCTION: recompute ALL singles with R2.consumption_graph
     (the exact stored machinery) and assert every entry matches the stored
     light_probe_E9a consumption_matrix to 1e-3 (and base CE to 1e-3).
  3. Built-in consistency: sources 23/24 (mlp10-slot? no -- streams 23 =
     attn11, 24 = mlp11) are visible ONLY to the readout, so their
     module-tier single must equal the edge-tier single bit-exactly.

REGISTERED PREDICTIONS (merged before computing):
  (i)   superadditive pairs (I > 0.01) concentrate among same-type
        module pairs (mlp-mlp) and adjacent depths (|depth diff| <= 1);
  (ii)  the S2 write-lasso broadcast cast (mlp11/mlp0/attn9/mlp10) shows
        the LARGEST positive interactions;
  (iii) >= 70% of pairs are near-additive (|I| < 0.005) -- the partition
        made most attribution locally additive.

OUTPUT qk_e26.json: full interaction matrices (module 25x25, readout-edge
24x24), top-15 superadditive and top-15 subadditive pairs with numbers,
additivity fractions, and the first-order vs interaction-aware ranking
comparison: Spearman of the weight table (weight_support) against plain
singles vs against interaction-adjusted importances adj(i) = dCE(i) +
0.5 * sum_j I(i,j), at both tiers. Partial pair CEs cached in
qk_e26_partial.json (resume-safe); idempotent on qk_e26.json keys."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, C, DEPTH, F, torch
import qk_e7_evenout_run as E7R
import qk_deeproute_train_2 as R2

E.DEV = 'cpu' if E.SMOKE else 'cuda'
JP = E.jpath('qk_e26.json')
PARTIAL = E.jpath('qk_e26_partial.json')
NS = 1 + 2 * DEPTH                      # 25 streams
ADD_TOL = 0.005                         # |I| below this = near-additive
SUPER_TOL = 0.01                        # I above this = superadditive
GATE_TOL = 1e-3
# S2 soft write-lasso broadcast cast (MAILBOX 2026-08-06 scale entry)
BROADCAST = {'mlp0': 2, 'attn9': 19, 'mlp10': 22, 'mlp11': 24}

if E.SMOKE:
    R2.DEV = 'cpu'
    R2.ABL_N = 8

sname = R2.stream_name


def depth_of(si):
    return (si - 1) // 2               # writer stream depth; si >= 1


def kind_of(si):
    return 'emb' if si == 0 else ('attn' if (si - 1) % 2 == 0 else 'mlp')


# ---------------- joint ablation (the singles machinery, plural) ------------
def joint_ce(m, means, sources, consumers):
    """R2.ce_with with several (consumer, source) mean substitutions in one
    forward; identical conventions to the stored singles."""
    sub = {}
    for li in consumers:
        d = {s: means[s] for s in sources if s in m.vis[li]}
        if d:
            sub[li] = d
    assert sub, f'no visible consumer for sources {sources}'
    return R2.ce_with(m, sub_entry=sub)


# ---------------- control 1: identity substitution == base ------------------
@torch.no_grad()
def identity_control(m):
    key = 'control_identity_substitution'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    b = R2.HELD[:8]
    col = {'entry_norm': [], 'attn_write': [], 'mlp_write': []}
    base_logits = m(b[:, :R2.T], collect=col)
    Dm = m.wte.weight.shape[1]
    streams = [F.rms_norm(m.wte(b[:, :R2.T]), (Dm,))]
    for l in range(DEPTH):
        streams.append(col['attn_write'][l])
        streams.append(col['mlp_write'][l])
    checks = {}
    for li, si in ((DEPTH, 24), (6, 3), (3, 0)):
        lg = m(b[:, :R2.T], sub_entry={li: {si: streams[si]}})
        checks[f'consumer{li}:{sname(si)}'] = float(
            (lg - base_logits).abs().max())
    rec = {'checks_max_abs_logit_diff': checks,
           'pass': bool(all(v == 0.0 for v in checks.values())),
           'note': 'substituting a stream with its own true tensor at one '
                   'consumer must be an exact identity (protocol: identity '
                   'reductions pass at exactly zero)'}
    E.merge(JP, key, rec)
    print(f"{key}: {checks} -> {'PASS' if rec['pass'] else 'FAIL'}",
          flush=True)
    assert rec['pass'], f'{key} FAILED'


# ---------------- control 2: singles reproduction gate ----------------------
def singles_and_gate(m):
    j = E.loadj(JP)
    base, means = R2.base_and_means(m)
    if 'singles_recomputed' in j and j.get('control_singles_gate', {}).get(
            'pass', E.SMOKE):
        sr = j['singles_recomputed']
        assert abs(sr['base_ce'] - base) < 1e-6, \
            (sr['base_ce'], base, 'base CE drifted between resumed runs')
        dce = {int(li): {int(si): v for si, v in row.items()}
               for li, row in sr['matrix'].items()}
        print("singles: cached (base reproduced) -- skip recompute",
              flush=True)
        return base, means, dce
    print("recomputing ALL singles via R2.consumption_graph ...", flush=True)
    dce = R2.consumption_graph(m, base, means)
    E.merge(JP, 'singles_recomputed', {
        'base_ce': base,
        'matrix': {str(li): {str(si): dce[li][si] for si in dce[li]}
                   for li in dce}})
    if E.SMOKE:
        E.merge(JP, 'control_singles_gate',
                {'pass': True, 'note': 'SMOKE: no stored reference'})
        return base, means, dce
    lp = E.loadj(E.jpath('qk_e9.json'))['light_probe_E9a']
    smat = {int(li): {int(si): v for si, v in row.items()}
            for li, row in lp['consumption_matrix'].items()}
    pairs_new = {(li, si) for li in dce for si in dce[li]}
    pairs_old = {(li, si) for li in smat for si in smat[li]}
    assert pairs_new == pairs_old, \
        f'edge sets differ: {pairs_new ^ pairs_old}'
    diffs = [(abs(dce[li][si] - smat[li][si]), li, si)
             for li, si in pairs_new]
    mx, mli, msi = max(diffs)
    bd = abs(base - lp['base_ce_fp32_abl_oldheld'])
    rec = {'n_edges': len(diffs), 'base_ce': base,
           'stored_base_ce': lp['base_ce_fp32_abl_oldheld'],
           'base_abs_diff': bd,
           'max_single_abs_diff': mx,
           'max_at': f'consumer{mli}:{sname(msi)}',
           'pass': bool(mx < GATE_TOL and bd < GATE_TOL)}
    E.merge(JP, 'control_singles_gate', rec)
    print(f"singles gate: {len(diffs)} edges, max |diff| {mx:.2e} at "
          f"consumer{mli}:{sname(msi)}, base diff {bd:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], 'singles reproduction gate FAILED'
    return base, means, dce


# ---------------- pair evaluation with resume-safe caching ------------------
def run_jobs(m, means, jobs, cache_key):
    part = E.loadj(PARTIAL)
    d = part.get(cache_key, {})
    todo = [(jid, srcs, cons) for jid, srcs, cons in jobs if jid not in d]
    if todo:
        t0 = time.time()
        for n, (jid, srcs, cons) in enumerate(todo):
            d[jid] = joint_ce(m, means, srcs, cons)
            if (n + 1) % 20 == 0 or n + 1 == len(todo):
                part = E.loadj(PARTIAL)
                part[cache_key] = d
                json.dump(part, open(PARTIAL, 'w'))
                el = time.time() - t0
                print(f"  {cache_key}: {n + 1}/{len(todo)} evals "
                      f"({el:.0f}s, {el / (n + 1):.2f}s/eval, "
                      f"~{el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min "
                      f"left)", flush=True)
    return d


def top_pairs(inter, k=15, reverse=True):
    rows = sorted(inter.items(), key=lambda kv: kv[1]['I'],
                  reverse=reverse)[:k]
    return [dict(pair=f'{sname(i)}+{sname(j)}',
                 **{kk: round(vv, 6) if isinstance(vv, float) else vv
                    for kk, vv in v.items()})
            for (i, j), v in rows]


# ---------------- tiers ----------------------------------------------------
def module_tier(m, means, base, dce):
    if 'module_tier' in E.loadj(JP):
        print("module_tier: done -- skip", flush=True)
        return
    srcs = list(range(7)) if E.SMOKE else list(range(NS))
    cons = list(range(DEPTH + 1))
    single_jobs = [(f's{i}', [i], cons) for i in srcs]
    sing = run_jobs(m, means, single_jobs, 'module_singles')
    dmod = {i: sing[f's{i}'] - base for i in srcs}
    # control 3: readout-only streams must equal the edge single bit-exactly
    consist = {}
    for si in (23, 24):
        if si in srcs:
            consist[sname(si)] = abs(dmod[si] - dce[DEPTH][si])
            assert consist[sname(si)] < 1e-9, \
                (si, dmod[si], dce[DEPTH][si])
    pair_jobs = [(f'p{i}_{j}', [i, j], cons)
                 for a, i in enumerate(srcs) for j in srcs[a + 1:]]
    pce = run_jobs(m, means, pair_jobs, 'module_pairs')
    inter = {}
    for a, i in enumerate(srcs):
        for j in srcs[a + 1:]:
            dj = pce[f'p{i}_{j}'] - base
            inter[(i, j)] = {'dce_i': dmod[i], 'dce_j': dmod[j],
                             'dce_joint': dj,
                             'I': dj - dmod[i] - dmod[j]}
    rec = {'sources': [sname(i) for i in srcs],
           'ablation': 'source mean-substituted jointly at EVERY visible '
                       'consumer (downstream writes recomputed)',
           'singles_dce': {sname(i): round(dmod[i], 6) for i in srcs},
           'consistency_readout_only_streams': {k: float(v) for k, v
                                                in consist.items()},
           'interaction_matrix': {f'{sname(i)}|{sname(j)}':
                                  round(v['I'], 6)
                                  for (i, j), v in inter.items()},
           'joint_dce_matrix': {f'{sname(i)}|{sname(j)}':
                                round(v['dce_joint'], 6)
                                for (i, j), v in inter.items()},
           'top15_superadditive': top_pairs(inter, 15, True),
           'top15_subadditive': top_pairs(inter, 15, False),
           'n_pairs': len(inter)}
    E.merge(JP, 'module_tier', rec)
    return


def edge_tier(m, means, base, dce):
    if 'edge_tier_readout' in E.loadj(JP):
        print("edge_tier_readout: done -- skip", flush=True)
        return
    totals = {li: sum(dce[li].values()) for li in dce}
    top_li = max(totals, key=totals.get)
    srcs = list(range(1, 7)) if E.SMOKE else list(range(1, NS))
    pair_jobs = [(f'p{i}_{j}', [i, j], [DEPTH])
                 for a, i in enumerate(srcs) for j in srcs[a + 1:]]
    pce = run_jobs(m, means, pair_jobs, 'edge_pairs')
    inter = {}
    for a, i in enumerate(srcs):
        for j in srcs[a + 1:]:
            dj = pce[f'p{i}_{j}'] - base
            inter[(i, j)] = {'dce_i': dce[DEPTH][i], 'dce_j': dce[DEPTH][j],
                             'dce_joint': dj,
                             'I': dj - dce[DEPTH][i] - dce[DEPTH][j]}
    rec = {'consumer': 'readout',
           'per_consumer_total_consumption':
               {('readout' if li == DEPTH else f'block{li}'):
                round(v, 5) for li, v in totals.items()},
           'readout_is_top_consumer': bool(top_li == DEPTH),
           'sources': [sname(i) for i in srcs],
           'singles_dce': {sname(i): round(dce[DEPTH][i], 6) for i in srcs},
           'interaction_matrix': {f'{sname(i)}|{sname(j)}':
                                  round(v['I'], 6)
                                  for (i, j), v in inter.items()},
           'joint_dce_matrix': {f'{sname(i)}|{sname(j)}':
                                round(v['dce_joint'], 6)
                                for (i, j), v in inter.items()},
           'top15_superadditive': top_pairs(inter, 15, True),
           'top15_subadditive': top_pairs(inter, 15, False),
           'n_pairs': len(inter)}
    E.merge(JP, 'edge_tier_readout', rec)


# ---------------- analysis --------------------------------------------------
def _iter_inter(rec):
    name2si = {sname(i): i for i in range(NS)}
    for k, v in rec['interaction_matrix'].items():
        a, b = k.split('|')
        yield name2si[a], name2si[b], v


def analysis(m, dce):
    j = E.loadj(JP)
    if 'analysis' in j:
        print("analysis: done -- skip", flush=True)
        return
    mod, edg = j['module_tier'], j['edge_tier_readout']
    out = {}
    # additivity fractions (registered prediction iii)
    for nm, rec in (('module', mod), ('edge_readout', edg)):
        Is = [v for _, _, v in _iter_inter(rec)]
        out[f'{nm}_additive_frac_absI_lt_{ADD_TOL}'] = round(
            sum(1 for x in Is if abs(x) < ADD_TOL) / len(Is), 4)
        out[f'{nm}_n_superadditive_gt_{SUPER_TOL}'] = sum(
            1 for x in Is if x > SUPER_TOL)
        out[f'{nm}_n_subadditive_lt_-{SUPER_TOL}'] = sum(
            1 for x in Is if x < -SUPER_TOL)
    allI = [v for rec in (mod, edg) for _, _, v in _iter_inter(rec)]
    out['combined_additive_frac'] = round(
        sum(1 for x in allI if abs(x) < ADD_TOL) / len(allI), 4)
    # prediction (i): type/depth structure of superadditive pairs (module
    # tier, writer-writer pairs only)
    ww = [(i, jx, v) for i, jx, v in _iter_inter(mod) if i > 0 and jx > 0]
    sup = [(i, jx, v) for i, jx, v in ww if v > SUPER_TOL]

    def frac(pairs, pred):
        return round(sum(1 for p in pairs if pred(p)) / len(pairs), 4) \
            if pairs else None
    is_mm = lambda p: kind_of(p[0]) == 'mlp' and kind_of(p[1]) == 'mlp'
    is_adj = lambda p: abs(depth_of(p[0]) - depth_of(p[1])) <= 1
    out['pred_i_type_depth'] = {
        'n_superadditive_writer_pairs': len(sup),
        'superadditive_frac_mlp_mlp': frac(sup, is_mm),
        'base_frac_mlp_mlp': frac(ww, is_mm),
        'superadditive_frac_adjacent_depth': frac(sup, is_adj),
        'base_frac_adjacent_depth': frac(ww, is_adj)}
    # prediction (ii): broadcast-cast pairs' positive-interaction ranks
    pos_sorted = sorted(_iter_inter(mod), key=lambda t: -t[2])
    rank = {(min(i, jx), max(i, jx)): r + 1
            for r, (i, jx, _) in enumerate(pos_sorted)}
    bset = sorted(BROADCAST.values())
    brows = []
    for a, i in enumerate(bset):
        for jx in bset[a + 1:]:
            if (i, jx) in rank:
                brows.append({'pair': f'{sname(i)}+{sname(jx)}',
                              'I': mod['interaction_matrix'][
                                  f'{sname(i)}|{sname(jx)}'],
                              'rank_among_positive_desc': rank[(i, jx)],
                              'n_pairs': len(pos_sorted)})
    out['pred_ii_broadcast_cast_pairs'] = brows
    # first-order vs interaction-aware ranking (both tiers)
    ws = E.weight_support(m)
    name2si = {sname(i): i for i in range(NS)}

    def ranking_block(rec, singles, sup_of):
        srcs = [name2si[s] for s in rec['sources'] if name2si[s] > 0]
        Imap = {}
        for i, jx, v in _iter_inter(rec):
            Imap.setdefault(i, {})[jx] = v
            Imap.setdefault(jx, {})[i] = v
        first = [singles[i] for i in srcs]
        adj = [singles[i] + 0.5 * sum(Imap.get(i, {}).values())
               for i in srcs]
        supv = [sup_of(i) for i in srcs]
        o = {'sources': [sname(i) for i in srcs],
             'first_order': [round(x, 6) for x in first],
             'interaction_adjusted': [round(x, 6) for x in adj],
             'spearman_weight_vs_first_order':
                 round(R2.spearman(supv, first), 4),
             'spearman_weight_vs_adjusted':
                 round(R2.spearman(supv, adj), 4),
             'spearman_first_order_vs_adjusted':
                 round(R2.spearman(first, adj), 4),
             'top5_first_order': [sname(srcs[k]) for k in
                                  np.argsort(first)[::-1][:5]],
             'top5_adjusted': [sname(srcs[k]) for k in
                               np.argsort(adj)[::-1][:5]]}
        return o
    out['ranking_edge_readout'] = ranking_block(
        edg, {i: dce[DEPTH][i] for i in dce[DEPTH]},
        lambda i: ws[DEPTH][i])
    mod_singles = {name2si[s]: v for s, v in mod['singles_dce'].items()}
    out['ranking_module'] = ranking_block(
        mod, mod_singles,
        lambda i: math.sqrt(sum(ws[li][i] ** 2 for li in ws
                                if i in ws[li])))
    # does the causal ground truth change MATERIALLY? (R1: Spearman SE ~0.08
    # at n=156, so a shift below that is a tie, not a change)
    for nm in ('ranking_edge_readout', 'ranking_module'):
        b = out[nm]
        d = (b['spearman_weight_vs_adjusted']
             - b['spearman_weight_vs_first_order'])
        b['delta_spearman_adjusted_minus_first_order'] = round(d, 4)
        b['material_change_abs_delta_gt_R1_SE_0.08'] = bool(abs(d) > 0.08)
        b['top5_reordered'] = bool(b['top5_first_order'] != b['top5_adjusted'])
    if not E.SMOKE:
        lp = E.loadj(E.jpath('qk_e9.json'))['light_probe_E9a']
        out['stored_light_probe_E9a_reference'] = {
            'wiring_spearman_all_156_edges': lp['wiring_spearman_all'],
            'wiring_spearman_effectual': lp['wiring_spearman_effectual'],
            'wiring_top10_precision': lp['wiring_top10_precision'],
            'note': 'the headline readability metric runs over all 156 '
                    'consumer-source edges; E26 tier (b) rescores the READOUT '
                    'ROW only (24 edges) and tier (a) the 25 module-level '
                    'sources, so these numbers anchor scale, not equality'}
    out['note'] = ('adj(i) = dCE(i) + 0.5 * sum_j I(i,j), the second-order '
                   'Shapley-style correction; weight table = '
                   'weight_support Frobenius norms (readout row for the '
                   'edge tier, RMS-combined over consumers for the module '
                   'tier)')
    E.merge(JP, 'analysis', out)
    print(json.dumps(out, indent=2, default=str)[:2000], flush=True)


def verdicts():
    j = E.loadj(JP)
    if 'verdicts' in j or E.SMOKE:
        return
    a = j['analysis']
    p1 = a['pred_i_type_depth']
    v1 = (None if p1['n_superadditive_writer_pairs'] == 0 else
          bool((p1['superadditive_frac_mlp_mlp'] or 0)
               > (p1['base_frac_mlp_mlp'] or 0)
               and (p1['superadditive_frac_adjacent_depth'] or 0)
               > (p1['base_frac_adjacent_depth'] or 0)))
    br = a['pred_ii_broadcast_cast_pairs']
    v2 = bool(br and min(b['rank_among_positive_desc'] for b in br) <= 15)
    v3 = bool(a['combined_additive_frac'] >= 0.70)
    mat = bool(a['ranking_edge_readout'][
                   'material_change_abs_delta_gt_R1_SE_0.08']
               or a['ranking_module'][
                   'material_change_abs_delta_gt_R1_SE_0.08'])
    E.merge(JP, 'verdicts', {
        'pred_i_superadditive_concentrate_mlpmlp_adjacent': v1,
        'pred_ii_broadcast_cast_among_largest_positive': v2,
        'pred_iii_ge70pct_near_additive': v3,
        'combined_additive_frac': a['combined_additive_frac'],
        'causal_ground_truth_changes_materially': mat,
        'consequence': (
            'second-order correction moves the weight-vs-causal Spearman by '
            'more than the R1 sampling SE (0.08) -- first-order single '
            'ablations are NOT an adequate causal ground truth and every '
            'readability number resting on them needs the interaction-'
            'adjusted version' if mat else
            'second-order correction leaves the weight-vs-causal Spearman '
            'inside the R1 sampling SE (0.08) -- first-order single '
            'ablations stand as the causal ground truth at this width')})


if __name__ == '__main__':
    E.setup()
    if E.SMOKE:
        R2.T = Q.T
    if 'registered_predictions' not in E.loadj(JP):
        E.merge(JP, 'registered_predictions', {
            'registered_before_computation': True,
            'i': f'superadditive pairs (I > {SUPER_TOL}) concentrate among '
                 'same-type (mlp-mlp) and adjacent-depth module pairs',
            'ii': 'the S2 write-lasso broadcast cast (mlp11/mlp0/attn9/'
                  'mlp10) shows the largest positive interactions',
            'iii': f'>= 70% of pairs are near-additive (|I| < {ADD_TOL})',
            'budget': '25 module singles + 300 module pairs + 276 readout '
                      'edge pairs + 169 recomputed singles (gate)'})
    if E.SMOKE:
        m = E7R.make_e7m1().eval().float()
    else:
        assert os.path.exists(E.ckpath('qk_e9_a')), 'qk_e9_a.pt missing'
        m, _ = E.load_arm('qk_e9_a', E7R.make_e7m1)
    identity_control(m)
    t0 = time.time()
    base, means, dce = singles_and_gate(m)
    print(f"singles + gate done in {time.time() - t0:.0f}s", flush=True)
    module_tier(m, means, base, dce)
    edge_tier(m, means, base, dce)
    analysis(m, dce)
    verdicts()
    print('e26 pairablate run done', flush=True)
