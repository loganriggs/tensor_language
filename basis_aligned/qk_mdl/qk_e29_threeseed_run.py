"""E29 MULTI-SEED PROTOCOL ON THE THREE ARMS THAT MATTER (training; results ->
qk_e29.json).

WHY (BRAINSTORM_STATE "FOUNDATIONS CORRECTION 2026-08-07", item a): E27's two
seed replicates showed CE is seed-stable (+0.019 / -0.012) but the
covariance-composed wiring Spearman moved 0.128 on the recipe arm and the
recipe-vs-frontier readability ORDERING REVERSED between seeds. With n = 2 per
arm nothing can be tested. This run takes the readability axis to n = 4 seeds
on the two recommendation arms and n = 3 on the best-CE arm, and reports which
frontier differences SURVIVE seed variation.

WHAT IS TRAINED (six runs; every other thing verbatim from the parent):
  frontier   (E15c bandwidth-reinvestment @ in-loss group-lasso 1e-4)
             seeds 2, 3   -> qk_e29_frontier_s2/_s3
             existing: seed 0 = qk_e19_a (E19a), seed 1 = qk_e27_frontier_s1
  recipe     (per-slot RMSNorm + Muon + in-loss group-lasso 3e-5)
             seeds 2, 3   -> qk_e29_recipe_s2/_s3
             existing: seed 0 = qk_e9_a (E9a), seed 1 = qk_e27_recipe_s1
  predicate-basis (E22a: named signed positional profile + MATCH_prev +
             MATCH_same terms on the frontier architecture) -- the program's
             BEST CE (4.8957, -0.159 vs the recipe) and it has exactly ONE
             seed, so it cannot enter any recommendation.
             seeds 1, 2   -> qk_e29_predbasis_s1/_s2
             existing: seed 0 = qk_e22_a (E22a)

WHAT VARIES: Q.SEED only, exactly as in E27. Q.SEED is consumed ONLY by the
model factories; the data order comes from the SEPARATE, UNCHANGED
Q.DATA_SEED = 1234, so every run sees the identical 132,000-row single-epoch
stream in the identical order. These are therefore INIT-sensitivity spreads and
UNDERSTATE full run-to-run spread -- stated in the JSON, not buried here.

POSITIVE CONTROLS, BEFORE TRAINING (per architecture):
  1. CONFIG EQUALITY: lr / group_coeff / width / batch / steps / epochs /
     corpus / held_fresh must equal the stored parent record field for field.
  2. SEED-0 REPRODUCTION + SEED MOVES: 3 steps through THIS runner's training
     call at Q.SEED = 0 must reproduce the parent's stored step-0 training CE
     at its stored 4-dp resolution, and 3 steps at each NEW seed must differ.
  3. AFTER TRAINING: the saved checkpoint's config dict must equal the
     parent's on every key except SEED.

STATISTICS (the deliverable is the survival table):
  per arm x axis (fresh held CE, plain wiring Spearman, covariance-composed
  wiring Spearman): n, mean, sample sd, and the 95% CI of the MEAN using the
  t multiplier at n-1 df (12.706 / 4.303 / 3.182 for n = 2 / 3 / 4).
  pairwise between arms, on every axis: an EXACT two-sided permutation test
  over all group-label assignments (n is 3-4, so the exact test is cheap and
  makes no distributional assumption) AND, on the seeds that align across
  arms, an EXACT paired sign-flip test. No normal approximations are used.

REGISTERED PREDICTIONS (merged before training):
  (i)   CE sample sd <= 0.015 nats per arm;
  (ii)  wiring-Spearman sample sd >= 0.04 (i.e. E27's spread was not a fluke);
  (iii) the predicate-basis arm's CE advantage over the recipe (-0.159 at
        seed 0) survives at more than 3 pooled standard deviations.

Standing logging applies throughout: per-arm train curve every 200 steps,
held-100 curve, per-sequence heldloss npy, wiring-trajectory npz, old-cooc held
record, paired deltas with sequence-clustered SEs, seed + data-order ids.
Idempotent on qk_e29.json keys and on checkpoints; smoke-gated via QK_SMOKE=1.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import itertools
import json
import math

import numpy as np

import qk_e_common as E
from qk_e_common import Q, C, DEPTH, torch
import qk_e7_evenout_run as E7R
import qk_e9_compose_run as E9R             # muon_lasso_trainer (recipe)
import qk_e15_reinvest_run as E15R          # make_e15c
import qk_e19_dial_run as E19R              # frontier trainer + dial
import qk_e22_predbasis_run as E22R         # predicate-basis arch + trainer
import qk_e18_probe_upgrades as E18U        # generalized + cov-composed probes
import qk_e17_composed_wiring as E17
import qk_deeproute_train_2 as R2

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e29.json')
SEED0 = 0
GATE_TOL = 1e-3
NG = E.NGROUP
CE_SD_PRED = 0.015
SP_SD_PRED = 0.04
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
CFG_FIELDS = ('lr', 'group_coeff', 'width', 'batch', 'steps', 'epochs',
              'corpus', 'held_fresh')


def set_seed(s):
    Q.SEED = s


# ============================ arm definitions ===============================
def architectures(s_c):
    """One entry per ARCHITECTURE; 'existing' lists the (seed, stem, json,
    record key, light-probe key, composed-wiring location) of the checkpoints
    already on disk, 'new_seeds' the ones this run trains."""
    return [
        dict(name='frontier', gc=E19R.GC19, dims=[s_c] * NG, slot=s_c,
             factory=(lambda: E15R.make_e15c(s=s_c)), trainer=E19R.trainer,
             parent_json='qk_e19.json', parent_key='E19a',
             design='E15c bandwidth-reinvestment architecture at the '
                    'readability dial (in-loss group-lasso 1e-4), verbatim '
                    'qk_e19_dial_run arm E19a',
             new_seeds=[2, 3], stem_fmt='qk_e29_frontier_s{}',
             existing=[(0, 'qk_e19_a', 'qk_e19.json', 'E19a',
                        'light_probe_E19a_var_dims',
                        ('qk_e19.json', 'composed_wiring_E19a')),
                       (1, 'qk_e27_frontier_s1', 'qk_e27.json',
                        'E27a_frontier', 'light_probe_E27a_frontier',
                        ('qk_e27.json', 'composed_wiring_E27a_frontier'))]),
        dict(name='recipe', gc=E9R.GC9, dims=[E.SUB] * NG, slot=E.SUB,
             factory=E7R.make_e7m1, trainer=E9R.muon_lasso_trainer,
             parent_json='qk_e9.json', parent_key='E9a',
             design='readable recipe: partitioned write slots + per-slot '
                    'RMSNorm + Muon 0.02 (embedding AdamW) + in-loss '
                    'group-lasso 3e-5, verbatim qk_e9_compose_run arm E9a',
             new_seeds=[2, 3], stem_fmt='qk_e29_recipe_s{}',
             existing=[(0, 'qk_e9_a', 'qk_e9.json', 'E9a',
                        'light_probe_E9a', ('qk_e17.json', None)),
                       (1, 'qk_e27_recipe_s1', 'qk_e27.json', 'E27b_recipe',
                        'light_probe_E27b_recipe',
                        ('qk_e27.json', 'composed_wiring_E27b_recipe'))]),
        dict(name='predicate_basis', gc=E22R.GC22, dims=[s_c] * NG, slot=s_c,
             factory=(lambda: E22R.make_e22(s=s_c)), trainer=E22R.trainer,
             parent_json='qk_e22.json', parent_key='E22a',
             design='predicate-basis attention on the frontier architecture: '
                    'per-head signed positional profile + b_h MATCH_prev + '
                    'c_h MATCH_same (init 0, AdamW) + bilinear residual, '
                    'verbatim qk_e22_predbasis_run arm E22a',
             new_seeds=[1, 2], stem_fmt='qk_e29_predbasis_s{}',
             existing=[(0, 'qk_e22_a', 'qk_e22.json', 'E22a',
                        'light_probe_E22a_var_dims',
                        ('qk_e22.json', 'composed_wiring_E22a'))]),
    ]


def parent_rec(a):
    return E.loadj(E.jpath(a['parent_json']))[a['parent_key']]


# ============================ controls ======================================
def control_config_equality(a, mlr):
    key = f"control_config_equality_{a['name']}"
    if E.SMOKE or E.loadj(JP).get(key, {}).get('pass'):
        return
    p = parent_rec(a)
    mine = {'lr': mlr, 'group_coeff': a['gc'], 'width': Q.D,
            'batch': Q.BATCH, 'steps': E.V8T.STEPS, 'epochs': 1,
            'corpus': f'corpus_fresh shards prefix [{E.V8T.STEPS * Q.BATCH}], '
                      'single pass',
            'held_fresh': 'corpus_fresh/fresh34k.npy rows [33000:34500] '
                          '(pure eval corpus)'}
    diffs = {f: [p.get(f), mine[f]] for f in CFG_FIELDS if p.get(f) != mine[f]}
    rec = {'parent': a['parent_key'], 'parent_json': a['parent_json'],
           'fields_checked': list(CFG_FIELDS),
           'stored_parent_config': {f: p.get(f) for f in CFG_FIELDS},
           'this_run_config': mine, 'mismatches': diffs,
           'new_seeds': a['new_seeds'], 'data_seed_all': Q.DATA_SEED,
           'note': 'the ONLY intended difference is the init seed; the data '
                   'order comes from Q.DATA_SEED which is unchanged',
           'pass': bool(not diffs)}
    E.merge(JP, key, rec)
    print(f"{key}: {len(diffs)} mismatches -> "
          f"{'PASS' if rec['pass'] else 'FAIL'} {diffs}", flush=True)
    assert rec['pass'], f'{key} FAILED: {diffs}'


def three_step(a, mlr, seed):
    set_seed(seed)
    try:
        log = a['trainer'](mlr, a['gc'], 3, log_every=1, save=False,
                           factory=a['factory'])
    finally:
        set_seed(SEED0)
    return [x[1] for x in log['train_loss']], log['final_held_ce']


def control_seed_gate(a, mlr):
    key = f"control_seed_gate_{a['name']}"
    if E.SMOKE or E.loadj(JP).get(key, {}).get('pass'):
        return
    p = parent_rec(a)
    stored0 = p['train_curve_every200'][0]
    assert stored0[0] == 0, stored0
    ce0, held0 = three_step(a, mlr, SEED0)
    d_repro = abs(round(ce0[0], 4) - stored0[1])
    per_seed = {}
    ok_moved = True
    for s in a['new_seeds']:
        ces, held = three_step(a, mlr, s)
        moved = abs(ces[0] - ce0[0])
        per_seed[str(s)] = {'per_step_ce': ces, 'held100': held,
                            'step0_abs_diff_vs_seed0': moved}
        ok_moved = ok_moved and moved > 1e-4
    rec = {'stored_parent_step0_ce': stored0[1],
           'rerun_seed0_per_step_ce': ce0, 'rerun_seed0_held100': held0,
           'seed0_step0_abs_diff_vs_stored': d_repro,
           'new_seeds': per_seed, 'repro_tol': 1e-4,
           'note': 'control 2: the parent logs step-0 CE at 4 dp, so seed-0 '
                   'reproduction is asserted at that resolution (step-0 CE '
                   'depends only on the init and the first batch); plus the '
                   'guard that each new seed actually moves the init',
           'pass': bool(d_repro <= 1e-4 and ok_moved)}
    E.merge(JP, key, rec)
    print(f"{key}: seed0 step0 {ce0[0]:.4f} vs stored {stored0[1]} "
          f"(diff {d_repro:.1e}); new seeds moved "
          f"{[round(v['step0_abs_diff_vs_seed0'], 4) for v in per_seed.values()]}"
          f" -> {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_ckpt_config(a, seed, stem):
    key = f"control_ckpt_config_{a['name']}_s{seed}"
    if E.SMOKE or E.loadj(JP).get(key, {}).get('pass'):
        return
    parent_stem = a['existing'][0][1]
    if not (os.path.exists(E.ckpath(stem))
            and os.path.exists(E.ckpath(parent_stem))):
        return
    cn = torch.load(E.ckpath(stem), map_location='cpu',
                    weights_only=False)['config']
    cp = torch.load(E.ckpath(parent_stem), map_location='cpu',
                    weights_only=False)['config']
    keys = sorted(set(cn) | set(cp))
    diffs = {k: [cp.get(k), cn.get(k)] for k in keys if cp.get(k) != cn.get(k)}
    rec = {'parent_checkpoint': f'{parent_stem}.pt',
           'replicate_checkpoint': f'{stem}.pt',
           'differing_keys': sorted(diffs), 'differences': diffs,
           'pass': bool(sorted(diffs) == ['SEED']
                        and cp.get('SEED') == SEED0
                        and cn.get('SEED') == seed)}
    E.merge(JP, key, rec)
    print(f"{key}: differing keys {sorted(diffs)} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED: {diffs}'


# ============================ probes ========================================
def probe(a, seed, stem, tag):
    """The generalized variable-slot-dim light probe (plain Spearman) + the
    covariance-composed re-scoring (the standard reported metric) -- the exact
    E27 probe path, which qk_e18 gates 1+2 validated."""
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    dims = a['dims']
    lp_key, cw_key = f'light_probe_{tag}', f'composed_wiring_{tag}'
    if lp_key not in E.loadj(JP):
        print(f"{tag} light probe (dims {dims[0]} x {NG}) ...", flush=True)
        m, _ = E.load_arm(stem, a['factory'])
        Ws = m.wte.weight.shape[1]
        assert Ws == sum(dims), (Ws, sum(dims))
        base, dce = E18U.gen_consumption(m, Ws)
        wp = E18U.wpairs(m, dims)
        sup = E18U.score(E18U.gen_gram_table(m, dims), wp)
        cau = [dce[li][si] for li, si in wp]
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        agr = E17.agreement(sup, cau, eff)
        rel = None
        if all(d == E.SUB for d in dims):
            ws = E.weight_support(m)
            rel = max(abs(sup[i] - ws[li][si]) / max(abs(ws[li][si]), 1e-12)
                      for i, (li, si) in enumerate(wp))
            assert rel < 1e-6, f'probe equivalence FAILED ({rel})'
        totals = {str(jj): round(sum(dce[li].get(si, 0.0) for li in dce
                                     for si in (1 + 2 * jj, 2 + 2 * jj)
                                     if si in dce[li]), 5)
                  for jj in range(DEPTH)}
        srt = sorted([(li, si, dce[li][si]) for li in dce for si in dce[li]],
                     key=lambda p: -p[2])
        E.merge(JP, lp_key, {
            'checkpoint': f'{stem}.pt', 'seed': seed, 'slot_dims': dims,
            'stream_width': Ws, 'compute_width': Q.D,
            'base_ce_fp32_abl_oldheld': round(base, 5),
            'wiring_n_pairs': len(wp),
            'wiring_spearman_all': agr['spearman_all'],
            'wiring_n_effectual': len(eff),
            'wiring_spearman_effectual': agr['spearman_effectual'],
            'wiring_top10_precision': agr['top10_precision'],
            'per_block_total_consumption': totals,
            'dead_blocks_below_0.001': [b for b in totals if totals[b] < 0.001],
            'consumption_top20': [
                {'consumer': ('readout' if li == DEPTH else f'block{li}'),
                 'source': R2.stream_name(si), 'dce': round(v, 5)}
                for li, si, v in srt[:20]],
            'consumption_matrix': {str(li): {str(si): round(v, 6)
                                             for si, v in dce[li].items()}
                                   for li in dce},
            'weight_support_matrix': {
                str(li): {str(si): round(sup[i], 3)
                          for i, (l2, si) in enumerate(wp) if l2 == li}
                for li in range(DEPTH + 1)},
            'probe_equivalence_vs_weight_support_max_rel': rel,
            'note': 'generalized variable-slot-dim probe (qk_e18 gate 1); '
                    'causal = mean-ablation on old cooc held [:96], fp32'})
        print(f"{tag} plain wiring Spearman {agr['spearman_all']} "
              f"effectual({len(eff)}) {agr['spearman_effectual']}", flush=True)
        del m
        torch.cuda.empty_cache()
    if cw_key not in E.loadj(JP):
        lp = E.loadj(JP)[lp_key]
        print(f"{tag} covariance-composed re-scoring ...", flush=True)
        m, _ = E.load_arm(stem, a['factory'])
        wp = E18U.wpairs(m, dims)
        cau = E18U.stored_cau(lp, wp)
        tables, meta, _ = E18U.composed_tables(m, dims, cau, wp, E.DEV,
                                               remnant=False)
        chk = abs(tables['plain']['spearman_all'] - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, f'plain does not reproduce the probe ({chk})'
        rec = {'checkpoint': f'{stem}.pt', 'seed': seed,
               'slot_dims_uniform': dims[0],
               'plain_reproduction_abs_diff': round(chk, 6), 'tables': tables}
        rec.update(meta)
        E.merge(JP, cw_key, rec)
        print(f"{tag} plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
        del m
        torch.cuda.empty_cache()


# ============================ statistics ====================================
def mean_ci(vals):
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return None
    m = float(v.mean())
    if n == 1:
        return {'n': 1, 'mean': round(m, 5), 'sd': None, 'ci95_mean': None}
    sd = float(v.std(ddof=1))
    t = T95.get(n - 1, 1.96)
    h = t * sd / math.sqrt(n)
    return {'n': n, 'mean': round(m, 5), 'sd': round(sd, 5),
            'ci95_mean': [round(m - h, 5), round(m + h, 5)],
            't_multiplier': t, 'values': [round(float(x), 5) for x in v]}


def exact_perm_test(a, b):
    """Exact two-sided permutation test on the difference of means over ALL
    label assignments (n is 3-4 here, so this is the whole null distribution,
    not a sample)."""
    a, b = list(map(float, a)), list(map(float, b))
    pool = a + b
    na, n = len(a), len(a) + len(b)
    obs = np.mean(a) - np.mean(b)
    cnt = tot = 0
    for idx in itertools.combinations(range(n), na):
        s = set(idx)
        d = (np.mean([pool[i] for i in idx])
             - np.mean([pool[i] for i in range(n) if i not in s]))
        tot += 1
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return {'diff_of_means': round(float(obs), 5),
            'p_two_sided_exact': round(cnt / tot, 4),
            'n_permutations': tot, 'na': na, 'nb': len(b)}


def exact_signflip_test(d):
    """Exact two-sided paired test: all 2^n sign patterns of the paired
    differences."""
    d = [float(x) for x in d]
    n = len(d)
    obs = float(np.mean(d))
    cnt = 0
    for bits in itertools.product([1, -1], repeat=n):
        m = float(np.mean([b * x for b, x in zip(bits, d)]))
        if abs(m) >= abs(obs) - 1e-12:
            cnt += 1
    return {'mean_paired_diff': round(obs, 5),
            'p_two_sided_exact': round(cnt / (2 ** n), 4),
            'n_pairs': n, 'paired_diffs': [round(x, 5) for x in d]}


# ============================ collection ====================================
def collect(a):
    """Every seed of one architecture: CE + plain Spearman + cov-composed
    Spearman, from the stored records (existing seeds) and this run's."""
    j = E.loadj(JP)
    rows = {}
    for seed, stem, pj, pkey, plp, (cwj, cwk) in a['existing']:
        pjj = E.loadj(E.jpath(pj))
        r = {'seed': seed, 'checkpoint': f'{stem}.pt', 'source': pj,
             'ce': pjj.get(pkey, {}).get('final_held_ce_fresh_bf16'),
             'plain': pjj.get(plp, {}).get('wiring_spearman_all')}
        cwjj = E.loadj(E.jpath(cwj))
        node = cwjj if cwk is None else cwjj.get(cwk, {})
        r['cov'] = (node.get('tables', {}).get('cov_composed', {})
                    .get('spearman_all'))
        rows[seed] = r
    for seed in a['new_seeds']:
        tag = f"{a['name']}_s{seed}"
        stem = a['stem_fmt'].format(seed)
        r = {'seed': seed, 'checkpoint': f'{stem}.pt', 'source': 'qk_e29.json',
             'ce': j.get(tag, {}).get('final_held_ce_fresh_bf16'),
             'plain': j.get(f'light_probe_{tag}', {}).get(
                 'wiring_spearman_all'),
             'cov': j.get(f'composed_wiring_{tag}', {}).get('tables', {})
                     .get('cov_composed', {}).get('spearman_all')}
        rows[seed] = r
    return rows


def summarize(archs):
    per_arm, seeds_of = {}, {}
    for a in archs:
        rows = collect(a)
        seeds_of[a['name']] = rows
        stats = {}
        for axis in ('ce', 'plain', 'cov'):
            vals = [rows[s][axis] for s in sorted(rows)
                    if rows[s].get(axis) is not None]
            stats[axis] = mean_ci(vals)
            stats[f'{axis}_seeds'] = [s for s in sorted(rows)
                                      if rows[s].get(axis) is not None]
        per_arm[a['name']] = {'design': a['design'], 'per_seed': rows,
                              'stats': stats}
    # ---- pairwise arm comparisons on every axis ----
    names = [a['name'] for a in archs]
    pw = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            block = {}
            for axis in ('ce', 'plain', 'cov'):
                sx = seeds_of[x]
                sy = seeds_of[y]
                ax = [sx[s][axis] for s in sorted(sx)
                      if sx[s].get(axis) is not None]
                ay = [sy[s][axis] for s in sorted(sy)
                      if sy[s].get(axis) is not None]
                if len(ax) < 2 or len(ay) < 2:
                    block[axis] = {'note': 'not enough seeds'}
                    continue
                b = {'unpaired_exact': exact_perm_test(ax, ay)}
                common = sorted(s for s in sx if s in sy
                                and sx[s].get(axis) is not None
                                and sy[s].get(axis) is not None)
                if len(common) >= 2:
                    b['paired_exact_on_shared_seeds'] = dict(
                        exact_signflip_test(
                            [sx[s][axis] - sy[s][axis] for s in common]),
                        shared_seeds=common)
                pooled_sd = math.sqrt(
                    (np.var(ax, ddof=1) * (len(ax) - 1)
                     + np.var(ay, ddof=1) * (len(ay) - 1))
                    / (len(ax) + len(ay) - 2))
                d = float(np.mean(ax) - np.mean(ay))
                b['pooled_sd'] = round(pooled_sd, 5)
                b['diff_in_pooled_sds'] = (round(d / pooled_sd, 3)
                                           if pooled_sd > 0 else None)
                b['survives_seed_variation_gt_2_pooled_sd'] = bool(
                    pooled_sd > 0 and abs(d) > 2 * pooled_sd)
                block[axis] = b
            pw[f'{x}_vs_{y}'] = block

    # ---- registered predictions ----
    ce_sds = {n: per_arm[n]['stats']['ce']['sd'] for n in names
              if per_arm[n]['stats']['ce']
              and per_arm[n]['stats']['ce'].get('sd') is not None}
    sp_sds = {}
    for n in names:
        for axis in ('plain', 'cov'):
            st = per_arm[n]['stats'][axis]
            if st and st.get('sd') is not None:
                sp_sds[f'{n}_{axis}'] = st['sd']
    pred = {
        'i_ce_sd_le_0.015': {
            'per_arm_ce_sd': ce_sds,
            'verdict': (None if not ce_sds else
                        bool(max(ce_sds.values()) <= CE_SD_PRED))},
        'ii_spearman_sd_ge_0.04': {
            'per_arm_axis_sd': sp_sds,
            'verdict': (None if not sp_sds else
                        bool(min(sp_sds.values()) >= SP_SD_PRED)),
            'note': 'registered as "E27\'s spread was not a fluke": the '
                    'prediction is that EVERY arm/axis sd clears 0.04'},
    }
    pv = pw.get('recipe_vs_predicate_basis')
    if pv and isinstance(pv.get('ce'), dict) and 'pooled_sd' in pv['ce']:
        d = pv['ce']['diff_in_pooled_sds'] or 0.0
        pred['iii_predbasis_ce_advantage_gt_3sd'] = {
            'sign_convention': 'diff = mean CE(recipe) - mean CE(predicate '
                               'basis); POSITIVE means the predicate-basis '
                               'arm is better (lower CE)',
            'diff_of_means_nats': pv['ce']['unpaired_exact']['diff_of_means'],
            'diff_in_pooled_sds': pv['ce']['diff_in_pooled_sds'],
            'pooled_sd': pv['ce']['pooled_sd'],
            'p_exact_unpaired': pv['ce']['unpaired_exact'][
                'p_two_sided_exact'],
            'seed0_reference_delta': -0.159,
            'verdict': bool(d > 3.0)}

    # ---- the survival table (the deliverable) ----
    survival = {}
    for pair, block in pw.items():
        for axis, b in block.items():
            if not isinstance(b, dict) or 'unpaired_exact' not in b:
                continue
            survival[f'{pair}::{axis}'] = {
                'diff_of_means': b['unpaired_exact']['diff_of_means'],
                'p_exact_unpaired': b['unpaired_exact']['p_two_sided_exact'],
                'p_exact_paired': (b.get('paired_exact_on_shared_seeds') or {})
                    .get('p_two_sided_exact'),
                'diff_in_pooled_sds': b['diff_in_pooled_sds'],
                'SURVIVES': b['survives_seed_variation_gt_2_pooled_sd']}
    out = {'per_arm': per_arm, 'pairwise': pw,
           'registered_prediction_verdicts': pred,
           'survival_table': survival,
           'what_varied': 'init seed only (Q.SEED); data order held FIXED via '
                          f'the unchanged Q.DATA_SEED = {Q.DATA_SEED}, so '
                          'these spreads isolate INIT sensitivity and '
                          'UNDERSTATE full run-to-run spread',
           'seed_counts': {n: len(seeds_of[n]) for n in names},
           'note': 'SURVIVES = |difference of arm means| exceeds 2 pooled '
                   'standard deviations AND should be read with the exact '
                   'permutation p-value in the same row; with n = 3-4 the '
                   'smallest attainable two-sided exact p is 1/35 = 0.029 '
                   '(4 vs 3) or 2/70 = 0.029 (4 vs 4), so nothing here can '
                   'be significant at 0.01 by construction'}
    E.merge(JP, 'summary_E29', out)
    print(json.dumps({'survival_table': survival,
                      'registered_prediction_verdicts': pred}, indent=2,
                     default=str), flush=True)


# ============================ main ==========================================
if __name__ == '__main__':
    E.setup()
    E.DEV = 'cpu' if E.SMOKE else 'cuda'
    if not E.SMOKE:
        e18 = E.loadj(E.jpath('qk_e18.json'))
        assert e18.get('gate1_uniform11_weight_support', {}).get('pass') \
            and e18.get('gate2_cov_composed_E9a', {}).get('pass'), \
            'qk_e18.json gates 1+2 not passed -- reused probes unvalidated'
        for st in ('qk_e9_a', 'qk_e19_a', 'qk_e22_a', 'qk_e27_recipe_s1',
                   'qk_e27_frontier_s1'):
            assert os.path.exists(E.ckpath(st)), f'{st}.pt missing'

    mlr = E7R.muon_lr()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c
        # the predicate-basis seed-0 parent has NO stored wiring table (the
        # E22 run OOMed in its residual-census step before probe_e22a).
        # E30 normally repairs this first; run it here too so E29 is not
        # silently reduced to n = 2 on that arm if E30 did not.
        e22 = E.loadj(E.jpath('qk_e22.json'))
        if os.path.exists(E.ckpath('qk_e22_a')) and (
                'light_probe_E22a_var_dims' not in e22
                or 'composed_wiring_E22a' not in e22):
            print('repairing the missing E22a wiring table via '
                  'qk_e22_predbasis_run.probe_e22a ...', flush=True)
            E22R.probe_e22a(s_c)
            torch.cuda.empty_cache()
    A = architectures(s_c)

    if 'E29_prediction' not in E.loadj(JP):
        E.merge(JP, 'E29_prediction', {
            'registered_before_training': True,
            'finding': 'E27: CE is seed-stable but the covariance-composed '
                       'wiring Spearman moved 0.128 on the recipe arm and the '
                       'recipe-vs-frontier ORDERING reversed between seeds',
            'i': f'CE sample sd <= {CE_SD_PRED} nats per arm',
            'ii': f'wiring-Spearman sample sd >= {SP_SD_PRED} on every '
                  'arm/axis (E27\'s spread was not a fluke)',
            'iii': 'the predicate-basis arm\'s CE advantage over the recipe '
                   '(-0.159 at seed 0) survives at more than 3 pooled sd',
            'design': 'init seed only; data order FIXED via the unchanged '
                      f'Q.DATA_SEED = {Q.DATA_SEED}',
            'arms': {a['name']: {'new_seeds': a['new_seeds'],
                                 'existing_seeds': [e[0] for e in
                                                    a['existing']]}
                     for a in A}})
    if 'E29_config' not in E.loadj(JP):
        E.merge(JP, 'E29_config', {
            'muon_lr': mlr, 'adamw_lr': E.get_lr(), 'data_seed': Q.DATA_SEED,
            'slot_frontier': s_c, 'slot_recipe': E.SUB,
            'group_coeffs': {a['name']: a['gc'] for a in A},
            'stems': {a['name']: [a['stem_fmt'].format(s)
                                  for s in a['new_seeds']] for a in A},
            'trainers': {'frontier': 'qk_e19_dial_run.trainer verbatim',
                         'recipe': 'qk_e9_compose_run.muon_lasso_trainer '
                                   'verbatim',
                         'predicate_basis': 'qk_e22_predbasis_run.trainer '
                                            'verbatim'}})

    # ---- positive controls, before any training ----
    for a in A:
        control_config_equality(a, mlr)
        control_seed_gate(a, mlr)

    # ---- train + measure, seed by seed ----
    for a in A:
        for seed in a['new_seeds']:
            tag = f"{a['name']}_s{seed}"
            stem = a['stem_fmt'].format(seed)
            set_seed(seed)
            print(f"=== Q.SEED {seed} for {tag} (data order unchanged, "
                  f"DATA_SEED {Q.DATA_SEED}) ===", flush=True)
            try:
                E.train_arm(stem, JP, tag, a['factory'], a['gc'], lr=mlr,
                            trainer=a['trainer'],
                            extra={'optimizer': 'muon', 'seed': seed,
                                   'data_seed': Q.DATA_SEED,
                                   'slot': a['slot'], 'arm': a['name'],
                                   'parent': f"{a['parent_key']} "
                                             f"({a['existing'][0][1]}, "
                                             f"seed {SEED0})",
                                   'design': a['design'],
                                   'varies_from_parent': 'init seed ONLY; '
                                                         'data order '
                                                         'identical by '
                                                         'construction'})
                control_ckpt_config(a, seed, stem)
                E.oldheld_record(stem, a['factory'], JP, f'{tag}_oldheld')
                E.paired_fresh(stem, JP, tag)
                if not E.SMOKE:
                    pstem = a['existing'][0][1]
                    fa, fb = f'{stem}_heldloss.npy', f'{pstem}_heldloss.npy'
                    if os.path.exists(f'{E.QK}/{fa}') \
                            and os.path.exists(f'{E.QK}/{fb}') \
                            and f'{tag}_minus_seed0' not in E.loadj(JP):
                        E.merge(JP, f'{tag}_minus_seed0',
                                dict(E.paired(fa, fb, len(Q.HELD), 'seed0'),
                                     parent=pstem,
                                     note='paired per-token fresh held CE, '
                                          'sequence-clustered SE; this seed '
                                          'minus the seed-0 twin'))
                    p = f'{E.QK}/{stem}_heldloss.npy'
                    q = f'{E.QK}/{stem}_heldloss_seq.npy'
                    if os.path.exists(p) and not os.path.exists(q):
                        np.save(q, np.load(p).reshape(
                            len(Q.HELD), Q.T).mean(1))
                    tp = f'{E.QK}/{stem}_traj.npz'
                    if os.path.exists(tp) and f'{tag}_traj' not in E.loadj(JP):
                        z = np.load(tp)
                        E.merge(JP, f'{tag}_traj',
                                {'path': os.path.basename(tp),
                                 'n_snapshots': int(len(z['step'])),
                                 'every': E.TRAJ_EVERY,
                                 'keys': sorted(z.files)})
                probe(a, seed, stem, tag)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                E.merge(JP, f'{tag}_FAILED',
                        {'error': f'{type(ex).__name__}: {str(ex)[:300]}'})
                if not E.SMOKE:
                    torch.cuda.empty_cache()
            finally:
                set_seed(SEED0)

    summarize(A)
    print('e29 threeseed run done', flush=True)
