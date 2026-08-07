"""E33 COMPOSITION-ARM SEED REPLICATES (training; results -> qk_e33.json).

WHY. The program leader is the composition stack E31a (qk_e31a_a.pt: bandwidth
architecture s=15 + in-loss group-lasso 1e-4 + named attention predicates +
variable-k codebook slots), fresh held CE 4.9785 -- and it has exactly ONE
seed, so by the E29/R4 rule it cannot enter any recommendation. Its predicate
parent E22a has three (4.9000 +- 0.0068, qk_e29.json). This run trains seeds 1
and 2 of the composition arm and takes it to n = 3 on all three reported axes
(fresh held CE, plain wiring Spearman, covariance-composed wiring Spearman).

WHAT IS TRAINED: two runs, machinery VERBATIM from qk_e31a_compose_run
(E31Route = variable-k codebook slots + predicate-basis attention on the E19a
base; its trainer, its group-lasso 1e-4, its Muon/AdamW split, its probe path),
parameterized by Q.SEED exactly as qk_e29_threeseed_run parameterized the other
three arms:
  seed 1 -> qk_e33_compose_s1,  seed 2 -> qk_e33_compose_s2
  existing: seed 0 = qk_e31a_a (qk_e31a.json)
WHAT VARIES: Q.SEED only. Q.SEED is consumed by the model factory alone; the
data order comes from the separate, unchanged Q.DATA_SEED = 1234, so all three
runs see the identical 132,000-row single-epoch stream in the identical order.
These are therefore INIT-sensitivity spreads and UNDERSTATE run-to-run spread.
(The codebook initialization uses its own fixed-seed generator inside E20Route,
so the seed moves the shared network parameters, not the codebook draw.)

POSITIVE CONTROLS BEFORE TRAINING (qk_e29_threeseed_run functions verbatim):
  1. CONFIG EQUALITY vs the stored E31a record, field by field (lr,
     group_coeff, width, batch, steps, epochs, corpus, held_fresh);
  2. SEED-0 REPRODUCTION + SEED MOVES: 3 steps through THIS runner's training
     call at Q.SEED = 0 reproduce E31a's stored step-0 training CE at its
     stored 4-dp resolution, and 3 steps at each new seed differ;
  3. AFTER TRAINING: the saved checkpoint's config dict equals E31a's on every
     key except SEED;
  4. the E31a-inherited gate chain is asserted (qk_e18 probe gates 1+2,
     qk_e22 control2 match kernels, qk_e20b planted-toy) via
     qk_e31a_compose_run.control_preconditions.

REGISTERED PREDICTIONS (merged before training):
  (i)   composition CE sample sd <= 0.015 nats (the E29 CE-stability rule);
  (ii)  the composition arm does NOT beat the predicate-basis arm on CE: the
        seed-0 gap of +0.0828 nats SURVIVES at more than 3 pooled sd, i.e.
        composing the codebook onto the predicate basis genuinely COSTS CE
        relative to the predicate basis alone;
  (iii) composition readability sample sd <= 0.04 on both Spearman axes (in
        E29 the newer architectures were the stable ones: frontier 0.032 /
        0.026, predicate-basis 0.050 / 0.029, against the recipe's 0.072).

STATISTICS: per axis n / mean / sample sd / 95% CI of the mean at n-1 df, and
against the predicate-basis arm's three seeds (qk_e29.json) an EXACT two-sided
permutation test over all label assignments plus an EXACT paired sign-flip test
on the shared seeds -- qk_e29_threeseed_run.mean_ci / exact_perm_test /
exact_signflip_test verbatim, no normal approximations.

Standing logging applies: train curve every 200 steps, held-100 curve,
per-sequence heldloss npy, wiring-trajectory npz, per-seed codebook snapshots
and dead-code event logs, old-cooc held record, paired deltas with
sequence-clustered SEs, seed + data-order ids. Idempotent on qk_e33.json keys
and on checkpoints; smoke-gated via QK_SMOKE=1."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math

import numpy as np

import qk_e_common as E
from qk_e_common import Q, DEPTH, torch
import qk_e7_evenout_run as E7R
import qk_e15_reinvest_run as E15R
import qk_e20_codebook_run as E20R
import qk_e29_threeseed_run as E29R           # controls + exact statistics
import qk_e31a_compose_run as E31aR           # the composition arm itself

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e33.json')
SEED0 = 0
NG = E.NGROUP
CE_SD_PRED = 0.015
SP_SD_PRED = 0.04
GAP_SEED0 = 0.0828                             # E31a minus E22a, paired
NEW_SEEDS = [1, 2]
STEM_FMT = 'qk_e33_compose_s{}'
T95 = E29R.T95


# ---------------- per-seed training-time logging (E31a's, per seed) --------
_SNAP = {}


def step_cb_for(seed):
    stem = STEM_FMT.format(seed)

    def cb(step, model):
        if not getattr(model, 'qz_on', False):
            return
        ev = getattr(model, 'qz_dead_events', None)
        if ev:
            with open(E.jpath(f'{stem}_deadcode_events.jsonl'), 'a') as f:
                for e_ in ev:
                    e_['phase'] = 'train'
                    f.write(json.dumps(e_) + '\n')
            model.qz_dead_events = []
        if step % 1000 == 0:
            _SNAP.setdefault(stem, {})[f'step{step:05d}'] = \
                model.qz_codebook.detach().cpu().numpy()
            np.savez(E.jpath(f'{stem}_codebook_snapshots.npz'), **_SNAP[stem])
    cb.every = E31aR.qz_step_cb.every
    return cb


def trainer_for(seed):
    """qk_e31a_compose_run.trainer with the codebook logging redirected to
    this seed's files (identical training mathematics)."""
    cb = step_cb_for(seed)

    def trainer(lr, gc, steps, **kw):
        kw.setdefault('step_cb', cb)
        return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)
    return trainer


def arch(s_c):
    return dict(
        name='composition', gc=E31aR.GC31, dims=[s_c] * NG, slot=s_c,
        factory=(lambda: E31aR.make_e31(s=s_c)), trainer=trainer_for(SEED0),
        parent_json='qk_e31a.json', parent_key='E31a',
        design='predicate-basis attention (signed positional profile + b_h '
               'MATCH_prev + c_h MATCH_same, AdamW, init 0) composed with '
               'variable-k codebook slots (n=256 per slot, k=4 attention / '
               'k=2 MLP) on the E19a bandwidth base (s=15, in-loss '
               'group-lasso 1e-4, Muon 0.02), verbatim qk_e31a_compose_run '
               'arm E31a',
        new_seeds=NEW_SEEDS, stem_fmt=STEM_FMT,
        existing=[(0, 'qk_e31a_a', 'qk_e31a.json', 'E31a',
                   'light_probe_E31a_var_dims',
                   ('qk_e31a.json', 'composed_wiring_E31a'))])


# ---------------- named-term mass per seed (cheap, weights only) -----------
def bmass(tag, stem, s_c):
    key = f'named_term_mass_{tag}'
    if E.SMOKE or key in E.loadj(JP) or not os.path.exists(E.ckpath(stem)):
        return
    ck = torch.load(E.ckpath(stem), map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    b = sd['pred_b'].float()
    c = sd['pred_c'].float()
    pr = sd['pred_prof'].float()
    E.merge(JP, key, {
        'checkpoint': f'{stem}.pt',
        'total_abs_b_mass': round(float(b.abs().sum()), 5),
        'total_abs_c_mass': round(float(c.abs().sum()), 5),
        'profile_l2_norm': round(float(pr.norm()), 5),
        'reference_E31a_seed0_total_abs_b_mass':
            E.loadj(E.jpath('qk_e31a.json'))
            .get('mixture_weights_E31a', {}).get('total_abs_b_mass'),
        'note': 'named MATCH_prev / MATCH_same coefficient mass straight off '
                'the checkpoint (no forward); the E31a seed-0 value came '
                'through qk_e31a_compose_run.mixture_tables'})


# ---------------- collection + statistics ----------------
def collect(a):
    j = E.loadj(JP)
    rows = {}
    for seed, stem, pj, pkey, plp, (cwj, cwk) in a['existing']:
        pjj = E.loadj(E.jpath(pj))
        node = E.loadj(E.jpath(cwj))
        node = node if cwk is None else node.get(cwk, {})
        rows[seed] = {
            'seed': seed, 'checkpoint': f'{stem}.pt', 'source': pj,
            'ce': pjj.get(pkey, {}).get('final_held_ce_fresh_bf16'),
            'plain': pjj.get(plp, {}).get('wiring_spearman_all'),
            'cov': (node.get('tables', {}).get('cov_composed', {})
                    .get('spearman_all'))}
    for seed in a['new_seeds']:
        tag = f"compose_s{seed}"
        stem = a['stem_fmt'].format(seed)
        rows[seed] = {
            'seed': seed, 'checkpoint': f'{stem}.pt', 'source': 'qk_e33.json',
            'ce': j.get(tag, {}).get('final_held_ce_fresh_bf16'),
            'plain': j.get(f'light_probe_{tag}_var_dims', {}).get(
                'wiring_spearman_all'),
            'cov': j.get(f'composed_wiring_{tag}', {}).get('tables', {})
                    .get('cov_composed', {}).get('spearman_all')}
    return rows


def predbasis_rows():
    """The predicate-basis arm's three seeds, straight out of qk_e29.json."""
    s = E.loadj(E.jpath('qk_e29.json')).get('summary_E29', {})
    per = s.get('per_arm', {}).get('predicate_basis', {}).get('per_seed', {})
    return {int(k): v for k, v in per.items()}


def summarize(a):
    rows = collect(a)
    pb = predbasis_rows()
    stats, pbstats = {}, {}
    for axis in ('ce', 'plain', 'cov'):
        vals = [rows[s][axis] for s in sorted(rows)
                if rows[s].get(axis) is not None]
        stats[axis] = E29R.mean_ci(vals)
        stats[f'{axis}_seeds'] = [s for s in sorted(rows)
                                  if rows[s].get(axis) is not None]
        pvals = [pb[s][axis] for s in sorted(pb) if pb[s].get(axis) is not None]
        pbstats[axis] = E29R.mean_ci(pvals)
    pw = {}
    for axis in ('ce', 'plain', 'cov'):
        ax = [rows[s][axis] for s in sorted(rows)
              if rows[s].get(axis) is not None]
        ay = [pb[s][axis] for s in sorted(pb) if pb[s].get(axis) is not None]
        if len(ax) < 2 or len(ay) < 2:
            pw[axis] = {'note': 'not enough seeds'}
            continue
        b = {'unpaired_exact': E29R.exact_perm_test(ax, ay)}
        common = sorted(s for s in rows if s in pb
                        and rows[s].get(axis) is not None
                        and pb[s].get(axis) is not None)
        if len(common) >= 2:
            b['paired_exact_on_shared_seeds'] = dict(
                E29R.exact_signflip_test(
                    [rows[s][axis] - pb[s][axis] for s in common]),
                shared_seeds=common)
        pooled = math.sqrt((np.var(ax, ddof=1) * (len(ax) - 1)
                            + np.var(ay, ddof=1) * (len(ay) - 1))
                           / (len(ax) + len(ay) - 2))
        d = float(np.mean(ax) - np.mean(ay))
        b['pooled_sd'] = round(pooled, 5)
        b['diff_in_pooled_sds'] = round(d / pooled, 3) if pooled > 0 else None
        b['survives_seed_variation_gt_2_pooled_sd'] = bool(
            pooled > 0 and abs(d) > 2 * pooled)
        pw[axis] = b
    pred = {}
    if stats['ce'] and stats['ce'].get('sd') is not None:
        pred['i_composition_ce_sd_le_0.015'] = {
            'ce_sd': stats['ce']['sd'], 'n': stats['ce']['n'],
            'verdict': bool(stats['ce']['sd'] <= CE_SD_PRED)}
    sp = {ax: stats[ax]['sd'] for ax in ('plain', 'cov')
          if stats[ax] and stats[ax].get('sd') is not None}
    if sp:
        pred['iii_composition_readability_sd_le_0.04'] = {
            'per_axis_sd': sp, 'verdict': bool(max(sp.values()) <= SP_SD_PRED)}
    if isinstance(pw.get('ce'), dict) and 'pooled_sd' in pw['ce']:
        d = pw['ce']['diff_in_pooled_sds'] or 0.0
        pred['ii_composition_does_not_beat_predicate_basis_on_ce'] = {
            'sign_convention': 'diff = mean CE(composition) - mean '
                               'CE(predicate basis); POSITIVE means the '
                               'composition is WORSE (higher CE), which is '
                               'what was predicted',
            'diff_of_means_nats': pw['ce']['unpaired_exact']['diff_of_means'],
            'diff_in_pooled_sds': pw['ce']['diff_in_pooled_sds'],
            'pooled_sd': pw['ce']['pooled_sd'],
            'p_exact_unpaired': pw['ce']['unpaired_exact'][
                'p_two_sided_exact'],
            'seed0_reference_gap': GAP_SEED0,
            'verdict': bool(d > 3.0)}
    out = {
        'design': a['design'],
        'what_varied': 'init seed only (Q.SEED); data order held FIXED via '
                       f'the unchanged Q.DATA_SEED = {Q.DATA_SEED}, so these '
                       'spreads isolate INIT sensitivity and UNDERSTATE full '
                       'run-to-run spread',
        'per_seed': rows, 'stats': stats,
        'predicate_basis_reference': {'per_seed': pb, 'stats': pbstats,
                                      'source': 'qk_e29.json::summary_E29'},
        'pairwise_vs_predicate_basis': pw,
        'registered_prediction_verdicts': pred,
        'named_term_mass_per_seed': {
            k: E.loadj(JP)[k] for k in E.loadj(JP)
            if k.startswith('named_term_mass_')},
        'note': 'with n = 3 vs 3 the smallest attainable two-sided exact '
                'permutation p is 2/20 = 0.10, so nothing here can be '
                'significant at 0.05 by construction; read the pooled-sd '
                'column with it'}
    E.merge(JP, 'summary_E33', out)
    print(json.dumps({'stats': stats, 'pairwise_vs_predicate_basis': pw,
                      'registered_prediction_verdicts': pred}, indent=2,
                     default=str), flush=True)


# ---------------- main ----------------
if __name__ == '__main__':
    E.setup()
    E.DEV = 'cpu' if E.SMOKE else 'cuda'
    mlr = E7R.muon_lr()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c
        assert os.path.exists(E.ckpath('qk_e31a_a')), 'qk_e31a_a.pt missing'
        for st in ('qk_e22_a', 'qk_e29_predbasis_s1', 'qk_e29_predbasis_s2'):
            assert os.path.exists(E.ckpath(st)), f'{st}.pt missing'
    a = arch(s_c)

    if 'E33_prediction' not in E.loadj(JP):
        E.merge(JP, 'E33_prediction', {
            'registered_before_training': True,
            'arm': 'composition (E31a): predicate-basis attention + '
                   'variable-k codebook slots on the bandwidth base',
            'new_seeds': NEW_SEEDS, 'existing_seeds': [0],
            'i': f'composition CE sample sd <= {CE_SD_PRED} nats',
            'ii': 'the composition arm does NOT beat the predicate-basis arm '
                  f'on CE: the seed-0 gap of +{GAP_SEED0} nats survives at '
                  'more than 3 pooled sd (composition genuinely costs CE '
                  'relative to the predicate basis alone)',
            'iii': f'composition readability sample sd <= {SP_SD_PRED} on '
                   'both wiring-Spearman axes (in qk_e29 the newer '
                   'architectures were the stable ones)',
            'design': 'init seed only; data order FIXED via the unchanged '
                      f'Q.DATA_SEED = {Q.DATA_SEED}'})
    if 'E33_config' not in E.loadj(JP):
        E.merge(JP, 'E33_config', {
            'muon_lr': mlr, 'adamw_lr': E.get_lr(), 'group_coeff': a['gc'],
            'slot': s_c, 'data_seed': Q.DATA_SEED,
            'stems': [a['stem_fmt'].format(s) for s in NEW_SEEDS],
            'trainer': 'qk_e31a_compose_run.trainer verbatim (E.train_muon '
                       'with the codebook step callback), with the codebook '
                       'snapshot / dead-code logs redirected per seed',
            'factory': 'qk_e31a_compose_run.make_e31(s=15)',
            'probe_path': 'qk_e31a_compose_run.probes (quantized + predicate '
                          'forward), parameterized by checkpoint'})

    # ---- controls, before any training ----
    E31aR.control_preconditions()
    E29R.control_config_equality(a, mlr)
    # the gate's 3-step runs use the same training call with the codebook
    # LOGGING callback dropped (it writes snapshot files and does not touch
    # the optimizer state), so the gate leaves no stray per-seed artefacts.
    a['trainer'] = (lambda lr, gc, steps, **kw:
                    E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw))
    E29R.control_seed_gate(a, mlr)

    # ---- train + measure, seed by seed ----
    for seed in NEW_SEEDS:
        tag = f'compose_s{seed}'
        stem = a['stem_fmt'].format(seed)
        E29R.set_seed(seed)
        print(f"=== Q.SEED {seed} for {tag} (data order unchanged, "
              f"DATA_SEED {Q.DATA_SEED}) ===", flush=True)
        try:
            E.train_arm(stem, JP, tag, a['factory'], a['gc'], lr=mlr,
                        trainer=trainer_for(seed),
                        extra={'optimizer': 'muon', 'seed': seed,
                               'data_seed': Q.DATA_SEED, 'slot': s_c,
                               'arm': 'composition',
                               'parent': 'E31a (qk_e31a_a, seed 0)',
                               'design': a['design'],
                               'varies_from_parent': 'init seed ONLY; data '
                                                     'order identical by '
                                                     'construction'})
            E29R.control_ckpt_config(a, seed, stem)
            E.oldheld_record(stem, a['factory'], JP, f'{tag}_oldheld')
            E.paired_fresh(stem, JP, tag)
            if not E.SMOKE:
                for pstem, lab in (('qk_e31a_a', 'seed0'),
                                   ('qk_e22_a', 'e22a')):
                    fa, fb = f'{stem}_heldloss.npy', f'{pstem}_heldloss.npy'
                    if os.path.exists(f'{E.QK}/{fa}') \
                            and os.path.exists(f'{E.QK}/{fb}') \
                            and f'{tag}_minus_{lab}' not in E.loadj(JP):
                        E.merge(JP, f'{tag}_minus_{lab}',
                                dict(E.paired(fa, fb, len(Q.HELD), lab),
                                     parent=pstem,
                                     note='paired per-token fresh held CE, '
                                          'sequence-clustered SE'))
                E20R.save_seq_heldloss(stem)
                tp = f'{E.QK}/{stem}_traj.npz'
                if os.path.exists(tp) and f'{tag}_traj' not in E.loadj(JP):
                    z = np.load(tp)
                    E.merge(JP, f'{tag}_traj',
                            {'path': os.path.basename(tp),
                             'n_snapshots': int(len(z['step'])),
                             'every': E.TRAJ_EVERY, 'keys': sorted(z.files)})
                bmass(tag, stem, s_c)
            E31aR.probes(s_c, stem=stem, jp=JP, tag=tag)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            E.merge(JP, f'{tag}_FAILED',
                    {'error': f'{type(ex).__name__}: {str(ex)[:300]}'})
            if not E.SMOKE:
                torch.cuda.empty_cache()
        finally:
            E29R.set_seed(SEED0)

    if not E.SMOKE:
        bmass('compose_s0', 'qk_e31a_a', s_c)
    summarize(a)
    print('e33 compose seeds run done', flush=True)
