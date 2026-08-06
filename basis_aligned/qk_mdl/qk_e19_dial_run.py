"""E19 READABILITY DIAL ON THE CHEAP PARTITIONS (fresh single-epoch batch-16,
recipe conventions; results -> qk_e19.json).

CONTEXT (qk_e18.json): the two cheap-partition arms pay a real readability
price vs the readable recipe (covariance-composed Spearman: recipe 0.8575;
E16b shrinking-channel+floor 0.6617; E15c bandwidth reinvestment 0.6728 with
155/156 edges causally effectual -- a nearly dense graph). Standing hypothesis
(RESULTS_l0_mdl.md section 112 update): a STRONGER in-loss lasso can buy the
readability back at modest CE, because on the recipe the dial 3e-5 -> 1e-4
bought Spearman 0.60 -> 0.78 at scale for +0.12 CE.

ARMS (both: full fresh single-epoch protocol, group-lasso 1e-4 instead of the
parents' 3e-5; everything else IDENTICAL to the parent -- same factories,
verbatim, same Muon/AdamW lrs, same seed and data order):
  E19a  E15c architecture (true-small decoders, 24 x 15-dim slots, stream 360,
        compute 264) -- the E15cRoute factory from qk_e15_reinvest_run reused
        verbatim (make_e15c(s=15)).
  E19b  E16b architecture (shrinking embedding channel, 44-dim floor) -- the
        E16 factory from qk_e16_shrinkemb_run reused verbatim (make_e16b).

PAIRINGS: each arm vs its 3e-5 parent (qk_e15_c / qk_e16_b), vs E0a vanilla
and E0b (paired_fresh), and vs the recipe qk_e9_a -- all on the fresh held
per-token losses with seq-clustered SEs.

POSITIVE CONTROLS (before training):
  1. DIAL-ONLY control per arm: this runner's training call at the PARENT's
     group_coeff 3e-5 for 3 steps must reproduce a rerun of the parent's
     training invocation (the exact expression from the parent runner:
     E.train_muon(E7R.muon_lr(), GC_parent, steps, lr_adamw=E.get_lr(),
     factory=<parent factory>)) to float tolerance -- same seed, same
     epoch_order(0) data. Proves the ONLY change is the dial.
  2. Group-lasso penalty vs a naive per-group loop on both architectures
     (E15c's custom seg_ind penalty via the V8T dispatch; E16b's standard
     V8T.group_penalty), rel < 1e-6; plus dispatch identity for E15c.
  3. Precondition (non-smoke): qk_e18.json gates 1 and 2 passed (the
     generalized variable-slot-dim probe and the covariance-composed pipeline
     reproduce the uniform-11 recipe numbers exactly) -- the probes below
     reuse those gated functions.

REGISTERED PREDICTION (merged into qk_e19.json BEFORE training): E19a
covariance-composed Spearman >= 0.75 at CE <= 4.99 (still below the recipe's
5.0547) CONFIRMS the readability-preserving cheap partition; below 0.72
REFUTES the dial hypothesis on widened slots.

PROBES (per arm, after training): the GENERALIZED variable-slot-dim light
probe (qk_e18_probe_upgrades.gen_consumption / gen_gram_table / wpairs /
score, reused) for E19a at dims [15]*24; the E16-corrected light probe
(qk_e16_shrinkemb_run.e16_light_probe, per-consumer remnant stream-0 means)
plus the remnant token-recovery probe for E19b; and the covariance-composed
re-scoring (qk_e18_probe_upgrades.composed_tables, remnant=False for E19a /
True for E19b). Plain AND covariance-composed Spearman reported for both.
Idempotent on qk_e19.json keys and checkpoints."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, V8T, C, DEPTH, F, torch
import qk_e7_evenout_run as E7R
import qk_e15_reinvest_run as E15R        # E15cRoute + make_e15c + solve_slot_c
import qk_e16_shrinkemb_run as E16R       # make_e16b + e16_light_probe + remnant_probe
import qk_e18_probe_upgrades as E18U      # generalized probe + cov-composed (gated)
import qk_e17_composed_wiring as E17      # agreement (already imported by E18U)
import qk_deeproute_train_2 as R2

# The E17 import (pulled in by E18U) sets E.DEV='cpu' for its own weight-only
# use -- restore the runner convention.
E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e19.json')
GC19 = 1e-4                               # the dial
GC_PARENT = 3e-5
GATE_TOL = 1e-3
NG = E.NGROUP

assert E15R.GC15 == GC_PARENT and E16R.GC16 == GC_PARENT


def trainer(lr, gc, steps, **kw):
    """EXACTLY the parents' trainer expression (E15/E16 runners)."""
    return E.train_muon(lr, gc, steps, lr_adamw=E.get_lr(), **kw)


def three_step(factory, gc, steps=3):
    """3-step truncated run with per-step CE + full-precision held-100 CE."""
    log = E.train_muon(E7R.muon_lr(), gc, steps, log_every=1, save=False,
                       factory=factory, lr_adamw=E.get_lr())
    return {'per_step_ce': [x[1] for x in log['train_loss']],
            'held100_ce': log['final_held_ce']}


# ---------------- controls ----------------
def control_dial_only(name, factory):
    """Rerun of the parent's first `steps` steps at gc 3e-5 vs this runner's
    training call at gc 3e-5: trajectories must match to float tolerance."""
    key = f'control_dial_only_{name}'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    # parent side: the exact invocation the parent runner makes (train_arm ->
    # trainer lambda -> E.train_muon(muon_lr, GC_PARENT, steps, lr_adamw=...))
    parent = three_step(factory, GC_PARENT)
    # E19 side: this runner's own trainer at the parent's coefficient
    log19 = trainer(E7R.muon_lr(), GC_PARENT, 3, log_every=1, save=False,
                    factory=factory)
    e19 = {'per_step_ce': [x[1] for x in log19['train_loss']],
           'held100_ce': log19['final_held_ce']}
    step_diff = max(abs(a - b) for a, b in
                    zip(parent['per_step_ce'], e19['per_step_ce']))
    held_diff = abs(parent['held100_ce'] - e19['held100_ce'])
    rec = {'parent_per_step_ce': parent['per_step_ce'],
           'e19_per_step_ce': e19['per_step_ce'],
           'max_per_step_abs_diff': step_diff,
           'parent_held100_ce': parent['held100_ce'],
           'e19_held100_ce': e19['held100_ce'],
           'held100_abs_diff': held_diff,
           'group_coeff': GC_PARENT, 'steps': 3,
           'note': 'both runs: same seed (factory reseeds), same '
                   'epoch_order(0) data; proves the only E19 change vs the '
                   'parent is the lasso coefficient',
           'pass': bool(step_diff == 0.0 and held_diff < 1e-6)}
    E.merge(JP, key, rec)
    print(f"{key}: per-step diff {step_diff:.2e}, held100 diff "
          f"{held_diff:.2e} -> {'PASS' if rec['pass'] else 'FAIL'}",
          flush=True)
    assert rec['pass'], f'{key} FAILED'


@torch.no_grad()
def control_penalties(s_c):
    key = 'control_penalty_vs_naive'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    rec = {}
    # E19a architecture: custom seg_ind penalty via the V8T dispatch
    mc = E15R.make_e15c(s=s_c)
    S = mc.s
    p_fast = float(V8T.group_penalty(mc))
    p_custom = float(mc.custom_group_penalty())
    p_naive = 0.0
    for blk in mc.h:
        for nm in E.READ_NAMES:
            M = getattr(blk, nm).weight
            for k in range(2 * DEPTH):
                p_naive += float(M[:, S * k:S * (k + 1)].pow(2).sum()) ** 0.5
    rel_a = abs(p_fast - p_naive) / p_naive
    rec['E19a'] = {'fast': p_fast, 'naive': p_naive, 'rel': rel_a,
                   'dispatch_abs_diff': abs(p_fast - p_custom)}
    print(f"control E19a penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel_a:.2e} (dispatch diff {abs(p_fast - p_custom):.1e})",
          flush=True)
    del mc
    torch.cuda.empty_cache()
    # E19b architecture: standard slot-column group penalty
    mb = E16R.make_e16b()
    Sb = mb.slot
    p_fast_b = float(V8T.group_penalty(mb))
    p_naive_b = 0.0
    for blk in mb.h:
        for nm in E.READ_NAMES:
            M = getattr(blk, nm).weight
            for k in range(NG):
                p_naive_b += float(
                    M[:, Sb * k:Sb * (k + 1)].pow(2).sum()) ** 0.5
    rel_b = abs(p_fast_b - p_naive_b) / p_naive_b
    rec['E19b'] = {'fast': p_fast_b, 'naive': p_naive_b, 'rel': rel_b}
    print(f"control E19b penalty fast {p_fast_b:.4f} vs naive "
          f"{p_naive_b:.4f} rel {rel_b:.2e}", flush=True)
    del mb
    torch.cuda.empty_cache()
    rec['pass'] = bool(rel_a < 1e-6 and rel_b < 1e-6
                       and abs(p_fast - p_custom) == 0.0)
    E.merge(JP, key, rec)
    assert rec['pass'], 'penalty control FAILED'


def control_e18_gates():
    """Precondition: the generalized-probe and cov-composed gates passed."""
    if E.SMOKE:
        return
    e18 = E.loadj(E.jpath('qk_e18.json'))
    g1 = e18.get('gate1_uniform11_weight_support', {}).get('pass')
    g2 = e18.get('gate2_cov_composed_E9a', {}).get('pass')
    assert g1 and g2, ('qk_e18.json gates not passed -- the reused probe '
                       'functions are not validated', g1, g2)
    print("precondition: qk_e18.json gates 1+2 passed (generalized probe + "
          "cov-composed pipeline validated on the uniform-11 recipe)",
          flush=True)


# ---------------- probes ----------------
def pair_extra(stem, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') \
                and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(JP, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


def probe_e19a(s_c):
    """E18 job-3 style variable-slot-dim light probe + job-4 cov-composed."""
    stem = 'qk_e19_a'
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    dims = [s_c] * NG
    j = E.loadj(JP)
    if 'light_probe_E19a_var_dims' not in j:
        print('E19a light probe (variable slot dims) ...', flush=True)
        m, _ = E.load_arm(stem, lambda: E15R.make_e15c(s=s_c))
        Ws = m.wte.weight.shape[1]
        assert Ws == 2 * DEPTH * s_c
        base, dce = E18U.gen_consumption(m, Ws)
        wp = E18U.wpairs(m, dims)
        G = E18U.gen_gram_table(m, dims)
        sup = E18U.score(G, wp)
        cau = [dce[li][si] for li, si in wp]
        eff = [k for k in range(len(wp)) if cau[k] > C.EFFECTUAL]
        agr = E17.agreement(sup, cau, eff)
        totals = {}
        for jj in range(DEPTH):
            totals[str(jj)] = round(sum(
                dce[li].get(si, 0.0) for li in dce
                for si in (1 + 2 * jj, 2 + 2 * jj) if si in dce[li]), 5)
        pairs_sorted = sorted([(li, si, dce[li][si]) for li in dce
                               for si in dce[li]], key=lambda p: -p[2])
        rec = {'checkpoint': f'{stem}.pt', 'stream_width': Ws,
               'slot_dims': dims, 'compute_width': Q.D,
               'base_ce_fp32_abl_oldheld': round(base, 5),
               'wiring_n_pairs': len(wp),
               'wiring_spearman_all': agr['spearman_all'],
               'wiring_n_effectual': len(eff),
               'wiring_spearman_effectual': agr['spearman_effectual'],
               'wiring_top10_precision': agr['top10_precision'],
               'per_block_total_consumption': totals,
               'dead_blocks_below_0.001':
                   [b for b in totals if totals[b] < 0.001],
               'consumption_top20': [
                   {'consumer': ('readout' if li == DEPTH else f'block{li}'),
                    'source': R2.stream_name(si), 'dce': round(v, 5)}
                   for li, si, v in pairs_sorted[:20]],
               'consumption_matrix': {str(li): {str(si): round(v, 6)
                                                for si, v in dce[li].items()}
                                      for li in dce},
               'weight_support_matrix': {
                   str(li): {str(si): round(sup[i], 3)
                             for i, (l2, si) in enumerate(wp) if l2 == li}
                   for li in range(DEPTH + 1)},
               'note': 'generalized variable-slot-dim probe (qk_e18 gate 1 '
                       'validated); causal = mean-ablation on old cooc held '
                       '[:96], fp32, light-probe conventions'}
        E.merge(JP, 'light_probe_E19a_var_dims', rec)
        print(f"E19a wiring Spearman all {agr['spearman_all']} "
              f"effectual({len(eff)}) {agr['spearman_effectual']} "
              f"top10 {agr['top10_precision']}", flush=True)
        del m
        torch.cuda.empty_cache()
    j = E.loadj(JP)
    if 'composed_wiring_E19a' not in j:
        lp = j['light_probe_E19a_var_dims']
        print('E19a covariance-composed re-scoring ...', flush=True)
        m, _ = E.load_arm(stem, lambda: E15R.make_e15c(s=s_c))
        wp = E18U.wpairs(m, dims)
        cau = E18U.stored_cau(lp, wp)
        tables, meta, _ = E18U.composed_tables(m, dims, cau, wp, E.DEV,
                                               remnant=False)
        chk = abs(tables['plain']['spearman_all']
                  - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, \
            f'E19a plain does not reproduce the light probe ({chk})'
        rec = {'checkpoint': f'{stem}.pt', 'slot_dims_uniform': s_c,
               'plain_reproduction_abs_diff': round(chk, 6),
               'tables': tables}
        rec.update(meta)
        E.merge(JP, 'composed_wiring_E19a', rec)
        print(f"E19a plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
        del m
        torch.cuda.empty_cache()


def probe_e19b():
    """E16-corrected light probe + remnant probe + cov-composed (remnant)."""
    stem = 'qk_e19_b'
    if E.SMOKE or not os.path.exists(E.ckpath(stem)):
        return
    dims11 = [E.SUB] * NG
    j = E.loadj(JP)
    m = None
    if 'remnant_probe_E19b' not in j:
        m, _ = E.load_arm(stem, E16R.make_e16b)
        E.merge(JP, 'remnant_probe_E19b', E16R.remnant_probe(m))
    if 'light_probe_E19b' not in E.loadj(JP):
        if m is None:
            m, _ = E.load_arm(stem, E16R.make_e16b)
        print('E19b light probe (E16 stream-0 remnant correction) ...',
              flush=True)
        E.merge(JP, 'light_probe_E19b', E16R.e16_light_probe(m))
    j = E.loadj(JP)
    if 'composed_wiring_E19b' not in j:
        if m is None:
            m, _ = E.load_arm(stem, E16R.make_e16b)
        lp = j['light_probe_E19b']
        print('E19b covariance-composed re-scoring ...', flush=True)
        wp = E18U.wpairs(m, dims11)
        cau = E18U.stored_cau(lp, wp)
        tables, meta, _ = E18U.composed_tables(m, dims11, cau, wp, E.DEV,
                                               remnant=True)
        chk = abs(tables['plain']['spearman_all']
                  - lp['wiring_spearman_all'])
        assert chk <= GATE_TOL, \
            f'E19b plain does not reproduce the light probe ({chk})'
        rec = {'checkpoint': f'{stem}.pt', 'slot_dims_uniform': E.SUB,
               'plain_reproduction_abs_diff': round(chk, 6),
               'tables': tables}
        rec.update(meta)
        E.merge(JP, 'composed_wiring_E19b', rec)
        print(f"E19b plain {tables['plain']['spearman_all']} -> cov "
              f"{tables['cov_composed']['spearman_all']}", flush=True)
    if m is not None:
        del m
        torch.cuda.empty_cache()


def summarize():
    j = E.loadj(JP)
    summary = {'parents': {
        'E15c_gc3e-5': {'ce': 4.9038, 'cov_composed': 0.6728,
                        'plain': 0.6294},
        'E16b_gc3e-5': {'ce': 5.0231, 'cov_composed': 0.6617,
                        'plain': 0.5946},
        'E9a_recipe': {'ce': 5.0547, 'cov_composed': 0.8575,
                       'plain': 0.7711}}}
    for arm, cw in (('E19a', 'composed_wiring_E19a'),
                    ('E19b', 'composed_wiring_E19b')):
        if arm not in j:
            continue
        row = {'final_held_ce_fresh_bf16': j[arm]['final_held_ce_fresh_bf16'],
               'diverged': j[arm]['diverged']}
        if cw in j:
            t = j[cw]['tables']
            row.update({
                'plain': t['plain']['spearman_all'],
                'cov_composed': t['cov_composed']['spearman_all'],
                'plain_effectual': t['plain']['spearman_effectual'],
                'cov_effectual': t['cov_composed']['spearman_effectual'],
                'plain_top10': t['plain']['top10_precision'],
                'cov_top10': t['cov_composed']['top10_precision']})
        summary[arm] = row
    if 'E19a' in summary and 'cov_composed' in summary['E19a']:
        cov = summary['E19a']['cov_composed']
        ce = summary['E19a']['final_held_ce_fresh_bf16']
        if cov >= 0.75 and ce is not None and ce <= 4.99:
            v = ('CONFIRMED: readability-preserving cheap partition '
                 f'(cov {cov} >= 0.75 at CE {round(ce, 4)} <= 4.99)')
        elif cov < 0.72:
            v = (f'REFUTED: dial hypothesis fails on widened slots '
                 f'(cov {cov} < 0.72)')
        else:
            v = (f'INTERMEDIATE: cov {cov}, CE '
                 f'{None if ce is None else round(ce, 4)} -- neither '
                 'confirmation (>= 0.75 at CE <= 4.99) nor refutation '
                 '(< 0.72) threshold met')
        summary['E19a_prediction_verdict'] = v
    E.merge(JP, 'summary_E19', summary)
    print(json.dumps({'summary_E19': summary}, indent=2), flush=True)


if __name__ == '__main__':
    E.setup()
    control_e18_gates()

    s_c, tgt = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c                     # the E15c-config slot size

    # ---- positive controls (before training) ----
    control_penalties(s_c)
    control_dial_only('E19a', lambda: E15R.make_e15c(s=s_c))
    control_dial_only('E19b', E16R.make_e16b)

    # ---- registered prediction (before training) ----
    if 'E19_prediction' not in E.loadj(JP):
        E.merge(JP, 'E19_prediction', {
            'registered_before_training': True,
            'hypothesis': 'a stronger in-loss group-lasso (1e-4 vs the '
                          'parents\' 3e-5) buys the cheap partitions\' '
                          'readability back at modest CE -- on the recipe '
                          'the same dial bought Spearman 0.60 -> 0.78 at '
                          'scale for +0.12 CE (RESULTS_l0_mdl.md sec 112)',
            'E19a_confirm': 'covariance-composed Spearman >= 0.75 at final '
                            'fresh held CE <= 4.99 (still below the '
                            'recipe\'s 5.0547) CONFIRMS the readability-'
                            'preserving cheap partition',
            'E19a_refute': 'covariance-composed Spearman < 0.72 REFUTES '
                           'the dial hypothesis on widened slots',
            'E19b_status': 'secondary arm, no registered threshold -- '
                           'reported alongside',
            'parent_reference': {'E15c': {'ce': 4.9038, 'cov': 0.6728},
                                 'E16b': {'ce': 5.0231, 'cov': 0.6617},
                                 'E9a_recipe': {'ce': 5.0547,
                                                'cov': 0.8575}}})

    if 'E19_config' not in E.loadj(JP):
        E.merge(JP, 'E19_config', {
            'group_coeff': GC19, 'parent_group_coeff': GC_PARENT,
            'muon_lr': E7R.muon_lr(), 'adamw_lr': E.get_lr(),
            'E19a_factory': 'qk_e15_reinvest_run.make_e15c(s=15) verbatim '
                            '(true-small decoders, 24 x 15-dim slots, '
                            'stream 360, compute 264, hidden 1056)',
            'E19b_factory': 'qk_e16_shrinkemb_run.make_e16b verbatim '
                            '(shrinking embedding channel, 44-dim floor; '
                            'checkpoint variant string stays E16b)',
            'slot_E19a': s_c})

    # ---- E19a ----
    E.train_arm('qk_e19_a', JP, 'E19a', lambda: E15R.make_e15c(s=s_c), GC19,
                lr=E7R.muon_lr(), trainer=trainer,
                extra={'optimizer': 'muon', 'slot': s_c,
                       'parent': 'E15c (qk_e15_c, gc 3e-5)',
                       'design': 'E15c bandwidth-reinvestment architecture '
                                 'with the readability dial: in-loss '
                                 'group-lasso 1e-4'})
    E.oldheld_record('qk_e19_a', lambda: E15R.make_e15c(s=s_c), JP,
                     'E19a_oldheld')
    E.paired_fresh('qk_e19_a', JP, 'E19a')
    pair_extra('qk_e19_a', 'E19a', (('qk_e15_c', 'e15c'),
                                    ('qk_e9_a', 'e9a')))
    probe_e19a(s_c)

    # ---- E19b ----
    E.train_arm('qk_e19_b', JP, 'E19b', E16R.make_e16b, GC19,
                lr=E7R.muon_lr(), trainer=trainer,
                extra={'optimizer': 'muon',
                       'parent': 'E16b (qk_e16_b, gc 3e-5)',
                       'design': 'E16b shrinking-channel + 44-dim-floor '
                                 'architecture with the readability dial: '
                                 'in-loss group-lasso 1e-4'})
    E.oldheld_record('qk_e19_b', E16R.make_e16b, JP, 'E19b_oldheld')
    E.paired_fresh('qk_e19_b', JP, 'E19b')
    pair_extra('qk_e19_b', 'E19b', (('qk_e16_b', 'e16b'),
                                    ('qk_e9_a', 'e9a')))
    probe_e19b()

    summarize()
    print('e19 dial run done', flush=True)
