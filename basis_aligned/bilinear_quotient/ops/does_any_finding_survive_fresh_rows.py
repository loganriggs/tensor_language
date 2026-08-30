# DOES ANY OF THE FOUR PARAMETER FINDINGS SURVIVE FRESH ROWS INDIVIDUALLY?
#
# §2037 retracted the converged build: on 98,304 fresh positions it LOSES to §1959's by 11.770 milli-nats
# at t = -28.71, against +3.064 in-sample. That is the COMPOSED build. Each of the four changes was
# measured separately and each was significant on the selecting roles, so the composition failing does not
# by itself say which components were fitted -- §2014 measured the program super-additive by 22% in the
# loss, so components need not fail the way their sum does.
#
# This scores each change ALONE against §1959's build on the fresh window. §2020's table raise was the
# largest single gain in the arc (+3.300 milli-nats in-sample) and is the one worth knowing about; the map
# cut is the parameter SALE whose cost §2030-§2032 showed concentrated in the uncovered arm.
#
# Registered honestly: after §2037 I expect these to fail, and the point of running it is that "the
# composition was fitted" and "every component was fitted" are different claims and only one is measured.
#
# ARMS. §1959's build; §2020's table raise alone; §2024's map cut alone; the converged build (both), which
# §2037 measured at -11.770 and which anchors this run. A fallback variant of the shipped build for the
# inert half of the control, and one differing-table-rank arm for the other half.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2037's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
CUT = 8                                                  # §2024
S1959, TABS, MAPCUT, CONVERGED = 'build_1959', 'tables_only', 'mapcut_only', 'converged'
FRESH = ('fresh',)

CONVERGED_FRESH = -0.011770      # §2037, nats, converged over §1959 on this window
TABS_IN_SAMPLE = 0.003300        # §2020, nats, table raise over §1959 pooled on the three roles


def _cut_arm():
    early = ','.join(f'mlp{L}' for L in range(CUT))
    late = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < CUT))
    return f'mix30m256@{early}+mix30m640@{late}'


PLAN = [('mix30m640', BASE, S1959, None),
        ('mix30m640', {**BASE, **TABLES}, TABS, None),
        (_cut_arm(), BASE, MAPCUT, B.SITES),
        (_cut_arm(), {**BASE, **TABLES}, CONVERGED, B.SITES),
        ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites: the INERT pair
        ('mix30m640', A256, 'rank_control', None)]         # differing table rank: the other half


def _beats(x, a, b):
    return -x.tpool_full(LO, a, b)['mean']


def _the_2037_anchor_reproduces(x):
    """the converged build's fresh-window deficit rebuilds to §2037's -11.770 milli-nats within 0.5.
    Every component figure below is measured on the same rows and needs that anchor"""
    return abs(_beats(x, CONVERGED, S1959) - CONVERGED_FRESH) < 0.0005


def _the_table_raise_fails_fresh(x):
    """and §2020's table raise ALONE is negative on the fresh window -- it gained +3.300 milli-nats
    in-sample and was the largest single finding of the arc. Registered in the direction §2037 implies:
    if FALSE the table raise generalises and only its composition with the map cut was fitted, which
    would make the retraction narrower than §2037 stated"""
    return _beats(x, TABS, S1959) < 0


def _the_map_cut_fails_fresh(x):
    """and §2024's map cut alone is negative here too. §2030-§2032 showed its cost concentrated in the
    uncovered arm at both coverages, so it is the component least likely to have been a coverage
    artefact and most likely to reproduce as a genuine loss"""
    return _beats(x, MAPCUT, S1959) < 0


def _the_components_explain_the_composition(x):
    """and the two components sum to within 3 milli-nats of the composed deficit. §2014 measured the
    shipped program super-additive by 22% in the loss, so exact additivity is not expected; a sum that
    misses by much more would mean the fresh-window failure is an interaction rather than two independent
    regressions"""
    return abs((_beats(x, TABS, S1959) + _beats(x, MAPCUT, S1959))
               - _beats(x, CONVERGED, S1959)) < 0.003


B.run(
    name='does_any_finding_survive_fresh_rows',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_2037_anchor_reproduces',
         '§2037\'s -11.770 milli-nat fresh-window deficit rebuilds within 0.5 milli-nats',
         _the_2037_anchor_reproduces),
        ('pred_b_the_table_raise_fails_fresh',
         'and §2020\'s table raise alone is negative on the fresh window', _the_table_raise_fails_fresh),
        ('pred_c_the_map_cut_fails_fresh',
         'and §2024\'s map cut alone is negative there too', _the_map_cut_fails_fresh),
        ('pred_d_the_components_explain_the_composition',
         'and the two components sum to within 3 milli-nats of the composed deficit',
         _the_components_explain_the_composition),
    ],
    paired_pairs=[(CONVERGED, S1959), (TABS, S1959), (MAPCUT, S1959)],
)
