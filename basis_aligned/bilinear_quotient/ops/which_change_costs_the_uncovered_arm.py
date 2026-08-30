# WHICH CHANGE COSTS THE UNCOVERED ARM?
#
# §2029 found the converged build a net LOSS at uncovered inputs on all three roles (−0.734, −0.790,
# −4.693 milli-nats) while gaining +3.3 to +5.0 at covered ones. The pooled +3.064 §2028 recorded is the
# net of both.
#
# Two changes separate the converged build from §1959's. §2020 raised the MLP table rank to 1152 at layers
# 10-17, which acts on covered AND uncovered rows. §2024 cut the map from 640 to 256 at MLP layers 0-7,
# and the map acts ONLY on uncovered rows. The second is the obvious suspect and §2029 said so without
# testing it; this tests it, because "obvious" is how §2020's registered direction came to be wrong.
#
# ARMS. §1959's build; §2020's build (tables raised, map untouched); §2024's map cut alone (map cut,
# tables at 768); and the converged build (both). A fallback variant of the shipped build for the inert
# half of the control, and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2029's open question.
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

# §2029, milli-nats the converged build beats §1959 by at UNCOVERED inputs
UNC_TOTAL = (-0.734, -0.790, -4.693)


def _cut_arm():
    early = ','.join(f'mlp{L}' for L in range(CUT))
    late = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < CUT))
    return f'mix30m256@{early}+mix30m640@{late}'


PLAN = [('mix30m640', BASE, S1959, None),
        ('mix30m640', {**BASE, **TABLES}, TABS, None),               # §2020: tables only
        (_cut_arm(), BASE, MAPCUT, B.SITES),                         # §2024's cut, tables at 768
        (_cut_arm(), {**BASE, **TABLES}, CONVERGED, B.SITES),        # both
        ('map512', BASE, 'shipped_fb_control', None),                # all 36 sites: the INERT pair
        ('mix30m640', A256, 'rank_control', None)]                   # differing table rank: other half


def _unc(x, role, lab):
    """milli-nats by which `lab` beats §1959's build at UNCOVERED inputs"""
    return 1000.0 * (x.ce(LO, role, S1959, 'uncovered_input')
                     - x.ce(LO, role, lab, 'uncovered_input'))


def _the_total_reproduces(x):
    """the converged build's uncovered-input deficit rebuilds to §2029's −0.734 / −0.790 / −4.693
    milli-nats within 0.05, on all three roles"""
    return all(abs(_unc(x, r, CONVERGED) - v) < 0.05 for r, v in zip(x.roles, UNC_TOTAL))


def _the_map_cut_is_the_cost(x):
    """and the map cut alone is negative at uncovered inputs on all three roles -- the map acts only on
    uncovered rows, so if the deficit is its doing it must show here in isolation"""
    return all(_unc(x, r, MAPCUT) < 0 for r in x.roles)


def _the_tables_are_not(x):
    """and §2020's table raise alone is POSITIVE at uncovered inputs on >=2 roles -- it raises the covered
    tables at layers 10-17, and the uncovered rows are built from those same tables by the neighbour and
    map arms, so it should help there too. If FALSE the table raise also costs the uncovered arm and
    §2029's named suspect was wrong"""
    return sum(1 for r in x.roles if _unc(x, r, TABS) > 0) >= 2


def _the_two_roughly_add(x):
    """and the two effects roughly add at uncovered inputs: their sum is within 1.0 milli-nat of the
    converged build's deficit, on >=2 roles. §2014 measured the shipped program super-additive by 22% in
    the loss, so this is not arithmetic and a large gap would mean the changes interact where they meet"""
    return sum(1 for r in x.roles
               if abs((_unc(x, r, TABS) + _unc(x, r, MAPCUT)) - _unc(x, r, CONVERGED)) < 1.0) >= 2


B.run(
    name='which_change_costs_the_uncovered_arm',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_the_total_reproduces',
         '§2029\'s uncovered deficit rebuilds to −0.734/−0.790/−4.693 milli-nats within 0.05, 3/3 roles',
         _the_total_reproduces),
        ('pred_b_the_map_cut_is_the_cost',
         'and §2024\'s map cut alone is negative at uncovered inputs on all three roles',
         _the_map_cut_is_the_cost),
        ('pred_c_the_tables_are_not',
         'and §2020\'s table raise alone is positive there (>=2 roles)', _the_tables_are_not),
        ('pred_d_the_two_roughly_add',
         'and the two effects sum to within 1.0 milli-nat of the total (>=2 roles)', _the_two_roughly_add),
    ],
    refs=[(S1959, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(CONVERGED, S1959), (TABS, S1959), (MAPCUT, S1959)],
)
