# WHICH CHANGE LOSES THE UNSEEN-TARGET BUCKET?
#
# §2029 found the converged build losing the unseen-target bucket at skip1200 by 3.948 milli-nats while
# gaining +2.060 and +2.963 on the other two roles. §2030 and §2031 attributed the COVERED/UNCOVERED axis
# and never decomposed the FREQUENCY axis, and the unseen bucket is the one cell the fallback exists to
# serve -- a target the program has never seen at fit time is predicted entirely by the neighbour and map
# arms.
#
# Two changes separate the converged build from §1959's. §2020 raised the MLP table rank to 1152 at layers
# 10-17. §2024 cut the map from 640 to 256 at MLP layers 0-7. §2031 showed the map cut's uncovered-input
# tax is uniform and the table raise's is unstable in sign; this asks the same question on the frequency
# axis.
#
# THIS SCRIPT WAS REWRITTEN. Its first version was derived from the uncovered-arm script by string
# replacement, several replacements silently did not match, and every registered text described the
# uncovered arm while the helper measured the unseen bucket. Scored as written it failed pred_a against a
# stale constant. Nothing here is derived; the texts and the helper are written together.
#
# ARMS. §1959's build; §2020's table raise alone; §2024's map cut alone; the converged build (both).
# A fallback variant of the shipped build for the inert half of the control, and one differing-table-rank
# arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2032's open question.
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
UNSEEN = '0-0'

# §2029 at 5,419: the converged build against §1959's, ON THE UNSEEN BUCKET, milli-nats
UNSEEN_TOTAL = (2.060, 2.963, -3.948)


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


def _unseen(x, role, lab):
    """milli-nats by which `lab` beats §1959's build ON THE UNSEEN-TARGET BUCKET"""
    return 1000.0 * (x.ce(LO, role, S1959, 'pooled', UNSEEN)
                     - x.ce(LO, role, lab, 'pooled', UNSEEN))


def _the_unseen_total_reproduces(x):
    """the converged build's unseen-bucket figures rebuild to §2029's +2.060 / +2.963 / -3.948 milli-nats
    within 0.05, on all three roles. Every attribution below is measured against them"""
    return all(abs(_unseen(x, r, CONVERGED) - v) < 0.05
               for r, v in zip(x.roles, UNSEEN_TOTAL))


def _the_map_cut_loses_the_unseen_bucket(x):
    """and §2024's map cut alone is negative on the unseen bucket on all three roles. An unseen target is
    predicted entirely by the fallback, and the map cut makes the shallow map poorer, so this is where it
    should show. If FALSE the map cut is not what loses this cell"""
    return all(_unseen(x, r, MAPCUT) < 0 for r in x.roles)


def _the_table_raise_wins_it_on_two_roles(x):
    """and §2020's table raise is positive on the unseen bucket on >=2 roles -- the same 2-of-3 shape
    §2030 found on the uncovered axis, which §2031 then showed was a coincidence of coverage. If this
    repeats the shape at 5,419, the two axes are telling one story about skip1200"""
    return sum(1 for r in x.roles if _unseen(x, r, TABS) > 0) >= 2


def _the_two_effects_add_on_this_bucket(x):
    """and the two effects sum to within 1.0 milli-nat of the converged build's unseen-bucket figure on
    >=2 roles. §2014 measured the shipped program super-additive by 22% in the loss, so additivity here
    is a finding rather than arithmetic"""
    return sum(1 for r in x.roles
               if abs((_unseen(x, r, TABS) + _unseen(x, r, MAPCUT))
                      - _unseen(x, r, CONVERGED)) < 1.0) >= 2


B.run(
    name='which_change_loses_the_unseen_bucket',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_the_unseen_total_reproduces',
         '§2029\'s unseen-bucket figures rebuild to +2.060/+2.963/-3.948 milli-nats within 0.05, 3/3 roles',
         _the_unseen_total_reproduces),
        ('pred_b_the_map_cut_loses_the_unseen_bucket',
         'and §2024\'s map cut alone is negative on the unseen bucket, on all three roles',
         _the_map_cut_loses_the_unseen_bucket),
        ('pred_c_the_table_raise_wins_it_on_two_roles',
         'and §2020\'s table raise is positive on the unseen bucket on >=2 roles',
         _the_table_raise_wins_it_on_two_roles),
        ('pred_d_the_two_effects_add_on_this_bucket',
         'and the two effects sum to within 1.0 milli-nat of the total on the unseen bucket (>=2 roles)',
         _the_two_effects_add_on_this_bucket),
    ],
    refs=[(S1959, B.PT + 'ops/the_converged_build_end_to_end_results.json', 'build_1959', LO, 0.0005)],
    paired_pairs=[(CONVERGED, S1959), (TABS, S1959), (MAPCUT, S1959)],
)
