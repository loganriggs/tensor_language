# THE UNSEEN BUCKET AT THE HIGHER COVERAGE -- THE LAST CELL WHERE THE TWO AXES CAN AGREE OR DISAGREE.
#
# §2033 decomposed the unseen-target bucket at 5,419: §2024's map cut is negative on every role (-0.402,
# -0.855, -1.642) and §2020's table raise is positive on two (+2.453, +3.806) and -2.318 at skip1200 --
# the same 2-of-3 shape §2030 found on the covered/uncovered axis, with the same role dissenting.
#
# §2031 already showed that shape is COVERAGE-UNSTABLE on the uncovered axis: the table raise's sign flips
# at skip11000 between 5,419 and 16,110. So the open question is whether the frequency axis behaves the
# same way. If the unseen bucket's 2-of-3 shape also breaks at 16,110, then skip1200 is not special and
# both axes have been reporting coverage noise about that role. If it holds, the two axes disagree about
# stability and skip1200 is a property of the role on the frequency axis specifically.
#
# WRITTEN FRESH, not derived (LESSON 105): §2033's first version was produced from a sibling by string
# replacement, several replacements silently missed, and every registered text described the wrong
# quantity. Texts, constants and helper below were written together.
#
# ARMS. §1959's build; §2020's table raise alone; §2024's map cut alone; the converged build. A fallback
# variant of the shipped build for the inert half of the control, and one differing-table-rank arm for the
# other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2033's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

HI = 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
CUT = 8                                                  # §2024
S1959, TABS, MAPCUT, CONVERGED = 'build_1959', 'tables_only', 'mapcut_only', 'converged'
UNSEEN = '0-0'

# §2033 at 5,419, on the unseen bucket, milli-nats over §1959's build
TABS_LO = (2.453, 3.806, -2.318)
MAPCUT_LO = (-0.402, -0.855, -1.642)


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
    """milli-nats by which `lab` beats §1959's build ON THE UNSEEN-TARGET BUCKET, at 16,110"""
    return 1000.0 * (x.ce(HI, role, S1959, 'pooled', UNSEEN)
                     - x.ce(HI, role, lab, 'pooled', UNSEEN))


def _the_map_cut_still_loses_it(x):
    """§2024's map cut is still negative on the unseen bucket on all three roles at 16,110. §2031 found
    its uncovered-input tax uniform across both coverages and §2033 found it negative on this bucket at
    5,419; if the cut is the steady object both sections claim, it must hold here"""
    return all(_unseen(x, r, MAPCUT) < 0 for r in x.roles)


def _the_table_raise_keeps_its_two_of_three(x):
    """and §2020's table raise is still positive on the unseen bucket on exactly the same two roles --
    skip7000 and skip11000 -- and negative at skip1200. If TRUE the frequency axis is coverage-STABLE
    where §2031 showed the uncovered axis is not, and skip1200 is a property of the role on this axis"""
    return (_unseen(x, x.roles[0], TABS) > 0 and _unseen(x, x.roles[1], TABS) > 0
            and _unseen(x, x.roles[2], TABS) < 0)


def _skip1200_stays_the_losing_role(x):
    """and the converged build is negative on the unseen bucket at skip1200 and positive on the other two
    -- §2029's shape at 5,419 was +2.060 / +2.963 / -3.948. If FALSE the losing role moves with coverage
    and skip1200 was never special"""
    return (_unseen(x, x.roles[0], CONVERGED) > 0 and _unseen(x, x.roles[1], CONVERGED) > 0
            and _unseen(x, x.roles[2], CONVERGED) < 0)


def _the_map_cut_tax_grows_with_coverage(x):
    """and the map cut's unseen-bucket tax is larger here than at 5,419 on >=2 roles, against -0.402 /
    -0.855 / -1.642. §2031 measured exactly that on the uncovered axis (-0.9/-1.1 becoming -1.0/-1.8),
    which is the wrong direction for a fallback whose uncovered arm has halved and worth confirming"""
    return sum(1 for r, v in zip(x.roles, MAPCUT_LO) if _unseen(x, r, MAPCUT) < v) >= 2


B.run(
    name='the_unseen_bucket_at_high_coverage',
    plan=PLAN,
    coverages=[(HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_map_cut_still_loses_it',
         'the map cut is still negative on the unseen bucket at 16,110, on all three roles',
         _the_map_cut_still_loses_it),
        ('pred_b_the_table_raise_keeps_its_two_of_three',
         'and the table raise is positive on the same two roles and negative at skip1200',
         _the_table_raise_keeps_its_two_of_three),
        ('pred_c_skip1200_stays_the_losing_role',
         'and the converged build still loses the unseen bucket only at skip1200',
         _skip1200_stays_the_losing_role),
        ('pred_d_the_map_cut_tax_grows_with_coverage',
         'and the map cut\'s unseen-bucket tax is larger here than at 5,419 (>=2 roles)',
         _the_map_cut_tax_grows_with_coverage),
    ],
    refs=[(S1959, B.PT + 'ops/the_converged_build_end_to_end_results.json', 'build_1959', HI, 0.0005)],
    paired_pairs=[(CONVERGED, S1959), (TABS, S1959), (MAPCUT, S1959)],
)
