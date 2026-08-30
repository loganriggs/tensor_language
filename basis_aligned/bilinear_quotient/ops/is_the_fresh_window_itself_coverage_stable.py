# CHECK THE INSTRUMENT THAT OVERTURNED FIFTEEN SECTIONS.
#
# §2037 through §2043 used one held-out window at ONE coverage (5,419) to retract a build, close an axis,
# validate an ancestry and declare a parameter unidentifiable. That is a great deal of weight on a single
# row set measured one way, and the discipline this line applies to every other claim -- does it hold at
# the other coverage? -- has never been applied to the fresh window itself.
#
# §1963 and §1965 each reversed a 5,419 claim at 16,110. If the fresh window's verdicts move the same way,
# then §2037's retraction is itself coverage-contingent and the honest statement is weaker than the one
# now in the ledger. If they hold, the retraction rests on two independent axes -- different rows AND a
# different fit set -- which is stronger than anything the retracted arc ever had.
#
# Registered expectation: they hold. §2037's margins are 10 to 130 milli-nats, an order of magnitude above
# the scale §1963 and §1965 reversed, and §2041's ancestry margins are larger still.
#
# ARMS. §1789's deployed design; §1959's build; the retracted converged build; §2020's table raise alone.
# A fallback variant of the shipped build for the inert half of the control, and one differing-table-rank
# arm for the other half.
#
# ROLES. 'fresh' ONLY, at 16,110. DISCOVERY ONLY. Rung 3 -- checking §2037's instrument.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

HI = 'c16110'
BASE = {'mlp': 768, 'attn': 384}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
CUT = 8                                                  # §2024
DEPLOYED, S1959, CONVERGED, TABS = 'deployed_1789', 'build_1959', 'converged', 'tables_only'
FRESH = ('fresh',)

# §2037/§2038 on the fresh window at 5,419, nats
CONVERGED_OVER_S1959_LO = -0.011770
S1959_OVER_DEPLOYED_LO = 0.127889
TABS_OVER_S1959_LO = -0.011578


def _cut_arm():
    early = ','.join(f'mlp{L}' for L in range(CUT))
    late = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < CUT))
    return f'mix30m256@{early}+mix30m640@{late}'


PLAN = [('map64', None, DEPLOYED, None),
        ('mix30m640', BASE, S1959, None),
        (_cut_arm(), {**BASE, **TABLES}, CONVERGED, B.SITES),
        ('mix30m640', {**BASE, **TABLES}, TABS, None),
        ('map512', BASE, 'shipped_fb_control', None),                       # all 36 sites: INERT pair
        ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _beats(x, a, b):
    """nats by which a beats b on the fresh window at 16,110"""
    return -x.tpool_full(HI, a, b)['mean']


def _the_retraction_holds_at_the_other_coverage(x):
    """the converged build still LOSES to §1959's on fresh rows at 16,110. §2037's retraction rests on
    this comparison at 5,419 alone; if it reverses here the retraction is coverage-contingent and the
    ledger overstates it"""
    return _beats(x, CONVERGED, S1959) < 0


def _the_table_raise_still_fails(x):
    """and §2020's table raise alone is still negative there -- §2038 attributed 98.4% of the deficit to
    it at 5,419, and §2039 found every depth harmful. If the attribution moves with coverage, the axis was
    closed on one measurement rather than on a property of the program"""
    return _beats(x, TABS, S1959) < 0


def _the_big_anchor_still_transports(x):
    """and §1959's build still beats §1789's deployed design by more than half its 5,419 fresh figure of
    127.889 milli-nats. That margin is the reason to trust the window at all; if it collapses at the other
    coverage the window is measuring something coverage-specific"""
    return _beats(x, S1959, DEPLOYED) > 0.5 * S1959_OVER_DEPLOYED_LO


def _the_deficit_is_the_same_order(x):
    """and the converged build's deficit is within a factor of three of its 5,419 fresh figure of -11.770
    milli-nats. Sign agreement alone would leave open that the effect is ten times smaller here; a
    retraction should rest on a magnitude that travels, not only a direction"""
    m = _beats(x, CONVERGED, S1959)
    return CONVERGED_OVER_S1959_LO * 3.0 < m < CONVERGED_OVER_S1959_LO / 3.0


B.run(
    name='is_the_fresh_window_itself_coverage_stable',
    plan=PLAN,
    coverages=[(HI, B.FIT_16110, 16110)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_retraction_holds_at_the_other_coverage',
         'the converged build still loses to §1959\'s on fresh rows at 16,110',
         _the_retraction_holds_at_the_other_coverage),
        ('pred_b_the_table_raise_still_fails',
         'and §2020\'s table raise alone is still negative there', _the_table_raise_still_fails),
        ('pred_c_the_big_anchor_still_transports',
         'and §1959 still beats §1789\'s deployed design by more than half its 127.889 milli-nats',
         _the_big_anchor_still_transports),
        ('pred_d_the_deficit_is_the_same_order',
         'and the deficit is within a factor of three of its -11.770 milli-nat figure at 5,419',
         _the_deficit_is_the_same_order),
    ],
    paired_pairs=[(CONVERGED, S1959), (TABS, S1959), (S1959, DEPLOYED), (CONVERGED, DEPLOYED)],
)
