# WHERE DOES THE FRESH-WINDOW LOSS LIVE, AND DOES IT EXPLAIN THE COVERAGE GROWTH?
#
# §2044 found the table raise's fresh-window harm growing 2.8x between coverages: -11.578 milli-nats at
# 5,419 and -32.687 at 16,110. More coverage means more positions served by the COVERED tables, which is
# exactly what untruncating changes, so the mechanism predicts the harm sits in the covered arm and scales
# with it.
#
# §2029 ran this decomposition in-sample and found §2020's gain concentrated at covered inputs (+4.970 to
# +3.329) with a loss at uncovered ones. If the fresh-window LOSS is also concentrated at covered inputs,
# then the same cells that carried the in-sample gain carry the out-of-sample harm -- which is what a
# fitted table looks like, and it closes the account of what §2020 did.
#
# ARMS. §1959's build and §2020's table raise alone, at both coverages, on the fresh window. A fallback
# variant of the shipped build for the inert half of the control, and one differing-table-rank arm for
# the other half. Two arms suffice: §2044 showed the map cut is a rounding term here.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2044's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
S1959, TABS = 'build_1959', 'tables_only'
FRESH = ('fresh',)

# §2038 / §2044, the table raise over §1959's build on the fresh window, milli-nats
TABS_LO, TABS_HI = -11.578, -32.687


def _cell(x, cov, cls):
    """milli-nats by which the table raise beats §1959's build on one class of inputs"""
    return 1000.0 * (x.ce(cov, 'fresh', S1959, cls) - x.ce(cov, 'fresh', TABS, cls))


PLAN = [('mix30m640', BASE, S1959, None),
        ('mix30m640', {**BASE, **TABLES}, TABS, None),
        ('map512', BASE, 'shipped_fb_control', None),                       # all 36 sites: INERT pair
        ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _the_totals_reproduce(x):
    """the pooled figures rebuild to §2038's -11.578 and §2044's -32.687 milli-nats within 0.5 at their
    respective coverages. The decomposition below is of those numbers and needs them anchored"""
    return (abs(_cell(x, LO, 'pooled') - TABS_LO) < 0.5
            and abs(_cell(x, HI, 'pooled') - TABS_HI) < 0.5)


def _the_harm_is_at_covered_inputs(x):
    """and the harm is negative at COVERED inputs at both coverages. §2029 found §2020's in-sample GAIN
    concentrated there; if the out-of-sample LOSS is in the same cells, the tables that were fitted are
    the tables that fail"""
    return _cell(x, LO, 'covered_input') < 0 and _cell(x, HI, 'covered_input') < 0


def _the_covered_harm_grows_with_coverage(x):
    """and the covered-input harm is larger at 16,110 than at 5,419 -- more coverage means more positions
    served by the covered tables, so the mechanism predicts it scales. If FALSE the 2.8x growth §2044
    measured comes from somewhere else and the account is incomplete"""
    return _cell(x, HI, 'covered_input') < _cell(x, LO, 'covered_input')


def _covered_carries_most_of_it(x):
    """and covered inputs carry at least 70% of the total harm at both coverages. §2029 measured covered
    inputs as about three quarters of scored positions, so a harm spread evenly would land near 75% by
    weight alone; more than that means the covered cells are hit disproportionately"""
    return all(_cell(x, c, 'covered_input') / _cell(x, c, 'pooled') > 0.70 for c in (LO, HI))


B.run(
    name='where_the_fresh_loss_lives',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_totals_reproduce',
         '§2038\'s -11.578 and §2044\'s -32.687 milli-nat totals rebuild within 0.5',
         _the_totals_reproduce),
        ('pred_b_the_harm_is_at_covered_inputs',
         'and the harm is negative at covered inputs at both coverages', _the_harm_is_at_covered_inputs),
        ('pred_c_the_covered_harm_grows_with_coverage',
         'and the covered-input harm is larger at 16,110 than at 5,419',
         _the_covered_harm_grows_with_coverage),
        ('pred_d_covered_carries_most_of_it',
         'and covered inputs carry at least 70% of the total harm at both coverages',
         _covered_carries_most_of_it),
    ],
    paired_pairs=[(TABS, S1959)],
)
