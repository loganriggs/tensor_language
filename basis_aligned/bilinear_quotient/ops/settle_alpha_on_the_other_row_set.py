# SETTLE ALPHA ON THE ROW SET THAT DID NOT SELECT IT.
#
# §2042 found the fresh window's alpha optimum at 0.40, beating the shipped 0.30 by +1.744 milli-nats at
# t = +3.42, and refused to adopt it: 0.40 was chosen by looking at that window, and +1.744 is about 0.05%
# of the build's distance from the live model -- the regime LESSON 106 says is not a result until measured
# elsewhere.
#
# The three published roles are that elsewhere, FOR THIS VALUE. §2027 swept 0.24 to 0.36 on them and
# stopped; 0.40 was never scored there. So for alpha 0.40 specifically the three roles are an
# out-of-selection set, exactly as the fresh window is for 0.28-0.36.
#
# §2027's in-sample curve declines above 0.28 (+0.003082 at 0.28, +0.003064 at 0.30, +0.002852 at 0.32,
# +0.001895 at 0.36), so the registered expectation is that 0.40 LOSES here. If it wins on both row sets
# while neither selected it, it is the first parameter change in this line validated in both directions
# and the build's one free parameter moves. If it loses, 0.30 stands and §2042's peak was selection.
#
# ARMS. alpha 0.30, 0.36 and 0.40 on §1959's build, plus 0.28 as §2027's in-sample optimum. A fallback
# variant for the inert half of the control, and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200 -- the three published roles. DISCOVERY ONLY. Rung 3 -- §2042's
# open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
SHIPPED = 'a30'

# §2027 on these three roles, nats over the shipped build, from its published grid
A36_IN_SAMPLE = 0.001895
A30_IN_SAMPLE = 0.003064
FRESH_A40_GAIN = 0.001744        # §2042: alpha 0.40 over 0.30 on the fresh window

PLAN = [('mix28m640', BASE, 'a28', None),
        ('mix30m640', BASE, SHIPPED, None),
        ('mix36m640', BASE, 'a36', None),
        ('mix40m640', BASE, 'a40', None),
        ('map512', BASE, 'shipped_fb_control', None),                       # all 36 sites: INERT pair
        ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _beats(x, a, b):
    """nats by which arm a beats arm b, pooled over the three published roles"""
    if a == b:
        return 0.0
    return -x.tpool_full(LO, a, b)['mean']


def _the_2027_shape_reproduces(x):
    """§2027's in-sample ordering rebuilds: 0.28 beats 0.30 beats 0.36 on these roles. That grid is the
    reason to expect 0.40 to lose here, so it has to reproduce before its extrapolation means anything"""
    return _beats(x, 'a28', SHIPPED) > 0 and _beats(x, SHIPPED, 'a36') > 0


def _alpha_040_loses_here(x):
    """and alpha 0.40 loses to 0.30 on the three published roles. Registered in the direction §2027's
    declining curve implies. If FALSE -- if 0.40 wins on BOTH row sets while neither selected it -- it is
    the first parameter change in this line validated in both directions and the build's free parameter
    moves"""
    return _beats(x, 'a40', SHIPPED) < 0


def _the_disagreement_is_real_not_noise(x):
    """and the disagreement is larger than noise: |0.40 minus 0.30| here exceeds a third of the +1.744
    milli-nats the fresh window measured in the opposite direction. Two row sets differing by a hair
    would mean the axis is simply flat and there is nothing to settle"""
    return abs(_beats(x, 'a40', SHIPPED)) > FRESH_A40_GAIN / 3.0


def _the_flat_interval_holds_here_too(x):
    """and the 0.28-to-0.40 interval spans under 3.300 milli-nats on these roles, as it spans about 3 on
    the fresh window (§2042). If TRUE the axis is flat near its optimum on BOTH row sets and the
    disagreement is about which point in a flat region is highest, not about a real difference in blend"""
    vals = [_beats(x, l, SHIPPED) for l in ('a28', SHIPPED, 'a36', 'a40')]
    return max(vals) - min(vals) < 0.003300


B.run(
    name='settle_alpha_on_the_other_row_set',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_the_2027_shape_reproduces',
         '§2027\'s ordering rebuilds: 0.28 beats 0.30 beats 0.36 on the three published roles',
         _the_2027_shape_reproduces),
        ('pred_b_alpha_040_loses_here',
         'and alpha 0.40 loses to 0.30 there, as §2027\'s declining curve implies', _alpha_040_loses_here),
        ('pred_c_the_disagreement_is_real_not_noise',
         'and the disagreement exceeds a third of the fresh window\'s +1.744 milli-nats',
         _the_disagreement_is_real_not_noise),
        ('pred_d_the_flat_interval_holds_here_too',
         'and 0.28 to 0.40 spans under 3.300 milli-nats here as it does on the fresh window',
         _the_flat_interval_holds_here_too),
    ],
    paired_pairs=[('a28', SHIPPED), ('a36', SHIPPED), ('a40', SHIPPED), ('a40', 'a36')],
)
