# HALVE THE TRANSFER BRACKET: DOES §1947's TABLE KNEE HOLD ON FRESH ROWS?
#
# §2051 bracketed the transfer boundary empirically. Everything measured at or below 3.3 milli-nats
# in-sample REVERSED on fresh rows (§2018 +0.962 -> -10.371; §2020 +3.300 -> -11.578); everything at or
# above 24.4 AMPLIFIED (§1941 3.4x; §1970 1.8x). Nothing has been tested in between.
#
# §1947 sits inside the gap. Its claim is that the allocation {mlp 768, attn 256} sits ON the knee of the
# table-rank curve: the step down to it costs 0.0094 / 0.0090 / 0.0072 nats per 100M -- the last step under
# the 0.010 threshold -- and the next step to {640, 160} crosses at 0.0169 / 0.0177 / 0.0132. The decisive
# CE difference between those two arms is 0.0118 nats: 11.8 milli-nats, near the middle of [3.3, 24.4].
#
# §1947 is also the section §1947's price rule comes from, so it is load-bearing beyond its size: if the
# knee moves on rows that did not choose it, every later section that priced a purchase at 0.010 nats per
# 100M was using a threshold located by selection.
#
# ARMS. §1947's own ladder at its own coverage and fallback -- mix25m256 at {1024,256}, {768,256},
# {640,160} and {512,128}. A fallback variant for the inert half of the control, and one differing-table-
# rank arm for the other half.
#
# ROLES. 'fresh' ONLY, at 16,110 -- §1947's coverage. DISCOVERY ONLY. Rung 3 -- §2051's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

HI = 'c16110'
ARM = 'mix25m256'
FRESH = ('fresh',)
KNEE, NEXT = 'r768_256', 'r640_160'

# §1947's published costs, millions of values
COST = {'r1024_256': 419.101, 'r768_256': 339.558, 'r640_160': 269.958, 'r512_128': 220.243}
SPEC = {'r1024_256': {'mlp': 1024, 'attn': 256}, 'r768_256': {'mlp': 768, 'attn': 256},
        'r640_160': {'mlp': 640, 'attn': 160}, 'r512_128': {'mlp': 512, 'attn': 128}}
LADDER = ['r1024_256', 'r768_256', 'r640_160', 'r512_128']
PRICE = 0.010                       # §1947's threshold, nats per 100M
IN_SAMPLE_KNEE_STEP = 0.0118        # §1947: CE difference between {768,256} and {640,160}

PLAN = [(ARM, SPEC[l], l, None) for l in LADDER] + [
    ('map512', {'mlp': 768, 'attn': 256}, 'shipped_fb_control', None),   # same spec, other fallback
    (ARM, {'mlp': 896, 'attn': 224}, 'rank_control', None)]              # differing table rank


def _worse(x, lo, hi):
    """nats by which the CHEAPER arm `lo` is worse than the richer `hi` -- positive is the cost of the step.

    tpool_full(cov, a, b)['mean'] is penalty(a) - penalty(b), so this is the mean as stored, taken
    directly. Written out rather than as a double negation: a sign-carrying helper that needs two
    minus signs to read is how earlier sections got signs wrong."""
    return x.tpool_full(HI, lo, hi)['mean']


def _per_100m(x, hi, lo):
    """nats paid per 100M values saved by stepping from `hi` down to `lo`"""
    return _worse(x, lo, hi) / ((COST[hi] - COST[lo]) / 100.0)


def _the_ladder_orders_the_same(x):
    """the ladder still orders monotonically on fresh rows -- each cheaper arm strictly worse than the one
    above it. §2039 found the untruncation depths SCRAMBLING out of sample, so this is not automatic"""
    return all(_worse(x, LADDER[i + 1], LADDER[i]) > 0 for i in range(len(LADDER) - 1))


def _the_knee_step_is_still_under_price(x):
    """and the step down to {768,256} still costs under §1947's 0.010 nats per 100M on fresh rows --
    §1947 measured 0.0094 / 0.0090 / 0.0072 in-sample. If FALSE the knee is above 768 on rows that did not
    choose it"""
    return _per_100m(x, 'r1024_256', KNEE) < PRICE


def _the_next_step_still_crosses(x):
    """and the next step, to {640,160}, still crosses it -- §1947 measured 0.0169 / 0.0177 / 0.0132. Both
    halves are needed: the knee is where one step is under and the next is over"""
    return _per_100m(x, KNEE, NEXT) > PRICE


def _the_decisive_margin_does_not_reverse(x):
    """and the decisive CE difference between {768,256} and {640,160} is positive and at least a third of
    its in-sample 11.8 milli-nats. That quantity sits near the middle of §2051's [3.3, 24.4] bracket: if
    it amplifies the boundary is below 11.8, if it reverses the boundary is above"""
    return _worse(x, NEXT, KNEE) > IN_SAMPLE_KNEE_STEP / 3.0


B.run(
    name='is_the_table_knee_on_fresh_rows',
    plan=PLAN,
    coverages=[(HI, B.FIT_16110, 16110)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_ladder_orders_the_same',
         '§1947\'s ladder still orders monotonically on fresh rows', _the_ladder_orders_the_same),
        ('pred_b_the_knee_step_is_still_under_price',
         'and the step down to {768,256} still costs under 0.010 nats per 100M',
         _the_knee_step_is_still_under_price),
        ('pred_c_the_next_step_still_crosses',
         'and the next step to {640,160} still crosses it', _the_next_step_still_crosses),
        ('pred_d_the_decisive_margin_does_not_reverse',
         'and the decisive 11.8 milli-nat difference keeps at least a third of its size',
         _the_decisive_margin_does_not_reverse),
    ],
    paired_pairs=[(LADDER[i + 1], LADDER[i]) for i in range(len(LADDER) - 1)] + [(NEXT, KNEE)],
)
