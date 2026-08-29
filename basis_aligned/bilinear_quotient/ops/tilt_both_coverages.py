# THE TILT AT BOTH COVERAGES, DECLARATIVELY -- and run()'s second data point.
#
# §1963 tested the per-token alpha at §1960's compromise allocation and §1964 confirmed it on the shipped
# 5,419 build. Both found the same thing: the tilt raises the unseen bucket by its intended mechanism and
# costs more CE than it is worth. §1963 saw the price roughly halve at 16,110 (pred_b passed 3/3 there,
# and skip1200 was free), but that was on the compromise allocation, not the shipped build -- so whether
# the tilt is affordable at high coverage on the build we would actually ship is unmeasured, and it is
# the one place across five attempts where the sign came close to reversing.
#
# §1964 also left run() with a single data point and no exercise of its multi-coverage path. This is both.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §1964 at the other coverage.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
FLAT, WIDE, NARROW = 'flat30', 'pat10_40m640', 'pat20_30m640'
LO, HI = 'c5419', 'c16110'

PLAN = [('mix30m640', A384, FLAT),
        ('pat10_40m640', A384, WIDE),
        ('pat20_30m640', A384, NARROW),
        ('mix25m512', A256, 'rank_control')]     # differing table rank keeps the control two-sided


def _best(x, c, r):
    return max((WIDE, NARROW), key=lambda a: x.kf(c, r, a, x.bot))


def _helps_both(x):
    """the tilt raises the unseen bucket at BOTH coverages (>=2 roles each)"""
    n = x.count(lambda c, r: x.kf(c, r, _best(x, c, r), x.bot) > x.kf(c, r, FLAT, x.bot))
    return n[LO] >= 2 and n[HI] >= 2


def _price_halves(x):
    """and the CE price at 16,110 is strictly LOWER than at 5,419 on >=2 roles -- S1963 saw this on the
    compromise allocation and it is the one place the sign came close to reversing"""
    def pen(c, r):
        return x.ce(c, r, _best(x, c, r)) - x.ce(c, r, FLAT)
    return sum(1 for r in x.roles if pen(HI, r) < pen(LO, r)) >= 2


def _affordable_at_high_coverage(x):
    """and at 16,110 it finally clears the 0.002-nat bar on >=2 roles -- if TRUE the tilt is worth
    shipping at high coverage and five sections of negatives get a boundary; if FALSE the deficit is the
    blend's price everywhere and the line is closed for good"""
    def pen(c, r):
        return x.ce(c, r, _best(x, c, r)) - x.ce(c, r, FLAT)
    return sum(1 for r in x.roles if pen(HI, r) <= 0.002) >= 2


B.run(
    name='tilt_both_coverages',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_helps_at_both', 'the tilt raises the unseen bucket at both coverages (>=2 roles each)',
         _helps_both),
        ('pred_b_price_falls_with_coverage',
         'and the CE price is lower at 16,110 than at 5,419 (>=2 roles)', _price_halves),
        ('pred_c_affordable_at_16110',
         'and at 16,110 it clears the 0.002-nat bar (>=2 roles) -- the boundary, if there is one',
         _affordable_at_high_coverage),
    ],
    refs=[(FLAT, B.PT + 'ops/tilt_shipped_declarative_results.json', FLAT, LO, 0.0005)],
    paired_pairs=[(WIDE, FLAT), (NARROW, FLAT)],
)
