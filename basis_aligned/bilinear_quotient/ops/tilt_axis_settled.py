# SETTLE THE TILT AXIS IN ONE RUN, WITH A STOPPING RULE.
#
# §1964, §1965 and §1966 each moved the recommended fallback by one to two milli-nats, and §1966 found a
# better point BETWEEN two I had already called optimal. Every one was found by testing an untried point
# rather than by a mechanism, on a grid four points wide. §1961's alpha curve is the warning: it was flat
# near its optimum and two sections were spent learning that.
#
# This sweeps the tilt densely at both coverages and carries an explicit STOPPING RULE: if the curve is
# flat near its optimum at the resolution of a marginal purchase, the axis is settled and no further
# section on it is worth running. That is the conclusion §1961 reached about alpha only in hindsight.
#
# ARMS. flat (alpha = 0.30 constant) and tilts 28->32, 25->35, 22->38, 20->40, 15->45, 10->50, all at
# {mlp 768, attn 384} with a rank-640 map; plus one differing-table-rank arm so neither half of the
# derived control is vacuous.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1966's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
LO, HI = 'c5419', 'c16110'

# ordered narrowest (flat) to widest; WIDTH is the tilt span in points of alpha
GRID = [('flat', 'mix30m640', 0), ('t28_32', 'pat28_32m640', 4), ('t25_35', 'pat25_35m640', 10),
        ('t22_38', 'pat22_38m640', 16), ('t20_40', 'pat20_40m640', 20),
        ('t15_45', 'pat15_45m640', 30), ('t10_50', 'pat10_50m640', 40)]
LAB = [g[0] for g in GRID]

PLAN = [(arm, A384, lab) for lab, arm, _w in GRID] + [('mix25m512', A256, 'rank_control')]


def _argmin(x, cov, role):
    return min(range(len(LAB)), key=lambda i: x.ce(cov, role, LAB[i]))


def _interior(x):
    """the CE optimum is strictly inside the grid at both coverages -- not the flat end, not the widest"""
    n = x.count(lambda c, r: 0 < _argmin(x, c, r) < len(LAB) - 1)
    return n[LO] >= 2 and n[HI] >= 2


def _same_at_both(x):
    """and it sits at the same place at both coverages, within one grid step -- S1966 found no genuine
    coverage divergence on this axis and this tests that at full resolution"""
    return sum(1 for r in x.roles if abs(_argmin(x, HI, r) - _argmin(x, LO, r)) <= 1) >= 2


def _flat_near_optimum(x):
    """and the curve is flat near it: the three grid points around the argmin span under 0.5 milli-nats,
    at both coverages. THIS IS THE STOPPING RULE -- if it holds, the axis is settled at the resolution of
    a marginal purchase and no further section on the tilt is worth running"""
    def span(c, r):
        i = _argmin(x, c, r)
        near = [LAB[j] for j in (i - 1, i, i + 1) if 0 <= j < len(LAB)]
        v = [x.ce(c, r, a) for a in near]
        return max(v) - min(v)
    n = x.count(lambda c, r: span(c, r) < 0.0005)
    return n[LO] >= 2 and n[HI] >= 2


B.run(
    name='tilt_axis_settled',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_optimum_is_interior',
         'the tilt optimum is interior to the grid at both coverages (>=2 roles each)', _interior),
        ('pred_b_same_optimum_at_both',
         'and it is the same point at both coverages within one grid step (>=2 roles)', _same_at_both),
        ('pred_c_curve_is_flat_there',
         'and the curve is flat near it (<0.5 milli-nats over three points) -- the STOPPING RULE',
         _flat_near_optimum),
    ],
    refs=[('flat', B.PT + 'ops/one_build_with_tilt_results.json', 'spec_5419', LO, 0.0005)],
    paired_pairs=[('t25_35', 'flat'), ('t20_40', 'flat'), ('t10_50', 'flat')],
)
