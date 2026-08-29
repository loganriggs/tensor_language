# PIN ALPHA, THE ONE AXIS THAT FAILS THE STOPPING RULE.
#
# §1968 applied §1967's stopping rule to §1961's own artifact and alpha failed on 6 of 6 role-coverage
# cells: the three grid points around its optimum span 2.10 / 3.48 / 1.19 milli-nats at 5,419 and
# 0.57 / 1.08 / 0.51 at 16,110, against a 0.50 bar and up to seven times the tilt's. It is not a grid
# artefact -- the two grids have comparable spacing. §1961 swept 10/20/25/30/40/50 and never looked
# between 0.25 and 0.35, which is where the optimum sits on 5 of 6 cells.
#
# So alpha is the one axis worth pinning, and this pins it on a grid four times finer, carrying the same
# stopping rule so the axis either settles or says what a one-step error costs.
#
# ARMS. alpha = 24 / 26 / 28 / 30 / 32 / 34 / 36 percent, all with a rank-640 map at {mlp 768, attn 384};
# plus one differing-table-rank arm so neither half of the derived control is vacuous.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1968's open consequence.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
LO, HI = 'c5419', 'c16110'
AL = (24, 26, 28, 30, 32, 34, 36)
LAB = [f'a{a}' for a in AL]

PLAN = [(f'mix{a}m640', A384, f'a{a}') for a in AL] + [('mix25m512', A256, 'rank_control')]


def _argmin(x, cov, role):
    return min(range(len(AL)), key=lambda i: x.ce(cov, role, LAB[i]))


def _interior(x):
    """the optimum is interior to this finer grid at both coverages -- if it sits at an end, 0.24-0.36
    is still the wrong window and the axis is not pinned"""
    n = x.count(lambda c, r: 0 < _argmin(x, c, r) < len(AL) - 1)
    return n[LO] >= 2 and n[HI] >= 2


def _settles_now(x):
    """and at this resolution the three points around it span under 0.5 milli-nats -- §1967's stopping
    rule, which alpha failed on §1961's coarser grid. If TRUE the axis is settled and the failure was
    about resolution after all; if FALSE alpha is genuinely sharp and a one-step error costs real CE"""
    def span(c, r):
        i = _argmin(x, c, r)
        v = [x.ce(c, r, LAB[j]) for j in (i - 1, i, i + 1) if 0 <= j < len(AL)]
        return max(v) - min(v)
    n = x.count(lambda c, r: span(c, r) < 0.0005)
    return n[LO] >= 2 and n[HI] >= 2


def _same_at_both(x):
    """and the optimum is the same at both coverages within one grid step -- alpha trades the neighbour
    against the map and both act only on uncovered inputs, so §1961's coverage-independence should hold
    at four times the resolution"""
    return sum(1 for r in x.roles if abs(_argmin(x, HI, r) - _argmin(x, LO, r)) <= 1) >= 2


B.run(
    name='alpha_dense',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_optimum_is_interior',
         'the alpha optimum is interior to the 0.24-0.36 window at both coverages (>=2 roles each)',
         _interior),
        ('pred_b_settles_at_this_resolution',
         'and the three points around it now span <0.5 milli-nats -- §1967 stopping rule', _settles_now),
        ('pred_c_same_optimum_at_both',
         'and it is the same point at both coverages within one grid step', _same_at_both),
    ],
    refs=[('a30', B.PT + 'ops/tilt_axis_settled_results.json', 'flat', LO, 0.0005)],
    paired_pairs=[('a24', 'a30'), ('a36', 'a30')],
)
