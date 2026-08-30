# SWEEP THE BLEND ON ROWS THAT DID NOT CHOOSE IT.
#
# §2041 found the restored build's ancestry holding out-of-sample on three of four decisions, and one
# comparison inside the noise: §1961's alpha 0.30 beats the rejected 0.50 end by +1.321 milli-nats at
# t = +1.38 on 98,304 fresh positions. |t| < 2 -- 0.30 is not shown better than 0.50 there.
#
# That matters more than its size. §2026 measured alpha as the MOST SENSITIVE parameter in the build:
# moving it 0.20 at eight sites cost 11.0 milli-nats, where the entire retracted arc claimed 3.3. And
# alpha is free -- it reweights two fallback arms the build already pays for -- so any improvement is
# pure. §2027's grid put the optimum at 0.28 and §1967's stopping rule called the axis flat, but BOTH
# were measured on the three roles that chose 0.30, and §2037 showed that is the evidence which does not
# transfer.
#
# ARMS. alpha 0.10 / 0.20 / 0.30 / 0.40 / 0.50 / 0.60 / 0.70 on §1959's build. A fallback variant for the
# inert half of the control, and one differing-table-rank arm for the other half. The grid is deliberately
# wider than §1961's 0.24-0.36: after §2041 the fresh optimum cannot be assumed to lie near 0.30.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2041's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
FRESH = ('fresh',)
ALPHAS = (10, 20, 30, 40, 50, 60, 70)
LAB = [f'a{a}' for a in ALPHAS]
SHIPPED = 'a30'                       # §1959's blend, the build of record

PLAN = [(f'mix{a}m640', BASE, f'a{a}', None) for a in ALPHAS] + [
    ('map512', BASE, 'shipped_fb_control', None),                       # all 36 sites: the INERT pair
    ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _beats(x, a, b):
    """nats by which arm a beats arm b on the fresh window. An arm against itself is zero by definition:
    _argmax below ranges over the whole grid including the shipped blend, and ('a30','a30') is not a pair
    the derived control computes -- the first run raised on it, correctly, rather than scoring a False."""
    if a == b:
        return 0.0
    return -x.tpool_full(LO, a, b)['mean']


def _argmax(x):
    return max(LAB, key=lambda l: _beats(x, l, SHIPPED))


def _the_low_end_still_loses(x):
    """alpha 0.10 still loses decisively to 0.30 on fresh rows -- §2041 measured +34.213 milli-nats at
    t = +27.03, so this is a reproduction and an anchor for the rest of the grid"""
    return _beats(x, SHIPPED, 'a10') > 0.010


def _the_optimum_is_not_below_030(x):
    """and the fresh optimum is at 0.30 or above. §1961 and §2027 both put it at or just below 0.30 using
    the rows that chose it; if the fresh optimum sits lower, the blend was fitted downward and the
    build's most sensitive parameter is mis-set"""
    return ALPHAS[LAB.index(_argmax(x))] >= 30


def _no_alpha_beats_030_by_a_real_margin(x):
    """and no alpha beats 0.30 by more than 3.300 milli-nats -- the largest gain the retracted arc
    claimed, and the scale at which §2037 showed in-sample selection cannot be trusted. If FALSE the
    fresh rows want a materially different blend and the build of record needs its one free parameter
    re-set on evidence that transfers"""
    return all(_beats(x, l, SHIPPED) <= 0.003300 for l in LAB)


def _the_axis_is_flat_near_the_optimum(x):
    """and the three grid points around the fresh optimum span under 3.300 milli-nats -- §1967's stopping
    rule, rescaled from its 0.5 milli-nat bar to the scale §2037 established as the floor for a claim.
    If TRUE the axis is flat where it matters and 0.30 is as good as anything nearby"""
    i = LAB.index(_argmax(x))
    near = [_beats(x, LAB[j], SHIPPED) for j in (i - 1, i, i + 1) if 0 <= j < len(LAB)]
    return max(near) - min(near) < 0.003300


B.run(
    name='sweep_alpha_on_fresh_rows',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_low_end_still_loses',
         '§2041\'s +34.213 milli-nat margin over alpha 0.10 reproduces above 10 milli-nats',
         _the_low_end_still_loses),
        ('pred_b_the_optimum_is_not_below_030',
         'and the fresh optimum sits at 0.30 or above', _the_optimum_is_not_below_030),
        ('pred_c_no_alpha_beats_030_by_a_real_margin',
         'and no alpha beats 0.30 by more than 3.300 milli-nats', _no_alpha_beats_030_by_a_real_margin),
        ('pred_d_the_axis_is_flat_near_the_optimum',
         'and the three points around the fresh optimum span under 3.300 milli-nats',
         _the_axis_is_flat_near_the_optimum),
    ],
    paired_pairs=[(l, SHIPPED) for l in LAB if l != SHIPPED],
)
