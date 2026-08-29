# DOES §1963 HOLD ON THE BUILD WE WOULD ACTUALLY SHIP?  -- and the acid test for B.run().
#
# §1963 tested the per-token alpha at §1960's COMPROMISE allocation ({mlp 768, attn 320}, rank-576 map).
# The deployed-coverage build §1957/§1959 actually arrived at is {mlp 768, attn 384} with a rank-640 map,
# and §1963's conclusion -- the tilt raises the unseen bucket by the intended mechanism and still pays
# more CE than it is worth -- has never been checked there.
#
# This is also the first experiment written against B.run(). The hand-forked version of exactly this
# experiment (ops/tilt_on_the_shipped_build.py) died on `ncov['c16110']`, a two-coverage control clause
# inherited into a single-coverage fork -- the eighth distinct fork-residue failure of the session. Here
# the controls are DERIVED from the plan, so that clause cannot exist to be left behind.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 2 -- §1963 at the shipped point.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
FLAT, WIDE, NARROW = 'flat30', 'pat10_40m640', 'pat20_30m640'

PLAN = [('mix30m640', A384, FLAT),
        ('mix25m640', A384, 'flat25'),
        ('pat10_40m640', A384, WIDE),
        ('pat20_30m640', A384, NARROW),
        ('mix25m512', A256, 'rank_control')]     # differing table rank: keeps the control two-sided

C = 'c5419'


def _helps(x):
    """some tilt arm raises the unseen bucket above the flat build"""
    n = x.count(lambda c, r: max(x.kf(c, r, a, x.bot) for a in (WIDE, NARROW)) > x.kf(c, r, FLAT, x.bot))
    return n[C] >= 2


def _still_costs(x):
    """and it still costs more than 0.002 nats of pooled CE -- §1963's finding, reproduced"""
    def worst(c, r):
        return max(x.ce(c, r, a) - x.ce(c, r, FLAT) for a in (WIDE, NARROW))
    return x.count(lambda c, r: worst(c, r) > 0.002)[C] >= 2


def _widest_strongest(x):
    """and the widest tilt still beats the narrowest, which is what shows it acts through its mechanism"""
    return x.count(lambda c, r: x.kf(c, r, WIDE, x.bot) > x.kf(c, r, NARROW, x.bot))[C] >= 2


B.run(
    name='tilt_shipped_declarative',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_tilt_still_helps',
         'the tilt still raises the unseen bucket on the shipped build (>=2 roles)', _helps),
        ('pred_b_still_costs_too_much',
         'and it STILL costs >0.002 nats of pooled CE (>=2 roles) -- §1963 reproduced', _still_costs),
        ('pred_c_widest_still_strongest',
         'and the widest tilt still beats the narrowest (>=2 roles)', _widest_strongest),
    ],
    # the reproduction anchor is READ from the artifact that published it, never retyped
    refs=[('flat25', B.PT + 'ops/map_curve_turnover_results.json', 'map640', None, 0.0005)],
    paired_pairs=[(WIDE, FLAT), (NARROW, FLAT)],
)
