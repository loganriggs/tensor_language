# IS IT THREE LAYERS OR ONE?  -- resolving S1979's step to single layers.
#
# §1979 found that with all 18 MLPs substituted, compiling attention at layers 6, 7 and 8 recovers
# 88-89% of the 3.6x partial-compilation penalty. That step spans three layers because the sweep moved in
# threes. Whether the damage is one layer, two or genuinely all three is one sweep away and stays
# entirely inside §1765's frame -- unlike the architectural question §1979 also raised, which is a
# decision about what the program IS and is not mine to take.
#
# ARMS. all 18 MLPs substituted throughout; attention substituted at layers 0-5 in every arm (the cheap
# prefix), then adding 6, then 6-7, then 6-8, then 6-8 plus the rest. If one layer carries it, one step
# takes nearly the whole fall.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 3 -- §1979's within-frame open.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
C = 'c5419'
ARM = 'mix30m640'
MLPS = [('mlp', L) for L in range(18)]
BASE6 = [('attn', L) for L in range(6)]
STEPS = [('a6only', []), ('a7', [6]), ('a8', [6, 7]), ('a9', [6, 7, 8])]
LAB = ['n0', 'n1', 'n2', 'n3']

PLAN = ([(ARM, A384, LAB[i], MLPS + BASE6 + [('attn', L) for L in STEPS[i][1]])
         for i in range(4)]
        + [(ARM, A384, 'full', None), ('map512', A384, 'spec_partner', None)])


def _cost(x, r, lab):
    o = x.res[C][r][lab]['pooled']['overall']
    return o['ce_prog'] - o['ce_live']


def _one_layer_carries_it(x):
    """a single layer accounts for more than half the fall from n0 to n3 -- if TRUE the mixing damage is
    one attention layer reading context-free rows, which is a far sharper statement than S1979'sthree-layer
    step and would name a specific site"""
    def steps(r):
        return [_cost(x, r, LAB[i - 1]) - _cost(x, r, LAB[i]) for i in range(1, 4)]
    return sum(1 for r in x.roles if max(steps(r)) > 0.5 * sum(steps(r))) >= 2


def _all_three_are_positive(x):
    """and every one of the three layers helps -- if some step is negative the layers interact and
    'which layer' is not well posed"""
    return sum(1 for r in x.roles
               if all(_cost(x, r, LAB[i]) < _cost(x, r, LAB[i - 1]) for i in range(1, 4))) >= 2


def _step_matches_S1979(x):
    """and n0 -> n3 reproduces §1979's k=6 -> 9 fall of 6.61-6.77 nats within 0.05 -- the same two
    builds by a different route, so a mismatch means one of the two sweeps is mis-specified"""
    return sum(1 for r in x.roles
               if abs((_cost(x, r, 'n0') - _cost(x, r, 'n3')) - 6.68) < 0.15) >= 2


B.run(
    name='which_of_the_three',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_one_layer_carries_it',
         'a single attention layer carries more than half the mixing damage', _one_layer_carries_it),
        ('pred_b_every_step_helps',
         'and each of layers 6, 7, 8 helps when compiled', _all_three_are_positive),
        ('pred_c_reproduces_s1979',
         'and the n0->n3 fall reproduces §1979 k=6->9 within 0.15 nats', _step_matches_S1979),
    ],
    refs=[('full', B.PT + 'ops/what_we_built_results.json', 'converged', C, 0.0005)],
    paired_pairs=[('n3', 'n0'), ('n3', 'full')],
)
