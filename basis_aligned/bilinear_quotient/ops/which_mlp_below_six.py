# WHICH OF THE SIX MLPs?  -- resolving S1981's below_mlp to single sites.
#
# S1981: with attention at layer 6 live, compiling the six MLPs at layers 0-5 costs 10.60-10.96 nats --
# 3.9x the full 36-site program. Those six were compiled together. S1980 resolved a three-layer step to
# one layer by sweeping singles; this does the same for the six.
#
# Every arm leaves attention 6 live and compiles exactly ONE MLP, at layer 0 through 5, with all thirty-
# five other sites live. If one carries the damage, one arm approaches below_mlp's cost on its own.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 3 -- S1981's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
C = 'c5419'
ARM = 'mix30m640'
SINGLES = [f'mlp{L}' for L in range(6)]

PLAN = ([(ARM, A384, f'mlp{L}', [('mlp', L)]) for L in range(6)]
        + [(ARM, A384, 'all_six', [('mlp', L) for L in range(6)]),
           (ARM, A384, 'full_program', None),
           ('map512', A384, 'spec_partner', None)])


def _cost(x, r, lab):
    o = x.res[C][r][lab]['pooled']['overall']
    return o['ce_prog'] - o['ce_live']


def _one_mlp_carries_it(x):
    """a single compiled MLP below layer 6 costs more than half of all_six's price. If TRUE the damage
    resolves to one site on each side of the interface, which is as sharp as this line can get"""
    return x.count(lambda c, r: max(_cost(x, r, s) for s in SINGLES)
                   > 0.5 * _cost(x, r, 'all_six'))[C] >= 2


def _the_earliest_is_worst(x):
    """and it is the EARLIEST -- mlp0, whose compiled row propagates through the most live machinery
    before reaching layer 6. If FALSE the damage is not about propagation distance"""
    return x.count(lambda c, r: max(SINGLES, key=lambda s: _cost(x, r, s)) == 'mlp0')[C] >= 2


def _singles_are_superadditive(x):
    """and the six together cost MORE than the sum of the singles -- S1981 showed the cost is not
    additive over sites, and this asks which way it fails here. If FALSE they overlap instead, and one
    compiled MLP is already most of the damage the six do"""
    return x.count(lambda c, r: _cost(x, r, 'all_six')
                   > sum(_cost(x, r, s) for s in SINGLES))[C] >= 2


B.run(
    name='which_mlp_below_six',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_one_mlp_carries_it',
         'a single compiled MLP below layer 6 costs more than half of all six', _one_mlp_carries_it),
        ('pred_b_the_earliest_is_worst',
         'and the costliest single is mlp0, the earliest', _the_earliest_is_worst),
        ('pred_c_six_exceed_the_sum_of_singles',
         'and the six together cost more than the sum of the singles', _singles_are_superadditive),
    ],
    refs=[('full_program', B.PT + 'ops/what_we_built_results.json', 'converged', C, 0.0005)],
    paired_pairs=[('mlp0', 'full_program'), ('all_six', 'full_program')],
)
