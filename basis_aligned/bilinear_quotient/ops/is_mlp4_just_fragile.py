# IS mlp4 SPECIAL TO COMPILATION, OR JUST A FRAGILE SITE?
#
# S1982: compiling the MLP at layer 4 alone -- all thirty-five other sites live -- costs 10.58-10.94
# nats, 3.9x the full thirty-six-site program, while layer 5 costs 2.0. The obvious deflationary reading
# is that layer 4 is simply a high-sensitivity site and ANY perturbation there would be catastrophic, in
# which case the finding is about mlp4 and not about compilation at all.
#
# meanrow is the null that separates them: every token gets the SAME row, the mean of the covered table.
# It keeps none of the table's content and only its context-freeness. If mean-substituting mlp4 is as bad
# as compiling it, the damage is context-freeness per se. If it is mild, the compiled row's CONTENT is
# what layer 6 cannot use, and "fragile site" is refuted either way by the layer-5 contrast.
#
# ARMS. mlp4 and mlp5 each substituted alone, by the compiled table and by the mean row; plus the pair
# fix from S1980 (compile attention 6 as well) under the mean null; plus the full program as the anchor.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 3 -- S1982's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
C = 'c5419'
ARM = 'mix30m640'
M4, M5 = [('mlp', 4)], [('mlp', 5)]

PLAN = [(ARM, A384, 'tab_mlp4', M4),
        (ARM, A384, 'tab_mlp5', M5),
        ('meanrow', A384, 'mean_mlp4', M4),
        ('meanrow', A384, 'mean_mlp5', M5),
        ('meanrow', A384, 'mean_mlp4_attn6', M4 + [('attn', 6)]),
        (ARM, A384, 'full_program', None),
        ('map512', A384, 'spec_partner', None)]


def _cost(x, r, lab):
    o = x.res[C][r][lab]['pooled']['overall']
    return o['ce_prog'] - o['ce_live']


def _mean_is_also_catastrophic(x):
    """the mean row at mlp4 costs at least half what the compiled table there does. If TRUE the damage is
    CONTEXT-FREENESS per se and the table's content is beside the point; if FALSE the compiled row's
    content is specifically what layer 6 cannot use"""
    return x.count(lambda c, r: _cost(x, r, 'mean_mlp4') >= 0.5 * _cost(x, r, 'tab_mlp4'))[C] >= 2


def _layer5_is_still_mild_under_the_null(x):
    """and layer 5 stays mild under the SAME substitution -- mean_mlp5 costs under half of mean_mlp4. This
    is what refutes 'layer 4 is just a fragile site': the two layers are perturbed identically and only
    one is catastrophic"""
    return x.count(lambda c, r: _cost(x, r, 'mean_mlp5') < 0.5 * _cost(x, r, 'mean_mlp4'))[C] >= 2


def _the_pair_fix_still_works(x):
    """and S1980's fix still applies under the null: compiling attention 6 as well removes most of the
    mean-row penalty at mlp4. If FALSE the pair mechanism is specific to the compiled table and S1980's
    account does not generalise to context-freeness"""
    return x.count(lambda c, r: _cost(x, r, 'mean_mlp4_attn6')
                   < 0.5 * _cost(x, r, 'mean_mlp4'))[C] >= 2


B.run(
    name='is_mlp4_just_fragile',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_mean_is_also_catastrophic',
         'a constant row at mlp4 costs at least half what the compiled table costs',
         _mean_is_also_catastrophic),
        ('pred_b_layer5_mild_under_the_same_null',
         'and layer 5 stays mild under the identical substitution -- not a fragile-site artefact',
         _layer5_is_still_mild_under_the_null),
        ('pred_c_pair_fix_generalises',
         'and compiling attention 6 still removes most of it under the null',
         _the_pair_fix_still_works),
    ],
    refs=[('full_program', B.PT + 'ops/what_we_built_results.json', 'converged', C, 0.0005)],
    paired_pairs=[('mean_mlp4', 'full_program'), ('tab_mlp4', 'full_program')],
)
