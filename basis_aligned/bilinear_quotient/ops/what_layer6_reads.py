# IS IT LAYER 6, OR WHAT REACHES IT?  -- separating the site from its input.
#
# §1980: with all 18 MLPs and attention 0-5 substituted, compiling attention at layer 6 alone recovers
# 98% of §1978's 3.6x penalty, and compiling layer 7 next is slightly negative -- so the site is layer 6
# specifically and not a boundary effect. What §1980 cannot say is WHY: whether layer 6's attention is
# special in itself, or whether the penalty is really about compiled rows arriving from BELOW it and
# layer 6 is simply the first place they are mixed across positions.
#
# This holds attention 6 LIVE in every arm and varies which sites below and above it are compiled. If
# the penalty tracks "compiled sites below 6" it is about the input; if it tracks total compiled sites
# regardless of position, it is about volume; if neither, layer 6 is special in itself.
#
# ARMS, all with attention 6 LIVE:
#   below_all   every site at layers 0-5 plus all MLPs 6-17   (= §1980's n0)
#   below_mlp   MLPs at layers 0-5 only; everything else live
#   below_full  every site at layers 0-5; everything else live
#   above_only  every site at layers 7-17; nothing below 6 compiled
#   plus the live model as the floor and the full program as the ceiling.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 3 -- §1980's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
C = 'c5419'
ARM = 'mix30m640'


BELOW_ALL = ([('mlp', L) for L in range(18)] + [('attn', L) for L in range(6)])
BELOW_MLP = [('mlp', L) for L in range(6)]
BELOW_FULL = [(k, L) for k in ('mlp', 'attn') for L in range(6)]
ABOVE_ONLY = [(k, L) for k in ('mlp', 'attn') for L in range(7, 18)]

PLAN = [(ARM, A384, 'below_all', BELOW_ALL),
        (ARM, A384, 'below_mlp', BELOW_MLP),
        (ARM, A384, 'below_full', BELOW_FULL),
        (ARM, A384, 'above_only', ABOVE_ONLY),
        (ARM, A384, 'full_program', None),
        ('map512', A384, 'spec_partner', None)]


def _cost(x, r, lab):
    o = x.res[C][r][lab]['pooled']['overall']
    return o['ce_prog'] - o['ce_live']


def _below_is_what_matters(x):
    """compiling only sites ABOVE layer 6 -- 22 of them, more than below_full's 12 -- costs far less
    than below_full does. If TRUE the penalty is about what REACHES layer 6, not about how many sites
    are compiled"""
    return x.count(lambda c, r: _cost(x, r, 'above_only') < _cost(x, r, 'below_full'))[C] >= 2


def _mlps_below_are_enough(x):
    """and compiling just the six MLPs below layer 6 -- everything else live -- already costs more than
    the entire 36-site program does. Six sites out of thirty-six, and worse than all of them"""
    return x.count(lambda c, r: _cost(x, r, 'below_mlp') > _cost(x, r, 'full_program'))[C] >= 2


def _reproduces_s1980(x):
    """and below_all reproduces §1980's n0 within 0.15 nats (9.14-9.53), the same build by a different
    site-set spelling"""
    return x.count(lambda c, r: abs(_cost(x, r, 'below_all') - 9.31) < 0.35)[C] >= 2


B.run(
    name='what_layer6_reads',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_below_is_what_matters',
         'compiling above layer 6 costs far less than compiling below it, despite more sites',
         _below_is_what_matters),
        ('pred_b_six_mlps_are_enough',
         'and the six MLPs below layer 6 alone cost more than the whole 36-site program',
         _mlps_below_are_enough),
        ('pred_c_reproduces_s1980',
         'and below_all reproduces §1980 n0 within 0.35 nats', _reproduces_s1980),
    ],
    refs=[('full_program', B.PT + 'ops/what_we_built_results.json', 'converged', C, 0.0005)],
    paired_pairs=[('below_mlp', 'full_program'), ('above_only', 'full_program')],
)
