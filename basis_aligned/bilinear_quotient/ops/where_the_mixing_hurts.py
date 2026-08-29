# WHERE DOES LIVE ATTENTION READING COMPILED ROWS DO THE DAMAGE?
#
# §1978: substituting only the MLPs costs 10.2-10.6 nats -- 3.6x the full 36-site price -- while
# substituting only attention costs 96% of it. The two arms are the halves of one partition and differ
# by a factor of four, so the penalty is not symmetric "mixing": it is specifically LIVE ATTENTION
# READING CONTEXT-FREE ROWS. §1978 named the open question as whether that requirement is global or
# local.
#
# This traces the curve. Every arm substitutes all 18 MLPs; they differ only in how many ATTENTION sites
# are also substituted, from the bottom up. k = 0 is §1978's catastrophic mlp_only; k = 18 is the full
# compiled program. If the damage is global, the curve falls smoothly across the whole range. If it is
# local, a few specific layers carry it and the curve has a step.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 3 -- §1978's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
C = 'c5419'
ARM = 'mix30m640'
MLPS = [('mlp', L) for L in range(18)]
KS = (0, 3, 6, 9, 12, 15, 18)


def sites(k):
    return MLPS + [('attn', L) for L in range(k)]


PLAN = [(ARM, A384, f'k{k:02d}', sites(k)) for k in KS] + [('map512', A384, 'spec_partner', None)]
LAB = [f'k{k:02d}' for k in KS]


def _cost(x, r, lab):
    o = x.res[C][r][lab]['pooled']['overall']
    return o['ce_prog'] - o['ce_live']


def _monotone_fall(x):
    """the penalty falls monotonically as more attention is compiled -- every step from k=0 to k=18
    should help, since each one removes another live-attention-reads-compiled-rows interface"""
    return sum(1 for r in x.roles
               if all(_cost(x, r, LAB[i]) < _cost(x, r, LAB[i - 1]) for i in range(1, len(LAB)))) >= 2


def _damage_is_local(x):
    """and it is LOCAL, not global: one step accounts for more than half the total fall from k=0 to
    k=18. If FALSE the requirement is global -- every live attention layer reading compiled rows costs
    about the same, and S1765's consistency is a property of the whole stack rather than of a few sites"""
    def steps(r):
        return [_cost(x, r, LAB[i - 1]) - _cost(x, r, LAB[i]) for i in range(1, len(LAB))]
    return sum(1 for r in x.roles
               if max(steps(r)) > 0.5 * sum(steps(r))) >= 2


def _first_layers_matter_most(x):
    """and the biggest step is in the EARLY layers (k <= 9), consistent with §1978's finding that
    compiling the early half costs more than the late half"""
    def steps(r):
        return [_cost(x, r, LAB[i - 1]) - _cost(x, r, LAB[i]) for i in range(1, len(LAB))]
    return sum(1 for r in x.roles
               if steps(r).index(max(steps(r))) <= 2) >= 2       # steps 0-2 span k=0..9


B.run(
    name='where_the_mixing_hurts',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_penalty_falls_monotonically',
         'the penalty falls at every step as more attention is compiled', _monotone_fall),
        ('pred_b_damage_is_local',
         'and one step carries more than half the total fall -- the requirement is local',
         _damage_is_local),
        ('pred_c_early_layers_carry_it',
         'and that step is in the early layers (k <= 9), as §1978 implies', _first_layers_matter_most),
    ],
    refs=[('k18', B.PT + 'ops/what_we_built_results.json', 'converged', C, 0.0005)],
    paired_pairs=[('k00', 'k18'), ('k09', 'k18')],
)
