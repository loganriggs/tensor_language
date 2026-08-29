# THE PREMISE NOBODY QUESTIONED: buy coverage instead of a better fallback.
#
# §1975 left the covered-vs-uncovered gap at 7.5pp on average -- 40% closed, and the largest named gap
# between the compiled program and the live model. Every attempt on it since §1937 has built a better
# FALLBACK. None has questioned the premise that an uncovered token must be served by one. The
# alternative is to extend coverage until the uncovered arm shrinks, and §1923 priced that against a
# build 18% more expensive than the one we now have.
#
# The comparison the arc never made: at a fixed parameter budget, is a dollar better spent on MORE
# COVERED TYPES or on a better fallback for the ones left over? Raising coverage 5,419 -> 16,110 costs
# 36*(NCOV_delta)*D of table; the fallback lever costs 36*R*2*D of map. Both are priced in the same unit
# and have never been put on the same axis.
#
# ARMS. at 16,110 (the higher coverage) the converged build and the DEPLOYED-era rank-64 map; at 5,419
# the same two. The 16,110 build spends its extra parameters on coverage; the 5,419 build can spend the
# difference on the fallback. Read across coverages, the two rows are the two ways to spend.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1975's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
LO, HI = 'c5419', 'c16110'
RICH, POOR = 'rich_fallback', 'poor_fallback'

PLAN = [('mix30m640', A384, RICH),          # the converged fallback
        ('map64', A384, POOR),              # §1789's rank-64 map, same tables
        ('map512', A384, 'spec_partner'),   # same table rank as the two above
        ('map512', None, 'rank_partner')]   # DIFFERENT table rank -- the runner failed the first
# version of this plan with control_is_two_sided=False because all three arms shared a spec, so the
# differing-pair half had nothing to check. Both halves are now non-empty.


def _coverage_beats_fallback(x):
    """raising coverage 5,419 -> 16,110 buys more CE than the whole fallback lever does at 5,419.
    If TRUE the arc optimised the cheaper axis and coverage is where the money should have gone"""
    cov_gain = [x.ce(LO, r, RICH) - x.ce(HI, r, RICH) for r in x.roles]
    fb_gain = [x.ce(LO, r, POOR) - x.ce(LO, r, RICH) for r in x.roles]
    return sum(1 for c, f in zip(cov_gain, fb_gain) if c > f) >= 2


def _fallback_matters_less_at_high_coverage(x):
    """and the fallback lever is worth less at 16,110 than at 5,419 -- §1936's arithmetic, since the
    uncovered arm is ~10% of positions there against ~24%"""
    return sum(1 for r in x.roles
               if (x.ce(HI, r, POOR) - x.ce(HI, r, RICH))
               < (x.ce(LO, r, POOR) - x.ce(LO, r, RICH))) >= 2


def _coverage_gain_is_significant(x):
    """and the coverage gain is real, not a scoring artefact: the converged build's pooled CE at 16,110
    is below its pooled CE at 5,419 on 3/3 roles. The two coverages score the SAME positions, so this is
    a like-for-like comparison and not a change of denominator"""
    return sum(1 for r in x.roles if x.ce(HI, r, RICH) < x.ce(LO, r, RICH)) == 3


B.run(
    name='coverage_vs_fallback',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_coverage_beats_the_fallback_lever',
         'raising coverage buys more CE than the entire fallback lever at 5,419',
         _coverage_beats_fallback),
        ('pred_b_fallback_worth_less_when_covered',
         'and the fallback lever is worth less at 16,110 than at 5,419',
         _fallback_matters_less_at_high_coverage),
        ('pred_c_coverage_gain_is_real',
         'and the converged build is better at 16,110 than at 5,419 on 3/3 roles',
         _coverage_gain_is_significant),
    ],
    refs=[(RICH, B.PT + 'ops/what_we_built_results.json', 'converged', LO, 0.0005)],
    paired_pairs=[(POOR, RICH)],
)
