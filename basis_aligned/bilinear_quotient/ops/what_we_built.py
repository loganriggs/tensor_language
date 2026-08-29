# WHAT DID THE ARC ACTUALLY BUILD?  -- the converged build characterised, not tuned.
#
# §1946-§1974 moved the compiled program from §1789's deployed design to a build 66.3M cheaper and
# 69.238 milli-nats better (pooled t = -42.76, §1972), and settled all five parameters under the
# criterion each requires (§1969). Every one of those sections asked "is this point better than that
# one". None asked what the resulting object looks like against the live model.
#
# §1936 is the last section that did: it found the compiled program serves covered inputs at 37.5%
# kept-fraction and uncovered inputs at 28.7%, and named that gap as the open cost lever. The whole arc
# since then has spent on the uncovered arm (§1953/§1954: the fallback carries essentially the entire
# margin), so the gap should have narrowed -- and nobody has looked.
#
# ARMS. §1789's deployed design and the converged build. Both coverages.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- characterisation, not a parameter.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
LO, HI = 'c5419', 'c16110'
DEP, CONV = 'deployed', 'converged'

# a second arm at the SAME table rank as CONV, so the derived covered-input control has a same-spec
# pair to check and neither half is vacuous (§1957). The runner FAILED the first version of this plan
# for exactly that, which is the protection working.
PLAN = [('map64', None, DEP), ('mix30m640', A384, CONV), ('map512', A384, 'spec_partner')]


def _kfc(x, cov, r, arm, cls):
    return x.res[cov][r][arm][cls]['overall']['top1_acc_prog'] / max(
        x.res[cov][r][arm][cls]['overall']['top1_acc_live'], 1e-9)


def _tables_still_dominate(x):
    """the build still serves COVERED inputs better than uncovered ones, by >=5pp of kept-fraction --
    S1936 measured 37.5% against 28.7%. If FALSE the arc has closed the gap it named as the open lever"""
    n = x.count(lambda c, r: _kfc(x, c, r, CONV, 'covered_input')
                - _kfc(x, c, r, CONV, 'uncovered_input') >= 0.05)
    return n[LO] >= 2 and n[HI] >= 2


def _gap_narrowed(x):
    """but the gap is NARROWER than the deployed design's, because every gain since S1946 landed on the
    uncovered arm (S1953/S1954). If FALSE the arc improved both arms equally, which would contradict the
    attribution those two sections rest on"""
    def gap(c, r, a):
        return _kfc(x, c, r, a, 'covered_input') - _kfc(x, c, r, a, 'uncovered_input')
    n = x.count(lambda c, r: gap(c, r, CONV) < gap(c, r, DEP))
    return n[LO] >= 2 and n[HI] >= 2


def _unseen_is_still_the_floor(x):
    """and the unseen-target bucket is still the worst of the five, at both coverages -- the residual
    S1937/S1938 showed is structural and S1955/S1956/S1963/S1973 could not buy back"""
    n = x.count(lambda c, r: x.kf(c, r, CONV, x.bot) == min(x.kf(c, r, CONV, b) for b in x.buckets))
    return n[LO] >= 2 and n[HI] >= 2


B.run(
    name='what_we_built',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_covered_arm_still_ahead',
         'the build still serves covered inputs >=5pp better than uncovered ones',
         _tables_still_dominate),
        ('pred_b_the_gap_narrowed',
         'but the covered-vs-uncovered gap is narrower than the deployed design was', _gap_narrowed),
        ('pred_c_unseen_is_the_floor',
         'and the unseen-target bucket is still the worst of the five', _unseen_is_still_the_floor),
    ],
    refs=[(CONV, B.PT + 'ops/arc_under_pooling_results.json', 'converged', LO, 0.0005)],
    paired_pairs=[(CONV, DEP)],
)
