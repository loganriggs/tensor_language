# THE ARC'S HEADLINE CLAIMS UNDER THE INSTRUMENT IT SHOULD HAVE HAD.
#
# §1971 found that skip1200 carries exactly half the scored positions of the other two roles (18,432 vs
# 36,864), so every 2-of-3 vote since §1946 has counted a half-sized role as an equal voter -- making a
# claim stronger than it reads when skip1200 dissents and weaker when it supports. The per-position data
# for every arm is cached, so a single paired test pooled across all three roles is available and
# weights each role by the evidence it actually carries.
#
# This re-tests the three claims the build rests on, under pooling rather than voting. Nothing here is a
# new build; it is the arc's own conclusions asked again with the right instrument.
#
# ARMS. the deployed design (§1789), the converged build (§1959's {mlp 768, attn 384} + rank-640 map at
# alpha 0.30), and that build with the §1967 tilt. 5,419, where the margins are largest (§1951).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1971's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
C = 'c5419'
DEP, CONV, TILT = 'deployed', 'converged', 'tilted'

PLAN = [('map64', None, DEP),                    # §1789's deployed design, full-rank tables
        ('mix30m640', A384, CONV),               # §1959/§1970's converged build
        ('pat25_35m640', A384, TILT)]            # §1967's settled tilt on top of it


def _converged_beats_deployed(x):
    """the converged build beats the deployed design POOLED, at |t| >= 10 -- S1951 reported this as
    'paired t = -19 to -29 across roles' and never pooled it. A pooled test on 92,160 positions should
    be at least as strong as the weakest role was, and this asks for a bar no single role set"""
    return x.tpool(C, CONV, DEP) <= -10.0


def _tilt_gain_is_not_significant(x):
    """and S1967's tilt -- worth 0.1 to 0.5 milli-nats, the whole axis three sections were spent on --
    is NOT significant even pooled, |t| < 2. If FALSE the tilt matters more than S1967 concluded and
    the stopping rule fired too early; if TRUE, three sections of work were inside the noise of the one
    instrument that could have said so on day one"""
    return abs(x.tpool(C, TILT, CONV)) < 2.0


def _pooling_beats_the_weakest_role(x):
    """and pooling is a real gain over voting: |pooled t| for the converged-vs-deployed comparison
    exceeds the largest single-role |t|, because it uses every position instead of the best third"""
    best_role = max(abs(x.t(C, r, CONV, DEP)) for r in x.roles)
    return abs(x.tpool(C, CONV, DEP)) > best_role


B.run(
    name='arc_under_pooling',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_converged_beats_deployed_pooled',
         'the converged build beats the deployed design pooled at |t| >= 10',
         _converged_beats_deployed),
        ('pred_b_tilt_is_inside_the_noise',
         'and §1967\'s whole tilt axis is not significant even pooled (|t| < 2)',
         _tilt_gain_is_not_significant),
        ('pred_c_pooling_beats_the_best_role',
         'and pooling is stronger than the strongest single role, as using all the data should be',
         _pooling_beats_the_weakest_role),
    ],
    refs=[(CONV, B.PT + 'ops/alpha_dense_results.json', 'a30', C, 0.0005)],
    paired_pairs=[(CONV, DEP), (TILT, CONV), (TILT, DEP)],
)
