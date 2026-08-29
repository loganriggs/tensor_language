# S1965'S BOUNDARY CLAIM, RE-TESTED UNDER POOLING.
#
# §1965 concluded the per-token tilt "IS worth shipping at 16,110" because it cleared a 0.002-nat
# cheapness bar on 3/3 roles. §1966 corrected that on magnitude -- the flat build is better on CE on 2 of
# 3 roles. Neither section pooled, and §1971 then found that the role §1965 leaned on hardest,
# 16,110/skip1200 (where the tilt looked free at -0.49 milli-nats), carries HALF the positions of the
# other two. §1972 built the pooled instrument and named this as the claim that most needs it.
#
# ARMS. the flat converged build and the two tilts §1965 and §1967 argued over, at {mlp 768, attn 384}
# with a rank-640 map, at 16,110; plus one differing-table-rank arm so the derived control is two-sided.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §1965/§1966 on the right instrument.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
C = 'c16110'
FLAT, NARROW, BEST = 'flat30', 'pat20_30m640', 'pat25_35m640'

PLAN = [('mix30m640', A384, FLAT),
        ('pat20_30m640', A384, NARROW),          # §1965's arm
        ('pat25_35m640', A384, BEST),            # §1967's settled optimum
        ('mix25m512', A256, 'rank_control')]


def _s1965_arm_is_not_better(x):
    """§1965's arm is NOT better than flat once pooled -- pooled t >= 0, i.e. it does not improve CE.
    §1966 already said so on per-role magnitudes; this asks the instrument that weights by evidence"""
    return x.tpool(C, NARROW, FLAT) >= 0.0


def _s1967_arm_is_better(x):
    """but §1967's settled optimum IS better than flat pooled, at |t| >= 2 -- if FALSE, the tilt axis
    gains nothing at this coverage under any instrument and §1967's optimum is a 5,419-only result"""
    return x.tpool(C, BEST, FLAT) <= -2.0


def _pooling_disagrees_with_the_vote(x):
    """and the pooled verdict on §1965's arm differs from the 3-of-3 vote §1965 reported -- the concrete
    demonstration that a vote over unequal roles can point the other way from the evidence"""
    votes = sum(1 for r in x.roles if x.ce(C, r, NARROW) <= x.ce(C, r, FLAT) + 0.002)
    return votes >= 2 and x.tpool(C, NARROW, FLAT) > 0.0


B.run(
    name='s1965_repooled',
    plan=PLAN,
    coverages=[(C, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_s1965_arm_not_better',
         '§1965 arm is not an improvement once pooled (pooled t >= 0)', _s1965_arm_is_not_better),
        ('pred_b_s1967_arm_is_better',
         'but §1967 settled optimum is better pooled at |t| >= 2', _s1967_arm_is_better),
        ('pred_c_vote_and_pooling_disagree',
         'and the cheapness vote passes while pooling says the arm is worse', _pooling_disagrees_with_the_vote),
    ],
    refs=[(FLAT, B.PT + 'ops/tilt_axis_settled_results.json', 'flat', C, 0.0005)],
    paired_pairs=[(NARROW, FLAT), (BEST, FLAT)],
)
