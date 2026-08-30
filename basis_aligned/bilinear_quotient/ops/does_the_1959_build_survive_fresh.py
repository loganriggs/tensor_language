# DOES THE RESTORED BUILD OF RECORD SURVIVE THE SAME TEST IT WAS RESTORED BY?
#
# §2037 reverted the build of record to §1959's and §2039 closed the untruncation axis. But §1959's build
# was itself reached by supersessions -- §1946 through §1959 chose an allocation, a map rank and a blend,
# each on the same three eval roles that §2037 showed cannot certify a 0.1% margin. The restored build has
# exactly one out-of-sample measurement: §2037 saw it beat §1789's deployed design by +127.889 milli-nats.
# That is one comparison against one much older design, and it does not test the choices §1946-§1959 made.
#
# It would be a poor outcome to retract fifteen sections for in-sample selection and then leave the build
# they were measured against resting on the same method. This runs the §1959-era decisions on fresh rows:
# the map rank (§1959 chose 640 over 512), the allocation (§1957 chose attn 384 over 256), and the blend
# (§1961 chose alpha 0.30 over 0.10 and 0.50).
#
# Registered honestly: these were larger margins than the arc's -- §1959's map turnover was worth several
# milli-nats, not tenths -- so the expectation is that they hold. If they do not, the build of record has
# no validated ancestor and that is a much larger problem than §2037.
#
# ARMS. §1959's build; the same with map rank 512 (§1949's choice, superseded by §1959); with attention
# 256 (§1947's, superseded by §1957); with alpha 0.10 and 0.50 (§1961's rejected ends). A fallback variant
# for the inert half of the control, and one differing-table-rank arm for the other half.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2039's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
S1959 = 'build_1959'
FRESH = ('fresh',)

PLAN = [('mix30m640', BASE, S1959, None),
        ('mix30m512', BASE, 'map512_arm', None),          # §1949's map rank, superseded by §1959
        ('mix30m640', A256, 'attn256_arm', None),         # §1947's allocation, superseded by §1957
        ('mix10m640', BASE, 'alpha10_arm', None),         # §1961's rejected low end
        ('mix50m640', BASE, 'alpha50_arm', None),         # §1961's rejected high end
        ('map512', BASE, 'shipped_fb_control', None),     # all 36 sites, other fallback: the INERT pair
        ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _beats(x, lab):
    """nats by which §1959's build beats `lab` on the fresh window -- positive means §1959 was right"""
    return -x.tpool_full(LO, S1959, lab)['mean']


def _the_map_rank_choice_holds(x):
    """§1959's rank-640 map still beats §1949's rank-512 on fresh rows. This was the largest of the
    §1946-§1959 decisions and the one §1959 was named for; if it reverses, the restored build of record
    has no validated ancestor"""
    return _beats(x, 'map512_arm') > 0


def _the_allocation_choice_holds(x):
    """and §1957's attention 384 still beats §1947's attention 256 there. §2022 confirmed the attention
    axis out of sample at the rank level; this tests the allocation decision itself"""
    return _beats(x, 'attn256_arm') > 0


def _the_blend_choice_holds(x):
    """and §1961's alpha 0.30 still beats both rejected ends, 0.10 and 0.50. §2026 found alpha the one
    uniform sweep that was right on the selecting roles, and §2035 found the frequency axis coverage-
    stable; this is the same question on rows that had no say"""
    return _beats(x, 'alpha10_arm') > 0 and _beats(x, 'alpha50_arm') > 0


def _these_margins_are_larger_than_the_arcs(x):
    """and every one of these margins exceeds 3.300 milli-nats -- the largest single gain the retracted
    arc claimed. §2037's lesson was that a 0.1% margin selected on the measuring rows is not a result;
    if the §1946-§1959 decisions are an order of magnitude larger, they were never in that regime"""
    return all(_beats(x, l) > 0.003300
               for l in ('map512_arm', 'attn256_arm', 'alpha10_arm', 'alpha50_arm'))


B.run(
    name='does_the_1959_build_survive_fresh',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_map_rank_choice_holds',
         '§1959\'s rank-640 map still beats §1949\'s rank-512 on fresh rows', _the_map_rank_choice_holds),
        ('pred_b_the_allocation_choice_holds',
         'and §1957\'s attention 384 still beats §1947\'s attention 256 there',
         _the_allocation_choice_holds),
        ('pred_c_the_blend_choice_holds',
         'and §1961\'s alpha 0.30 still beats both rejected ends', _the_blend_choice_holds),
        ('pred_d_these_margins_are_larger_than_the_arcs',
         'and every margin exceeds 3.300 milli-nats, the largest gain the retracted arc claimed',
         _these_margins_are_larger_than_the_arcs),
    ],
    paired_pairs=[(S1959, 'map512_arm'), (S1959, 'attn256_arm'),
                  (S1959, 'alpha10_arm'), (S1959, 'alpha50_arm')],
)
