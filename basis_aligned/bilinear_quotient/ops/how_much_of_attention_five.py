# ARE THE TWO LAYERS OF THE THRESHOLD DOING THE SAME JOB?
#
# §1996 swept attention 6's table rank and found it needs almost no capacity: rank 16 lands at 2.137
# against rank 384's 1.971, so a 24-fold cut costs 0.166 nats where removing the site entirely costs
# 0.611. Whatever attention 6 supplies is low-dimensional.
#
# The threshold names TWO layers and only one has been swept. §1992 already showed they are not
# interchangeable — attention 5 alone gets mlp2 to 2.556 while attention 6 alone leaves it at 4.876, and
# at mlp4 the same attention-5-alone arm leaves 8.021. If attention 5 is equally shallow, the requirement
# is about presence at both sites and not about capacity at either. If it is steep, the two layers are
# doing different jobs, and that asymmetry is the first mechanical handle this line has offered.
#
# ARMS. mlp2 + attention 5,6 with attention 5's table at rank 16 / 32 / 64 / 128 / 384, everything else at
# {mlp 768, attn 384}; the rank-384 arm is §1992's mlp2_a56 exactly. Plus mlp2 + attention 6 alone (the
# arm that omits attention 5), the full 36-site program and a fallback variant of it for the inert half of
# the control, and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1996's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
SITES2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
RANKS = (16, 32, 64, 128, 384)
A256 = {'mlp': 768, 'attn': 256}
NO5 = 'mlp2_a6'

# §1996's attention-6 sweep, for the comparison the predicates make
SIX = {16: 2.137, 32: 2.109, 64: 2.060, 128: 2.011, 384: 1.971}     # skip7000, 5,419


def _spec(r):
    return {'mlp': 768, 'attn': 384, ('attn', 5): r}


PLAN = [(ARM, _spec(r), f'r{r}', SITES2) for r in RANKS] + [
    (ARM, BASE, NO5, [('mlp', 2), ('attn', 6)]),          # §1992: 4.876 / 5.355 / 5.016
    (ARM, BASE, 'full_program', None),                    # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),            # all 36 sites, other fallback: the INERT pair
    (ARM, A256, 'rank_control', None)]                    # differing table rank: the other half


def _full_rank_reproduces(x):
    """the rank-384 arm reproduces §1992's mlp2_a56 to 0.005 nats on all three roles -- naming attention 5
    at the rank its kind already carried must be a no-op, and if it is not, nothing else here means
    anything"""
    want = (1.971, 2.090, 1.952)
    return all(abs(x.penalty(C, r, 'r384') - v) < 0.005 for r, v in zip(x.roles, want))


def _attention_five_is_steeper(x):
    """and cutting attention 5 to rank 16 costs MORE than the same cut at attention 6 did -- §1996
    measured +0.166 nats there. Registered directionally on the skip7000 role where §1996's figures are
    quoted, and on >=2 roles by the same margin. If TRUE the two layers are doing different jobs; if
    FALSE the threshold is about presence at both and capacity at neither"""
    gap6 = SIX[16] - SIX[384]
    return sum(1 for r in x.roles
               if x.penalty(C, r, 'r16') - x.penalty(C, r, 'r384') > gap6) >= 2


def _low_rank_still_clears_the_threshold(x):
    """and even at rank 16 attention 5 keeps the arm on the good side: under 2.5 nats on >=2 roles,
    against 4.876 for the arm that omits attention 5 altogether. Whatever the capacity story, PRESENCE is
    worth far more than rank at both sites"""
    return sum(1 for r in x.roles if x.penalty(C, r, 'r16') < 2.5) >= 2


B.run(
    name='how_much_of_attention_five',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_rank_reproduces',
         'naming attention 5 at rank 384 reproduces §1992\'s mlp2_a56 to 0.005 nats on 3/3 roles',
         _full_rank_reproduces),
        ('pred_b_attention_five_is_steeper',
         'and cutting attention 5 to rank 16 costs more than the same cut at attention 6 (+0.166, >=2 roles)',
         _attention_five_is_steeper),
        ('pred_c_low_rank_still_clears_the_threshold',
         'and rank 16 at attention 5 still lands under 2.5 nats -- presence beats capacity at both sites',
         _low_rank_still_clears_the_threshold),
    ],
    refs=[(NO5, B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a6', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('r16', 'r384'), ('r32', 'r384'), ('r384', NO5)],
)
