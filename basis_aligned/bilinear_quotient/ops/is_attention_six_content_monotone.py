# IS ATTENTION 6'S CONTENT VALUE MONOTONE IN THE COMPILED SITE'S DAMAGE, OR JUST TWO POINTS?
#
# §1999 second-class confirmed §1998's presence/content split in SIGN but not in SIZE: a mean row at
# attention 6 costs +0.212 when the compiled MLP is mlp2 and +0.095 when it is mlp4. Two points make a
# difference, not a trend — and the direction is the surprising one. The lone-site damages run mlp2 4.813,
# mlp3 6.574, mlp4 10.669, so the site that is MOST broken by compilation gets the LEAST out of attention
# 6's content, which is the opposite of what "more damage needs more help" predicts.
#
# mlp3 is the one cheap point that separates a monotone relationship from two points on a line, and mlp5 —
# whose lone-site damage is 2.022, below all three — extends the range downward in the same run.
#
# ARMS. mlp3 and mlp5 with attention 5,6, each with all tables and with a mean row at attention 6; the mlp2
# and mlp4 versions rebuilt here so all four sites are measured in-run; the full 36-site program with a
# fallback variant for the inert half of the control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1999's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
MLPS = (2, 3, 4, 5)
LONE = {2: 4.813, 3: 6.574, 4: 10.669, 5: 2.022}      # §1987/§1988, skip7000, lone compiled MLP


def _sites(L):
    return [('mlp', L), ('attn', 5), ('attn', 6)]


PLAN = [(ARM, BASE, f'm{L}_tab', _sites(L)) for L in MLPS] + \
       [(f'meanrow@attn6+mix30m640@mlp{L},attn5', BASE, f'm{L}_mean6', _sites(L)) for L in MLPS] + [
    (ARM, BASE, 'full_program', None),                 # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),         # all 36 sites, other fallback: the INERT pair
    (ARM, A256, 'rank_control', None)]                 # differing table rank: the other half


def _d6(x, role, L):
    return x.penalty(C, role, f'm{L}_mean6') - x.penalty(C, role, f'm{L}_tab')


def _anchors_reproduce(x):
    """the mlp2 and mlp4 attention-6 content values rebuild to §1998/§1999's figures within 0.005 nats on
    all three roles -- +0.212/+0.216/+0.191 and +0.095/+0.098/+0.080. If they do not, the four-site
    comparison below is not measuring what it claims"""
    want = {2: (0.212, 0.216, 0.191), 4: (0.095, 0.098, 0.080)}
    return all(abs(_d6(x, r, L) - v) < 0.005
               for L, vs in want.items() for r, v in zip(x.roles, vs))


def _content_value_is_monotone(x):
    """and attention 6's content value falls monotonically as the compiled site's lone damage rises:
    ordering the four sites by LONE damage (mlp5 2.02 < mlp2 4.81 < mlp3 6.57 < mlp4 10.67) gives a
    strictly decreasing sequence of d6 on >=2 roles. If TRUE the inverse relationship §1999 found in two
    points is a trend; if FALSE it is two points and the pair dependence has no simple form"""
    order = sorted(MLPS, key=lambda L: LONE[L])
    return sum(1 for r in x.roles
               if all(_d6(x, r, order[i]) > _d6(x, r, order[i + 1]) for i in range(len(order) - 1))) >= 2


def _content_is_positive_everywhere(x):
    """and attention 6's content is worth something at every one of the four sites -- d6 strictly positive
    on all three roles at all four. §1998 and §1999 found it positive at two; a site where it is zero or
    negative would mean attention 6 is not always a content site"""
    return all(_d6(x, r, L) > 0 for r in x.roles for L in MLPS)


B.run(
    name='is_attention_six_content_monotone',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_anchors_reproduce',
         'the mlp2 and mlp4 attention-6 content values rebuild to §1998/§1999 within 0.005 nats, 3/3 roles',
         _anchors_reproduce),
        ('pred_b_content_value_is_monotone',
         'and it falls monotonically as the compiled site\'s lone damage rises, across four sites (>=2 roles)',
         _content_value_is_monotone),
        ('pred_c_content_is_positive_everywhere',
         'and it is strictly positive at all four sites on all three roles', _content_is_positive_everywhere),
    ],
    refs=[('m4_tab', B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', C, 0.0005),
          ('m2_tab', B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a56', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('m3_mean6', 'm3_tab'), ('m5_mean6', 'm5_tab'), ('m2_mean6', 'm4_mean6')],
)
