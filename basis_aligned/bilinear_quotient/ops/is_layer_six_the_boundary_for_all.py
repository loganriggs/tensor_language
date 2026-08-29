# IS LAYER 6 THE BOUNDARY FOR EVERY COMPILED MLP, OR DOES EACH SITE HAVE ITS OWN?
#
# §1985 and §1986 established a path rule from two sites: mlp4 needs attention 4–6 compiled and nothing
# else (1.745 nats, against 10.669 alone), and mlp12 — above layer 6 — needs nothing at all (0.034). Every
# result names layer 6 and none explains it, and two sites either side is thin evidence for a boundary.
#
# The rule makes two testable claims about sites it has never seen. A compiled mlp2 should be catastrophic
# alone, should STAY catastrophic if its path is closed only part-way, and should become cheap only when
# the path reaches 6. A compiled mlp8, above the boundary, should be nearly free like mlp12.
#
# §1986 also taught a bar lesson: its pred_a and pred_c were registered two-sided ("within 0.05 nats") and
# failed on results that strengthened the rule. Where the rule predicts a DIRECTION, this registers one.
#
# ARMS. mlp2 alone; mlp2 + attention 2–3 (path closed short of 6); mlp2 + attention 2–6 (path closed);
# mlp8 alone; mlp4 + attention 4–6 as the §1986 anchor; the full 36-site program; and one fallback variant
# of it so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1986's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
M2, M8 = [('mlp', 2)], [('mlp', 8)]
SHORT = [('attn', L) for L in (2, 3)]
LONG = [('attn', L) for L in (2, 3, 4, 5, 6)]
PATH4 = [('mlp', 4)] + [('attn', L) for L in (4, 5, 6)]

ONLY2, SHORT2, LONG2, ONLY8 = 'mlp2', 'mlp2_short', 'mlp2_path', 'mlp8'
ANCHOR, FULL = 'mlp4_path', 'full_program'

PLAN = [(ARM, A384, ONLY2, M2),
        (ARM, A384, SHORT2, M2 + SHORT),
        (ARM, A384, LONG2, M2 + LONG),
        (ARM, A384, ONLY8, M8),
        (ARM, A384, ANCHOR, PATH4),          # §1986: 1.745 / 1.853 / 1.682
        (ARM, A384, FULL),                   # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]  # same rank, same sites, other fallback: INERT


def _below_six_is_catastrophic(x):
    """a lone compiled mlp2 costs more than 5 nats on >=2 roles, as mlp4 did at 10.669 -- the damage
    below layer 6 is not specific to layer 4"""
    return sum(1 for r in x.roles if x.penalty(C, r, ONLY2) > 5.0) >= 2


def _the_path_must_reach_six(x):
    """and closing mlp2's path only to layer 3 does NOT rescue it -- mlp2_short still costs more than 5
    nats -- while closing it to layer 6 does, under 3 nats, on >=2 roles each. Registered directionally:
    §1986's two-sided bars failed on results that strengthened the rule"""
    short = sum(1 for r in x.roles if x.penalty(C, r, SHORT2) > 5.0)
    full = sum(1 for r in x.roles if x.penalty(C, r, LONG2) < 3.0)
    return short >= 2 and full >= 2


def _above_six_is_free(x):
    """and a lone compiled mlp8, above the boundary, costs under 0.5 nats on >=2 roles -- mlp12 measured
    0.034. If FALSE the boundary is not layer 6 for every MLP and each site has its own"""
    return sum(1 for r in x.roles if x.penalty(C, r, ONLY8) < 0.5) >= 2


B.run(
    name='is_layer_six_the_boundary_for_all',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_below_six_is_catastrophic',
         'a lone compiled mlp2 costs more than 5 nats (>=2 roles) -- not specific to layer 4',
         _below_six_is_catastrophic),
        ('pred_b_the_path_must_reach_six',
         'and mlp2 stays above 5 nats with its path closed to layer 3, and drops under 3 when it reaches 6',
         _the_path_must_reach_six),
        ('pred_c_above_six_is_free',
         'and a lone compiled mlp8 costs under 0.5 nats -- the boundary is layer 6 for every MLP',
         _above_six_is_free),
    ],
    refs=[(ANCHOR, B.PT + 'ops/how_far_does_the_prefix_reach_results.json', 'mlp4_path', C, 0.0005),
          (FULL, B.PT + 'ops/how_far_does_the_prefix_reach_results.json', FULL, C, 0.0005)],
    paired_pairs=[(LONG2, SHORT2), (LONG2, ANCHOR), (ONLY8, ONLY2)],
)
