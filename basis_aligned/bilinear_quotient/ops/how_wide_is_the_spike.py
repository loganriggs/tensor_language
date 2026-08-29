# IS THE mlp4 SPIKE ONE SITE WIDE, AND WHERE EXACTLY DOES THE BOUNDARY CUT?
#
# §1987 assembled the depth profile: mlp2 4.81, mlp4 10.67, mlp5 2.02, mlp8 0.05, mlp12 0.03. Layer 4 is a
# spike rather than the edge of a plateau — its neighbour one layer closer to attention 6 costs a fifth as
# much — and nothing in the path rule predicts a spike at all.
#
# Two gaps in that profile decide what it means. Layer 3 has never been measured, so "one site wide" is
# assumed rather than shown. And layers 6 and 7 sit on the far side of the boundary in BLOCK ORDER —
# attention 6 runs before mlp6 — so if the boundary is attention 6 itself, mlp6 should already be free,
# which is the crispest available statement of where the cut falls.
#
# ARMS. mlp3, mlp6, mlp7 alone; mlp4 and mlp5 alone as the §1982/§1987 anchors; the full 36-site program;
# and one fallback variant of it so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1987's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
M = {L: [('mlp', L)] for L in (3, 4, 5, 6, 7)}
L3, L4, L5, L6, L7 = 'mlp3', 'mlp4', 'mlp5', 'mlp6', 'mlp7'
FULL = 'full_program'

PLAN = [(ARM, A384, L3, M[3]),
        (ARM, A384, L4, M[4]),               # §1985: 10.669 / 10.937 / 10.580
        (ARM, A384, L5, M[5]),               # §1982: 2.022 / 2.141 / 1.994
        (ARM, A384, L6, M[6]),
        (ARM, A384, L7, M[7]),
        (ARM, A384, FULL),                   # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]  # same rank, same sites, other fallback: INERT


def _boundary_is_attention_six_itself(x):
    """mlp6 and mlp7 both cost under 0.5 nats on >=2 roles. Attention 6 runs BEFORE mlp6 in block order,
    so if the boundary is attention 6 itself then mlp6 is already above it and free, like mlp8 (0.051).
    If FALSE the cut is not at attention 6 but somewhere inside layers 6-7"""
    return all(sum(1 for r in x.roles if x.penalty(C, r, a) < 0.5) >= 2 for a in (L6, L7))


def _the_spike_is_one_site_wide(x):
    """and layer 3, never measured, costs less than half of mlp4 on >=2 roles -- as mlp5 does at 2.02
    against 10.67. If FALSE the spike is at least two sites wide and 'layer 4' is the wrong description"""
    return sum(1 for r in x.roles if x.penalty(C, r, L3) < 0.5 * x.penalty(C, r, L4)) >= 2


def _mlp4_is_the_maximum(x):
    """and mlp4 is strictly the worst of layers 3,4,5 on all three roles -- the spike is a single site and
    it is that one"""
    return all(x.penalty(C, r, L4) > max(x.penalty(C, r, a) for a in (L3, L5)) for r in x.roles)


B.run(
    name='how_wide_is_the_spike',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_boundary_is_attention_six_itself',
         'mlp6 and mlp7 both cost under 0.5 nats -- attention 6 runs before mlp6, so mlp6 is above the cut',
         _boundary_is_attention_six_itself),
        ('pred_b_the_spike_is_one_site_wide',
         'and mlp3 costs less than half of mlp4 (>=2 roles), as mlp5 does', _the_spike_is_one_site_wide),
        ('pred_c_mlp4_is_the_maximum',
         'and mlp4 is strictly the worst of layers 3, 4, 5 on all three roles', _mlp4_is_the_maximum),
    ],
    refs=[(L4, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005),
          (L5, B.PT + 'ops/is_mlp4_just_fragile_results.json', 'tab_mlp5', C, 0.0005),
          (FULL, B.PT + 'ops/where_the_cliff_is_results.json', FULL, C, 0.0005)],
    paired_pairs=[(L3, L4), (L5, L4), (L6, L7)],
)
