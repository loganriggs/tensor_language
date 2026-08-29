# HOW MUCH OF THE PREFIX DOES THE RULE ACTUALLY NEED, AND DOES IT POINT DOWNWARD ONLY?
#
# §1985 found that a compiled mlp4 is catastrophic (10.669) until every attention layer up to and
# including 6 is compiled too, at which point it is cheap (2.105) -- cheaper than the full 36-site
# program. But §1985 bought that with attention 0–6, and attention 0–3 sit BELOW the compiled site. If
# they are not needed, the rule tightens from "the whole prefix" to "from the site up to layer 6", which
# is a statement about a path rather than about a block.
#
# The rule is also stated only for a site beneath layer 6. A site ABOVE it -- mlp12 -- has no attention 6
# in front of it at all, and the rule as written predicts it should cost nothing special on its own. That
# is the sharpest thing the rule says, and it has never been tested.
#
# ARMS. mlp4 alone; mlp4 + attention 4–6 (the minimal path); mlp4 + attention 0–6 (§1985's recipe);
# mlp12 alone; mlp12 + attention 6; the full 36-site program; and one fallback variant of it so the inert
# half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1985's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
M4, M12 = [('mlp', 4)], [('mlp', 12)]
PATH = [('attn', L) for L in (4, 5, 6)]
A06 = [('attn', L) for L in range(7)]

ONLY4, PATH4, FULLPRE4 = 'mlp4', 'mlp4_path', 'mlp4_a06'
ONLY12, P12_A6, FULL = 'mlp12', 'mlp12_a6', 'full_program'

PLAN = [(ARM, A384, ONLY4, M4),                     # §1985: 10.669 / 10.937 / 10.580
        (ARM, A384, PATH4, M4 + PATH),              # the minimal path: attention 4–6 only
        (ARM, A384, FULLPRE4, M4 + A06),            # §1985: 2.105 / 2.250 / 2.026
        (ARM, A384, ONLY12, M12),                   # a compiled site ABOVE layer 6
        (ARM, A384, P12_A6, M12 + [('attn', 6)]),   # layer 6 compiled, but behind it
        (ARM, A384, FULL),                          # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]        # same rank, same sites, other fallback: INERT


def _path_is_enough(x):
    """attention 4–6 alone buys what attention 0–6 bought: within 0.05 nats on >=2 roles. If TRUE the
    rule is about a PATH from the compiled site up to layer 6, and the three attention layers below the
    site are no part of it; if FALSE the whole prefix is required and the rule is about a block"""
    return sum(1 for r in x.roles
               if abs(x.penalty(C, r, PATH4) - x.penalty(C, r, FULLPRE4)) < 0.05) >= 2


def _a_site_above_six_is_cheap(x):
    """and a lone compiled mlp12, with no attention 6 in front of it, costs under 3 nats on >=2 roles --
    against mlp4's 10.669 for the identical substitution one site lower. This is the sharpest thing
    §1985's rule says: the requirement points DOWNWARD only. If FALSE the rule is not about layer 6"""
    return sum(1 for r in x.roles if x.penalty(C, r, ONLY12) < 3.0) >= 2


def _layer_six_behind_it_does_nothing(x):
    """and compiling attention 6 does nothing for mlp12, since it sits behind it: within 0.05 nats of
    mlp12 alone on >=2 roles. Layer 6 is a gate for what flows INTO it, not a globally special site"""
    return sum(1 for r in x.roles
               if abs(x.penalty(C, r, P12_A6) - x.penalty(C, r, ONLY12)) < 0.05) >= 2


B.run(
    name='how_far_does_the_prefix_reach',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_the_path_is_enough',
         'attention 4-6 alone buys what attention 0-6 bought (within 0.05 nats, >=2 roles)',
         _path_is_enough),
        ('pred_b_a_site_above_six_is_cheap',
         'and a lone compiled mlp12 costs under 3 nats, against mlp4\'s 10.669 one site lower',
         _a_site_above_six_is_cheap),
        ('pred_c_layer_six_behind_it_does_nothing',
         'and compiling attention 6 does nothing for mlp12 -- the requirement points downward only',
         _layer_six_behind_it_does_nothing),
    ],
    refs=[(ONLY4, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005),
          (FULLPRE4, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4_a06', C, 0.0005),
          (FULL, B.PT + 'ops/where_the_cliff_is_results.json', FULL, C, 0.0005)],
    paired_pairs=[(PATH4, FULLPRE4), (ONLY12, ONLY4), (P12_A6, ONLY12)],
)
