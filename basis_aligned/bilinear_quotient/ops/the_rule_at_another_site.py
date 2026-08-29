# THE PATH RULE, STATED FOR ONE SITE AND TESTED AT ANOTHER.
#
# §1990 closed the mlp4 case: the minimal path is mlp4 + attention 5 + attention 6, attention 4 is below
# the compiled site and costs 0.198 to include, and neither remaining layer works alone. That gives a rule
# — every attention layer STRICTLY ABOVE the compiled site, up to and including attention 6, and nothing
# else — which was derived from a single site and has never been tested at another.
#
# §1987 bought mlp2 + attention 2–6 (1.977 nats, from 4.813) without asking whether attention 2 belonged.
# By block order it does not: attention 2 runs before mlp2. The rule predicts attention 3–6 is the minimal
# path for mlp2, and that dropping attention 3 as well breaks it, since the chain must be unbroken.
#
# ARMS. mlp2 alone; + attention 3–6 (the predicted minimal path); + attention 2–6 (§1987's version);
# + attention 4–6 (a gap at layer 3); mlp4 + attention 5,6 as the §1990 anchor; the full 36-site program;
# and one fallback variant of it so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1990's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
M2 = [('mlp', 2)]
ONLY2, MIN2, WIDE2, GAP2 = 'mlp2', 'mlp2_a36', 'mlp2_a26', 'mlp2_a46'
ANCHOR, FULL = 'mlp4_a56', 'full_program'


def _at(*ls):
    return [('attn', L) for L in ls]


PLAN = [(ARM, A384, ONLY2, M2),                                  # §1987: 4.813 / 5.291 / 4.958
        (ARM, A384, MIN2, M2 + _at(3, 4, 5, 6)),                 # the rule's prediction
        (ARM, A384, WIDE2, M2 + _at(2, 3, 4, 5, 6)),             # §1987: 1.977 / 2.117 / 1.906
        (ARM, A384, GAP2, M2 + _at(4, 5, 6)),                    # a gap at layer 3: chain broken
        (ARM, A384, ANCHOR, [('mlp', 4)] + _at(5, 6)),           # §1990: 1.555 / 1.640 / 1.498
        (ARM, A384, FULL),                                       # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]                     # other fallback, same sites: INERT


def _attention_two_is_not_needed(x):
    """dropping attention 2, which runs below mlp2, costs nothing: mlp2_a36 is no worse than §1987's
    mlp2_a26 plus 0.01 nats on >=2 roles. §1990 measured the same drop at mlp4 as worth 0.198"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, MIN2) <= x.penalty(C, r, WIDE2) + 0.01) >= 2


def _the_chain_must_be_unbroken(x):
    """and a gap breaks it: mlp2 + attention 4-6, skipping layer 3, costs more than 4 nats on >=2 roles,
    against the unbroken path's expected ~2. If FALSE the requirement is not a chain but a set, and only
    the layers nearest attention 6 matter"""
    return sum(1 for r in x.roles if x.penalty(C, r, GAP2) > 4.0) >= 2


def _the_minimal_path_works_here_too(x):
    """and the minimal path for mlp2 lands under 2.5 nats on >=2 roles -- the rule transfers to a site it
    was not derived from"""
    return sum(1 for r in x.roles if x.penalty(C, r, MIN2) < 2.5) >= 2


B.run(
    name='the_rule_at_another_site',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_attention_two_is_not_needed',
         'dropping attention 2, which runs below mlp2, costs nothing (>=2 roles, +0.01 slack)',
         _attention_two_is_not_needed),
        ('pred_b_the_chain_must_be_unbroken',
         'and a gap at layer 3 breaks it -- mlp2 + attention 4-6 stays above 4 nats',
         _the_chain_must_be_unbroken),
        ('pred_c_the_minimal_path_works_here_too',
         'and mlp2\'s minimal path lands under 2.5 nats -- the rule transfers to a new site',
         _the_minimal_path_works_here_too),
    ],
    refs=[(ONLY2, B.PT + 'ops/is_layer_six_the_boundary_for_all_results.json', 'mlp2', C, 0.0005),
          (WIDE2, B.PT + 'ops/is_layer_six_the_boundary_for_all_results.json', 'mlp2_path', C, 0.0005),
          (ANCHOR, B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', C, 0.0005),
          (FULL, B.PT + 'ops/the_minimal_path_results.json', FULL, C, 0.0005)],
    paired_pairs=[(MIN2, WIDE2), (MIN2, GAP2), (MIN2, ANCHOR)],
)
