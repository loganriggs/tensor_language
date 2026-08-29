# THE NARROW FORM: IS THE REQUIREMENT SIMPLY ATTENTION 5 AND ATTENTION 6?
#
# §1991 falsified the chain: removing attention 3 from mlp2's supposed path made it slightly BETTER
# (1.845 against 1.887). What every working arm shares is attention 5 and attention 6 — mlp4+{5,6} at
# 1.555, mlp2+{4,5,6} at 1.845, mlp2+{3,4,5,6} at 1.887 — and each layer beyond those two costs a little.
# No arm has ever compiled a site with attention 5 and 6 ALONE when the site sits further below, so the
# narrow form is unmeasured and no rule is currently in force.
#
# This measures it, at mlp2 and at mlp3, and asks whether the requirement is independent of how far below
# the boundary the compiled site sits.
#
# ARMS. mlp2 + attention 5,6; mlp3 + attention 5,6; mlp2 + attention 5 alone; mlp2 + attention 6 alone;
# mlp2 + attention 3–6 as the §1991 anchor; mlp4 + attention 5,6 as the §1990 anchor; the full 36-site
# program; and one fallback variant of it so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1991's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
A56 = [('attn', 5), ('attn', 6)]
N2, N3, ONLY5, ONLY6 = 'mlp2_a56', 'mlp3_a56', 'mlp2_a5', 'mlp2_a6'
WIDE2, ANCHOR4, FULL = 'mlp2_a36', 'mlp4_a56', 'full_program'

PLAN = [(ARM, A384, N2, [('mlp', 2)] + A56),                                  # the narrow form at mlp2
        (ARM, A384, N3, [('mlp', 3)] + A56),                                  # and at mlp3
        (ARM, A384, ONLY5, [('mlp', 2), ('attn', 5)]),
        (ARM, A384, ONLY6, [('mlp', 2), ('attn', 6)]),
        (ARM, A384, WIDE2, [('mlp', 2)] + [('attn', L) for L in (3, 4, 5, 6)]),  # §1991: 1.887/2.016/1.817
        (ARM, A384, ANCHOR4, [('mlp', 4)] + A56),                             # §1990: 1.555/1.640/1.498
        (ARM, A384, FULL),                                                    # §1985: 2.808/2.979/2.702
        ('map512', A384, 'full_fb_control')]                                  # other fallback: INERT


def _two_layers_are_enough(x):
    """attention 5 and 6 alone do what §1991's four-layer version did for mlp2: no worse than mlp2_a36
    plus 0.01 nats, on >=2 roles. Registered directionally -- §1991 measured every extra layer as a small
    cost, so the prediction has a sign. If FALSE the intervening layers do carry something after all"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, N2) <= x.penalty(C, r, WIDE2) + 0.01) >= 2


def _neither_layer_alone(x):
    """and it is still the pair: mlp2 with attention 5 alone, or attention 6 alone, stays above 4 nats on
    >=2 roles each, against the pair's expected ~1.8 -- §1990 measured the same at mlp4 (8.02 and 10.67
    against 1.56)"""
    return all(sum(1 for r in x.roles if x.penalty(C, r, a) > 4.0) >= 2 for a in (ONLY5, ONLY6))


def _independent_of_depth(x):
    """and the requirement does not depend on how far below the boundary the site sits: mlp2, mlp3 and
    mlp4 with attention 5,6 all land within 0.5 nats of one another on all three roles, though their lone
    costs span 4.81 / 6.57 / 10.67. If FALSE depth matters after all and the narrow form is incomplete"""
    return all(max(x.penalty(C, r, a) for a in (N2, N3, ANCHOR4))
               - min(x.penalty(C, r, a) for a in (N2, N3, ANCHOR4)) < 0.5 for r in x.roles)


B.run(
    name='is_it_just_attention_five_and_six',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_two_layers_are_enough',
         'attention 5 and 6 alone match §1991\'s four-layer version for mlp2 (>=2 roles, +0.01 slack)',
         _two_layers_are_enough),
        ('pred_b_neither_layer_alone',
         'and neither attention 5 nor attention 6 alone works for mlp2 (both above 4 nats)',
         _neither_layer_alone),
        ('pred_c_independent_of_depth',
         'and mlp2, mlp3, mlp4 with attention 5,6 land within 0.5 nats of one another on 3/3 roles',
         _independent_of_depth),
    ],
    refs=[(WIDE2, B.PT + 'ops/the_rule_at_another_site_results.json', 'mlp2_a36', C, 0.0005),
          (ANCHOR4, B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', C, 0.0005),
          (FULL, B.PT + 'ops/the_minimal_path_results.json', FULL, C, 0.0005)],
    paired_pairs=[(N2, WIDE2), (N2, N3), (N3, ANCHOR4)],
)
