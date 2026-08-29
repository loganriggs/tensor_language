# THE MINIMAL PATH: DOES mlp4 NEED ATTENTION 4 AT ALL?
#
# §1986 fixed a compiled mlp4 with attention 4–6 (1.745 nats, from 10.669) and §1989 closed the
# localisation: the spike is mlp4 itself, every lone attention layer costing 0.05–0.31. But block order is
# attention_L then mlp_L, so ATTENTION 4 RUNS BEFORE mlp4 — it sits below the compiled site, exactly where
# §1986 found compiled sites cost rather than help (attention 0–3 under mlp4 were worth −0.372).
#
# So §1986's four-site path probably carries a site it does not need, and the minimal path is
# mlp4 + attention 5 + attention 6. This tests that, and decomposes the remainder: attention 6 alone is
# already known not to work (10.666), so the question is whether attention 5 is doing real work or merely
# standing between them.
#
# ARMS. mlp4 alone; + attention 5; + attention 6; + attention 5 and 6 (the candidate minimal path);
# + attention 4, 5 and 6 (§1986's path); the full 36-site program; and one fallback variant of it so the
# inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1989's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
M4 = [('mlp', 4)]
ONLY, P5, P6, P56, P456 = 'mlp4', 'mlp4_a5', 'mlp4_a6', 'mlp4_a56', 'mlp4_a456'
FULL = 'full_program'

PLAN = [(ARM, A384, ONLY, M4),                                          # §1985: 10.669 / 10.937 / 10.580
        (ARM, A384, P5, M4 + [('attn', 5)]),
        (ARM, A384, P6, M4 + [('attn', 6)]),                            # §1984: 10.666 / 10.934 / 10.577
        (ARM, A384, P56, M4 + [('attn', 5), ('attn', 6)]),
        (ARM, A384, P456, M4 + [('attn', L) for L in (4, 5, 6)]),       # §1986: 1.745 / 1.853 / 1.682
        (ARM, A384, FULL),                                              # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]                            # other fallback, same sites: INERT


def _attention_four_is_not_needed(x):
    """dropping attention 4 does not make things worse: mlp4_a56 costs no more than §1986's mlp4_a456
    plus 0.01 nats, on >=2 roles. Registered directionally -- attention 4 runs BELOW the compiled site,
    and §1986 measured sites below as costly, so the prediction has a sign"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, P56) <= x.penalty(C, r, P456) + 0.01) >= 2


def _both_remaining_layers_are_needed(x):
    """and neither half of the pair works alone: mlp4_a5 and mlp4_a6 both stay above 5 nats on >=2 roles
    each, while the pair is under 3. If FALSE one of the two is doing all the work and the 'path' is a
    single extra site rather than a chain"""
    lone = all(sum(1 for r in x.roles if x.penalty(C, r, a) > 5.0) >= 2 for a in (P5, P6))
    return lone and sum(1 for r in x.roles if x.penalty(C, r, P56) < 3.0) >= 2


def _three_sites_are_the_whole_fix(x):
    """and the three-site program removes at least 80% of the lone-mlp4 penalty, measured down to the
    full 36-site program -- §1986's four-site version removed 108.9%"""
    def removed(r):
        hi, lo = x.penalty(C, r, ONLY), x.penalty(C, r, FULL)
        return (hi - x.penalty(C, r, P56)) / (hi - lo)
    return sum(1 for r in x.roles if removed(r) >= 0.80) >= 2


B.run(
    name='the_minimal_path',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_attention_four_is_not_needed',
         'dropping attention 4 costs nothing -- it runs below the compiled site (>=2 roles, +0.01 slack)',
         _attention_four_is_not_needed),
        ('pred_b_both_remaining_layers_are_needed',
         'and neither attention 5 nor attention 6 works alone (both >5 nats) while the pair is under 3',
         _both_remaining_layers_are_needed),
        ('pred_c_three_sites_are_the_whole_fix',
         'and three sites remove >=80% of the lone-mlp4 penalty', _three_sites_are_the_whole_fix),
    ],
    refs=[(ONLY, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005),
          (P456, B.PT + 'ops/how_far_does_the_prefix_reach_results.json', 'mlp4_path', C, 0.0005),
          (FULL, B.PT + 'ops/where_the_cliff_is_results.json', FULL, C, 0.0005)],
    paired_pairs=[(P56, P456), (P56, P5), (P56, P6)],
)
