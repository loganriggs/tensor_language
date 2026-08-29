# IS THE 3:1 FREQUENCY GRADIENT ABOUT ATTENTION 6, OR ABOUT COMPILATION IN GENERAL?
#
# §1994 found the threshold's gap graded by target frequency on all three roles — omitting attention 6
# costs 0.93 nats on an unseen target and 0.30 on a frequent one, a ratio near 0.29. That looks like a
# mechanism. §1986, §1990 and §1992 each looked like a mechanism too, and each was a generic feature of
# the sweep's shape rather than a fact about the model.
#
# So this is the deflationary control, run before anything is built on the gradient. If the full 36-site
# program's own damage and a lone compiled mlp2's have the SAME frequency shape, then the gradient is a
# property of compiling anything and says nothing about attention 6.
#
# ARMS. all cached: mlp2 + attention 5,6; mlp2 + attention 5; mlp2 alone; mlp4 alone; the full 36-site
# program; and one fallback variant of it so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1994's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
A56 = [('attn', 5), ('attn', 6)]
GOOD, BAD = 'mlp2_a56', 'mlp2_a5'
LONE2, LONE4, FULL = 'mlp2', 'mlp4', 'full_program'

PLAN = [(ARM, A384, GOOD, [('mlp', 2)] + A56),                # §1992: 1.971 / 2.090 / 1.952
        (ARM, A384, BAD, [('mlp', 2), ('attn', 5)]),          # §1992: 2.556 / 2.730 / 2.558
        (ARM, A384, LONE2, [('mlp', 2)]),                     # §1987: 4.813 / 5.291 / 4.958
        (ARM, A384, LONE4, [('mlp', 4)]),                     # §1985: 10.669 / 10.937 / 10.580
        (ARM, A384, FULL),                                    # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]                  # other fallback, same sites: INERT

THRESHOLD_SHAPE = 0.29    # §1994: 0.317 / 0.290 / 0.266, the top-bucket gap over the unseen-bucket gap


def _damage_shape(x, role, arm):
    """damage over the live model in the most frequent bucket, divided by the same in the unseen bucket"""
    def dmg(b):
        return x.ce(C, role, arm, 'pooled', b) - x.res[C][role][arm]['pooled'][b]['ce_live']
    return dmg(x.top) / dmg(x.bot)


def _full_program_has_the_same_shape(x):
    """the full 36-site program's own damage is graded the same way, within 0.15 of §1994's 0.29, on
    >=2 roles. If TRUE the gradient is a property of compiling anything and says nothing about attention
    6; if FALSE the threshold's gradient is specific and worth a mechanism"""
    return sum(1 for r in x.roles
               if abs(_damage_shape(x, r, FULL) - THRESHOLD_SHAPE) < 0.15) >= 2


def _lone_sites_have_it_too(x):
    """and so do both lone compiled MLPs, within the same 0.15 -- two sites whose total damage differs by
    a factor of 2.2 (4.81 against 10.67) should not share a frequency shape unless the shape is generic"""
    return all(sum(1 for r in x.roles if abs(_damage_shape(x, r, a) - THRESHOLD_SHAPE) < 0.15) >= 2
               for a in (LONE2, LONE4))


def _all_shapes_are_steep(x):
    """and every one of them is under 0.5 on all three roles: whatever is compiled, the damage is at least
    twice as large on unseen targets as on frequent ones"""
    return all(_damage_shape(x, r, a) < 0.5
               for r in x.roles for a in (GOOD, BAD, LONE2, LONE4, FULL))


B.run(
    name='is_the_frequency_gradient_generic',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_program_has_the_same_shape',
         'the full program\'s damage is graded like the threshold gap, within 0.15 of 0.29 (>=2 roles)',
         _full_program_has_the_same_shape),
        ('pred_b_lone_sites_have_it_too',
         'and both lone compiled MLPs do too, though their total damage differs by 2.2x',
         _lone_sites_have_it_too),
        ('pred_c_all_shapes_are_steep',
         'and every shape is under 0.5 on 3/3 roles -- damage is always at least 2x steeper on unseen targets',
         _all_shapes_are_steep),
    ],
    refs=[(GOOD, B.PT + 'ops/where_the_threshold_gap_lives_results.json', GOOD, C, 0.0005),
          (BAD, B.PT + 'ops/where_the_threshold_gap_lives_results.json', BAD, C, 0.0005),
          (LONE4, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005)],
    paired_pairs=[(GOOD, BAD), (LONE2, FULL), (LONE4, FULL)],
)
