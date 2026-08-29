# WHAT DOES ATTENTION 6 NEED BENEATH IT? THE PREFIX, OR THE MLPs?
#
# §1984 scoped §1980: compiling attention 6 removes 98% of the penalty when all 18 MLPs and attention 0–5
# are already compiled, and 0.0% when only mlp4 is. So the fix is real but conditional, and the condition
# is somewhere in what §1980 had already substituted and §1984 did not.
#
# There are two candidates and they are cleanly separable. Either attention 6 needs the ATTENTION LAYERS
# BELOW IT compiled -- a live attention 0–5 re-introduces context that a compiled attention 6 then cannot
# use -- or it needs the MLPs, and 17 live MLPs are what §1984 was missing.
#
# This is that 2x2, with §1980's own two arms rebuilt from scratch inside it as a second-class confirm.
#
# ARMS. mlp4 alone; mlp4 + attention 6; mlp4 + attention 0–5; mlp4 + attention 0–6; §1980's baseline
# (18 MLPs + attention 0–5) and its fix (+ attention 6); the full 36-site program; and one fallback
# variant of the full program so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1984's open question, and rung 2 for
# §1980's two published numbers.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
M4 = [('mlp', 4)]
A05 = [('attn', L) for L in range(6)]
A6 = [('attn', 6)]
ALLMLP = [('mlp', L) for L in range(18)]

ONLY4, P_A6, P_A05, P_A06 = 'mlp4', 'mlp4_a6', 'mlp4_a05', 'mlp4_a06'
S1980_N0, S1980_FIX, FULL = 's1980_baseline', 's1980_fix', 'full_program'

PLAN = [(ARM, A384, ONLY4, M4),                          # §1984: 10.669 / 10.937 / 10.580
        (ARM, A384, P_A6, M4 + A6),                      # §1984: 10.666 -- the fix alone buys nothing
        (ARM, A384, P_A05, M4 + A05),                    # the attention prefix, layer 6 still live
        (ARM, A384, P_A06, M4 + A05 + A6),               # prefix AND the fix, 17 MLPs live
        (ARM, A384, S1980_N0, ALLMLP + A05),             # §1980's baseline: published 9.266 / 9.531 / 9.141
        (ARM, A384, S1980_FIX, ALLMLP + A05 + A6),       # §1980's fix: published 2.733 / 2.889 / 2.681
        (ARM, A384, FULL),                               # all 36 sites: §1984 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]             # same rank, same sites, other fallback: INERT


def _fix_needs_the_attention_prefix(x):
    """with attention 0–5 compiled beneath it, compiling attention 6 finally pays: mlp4_a06 removes at
    least half the mlp4-alone penalty, measured down to the full program, on >=2 roles. §1984 measured
    0.0% for the same fix without the prefix. If FALSE the condition is not the attention prefix and 17
    live MLPs are what §1984 was missing"""
    def removed(r, a):
        hi, lo = x.penalty(C, r, ONLY4), x.penalty(C, r, FULL)
        return (hi - x.penalty(C, r, a)) / (hi - lo)
    return sum(1 for r in x.roles if removed(r, P_A06) >= 0.50) >= 2


def _the_prefix_is_what_does_it(x):
    """and it is specifically attention 6 ON TOP of the prefix, not the prefix by itself: mlp4_a06 beats
    mlp4_a05 by more than 1 nat on >=2 roles. If FALSE the prefix alone is the whole story and attention
    6 is not a distinguished site at all -- which would scope §1980 a second time"""
    return sum(1 for r in x.roles if x.penalty(C, r, P_A05) - x.penalty(C, r, P_A06) > 1.0) >= 2


def _s1980_replicates(x):
    """and §1980's two published triples rebuild from scratch here within 0.05 nats -- baseline
    9.266 / 9.531 / 9.141 and fix 2.733 / 2.889 / 2.681, on all three roles each"""
    want = {S1980_N0: (9.266, 9.531, 9.141), S1980_FIX: (2.733, 2.889, 2.681)}
    return all(abs(x.penalty(C, r, a) - v) < 0.05
               for a, vs in want.items() for r, v in zip(x.roles, vs))


B.run(
    name='where_the_cliff_is',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_fix_needs_the_attention_prefix',
         'with attention 0-5 compiled beneath it, compiling attention 6 removes >=50% of the mlp4 penalty',
         _fix_needs_the_attention_prefix),
        ('pred_b_attn6_is_still_distinguished',
         'and it is attention 6 on top of the prefix, not the prefix alone (>1 nat, >=2 roles)',
         _the_prefix_is_what_does_it),
        ('pred_c_s1980_replicates',
         'and §1980\'s two published triples rebuild within 0.05 nats on 3/3 roles each',
         _s1980_replicates),
    ],
    refs=[(ONLY4, B.PT + 'ops/does_the_layer6_fix_generalise_results.json', 'tab_mlp4', C, 0.0005),
          (FULL, B.PT + 'ops/does_the_layer6_fix_generalise_results.json', FULL, C, 0.0005)],
    paired_pairs=[(P_A06, P_A05), (P_A06, ONLY4), (S1980_FIX, S1980_N0), (P_A06, S1980_FIX)],
)
