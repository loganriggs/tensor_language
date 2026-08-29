# IS ATTENTION 5'S LEVER ACTION ALSO PRESENCE-ONLY?
#
# §2006 established attention 5 as the only site below attention 6 that moves anything: compiling it
# beneath a compiled mlp4 takes the damage from the 10.669 ceiling to 8.021 with attention 6 STILL LIVE,
# where attention 4 gives 0.000 and attention 0–3 give 0.018.
#
# §1998 showed attention 5's content is worthless when attention 6 is COMPILED — a mean row beats its
# table by 0.044 nats — and §1997 showed rank 16 costs 0.010 there. Neither touches the configuration in
# which attention 5 acts as a lever rather than half of a repair. If the lever is also presence-only, then
# every role attention 5 plays is about it not varying with context, and nothing about what it writes.
#
# ARMS. mlp4 + attention 5 with its compiled table, with a MEAN ROW, and with its table truncated to rank
# 16 — attention 6 live in all three; mlp4 alone as the ceiling anchor; mlp4 + attention 5,6 as the
# repaired anchor; the full 36-site program with a fallback variant for the inert half of the control; and
# one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2006's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
M4A5 = [('mlp', 4), ('attn', 5)]
CEIL, TAB, MEAN, R16, FIXED = 'm4', 'a5_table', 'a5_mean', 'a5_rank16', 'm4_a56'

PLAN = [(ARM, BASE, CEIL, [('mlp', 4)]),                                  # §1985: 10.669/10.937/10.580
        (ARM, BASE, TAB, M4A5),                                           # §1990:  8.021/ 8.300/ 7.962
        ('meanrow@attn5+mix30m640@mlp4', BASE, MEAN, M4A5),
        (ARM, {'mlp': 768, 'attn': 384, ('attn', 5): 16}, R16, M4A5),
        (ARM, BASE, FIXED, [('mlp', 4), ('attn', 5), ('attn', 6)]),       # §1990:  1.555/ 1.640/ 1.498
        (ARM, BASE, 'full_program', None),                                # §1985:  2.808/ 2.979/ 2.702
        ('map512', BASE, 'full_fb_control', None),                        # all 36 sites: the INERT pair
        (ARM, A256, 'rank_control', None)]                                # differing rank: other half


def _the_lever_anchor_reproduces(x):
    """§1990's mlp4 + attention 5 rebuilds to 8.021 / 8.300 / 7.962 within 0.005 nats on all three roles --
    the quantity both other arms are measured against"""
    want = (8.021, 8.300, 7.962)
    return all(abs(x.penalty(C, r, TAB) - v) < 0.005 for r, v in zip(x.roles, want))


def _a_mean_row_levers_too(x):
    """and a mean row at attention 5 -- zero information, only context-freeness -- keeps the damage within
    0.20 nats of the compiled table's 8.021, on >=2 roles. If TRUE every role attention 5 plays is about
    it not varying with context and nothing about what it writes; if FALSE the lever is the one place its
    content matters, which would be the first such place in six sections"""
    return sum(1 for r in x.roles
               if abs(x.penalty(C, r, MEAN) - x.penalty(C, r, TAB)) < 0.20) >= 2


def _the_lever_stays_far_below_the_ceiling(x):
    """and both low-content versions stay at least 2.0 nats below the ceiling on all three roles -- the
    lever is worth 2.65 with the full table, and a content-free version that lost most of that would mean
    the drop is not presence"""
    return all(x.penalty(C, r, CEIL) - x.penalty(C, r, a) > 2.0
               for r in x.roles for a in (MEAN, R16))


B.run(
    name='is_the_lever_also_presence_only',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_the_lever_anchor_reproduces',
         '§1990\'s mlp4 + attention 5 rebuilds to 8.021/8.300/7.962 within 0.005 nats on 3/3 roles',
         _the_lever_anchor_reproduces),
        ('pred_b_a_mean_row_levers_too',
         'and a mean row at attention 5 keeps the damage within 0.20 nats of its compiled table (>=2 roles)',
         _a_mean_row_levers_too),
        ('pred_c_the_lever_stays_far_below_the_ceiling',
         'and both low-content versions stay more than 2.0 nats below the ceiling, on 3/3 roles',
         _the_lever_stays_far_below_the_ceiling),
    ],
    refs=[(CEIL, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005),
          (FIXED, B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[(MEAN, TAB), (R16, TAB), (TAB, CEIL)],
)
