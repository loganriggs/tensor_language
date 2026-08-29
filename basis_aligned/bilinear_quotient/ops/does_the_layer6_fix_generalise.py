# DOES §1980'S FIX SURVIVE THE NULL -- AND IS IT ABOUT ATTENTION 6'S CONTENT OR ITS CONTEXT-FREENESS?
#
# §1983 showed the mlp4 catastrophe is context-freeness ITSELF: a constant row costs 10.675 against the
# compiled table's 10.669. It then tried to ask whether §1980's fix -- compiling attention 6 -- still
# applies under that null, and could not: run() applied ONE arm to the whole substituted set, so
# attention 6 received the MEAN ROW too. That measured a different intervention, and its answer was
# startling on its own terms: giving attention 6 a mean row changed nothing at all (10.675 vs 10.675).
#
# So the two readings of §1980 come apart, and §1983's accident already speaks against one of them:
#   (i)  attention 6 must stop MIXING     -- then any context-free row there would rescue it. It did not.
#   (ii) attention 6's COMPILED TABLE specifically carries what layer 6 needs.
#
# This asks (ii) directly, with the composite arm §1983's limit forced into the library: mlp4 takes the
# mean row while attention 6 takes its compiled table. If that rescues what the mean row could not, then
# at mlp4 the table's content is irrelevant and at attn6 it is the whole thing -- an asymmetry between
# two sites three layers apart, under the identical substitution.
#
# ARMS. mlp4 alone (table / mean), mlp4+attn6 (both table / mean at mlp4 + table at attn6), the full
# 36-site program, one fallback variant of it so the inert control half is real, and one differing-rank
# arm so the other half is.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1983's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
C = 'c5419'
M4 = [('mlp', 4)]
M4A6 = [('mlp', 4), ('attn', 6)]
TAB4, MEAN4 = 'tab_mlp4', 'mean_mlp4'
TABBOTH, MEANTAB = 'tab_mlp4_attn6', 'mean_mlp4_tab_attn6'
FULL = 'full_program'

PLAN = [('mix30m640', A384, TAB4, M4),                    # §1983: 10.669 / 10.937 / 10.580
        ('meanrow', A384, MEAN4, M4),                     # §1983: 10.675 / 10.942 / 10.586
        ('mix30m640', A384, TABBOTH, M4A6),               # §1980's fix, compiled throughout
        ('meanrow@mlp4+mix30m640@attn6', A384, MEANTAB, M4A6),   # the arm §1983 could not express
        ('mix30m640', A384, FULL),                        # all 36 sites
        ('map512', A384, 'full_fb_control'),              # same rank, same sites, other fallback: INERT
        ('mix25m512', A256, 'rank_control')]              # differing table rank: the other control half


def _removed(x, role, arm):
    """fraction of the mean-row penalty at mlp4 that `arm` removes, measured down to the full program"""
    hi, lo = x.penalty(C, role, MEAN4), x.penalty(C, role, FULL)
    return (hi - x.penalty(C, role, arm)) / (hi - lo)


def _fix_survives_the_null(x):
    """compiling attention 6 with its TABLE rescues mlp4 even when mlp4 carries only a mean row --
    at least 80% of the penalty removed, on >=2 roles. §1980 reported 98% for the all-compiled version.
    If FALSE, §1980's fix is specific to the compiled table at mlp4 and does not generalise to
    context-freeness, and §1983's pred_c was pointing at something real"""
    return sum(1 for r in x.roles if _removed(x, r, MEANTAB) >= 0.80) >= 2


def _content_at_attn6_not_mixing(x):
    """and it is attention 6's CONTENT, not merely its context-freeness, that does the rescuing: the
    compiled table there beats the mean row there by more than 5 nats, on >=2 roles. §1983 measured the
    mean-row version at 10.675 -- indistinguishable from no substitution at all"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, MEAN4) - x.penalty(C, r, MEANTAB) > 5.0) >= 2


def _mlp4_content_still_irrelevant(x):
    """and the asymmetry is exact: with attention 6 compiled, what mlp4 carries STILL does not matter --
    the mean-row and compiled-table versions agree within 0.05 nats on >=2 roles, as they did within
    0.006 at mlp4 alone. Two sites three layers apart, the same substitution, opposite verdicts"""
    return sum(1 for r in x.roles
               if abs(x.penalty(C, r, MEANTAB) - x.penalty(C, r, TABBOTH)) < 0.05) >= 2


B.run(
    name='does_the_layer6_fix_generalise',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_fix_survives_the_null',
         'compiling attention 6 with its table rescues a mean-row mlp4, >=80% of the penalty, >=2 roles',
         _fix_survives_the_null),
        ('pred_b_it_is_attn6_content_not_mixing',
         'and it is attention 6\'s content doing it, not its context-freeness (>5 nats over the mean row)',
         _content_at_attn6_not_mixing),
        ('pred_c_mlp4_content_still_irrelevant',
         'and with attn6 compiled, what mlp4 carries still does not matter (within 0.05 nats)',
         _mlp4_content_still_irrelevant),
    ],
    refs=[(MEAN4, B.PT + 'ops/is_mlp4_just_fragile_results.json', MEAN4, C, 0.0005),
          (FULL, B.PT + 'ops/is_mlp4_just_fragile_results.json', FULL, C, 0.0005)],
    paired_pairs=[(MEANTAB, MEAN4), (MEANTAB, TABBOTH), (TABBOTH, TAB4)],
)
