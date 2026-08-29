# ATTENTION 5: PRESENCE WITH NO CONTENT AT ALL?
#
# §1997 found the threshold's two layers doing different jobs. Cutting attention 5's table from rank 384
# to rank 16 costs 0.010 nats; the same cut at attention 6 costs 0.166. Removing attention 5 entirely
# costs 2.905. So attention 5 looks like pure presence — it must be compiled, and almost nothing about
# what its table contains matters.
#
# "Almost nothing" has a limit, and §1983 built the instrument for it. The `meanrow` null gives every
# token THE SAME ROW — the mean of the covered table — keeping zero information and only context-freeness.
# At mlp4 it cost the same 10.67 as the full compiled table, which is how §1983 showed the damage there
# was context-freeness itself. Here the question is the mirror image: does a site that must be present
# need to contain anything?
#
# ARMS. mlp2 + attention 5,6 with a MEAN ROW at attention 5; the same with a mean row at attention 6; the
# same with mean rows at both; §1992's all-table version; the arm that omits attention 5; the full 36-site
# program with a fallback variant for the inert half of the control; and one differing-rank arm.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1997's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
SITES2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
TABLES, MEAN5, MEAN6, MEANBOTH = 'all_tables', 'mean_a5', 'mean_a6', 'mean_both'
NO5 = 'mlp2_a6'

PLAN = [(ARM, BASE, TABLES, SITES2),                                          # §1992: 1.971/2.090/1.952
        ('meanrow@attn5+mix30m640@mlp2,attn6', BASE, MEAN5, SITES2),
        ('meanrow@attn6+mix30m640@mlp2,attn5', BASE, MEAN6, SITES2),
        ('meanrow@attn5,attn6+mix30m640@mlp2', BASE, MEANBOTH, SITES2),
        (ARM, BASE, NO5, [('mlp', 2), ('attn', 6)]),                          # §1992: 4.876/5.355/5.016
        (ARM, BASE, 'full_program', None),                                    # §1985: 2.808/2.979/2.702
        ('map512', BASE, 'full_fb_control', None),                            # all 36 sites: INERT pair
        (ARM, A256, 'rank_control', None)]                                    # differing rank: other half


def _mean_row_suffices_at_five(x):
    """a mean row at attention 5 -- zero information, only context-freeness -- costs under 0.20 nats
    against the full compiled table there, on >=2 roles. §1997 measured a 24-fold rank cut at 0.010 and
    removal at 2.905, so if presence really is the whole requirement the null should be nearly free"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, MEAN5) - x.penalty(C, r, TABLES) < 0.20) >= 2


def _mean_row_costs_more_at_six(x):
    """and the same null at attention 6 costs strictly more than at attention 5, on all three roles --
    §1996 showed attention 6's table carries real content and §1997 showed attention 5's does not, so the
    null should separate them. If FALSE the two layers are not distinguishable by content after all and
    §1997's asymmetry is about rank truncation rather than information"""
    return all(x.penalty(C, r, MEAN6) > x.penalty(C, r, MEAN5) for r in x.roles)


def _presence_still_beats_content(x):
    """and even mean rows at BOTH layers keep the arm far below the arm that omits attention 5: under
    3.5 nats on >=2 roles, against 4.876. Whatever content is worth, presence is worth more"""
    return sum(1 for r in x.roles if x.penalty(C, r, MEANBOTH) < 3.5) >= 2


B.run(
    name='presence_or_content_at_attention_five',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_mean_row_suffices_at_five',
         'a mean row at attention 5 costs under 0.20 nats against its full table (>=2 roles)',
         _mean_row_suffices_at_five),
        ('pred_b_mean_row_costs_more_at_six',
         'and the same null at attention 6 costs strictly more, on 3/3 roles',
         _mean_row_costs_more_at_six),
        ('pred_c_presence_still_beats_content',
         'and mean rows at both layers stay under 3.5 nats, against 4.876 for omitting attention 5',
         _presence_still_beats_content),
    ],
    refs=[(NO5, B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a6', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[(MEAN5, TABLES), (MEAN6, TABLES), (MEANBOTH, TABLES)],
)
