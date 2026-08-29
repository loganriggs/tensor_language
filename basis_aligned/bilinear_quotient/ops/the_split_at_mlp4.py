# THE PRESENCE/CONTENT SPLIT AT A SECOND COMPILED MLP.
#
# §1996–§1998 gave the threshold's two attention layers distinct roles: attention 5 must be present and its
# content is slightly HARMFUL (a mean row beats its table by 0.044 nats), while attention 6 must be present
# AND carries content worth 0.209 nats, low-dimensional enough that rank 16 recovers 72% of it.
#
# All three sections used mlp2 as the compiled MLP. §1990's minimal path used mlp4, whose lone-site damage
# is 2.2x larger (10.669 against 4.813), and the split has never been checked there. If it is a fact about
# attention 5 and 6 it should transfer unchanged; if the roles shift with the compiled site, then they are
# facts about the pair rather than about the layers.
#
# ARMS. mlp4 + attention 5,6 with all tables, a mean row at attention 5, a mean row at attention 6, and
# mean rows at both; the three mlp2 versions rebuilt here so the comparison is in-run on both sides; the full 36-site program with
# a fallback variant for the inert half of the control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §1998 second-class confirmed at mlp4.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
S4 = [('mlp', 4), ('attn', 5), ('attn', 6)]
S2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
T4, M4_5, M4_6, M4_BOTH = 'mlp4_tables', 'mlp4_mean_a5', 'mlp4_mean_a6', 'mlp4_mean_both'
T2, M2_5, M2_6 = 'mlp2_tables', 'mlp2_mean_a5', 'mlp2_mean_a6'

PLAN = [(ARM, BASE, T4, S4),                                                   # §1990: 1.555/1.640/1.498
        ('meanrow@attn5+mix30m640@mlp4,attn6', BASE, M4_5, S4),
        ('meanrow@attn6+mix30m640@mlp4,attn5', BASE, M4_6, S4),
        ('meanrow@attn5,attn6+mix30m640@mlp4', BASE, M4_BOTH, S4),
        (ARM, BASE, T2, S2),                                                   # §1992: 1.971/2.090/1.952
        ('meanrow@attn5+mix30m640@mlp2,attn6', BASE, M2_5, S2),                # §1998: 1.940/2.047/1.881
        ('meanrow@attn6+mix30m640@mlp2,attn5', BASE, M2_6, S2),                # §1998: 2.183/2.306/2.142
        (ARM, BASE, 'full_program', None),                                     # §1985: 2.808/2.979/2.702
        ('map512', BASE, 'full_fb_control', None),                             # all 36 sites: INERT pair
        (ARM, A256, 'rank_control', None)]                                     # differing rank: other half


def _d(x, role, mean_arm, table_arm):
    return x.penalty(C, role, mean_arm) - x.penalty(C, role, table_arm)


def _attention_five_still_pays(x):
    """a mean row at attention 5 is still no worse than attention 5's compiled table at mlp4 -- the
    difference is at most +0.01 nats on >=2 roles, where at mlp2 it was -0.031 to -0.070. Registered
    directionally: §1998 measured a NEGATIVE cost and a transfer should keep the sign"""
    return sum(1 for r in x.roles if _d(x, r, M4_5, T4) <= 0.01) >= 2


def _attention_six_still_costs(x):
    """and a mean row at attention 6 still costs real CE at mlp4 -- more than 0.10 nats on >=2 roles,
    against 0.19-0.22 at mlp2. If FALSE attention 6's content matters only for a compiled mlp2 and the
    split is a fact about the pair rather than about the layer"""
    return sum(1 for r in x.roles if _d(x, r, M4_6, T4) > 0.10) >= 2


def _the_split_transfers_in_size(x):
    """and the split has the same size at both sites: attention 6's content is worth within 0.10 nats of
    the same at mlp4 as at mlp2, with BOTH sides rebuilt in this run rather than one of them retyped from
    §1998's published triple"""
    return sum(1 for r in x.roles
               if abs(_d(x, r, M4_6, T4) - _d(x, r, M2_6, T2)) < 0.10) >= 2


B.run(
    name='the_split_at_mlp4',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_attention_five_still_pays',
         'a mean row at attention 5 is no worse than its table at mlp4 too (<=+0.01, >=2 roles)',
         _attention_five_still_pays),
        ('pred_b_attention_six_still_costs',
         'and a mean row at attention 6 still costs more than 0.10 nats at mlp4 (>=2 roles)',
         _attention_six_still_costs),
        ('pred_c_the_split_transfers_in_size',
         'and attention 6\'s content is worth the same at mlp4 as at mlp2 (within 0.10 nats, >=2 roles)',
         _the_split_transfers_in_size),
    ],
    refs=[(T2, B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a56', C, 0.0005),
          (M2_5, B.PT + 'ops/presence_or_content_at_attention_five_results.json', 'mean_a5', C, 0.0005),
          (T4, B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', C, 0.0005)],
    paired_pairs=[(M4_5, T4), (M4_6, T4), (M4_BOTH, T4), (M2_6, T2)],
)
