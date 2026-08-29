# IS "CONTENT NEEDS RANK" SPECIFIC TO ATTENTION 6, OR GENERIC TO COMPILED SITES?
#
# §2011 measured attention 6's content -- (mean row - table), the part a context-free row cannot supply --
# as high-dimensional: rank 4 worth exactly a constant row, rank 16 buying 17-22%, rank 64 54-58%, rank
# 128 79-81%. That reads like a fact about attention 6.
#
# §1995 is the standing warning. It took §1994's 3:1 frequency gradient, which also read like a mechanism,
# and showed every arm had the same shape -- the gradient was a property of compiling anything. The same
# control applies here and has not been run: if a compiled MLP's content also needs 64 directions for half
# of it, then "content needs rank" says nothing about attention 6.
#
# mlp2 is the site to ask, because it is the compiled MLP in every §2009-§2011 arm and its content in the
# REPAIRED configuration has never been measured -- §1983 measured mlp4's content in the LONE
# configuration at essentially zero, which is a different quantity.
#
# ARMS. mlp2 + attention 5,6 with mlp2's table at rank 1 / 16 / 64 / 128 / 768, and the same with a MEAN
# ROW at mlp2 as the zero-content baseline; the full 36-site program with a fallback variant for the inert
# half of the control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2011's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
S2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
RANKS = (1, 16, 64, 128, 768)
MEAN = 'm2_mean'

FULL_RANK = (1.971, 2.090, 1.952)          # §1992: mlp2 + attention 5,6, all tables
A6_AT_16 = (0.219, 0.208, 0.170)           # §2011: attention 6's content share at rank 16


def _spec2(r):
    return {'mlp': 768, 'attn': 384, ('mlp', 2): r}


PLAN = [(ARM, _spec2(r), f'r{r}', S2) for r in RANKS] + [
    ('meanrow@mlp2+mix30m640@attn5,attn6', BASE, MEAN, S2),
    (ARM, BASE, 'full_program', None),                        # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),                # all 36 sites: the INERT pair
    (ARM, A256, 'rank_control', None)]                        # differing rank: other half


def _content(x, role, lab):
    hi, lo = x.penalty(C, role, MEAN), x.penalty(C, role, 'r768')
    return (hi - x.penalty(C, role, lab)) / (hi - lo)


def _full_rank_reproduces(x):
    """the rank-768 arm reproduces §1992's mlp2_a56 to 0.005 nats on all three roles -- naming mlp2 at the
    rank its kind already carried must be a no-op"""
    return all(abs(x.penalty(C, r, 'r768') - v) < 0.005 for r, v in zip(x.roles, FULL_RANK))


def _mlp2_has_measurable_content(x):
    """and mlp2 has real content in this configuration -- its mean row costs at least 0.01 nats more than
    its table, on >=2 roles. §1983 measured mlp4's content in the LONE configuration at 0.006, so this is
    not guaranteed, and every share below is meaningless without it"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, MEAN) - x.penalty(C, r, 'r768') >= 0.01) >= 2


def _the_rank_curve_differs_from_attention_six(x):
    """and mlp2's content curve differs from attention 6's: at rank 16 the share differs by more than 15
    points on >=2 roles, against §2011's 21.9 / 20.8 / 17.0. If FALSE the high-dimensionality of content is
    generic to compiled sites and §2011 says nothing specific about attention 6 -- the §1995 control, one
    level up"""
    return sum(1 for r, v in zip(x.roles, A6_AT_16)
               if abs(_content(x, r, 'r16') - v) > 0.15) >= 2


B.run(
    name='is_high_dimensional_content_generic',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_rank_reproduces',
         'naming mlp2 at rank 768 reproduces §1992\'s mlp2_a56 to 0.005 nats on 3/3 roles',
         _full_rank_reproduces),
        ('pred_b_mlp2_has_measurable_content',
         'and mlp2 has at least 0.01 nats of content in this configuration (>=2 roles)',
         _mlp2_has_measurable_content),
        ('pred_c_the_rank_curve_differs_from_attention_six',
         'and its rank-16 content share differs from attention 6\'s by more than 15 points (>=2 roles)',
         _the_rank_curve_differs_from_attention_six),
    ],
    refs=[('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('r1', MEAN), ('r16', 'r768'), (MEAN, 'r768')],
)
