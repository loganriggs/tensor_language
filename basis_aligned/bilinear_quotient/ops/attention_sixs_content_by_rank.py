# ATTENTION 6'S CONTENT, MEASURED AGAINST THE RIGHT DENOMINATOR.
#
# §2009 swept attention 6's table rank against the arm that OMITS attention 6, and §2010 showed that
# denominator is 64-69% presence at mlp2 and 98.5% at mlp4 — so three sections of rank sweeps measured the
# presence floor and said "content". LESSON 99.
#
# The content is exactly (mean row − table): what a context-free row cannot supply. §1998 measured it at
# 0.212 nats for a compiled mlp2. This sweeps the rank against THAT baseline, which is the only one whose
# difference is content by construction.
#
# ARMS. mlp2 + attention 5,6 with attention 6's table at rank 1 / 4 / 16 / 64 / 128 / 384, and the same
# with a MEAN ROW at attention 6 as the zero-content baseline; the full 36-site program with a fallback
# variant for the inert half of the control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2010's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
S2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
RANKS = (1, 4, 16, 64, 128, 384)
MEAN = 'a6_mean'

# §1998 at 5,419: a mean row at attention 6 inside this arm. The zero-content baseline.
MEAN_ROW = (2.183, 2.306, 2.142)


def _spec6(r):
    return {'mlp': 768, 'attn': 384, ('attn', 6): r}


PLAN = [(ARM, _spec6(r), f'r{r}', S2) for r in RANKS] + [
    ('meanrow@attn6+mix30m640@mlp2,attn5', BASE, MEAN, S2),   # §1998: 2.183 / 2.306 / 2.142
    (ARM, BASE, 'full_program', None),                        # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),                # all 36 sites: the INERT pair
    (ARM, A256, 'rank_control', None)]                        # differing rank: other half


def _content(x, role, lab):
    """share of attention 6's CONTENT that `lab` recovers, measured from the mean row -- the only
    denominator whose difference is content by construction (LESSON 99)"""
    hi, lo = x.penalty(C, role, MEAN), x.penalty(C, role, 'r384')
    return (hi - x.penalty(C, role, lab)) / (hi - lo)


def _mean_row_baseline_reproduces(x):
    """§1998's mean row at attention 6 rebuilds to 2.183 / 2.306 / 2.142 within 0.005 nats on all three
    roles. Every share below is measured from it, so if it does not reproduce nothing else here holds"""
    return all(abs(x.penalty(C, r, MEAN) - v) < 0.005 for r, v in zip(x.roles, MEAN_ROW))


def _rank_one_carries_no_content(x):
    """and rank 1 recovers under 10% of the content on all three roles -- §2010 measured it as slightly
    WORSE than the mean row, so a negative share is expected and the bar is one-sided"""
    return all(_content(x, r, 'r1') < 0.10 for r in x.roles)


def _the_content_needs_high_rank(x):
    """and even rank 64 recovers under half the content on >=2 roles. If TRUE attention 6's content is
    genuinely high-dimensional and no small set of directions names it, which closes the rank line; if
    FALSE there is a mid-rank structure the whole-contribution denominator was hiding"""
    return sum(1 for r in x.roles if _content(x, r, 'r64') < 0.50) >= 2


B.run(
    name='attention_sixs_content_by_rank',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_mean_row_baseline_reproduces',
         '§1998\'s mean row at attention 6 rebuilds to 2.183/2.306/2.142 within 0.005 nats on 3/3 roles',
         _mean_row_baseline_reproduces),
        ('pred_b_rank_one_carries_no_content',
         'and rank 1 recovers under 10% of the CONTENT, measured from the mean row, on 3/3 roles',
         _rank_one_carries_no_content),
        ('pred_c_the_content_needs_high_rank',
         'and even rank 64 recovers under half the content (>=2 roles)', _the_content_needs_high_rank),
    ],
    refs=[(MEAN, B.PT + 'ops/is_attention_six_content_monotone_results.json', 'm2_mean6', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('r1', MEAN), ('r64', 'r384'), ('r16', MEAN)],
)
