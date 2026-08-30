# IS RANK 768 A PROPERTY OF THE MODEL, OR OF THE FIT-SET SIZE?
#
# Two independent measurements now say extra table rank above ~768 fails out-of-sample. §2020: untruncating
# the late MLPs to rank 1152 costs -11.578 milli-nats on fresh rows. §2052: uniform rank 1024 is worse than
# 768 by 17.054 (t = -41.82) where in-sample the richer arm won. And §2044 found the failure GROWING with
# coverage.
#
# There are two readings and they differ in what to do next. Either 768 is a property of bilin18 -- the
# tables genuinely have about that much structure and the rest is noise the fit picks up -- or it is a
# property of the FIT SET: with more rows to fit on, a richer table would be estimated well enough to
# transfer, and 768 is where 480 documents run out.
#
# The two fit sets differ threefold: fineweb_n96_skip80 gives 5,419 covered types and fineweb_n480_skip80
# gives 16,110. If 768 is about the fit set, the optimum should sit LOWER at 5,419 than at 16,110, because
# the smaller fit set supports less rank. If it is about the model, the optimum should sit at the same
# place at both.
#
# Registered expectation: the fit-set reading. §2044 measured the untruncation harm growing 2.8x between
# coverages, which is the direction more-fit-data-supports-more-rank predicts.
#
# ARMS. mix25m256 at {512,128}, {640,160}, {768,256}, {1024,256} -- §1947's ladder -- scored at BOTH
# coverages on the fresh window. A fallback variant for the inert half of the control, and one differing-
# table-rank arm for the other half.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2052's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
ARM = 'mix25m256'
FRESH = ('fresh',)
LADDER = ['r512_128', 'r640_160', 'r768_256', 'r1024_256']
SPEC = {'r512_128': {'mlp': 512, 'attn': 128}, 'r640_160': {'mlp': 640, 'attn': 160},
        'r768_256': {'mlp': 768, 'attn': 256}, 'r1024_256': {'mlp': 1024, 'attn': 256}}
KNEE = 'r768_256'
RICHEST_PENALTY_HI = 0.017054      # §2052: {1024,256} worse than {768,256} at 16,110, nats


PLAN = [(ARM, SPEC[l], l, None) for l in LADDER] + [
    ('map512', SPEC[KNEE], 'shipped_fb_control', None),                  # same spec, other fallback
    (ARM, {'mlp': 896, 'attn': 224}, 'rank_control', None)]              # differing table rank


def _beats(x, cov, a, b):
    if a == b:
        return 0.0
    return -x.tpool_full(cov, a, b)['mean']


def _argmax(x, cov):
    return max(LADDER, key=lambda l: _beats(x, cov, l, KNEE))


def _the_2052_figure_reproduces(x):
    """§2052's -17.054 milli-nat penalty for rank 1024 over 768 at 16,110 rebuilds within 1 milli-nat.
    The comparison between coverages below is a difference of two such quantities"""
    return abs(_beats(x, HI, KNEE, 'r1024_256') - RICHEST_PENALTY_HI) < 0.001


def _the_richest_arm_loses_at_both(x):
    """and rank 1024 loses to 768 at BOTH coverages on fresh rows. §2052 measured it only at 16,110; if it
    wins at 5,419 the effect is coverage-specific and neither reading holds as stated"""
    return all(_beats(x, c, KNEE, 'r1024_256') > 0 for c in (LO, HI))


def _the_optimum_sits_lower_at_the_smaller_fit_set(x):
    """and the fresh optimum is LOWER at 5,419 than at 16,110 -- the fit-set reading. The smaller fit set
    supports less rank, so if 768 is where 480 documents run out, 96 documents should run out sooner. If
    FALSE and the optimum is the same at both, 768 is a property of the model rather than of the fit"""
    return LADDER.index(_argmax(x, LO)) < LADDER.index(_argmax(x, HI))


def _the_penalty_for_excess_rank_shrinks_with_fit_data(x):
    """and the penalty for the richest arm is SMALLER at 16,110 than at 5,419 -- more fit data should make
    excess rank less harmful even if it never becomes profitable. This is the fit-set reading's quantitative
    consequence, and it is independent of where the argmax lands"""
    return _beats(x, HI, KNEE, 'r1024_256') < _beats(x, LO, KNEE, 'r1024_256')


B.run(
    name='is_768_the_model_or_the_fit_set',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_2052_figure_reproduces',
         '§2052\'s -17.054 milli-nat penalty for rank 1024 rebuilds within 1 milli-nat',
         _the_2052_figure_reproduces),
        ('pred_b_the_richest_arm_loses_at_both',
         'and rank 1024 loses to 768 at both coverages on fresh rows', _the_richest_arm_loses_at_both),
        ('pred_c_the_optimum_sits_lower_at_the_smaller_fit_set',
         'and the fresh optimum is lower at 5,419 than at 16,110 -- the fit-set reading',
         _the_optimum_sits_lower_at_the_smaller_fit_set),
        ('pred_d_the_penalty_for_excess_rank_shrinks_with_fit_data',
         'and the penalty for rank 1024 is smaller at 16,110 than at 5,419',
         _the_penalty_for_excess_rank_shrinks_with_fit_data),
    ],
    paired_pairs=[(l, KNEE) for l in LADDER if l != KNEE],
)
