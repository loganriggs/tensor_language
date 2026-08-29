# THE LAST UNTESTED DIRECTION: DO THE SIXTEEN SHALLOW MLP SITES WANT RANK ABOVE 768?
#
# §2019 recorded the best-known build with mlp16 and mlp17 untruncated to rank 1152, closing the rank axis
# at the top for the two sites that carry the program. The other sixteen MLP sites have never been tested
# above 768 in either direction.
#
# §2015 measured their table CONTENT as small -- the eight sampled sites at layers 0-14 sum to 0.055 nats
# against mlp16 and mlp17's 1.17 -- which is the reason to expect little here. But that was measured AT
# rank 768: it says what those tables carry, not what a fuller table there would carry. And §2017's
# sub-additivity (improvements 10-28% short of additive) means sixteen sites are not sixteen times one.
#
# Untruncating all sixteen adds 16 x 2.52M = 40.4M values, worth 0.00404 nats at §1947's 0.010-per-100M
# price -- sixteen times the price of the purchase §2019 just made, so it needs a much larger gain.
#
# ARMS. the shipped build; §2019's build (mlp16+17 at 1152); §2019's build with layers 10-15 also at 1152;
# all eighteen MLPs at 1152; a fallback variant of the shipped build for the inert half of the control;
# and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2019's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
SHIPPED, BEST, MID, ALL18 = 'shipped', 'late_two', 'late_eight', 'all_eighteen'
PRICE_MID = 0.00150          # §1947: six extra sites, 15.1M values
PRICE_ALL = 0.00404          # sixteen extra sites, 40.4M values

# §2019, pooled mean over all three roles, best-known build against the shipped one
BEST_POOLED = {LO: 0.000962, HI: 0.002340}

LATE2 = {('mlp', L): 1152 for L in (16, 17)}
LATE8 = {('mlp', L): 1152 for L in range(10, 18)}
ALL = {('mlp', L): 1152 for L in range(18)}

PLAN = [(ARM, BASE, SHIPPED, None),
        (ARM, {**BASE, **LATE2}, BEST, None),
        (ARM, {**BASE, **LATE8}, MID, None),
        (ARM, {**BASE, **ALL}, ALL18, None),
        ('map512', BASE, 'shipped_fb_control', None),   # all 36 sites, other fallback: the INERT pair
        (ARM, A256, 'rank_control', None)]              # differing table rank: the other half


def _pooled_gain(x, cov, lab):
    """nats bought over the shipped build, pooled across all three roles -- LESSON 101: a shipping
    decision is made on the pooled test, not a per-role vote"""
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _best_known_reproduces(x):
    """§2019's build rebuilds to 0.000962 pooled at 5,419 and 0.002340 at 16,110, within 0.0002 nats.
    Everything below is measured as an increment over it"""
    return all(abs(_pooled_gain(x, c, BEST) - v) < 0.0002 for c, v in BEST_POOLED.items())


def _more_rank_still_buys_something(x):
    """and untruncating all eighteen buys strictly more than §2019's two, pooled, at both coverages. If it
    does not, rank 768 is already past the useful point at the shallow sites and the axis is closed"""
    return all(_pooled_gain(x, c, ALL18) > _pooled_gain(x, c, BEST) for c in (LO, HI))


def _it_does_not_pay_for_itself(x):
    """but the sixteen extra sites do NOT clear their 0.00404 price at the deployed 5,419 coverage: the
    increment over §2019's build is under that. Registered directionally -- §2015 measured the shallow
    sites' whole content at 0.055 nats, so an increment of four milli-nats from raising their rank would
    be most of what they have. If FALSE the build should take all eighteen and §2019 under-bought again"""
    return _pooled_gain(x, LO, ALL18) - _pooled_gain(x, LO, BEST) < PRICE_ALL


def _the_middle_option_is_not_better(x):
    """and the six-site middle option does not clear its own 0.00150 price either, at 5,419 -- if the
    curve had a knee between two sites and eighteen, this is where it would show"""
    return _pooled_gain(x, LO, MID) - _pooled_gain(x, LO, BEST) < PRICE_MID


B.run(
    name='do_the_shallow_sites_want_rank_too',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_best_known_reproduces',
         '§2019\'s build rebuilds to 0.000962 / 0.002340 pooled within 0.0002 nats',
         _best_known_reproduces),
        ('pred_b_more_rank_still_buys_something',
         'and untruncating all eighteen buys strictly more than the late two, pooled, at both coverages',
         _more_rank_still_buys_something),
        ('pred_c_it_does_not_pay_for_itself',
         'but the sixteen extra sites do not clear their 0.00404 price at 5,419',
         _it_does_not_pay_for_itself),
        ('pred_d_the_middle_option_is_not_better',
         'and the six-site middle option does not clear its 0.00150 price either',
         _the_middle_option_is_not_better),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(BEST, SHIPPED), (MID, SHIPPED), (ALL18, SHIPPED), (ALL18, BEST)],
)
