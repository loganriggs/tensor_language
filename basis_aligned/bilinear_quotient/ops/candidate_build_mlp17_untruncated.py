# THE CANDIDATE BUILD, SCORED END-TO-END AT BOTH COVERAGES.
#
# §2017 found the shipped rank 768 slightly under-bought at mlp17: untruncating it to rank 1152 buys
# 0.00030-0.00063 nats against §1947's 0.00025 price, on 3/3 roles, at pooled t = -3.58. mlp16 does not
# clear the bar (one role negative). That is a discovery-run margin of half a milli-nat, and the house
# pattern is not to edit a build on one.
#
# This scores the candidate against the deployed design as a BUILD: both coverages, all three roles, with
# the price stated. §1993 and §2008 both found this family stable at 16,110, but §1963 and §1965 reversed
# 5,419 claims twice, and a 0.0005-nat margin is exactly the size a coverage change could erase.
#
# ARMS. the shipped build {mlp 768, attn 384} with a rank-640 map; the candidate, identical but with mlp17
# untruncated at 1152; mlp16 untruncated as the arm §2017 declined to recommend, carried so the comparison
# is three-way; a fallback variant of the shipped build for the inert half of the control; and one
# differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2017's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
SHIPPED, CAND, ALT = 'shipped', 'mlp17_1152', 'mlp16_1152'
PRICE = 0.00025          # §1947: +2.52M values at 0.010 nats per 100M

# §2017 at 5,419, skip7000 / skip11000 / skip1200
GAIN_5419 = (0.00063, 0.00030, 0.00051)

PLAN = [(ARM, BASE, SHIPPED, None),
        (ARM, {**BASE, ('mlp', 17): 1152}, CAND, None),
        (ARM, {**BASE, ('mlp', 16): 1152}, ALT, None),
        ('map512', BASE, 'shipped_fb_control', None),   # all 36 sites, other fallback: the INERT pair
        (ARM, A256, 'rank_control', None)]              # differing table rank: the other half


def _gain(x, cov, role, lab):
    return x.penalty(cov, role, SHIPPED) - x.penalty(cov, role, lab)


def _the_5419_gain_reproduces(x):
    """§2017's mlp17 gain rebuilds to 0.00063 / 0.00030 / 0.00051 within 0.0002 nats on all three roles.
    The whole claim is half a milli-nat, so it has to reproduce before it can transfer"""
    return all(abs(_gain(x, LO, r, CAND) - v) < 0.0002 for r, v in zip(x.roles, GAIN_5419))


def _it_survives_the_coverage_change(x):
    """and at 16,110 the candidate still beats the shipped build on all three roles -- strictly positive
    gain. §1963 and §1965 each reversed a 5,419 claim at this coverage, and a 0.0005-nat margin is exactly
    the size that could erase"""
    return all(_gain(x, HI, r, CAND) > 0 for r in x.roles)


def _it_still_clears_the_price(x):
    """and it clears §1947's 0.00025 price at 16,110 on >=2 roles, as it did on 3/3 at 5,419. If FALSE the
    purchase is justified only at the lower coverage and the shipped build should not move"""
    return sum(1 for r in x.roles if _gain(x, HI, r, CAND) > PRICE) >= 2


def _mlp16_still_does_not_qualify(x):
    """and mlp16 still fails, at both coverages: it clears the price on fewer roles than mlp17 does.
    §2017 measured it negative on one role at 5,419, and carrying it here keeps the recommendation
    honest rather than resting on the single site that happened to pass"""
    def n(cov, lab):
        return sum(1 for r in x.roles if _gain(x, cov, r, lab) > PRICE)
    return n(LO, ALT) < n(LO, CAND) or n(HI, ALT) < n(HI, CAND)


B.run(
    name='candidate_build_mlp17_untruncated',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_5419_gain_reproduces',
         '§2017\'s mlp17 gain rebuilds to 0.00063/0.00030/0.00051 within 0.0002 nats on 3/3 roles',
         _the_5419_gain_reproduces),
        ('pred_b_it_survives_the_coverage_change',
         'and the candidate still beats the shipped build at 16,110, on 3/3 roles',
         _it_survives_the_coverage_change),
        ('pred_c_it_still_clears_the_price',
         'and it still clears §1947\'s 0.00025 price at 16,110 (>=2 roles)', _it_still_clears_the_price),
        ('pred_d_mlp16_still_does_not_qualify',
         'and mlp16 clears the price on fewer roles than mlp17, at one coverage or the other',
         _mlp16_still_does_not_qualify),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(CAND, SHIPPED), (ALT, SHIPPED), (CAND, ALT)],
)
