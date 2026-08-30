# DOES "ONE BUILD SUFFICES" SURVIVE ROWS THAT DID NOT CHOOSE IT?
#
# §2046's audit left thirteen in-sample selections standing and §2047 cleared three of them. The largest
# remaining is §1960's: that a single compromise build sits within 0.002 nats of the coverage-specific
# optimum at BOTH coverages, decided on a 3.0 milli-nat margin on the three published roles.
#
# It is under pressure from inside the ledger already. §2024 found the map cut's knee at MLP layer 8 at
# 5,419 and beyond layer 9 at 16,110 -- the first parameter where the two coverages want different things
# -- and §2025 showed that disagreement large enough to flip a build decision, though by only 0.04
# milli-nats. §1960 predates both.
#
# The claim is about the build of record, so it matters: if one build does NOT suffice on fresh rows, the
# deployed program should differ by coverage and §1960's convenience was selection.
#
# ARMS. §1959's build (attn 384, map 640) as the compromise; the coverage-specific alternatives §1960
# weighed -- attention 256 with map 640, and attention 384 with map 512. A fallback variant for the inert
# half of the control, and one differing-table-rank arm for the other half. Scored at BOTH coverages on
# the fresh window, which is the comparison §1960 made in-sample.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2047's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
COMPROMISE, ALT_ATTN, ALT_MAP = 'build_1959', 'attn256_map640', 'attn384_map512'
FRESH = ('fresh',)
BAR = 0.002        # §1960's own bar, nats: "within 0.002 of the coverage-specific optimum"

PLAN = [('mix30m640', BASE, COMPROMISE, None),
        ('mix30m640', A256, ALT_ATTN, None),
        ('mix30m512', BASE, ALT_MAP, None),
        ('map512', BASE, 'shipped_fb_control', None),                       # all 36 sites: INERT pair
        ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _beats(x, cov, a, b):
    if a == b:
        return 0.0
    return -x.tpool_full(cov, a, b)['mean']


def _gap(x, cov):
    """how far the compromise sits below the best of the three at this coverage, in nats"""
    best = max(_beats(x, cov, l, COMPROMISE) for l in (ALT_ATTN, ALT_MAP, COMPROMISE))
    return best - 0.0


def _the_compromise_wins_at_low_coverage(x):
    """§1959's build beats both alternatives at 5,419 on fresh rows. It was chosen there, so this is the
    weaker half of the test -- but if it loses even here, the build of record is wrong on rows that did
    not select it and that is a larger finding than §1960"""
    return all(_beats(x, LO, COMPROMISE, l) > 0 for l in (ALT_ATTN, ALT_MAP))


def _the_compromise_wins_at_high_coverage(x):
    """and it beats both at 16,110 too. §1960's claim is that ONE build serves both; §2024 found the first
    parameter where the coverages disagree, so this is where the claim is most exposed"""
    return all(_beats(x, HI, COMPROMISE, l) > 0 for l in (ALT_ATTN, ALT_MAP))


def _the_gap_clears_1960s_own_bar(x):
    """and at both coverages the compromise is within §1960's own 0.002-nat bar of the best arm -- which
    it is trivially if it IS the best arm, and this is the form §1960 registered. Scored as written"""
    return _gap(x, LO) <= BAR and _gap(x, HI) <= BAR


def _the_alternatives_are_not_within_noise(x):
    """and the compromise's margin over the better alternative exceeds a third of a milli-nat at both
    coverages. §2043 and §2047 both found the marginal comparison unidentifiable; if this one is too,
    §1960's 'one build suffices' is true only because the alternatives are indistinguishable, which is a
    different claim from the one it made"""
    def m(cov):
        return min(_beats(x, cov, COMPROMISE, l) for l in (ALT_ATTN, ALT_MAP))
    return m(LO) > 0.00033 and m(HI) > 0.00033


B.run(
    name='does_one_build_still_suffice_fresh',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_compromise_wins_at_low_coverage',
         '§1959\'s build beats both alternatives at 5,419 on fresh rows',
         _the_compromise_wins_at_low_coverage),
        ('pred_b_the_compromise_wins_at_high_coverage',
         'and beats both at 16,110 too -- where §2024 found the coverages disagreeing',
         _the_compromise_wins_at_high_coverage),
        ('pred_c_the_gap_clears_1960s_own_bar',
         'and it is within §1960\'s own 0.002-nat bar of the best arm at both coverages',
         _the_gap_clears_1960s_own_bar),
        ('pred_d_the_alternatives_are_not_within_noise',
         'and its margin over the better alternative exceeds a third of a milli-nat at both coverages',
         _the_alternatives_are_not_within_noise),
    ],
    paired_pairs=[(ALT_ATTN, COMPROMISE), (ALT_MAP, COMPROMISE), (ALT_ATTN, ALT_MAP)],
)
