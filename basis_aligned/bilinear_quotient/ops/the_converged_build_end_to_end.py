# THE CONVERGED BUILD, SCORED END-TO-END AGAINST WHAT WAS DEPLOYED.
#
# §2013-§2027 moved the build from 189.5M to 202.6M in fifteen sections, each scored against the shipped
# design one parameter at a time: per-site table ranks (§2020, knee at layer 10), attention rank left alone
# (§2022), a per-site map cut (§2024, knee at layer 8), the two knees shown distinct (§2025), and alpha
# re-checked and left at 0.30 (§2026, §2027).
#
# Every one of those was an increment. The converged build has never been scored against §1789's DEPLOYED
# design -- the comparison §1970 last made, at 230.087M values, before any of this line existed. That is
# the number the whole arc exists to produce, and it is one run.
#
# ARMS. §1789's deployed design (full-rank tables, rank-64 map); §1959's build, which §1970 recorded as
# beating it by 69.238 milli-nats; the converged build; a fallback variant of the shipped build for the
# inert half of the control; and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- the converged build second-class
# confirmed against the deployed design at both coverages.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
CUT = 8                                                  # §2024: map 256 at MLP 0-7
DEPLOYED, S1959, CONVERGED = 'deployed_1789', 'build_1959', 'converged'

# §1970: the §1959 build beats §1789's deployed design by 69.238 milli-nats pooled at 5,419
S1970_MARGIN = 0.069238


def _converged_arm():
    early = ','.join(f'mlp{L}' for L in range(CUT))
    late = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < CUT))
    return f'mix30m256@{early}+mix30m640@{late}'


PLAN = [('map64', None, DEPLOYED, None),                          # §1789's deployed design
        ('mix30m640', BASE, S1959, None),                         # §1959 / §1970's build
        (_converged_arm(), {**BASE, **TABLES}, CONVERGED, B.SITES),
        ('map512', BASE, 'shipped_fb_control', None),             # all 36 sites: the INERT pair
        ('mix30m640', A256, 'rank_control', None)]                # differing table rank: the other half


def _beats(x, cov, a, b):
    """nats by which a beats b, pooled across all three roles"""
    return -x.tpool_full(cov, a, b)['mean']


def _s1970_reproduces(x):
    """§1970's headline rebuilds: the §1959 build beats §1789's deployed design by 69.238 milli-nats
    pooled at 5,419, within 2 milli-nats. If it does not, the two anchors are not the objects §1970
    scored and the total below is not comparable to anything published"""
    return abs(_beats(x, LO, S1959, DEPLOYED) - S1970_MARGIN) < 0.002


def _the_converged_build_wins(x):
    """and the converged build beats the deployed design at both coverages, on the pooled test"""
    return all(_beats(x, c, CONVERGED, DEPLOYED) > 0 for c in (LO, HI))


def _it_beats_the_1959_build_too(x):
    """and it beats §1959's build, at both coverages -- fifteen sections of increments should sum to a
    positive whole, and §2014's super-additivity means that is not arithmetic"""
    return all(_beats(x, c, CONVERGED, S1959) > 0 for c in (LO, HI))


def _the_increments_roughly_sum(x):
    """and the total over §1959's build is within 1 milli-nat of the sum of what §2020 (+3.300) and §2024
    (-0.236 CE) recorded at 5,419 -- 3.064 milli-nats. If FALSE the increments do not compose and one of
    the fifteen sections was measuring something the whole build does not keep"""
    return abs(_beats(x, LO, CONVERGED, S1959) - 0.003064) < 0.001


B.run(
    name='the_converged_build_end_to_end',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_s1970_reproduces',
         '§1970\'s 69.238 milli-nat margin rebuilds within 2 milli-nats at 5,419', _s1970_reproduces),
        ('pred_b_the_converged_build_wins',
         'and the converged build beats §1789\'s deployed design at both coverages',
         _the_converged_build_wins),
        ('pred_c_it_beats_the_1959_build_too',
         'and it beats §1959\'s build at both coverages', _it_beats_the_1959_build_too),
        ('pred_d_the_increments_roughly_sum',
         'and its margin over §1959 is within 1 milli-nat of the 3.064 the increments recorded',
         _the_increments_roughly_sum),
    ],
    refs=[(S1959, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(S1959, DEPLOYED), (CONVERGED, DEPLOYED), (CONVERGED, S1959)],
)
