# DOES ONE BUILD STILL SUFFICE NOW THAT THE TILT IS COVERAGE-DEPENDENT?
#
# §1960 established that a single compromise build sits within 0.002 nats of the coverage-specific
# optimum at both coverages, when the parameters in play were the allocation and the map rank. §1965 has
# since added a fourth: the per-token tilt is worth shipping at 16,110 (+0.47/+0.39/-0.49 milli-nats,
# clearing the bar 3/3) and is not at 5,419 (+1.45 to +2.94). So the coverage-specific optima now differ
# in the tilt as well, and §1960's answer was obtained before that parameter existed.
#
# The question §1960 asked, re-asked with the fourth parameter: is there still ONE build within 0.002
# nats of the coverage-specific optimum at both coverages? 0.002 is the bar §1960 used and the size of
# the marginal purchases §1957-§1961 were making.
#
# ARMS. spec_5419 (flat α=0.30) and spec_16110 (the narrow tilt), both at {mlp 768, attn 384} with a
# rank-640 map; plus the two intermediate tilts as candidate compromises, and one differing-table-rank
# arm so neither half of the derived control is vacuous.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1965's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
LO, HI = 'c5419', 'c16110'
SPEC = {LO: 'spec_5419', HI: 'spec_16110'}
MID = ('mid_2535', 'mid_2832')

PLAN = [('mix30m640', A384, 'spec_5419'),          # §1964: the 5,419 optimum is the flat blend
        ('pat20_30m640', A384, 'spec_16110'),      # §1965: the 16,110 optimum is the narrow tilt
        ('pat25_35m640', A384, 'mid_2535'),
        ('pat28_32m640', A384, 'mid_2832'),
        ('mix25m512', A256, 'rank_control')]


def _gap(x, cov, arm):
    """how far arm sits from that coverage's own specific build, in nats, worst over roles"""
    return max(x.ce(cov, r, arm) - x.ce(cov, r, SPEC[cov]) for r in x.roles)


def _one_build_suffices(x):
    """some compromise is within 0.002 nats of the coverage-specific optimum at BOTH coverages"""
    return any(_gap(x, LO, a) <= 0.002 and _gap(x, HI, a) <= 0.002 for a in MID)


def _spec_arms_disagree(x):
    """and the two coverage-specific builds really do differ -- if each is already within 0.002 of the
    other there was never a compromise to find, and §1965's boundary is smaller than it looked"""
    return _gap(x, LO, SPEC[HI]) > 0.002 or _gap(x, HI, SPEC[LO]) > 0.002


def _tilt_direction_holds(x):
    """and the tilt still helps the unseen bucket at 16,110 and not at 5,419 -- §1965's finding, which
    is what makes this question live"""
    hi = sum(1 for r in x.roles
             if x.kf(HI, r, SPEC[HI], x.bot) >= x.kf(HI, r, SPEC[LO], x.bot))
    return hi >= 2


B.run(
    name='one_build_with_tilt',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_one_build_still_suffices',
         'one compromise build is within 0.002 nats of both coverage-specific optima',
         _one_build_suffices),
        ('pred_b_the_specs_really_differ',
         'and the two coverage-specific builds genuinely differ by more than that bar',
         _spec_arms_disagree),
        ('pred_c_tilt_direction_holds',
         'and the tilt still helps the unseen bucket at 16,110 (>=2 roles) -- §1965 reproduced',
         _tilt_direction_holds),
    ],
    refs=[('spec_5419', B.PT + 'ops/tilt_both_coverages_results.json', 'flat30', LO, 0.0005)],
    paired_pairs=[('mid_2535', 'spec_5419'), ('mid_2832', 'spec_5419'),
                  ('spec_16110', 'spec_5419')],
)
