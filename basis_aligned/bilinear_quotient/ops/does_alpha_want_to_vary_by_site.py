# THE FOURTH AND LAST UNIFORM SWEEP: DOES THE BLEND ALPHA WANT TO VARY BY SITE?
#
# §2016-§2025 audited three parameters set by sweeps that were UNIFORM over per-site quantities. The MLP
# table rank had an unexplored region and a knee at layer 10 (§2020, +3.30 milli-nats). The attention rank
# had neither (§2022, 0.03-0.12x its price everywhere) -- and that is the axis §1989 measured as
# homogeneous. The map rank was over-bought below layer 8 (§2024, +0.47 net).
#
# Alpha is the last. §1961 and §1967 fixed the neighbour/map blend at 0.30 by sweeping it uniformly across
# all thirty-six sites. Alpha costs NOTHING -- it reweights two fallback arms the build already pays for --
# so unlike the three rank axes there is no price to clear: any net CE gain is free, and any loss is a
# pure loss. That makes it the cleanest of the four to score.
#
# ARMS. §2024's build at the uniform alpha 0.30; the same with alpha 0.10 and 0.50 at the LATE MLP sites
# (10-17, where the tables are untruncated and carry the content); and with alpha 0.10 at the EARLY sites
# (0-9, where §2024 already cut the map). A fallback variant of the shipped build for the inert half of
# the control, and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2025's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020's table ranks
SHIPPED, B2024 = 'shipped', 'best_2024'
CUT = 8                                                  # §2024: map rank 256 at MLP layers 0-7

BEST_2024 = {LO: 0.003064, HI: 0.007486}                 # §2024 / §2025, pooled over three roles


def _sites(pred):
    return ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18) if pred(k, L))


def _arm(a_early, a_late):
    """alpha a_early with the rank-256 map at MLP 0-7, a_late with the rank-640 map elsewhere"""
    early = _sites(lambda k, L: k == 'mlp' and L < CUT)
    late = _sites(lambda k, L: not (k == 'mlp' and L < CUT))
    return f'mix{a_early}m256@{early}+mix{a_late}m640@{late}'


PLAN = [('mix30m640', BASE, SHIPPED, None),
        (_arm(30, 30), {**BASE, **TABLES}, B2024, B.SITES),          # §2024: 0.003064 / 0.007486
        (_arm(30, 10), {**BASE, **TABLES}, 'late_a10', B.SITES),
        (_arm(30, 50), {**BASE, **TABLES}, 'late_a50', B.SITES),
        (_arm(10, 30), {**BASE, **TABLES}, 'early_a10', B.SITES),
        ('map512', BASE, 'shipped_fb_control', None),     # all 36 sites, other fallback: the INERT pair
        ('mix30m640', A256, 'rank_control', None)]        # differing table rank: the other half


def _gain(x, cov, lab):
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _delta(x, cov, lab):
    """CE gain over §2024's build. Alpha is free, so this IS the net -- no price to clear"""
    return _gain(x, cov, lab) - _gain(x, cov, B2024)


def _the_2024_build_reproduces(x):
    """§2024's build, rebuilt here as a composite arm at uniform alpha 0.30, reproduces its recorded
    pooled gain within 0.0002 nats at both coverages. If it does not, the composite is not expressing the
    same program and nothing below means anything"""
    return all(abs(_gain(x, c, B2024) - v) < 0.0002 for c, v in BEST_2024.items())


def _some_per_site_alpha_beats_uniform(x):
    """and at least one per-site alpha beats uniform 0.30 at the deployed coverage. Alpha costs nothing,
    so any positive delta is a free improvement. If FALSE the fourth uniform sweep was correct and the
    audit closes with three of four axes re-examined and one gap found in each of two"""
    return any(_delta(x, LO, lab) > 0 for lab in ('late_a10', 'late_a50', 'early_a10'))


def _the_late_sites_are_where_it_matters(x):
    """and whatever moves, moves more at the LATE sites than the early ones: the larger of the two late
    deltas exceeds the early one in absolute size, at 5,419. §2015 put 96% of the table content at the
    late sites, so a fallback parameter that mattered equally everywhere would be surprising"""
    late = max(abs(_delta(x, LO, l)) for l in ('late_a10', 'late_a50'))
    return late > abs(_delta(x, LO, 'early_a10'))


def _the_sign_agrees_across_coverages(x):
    """and no alpha move that helps at 5,419 hurts at 16,110 -- §2024 and §2025 found the MAP knee
    coverage-dependent, and alpha trades against the map, so this asks whether the dependence follows"""
    return all((_delta(x, LO, lab) > 0) == (_delta(x, HI, lab) > 0)
               for lab in ('late_a10', 'late_a50', 'early_a10'))


B.run(
    name='does_alpha_want_to_vary_by_site',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_2024_build_reproduces',
         '§2024\'s build rebuilds as a composite at uniform alpha to 0.003064 / 0.007486 within 0.0002',
         _the_2024_build_reproduces),
        ('pred_b_some_per_site_alpha_beats_uniform',
         'and at least one per-site alpha beats uniform 0.30 at 5,419 -- alpha is free, so any gain counts',
         _some_per_site_alpha_beats_uniform),
        ('pred_c_the_late_sites_are_where_it_matters',
         'and alpha moves more at the late sites than the early ones', _the_late_sites_are_where_it_matters),
        ('pred_d_the_sign_agrees_across_coverages',
         'and no alpha move that helps at 5,419 hurts at 16,110', _the_sign_agrees_across_coverages),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(B2024, SHIPPED), ('late_a10', B2024), ('late_a50', B2024), ('early_a10', B2024)],
)
