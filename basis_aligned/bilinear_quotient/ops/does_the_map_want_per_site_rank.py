# THE THIRD UNIFORM SWEEP: DOES THE MAP WANT A PER-SITE RANK?
#
# §2020 found the MLP table rank had an unexplored region above 768 because §1947/§1959 swept it
# UNIFORMLY over eighteen heterogeneous sites. §2022 found the attention axis had no such region -- 384 is
# correctly bought -- because attention contributions are small and flat where MLP ones are back-loaded and
# span three orders of magnitude.
#
# §1870's map is the third per-site quantity in this build, fit separately at all thirty-six sites, and
# §1959 swept MAP_RANK uniformly and put the turnover at 640. It has never been re-examined per-site. The
# map acts only on UNCOVERED rows, so the question is whether the sites whose tables carry nothing (§2015:
# layers 0-9 at under 0.005 nats each) also need a rank-640 map for the tokens they have not seen.
#
# A rank-r map at one site costs r x 2 x D = 2304r values, so 640 -> 256 SAVES 0.885M per site: cutting it
# at the ten shallow MLP sites saves 8.85M, worth 0.00089 nats at §1947's price. This is the first arm in
# this line that tries to make the build CHEAPER rather than better.
#
# ARMS. §2020's best-known build; the same with a rank-256 map at MLP layers 0-9, at MLP layers 0-5, and
# at all eighteen MLP sites; a fallback variant of the shipped build for the inert half of the control;
# and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2022's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
BEST = {('mlp', L): 1152 for L in range(10, 18)}        # §2020's build
RICH, POOR = 'mix30m640', 'mix30m256'
SHIPPED, B2020 = 'shipped', 'best_2020'

BEST_POOLED = {LO: 0.003300, HI: 0.007621}              # §2020 / §2021 / §2022
SAVING = {'cheap10': 0.00089, 'cheap6': 0.00053, 'cheap18': 0.00160}


def _mixed(cheap):
    """rank-256 map at the named MLP layers, rank-640 everywhere else"""
    poor = ','.join(f'mlp{L}' for L in cheap)
    rich = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L in cheap))
    return f'{POOR}@{poor}+{RICH}@{rich}'


PLAN = [(RICH, BASE, SHIPPED, None),
        (RICH, {**BASE, **BEST}, B2020, None),                                    # §2020: 0.003300
        (_mixed(range(10)), {**BASE, **BEST}, 'cheap10', B.SITES),
        (_mixed(range(6)), {**BASE, **BEST}, 'cheap6', B.SITES),
        (_mixed(range(18)), {**BASE, **BEST}, 'cheap18', B.SITES),
        ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites, other fallback: the INERT pair
        (RICH, A256, 'rank_control', None)]                # differing table rank: the other half


def _gain(x, cov, lab):
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _the_best_known_reproduces(x):
    """§2020's build rebuilds to 0.003300 pooled at 5,419 and 0.007621 at 16,110 within 0.0002 nats"""
    return all(abs(_gain(x, c, B2020) - v) < 0.0002 for c, v in BEST_POOLED.items())


def _cutting_the_shallow_map_is_free(x):
    """and cutting the map to rank 256 at MLP layers 0-9 costs less than the 0.00089 nats its 8.85M saved
    values are worth, at the deployed 5,419 coverage. If TRUE the build gets cheaper for free and §1959's
    uniform map rank was over-bought at the sites whose tables carry nothing"""
    return _gain(x, LO, B2020) - _gain(x, LO, 'cheap10') < SAVING['cheap10']


def _cutting_it_everywhere_is_not(x):
    """but cutting it at ALL eighteen MLP sites costs more than its 0.00160 saving, at 5,419 -- §1959 put
    the map turnover at 640 on a uniform sweep, so the late sites at least should want the richer map. If
    FALSE the map rank is over-bought everywhere and §1959's turnover needs re-deriving, not refining"""
    return _gain(x, LO, B2020) - _gain(x, LO, 'cheap18') > SAVING['cheap18']


def _the_saving_survives_the_coverage_change(x):
    """and the shallow cut is still free at 16,110, where the uncovered arm is half the size and the map
    matters less. §1963 and §1965 both reversed a 5,419 claim at this coverage; a cut that is free at the
    deployed coverage and costly at the other is not a cut worth making"""
    return _gain(x, HI, B2020) - _gain(x, HI, 'cheap10') < SAVING['cheap10']


B.run(
    name='does_the_map_want_per_site_rank',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_best_known_reproduces',
         '§2020\'s build rebuilds to 0.003300 / 0.007621 pooled within 0.0002 nats',
         _the_best_known_reproduces),
        ('pred_b_cutting_the_shallow_map_is_free',
         'and a rank-256 map at MLP layers 0-9 costs less than its 0.00089 saving at 5,419',
         _cutting_the_shallow_map_is_free),
        ('pred_c_cutting_it_everywhere_is_not',
         'but cutting it at all eighteen costs more than its 0.00160 saving', _cutting_it_everywhere_is_not),
        ('pred_d_the_saving_survives_the_coverage_change',
         'and the shallow cut is still free at 16,110', _the_saving_survives_the_coverage_change),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(B2020, SHIPPED), ('cheap10', SHIPPED), ('cheap18', SHIPPED), ('cheap6', SHIPPED)],
)
