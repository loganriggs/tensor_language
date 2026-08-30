# DID ALPHA'S OPTIMUM MOVE WHEN THE RANKS DID?
#
# §2026 showed alpha wants to be UNIFORM -- every per-site move loses 1.7 to 11.0 milli-nats -- and that
# it is the most sensitive parameter in the build: a 0.20 move at eight sites costs 11.0 milli-nats, where
# untruncating those same eight tables bought 3.30 (§2020) and the whole map refinement was worth 0.47
# (§2024).
#
# But §2026 only tested 0.10 / 0.30 / 0.50. The VALUE 0.30 was located by §1961 and §1967 on a fine grid
# at the OLD allocation -- uniform mlp 768, uniform map 640 -- before per-site ranks existed. §2020 and
# §2024 have since changed what the covered rows carry at the late sites and what the map carries at the
# early ones, and alpha trades exactly those two things off. If the optimum moved even one grid step, that
# is worth more than everything §2020-§2024 bought together, and alpha is free.
#
# ARMS. §2024's build at alpha 24 / 26 / 28 / 30 / 32 / 34 / 36 -- §1961's own grid spacing, which §1968
# and the alpha_dense work established as the resolution where this axis is flat. A fallback variant of
# the shipped build for the inert half of the control, and one differing-table-rank arm for the other.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2026's open question, corrected.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
CUT = 8                                                  # §2024: map 256 at MLP 0-7
SHIPPED = 'shipped'
ALPHAS = (24, 26, 28, 30, 32, 34, 36)
LAB = [f'a{a}' for a in ALPHAS]

BEST_2024 = {LO: 0.003064, HI: 0.007486}                 # §2024 / §2025 / §2026, the a30 arm


def _arm(a):
    early = ','.join(f'mlp{L}' for L in range(CUT))
    late = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < CUT))
    return f'mix{a}m256@{early}+mix{a}m640@{late}'


PLAN = [('mix30m640', BASE, SHIPPED, None)] + \
       [(_arm(a), {**BASE, **TABLES}, f'a{a}', B.SITES) for a in ALPHAS] + [
    ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites, other fallback: the INERT pair
    ('mix30m640', A256, 'rank_control', None)]         # differing table rank: the other half


def _gain(x, cov, lab):
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _argmax(x, cov):
    return max(LAB, key=lambda l: _gain(x, cov, l))


def _the_a30_arm_reproduces(x):
    """the alpha-30 arm reproduces §2024's build to 0.0002 nats at both coverages -- it IS that build, so
    if it does not rebuild, the composite is not expressing what §2020-§2024 recorded"""
    return all(abs(_gain(x, c, 'a30') - v) < 0.0002 for c, v in BEST_2024.items())


def _the_optimum_is_still_30(x):
    """and the optimum is still alpha 0.30 at the deployed coverage, or within one grid step of it. If
    FALSE the ranks moved the blend and the build takes a free correction larger than everything
    §2020-§2024 bought"""
    return abs(ALPHAS.index(int(_argmax(x, LO)[1:])) - ALPHAS.index(30)) <= 1


def _the_axis_is_still_flat_here(x):
    """and the three points around the optimum span under 0.5 milli-nats at 5,419 -- §1967's stopping
    rule, which §1968 applied to this axis at the old allocation. If FALSE the new ranks have made alpha
    sharper and it needs a finer grid, not just a re-check"""
    i = LAB.index(_argmax(x, LO))
    near = [_gain(x, LO, LAB[j]) for j in (i - 1, i, i + 1) if 0 <= j < len(LAB)]
    return max(near) - min(near) < 0.0005


def _both_coverages_agree(x):
    """and the two coverages put the optimum within one grid step of each other -- §2024 and §2025 found
    the MAP knee coverage-dependent, and alpha trades against the map, so this asks whether that
    dependence propagated into the blend"""
    return abs(ALPHAS.index(int(_argmax(x, LO)[1:]))
               - ALPHAS.index(int(_argmax(x, HI)[1:]))) <= 1


B.run(
    name='did_the_alpha_optimum_move',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_a30_arm_reproduces',
         'the alpha-30 arm rebuilds §2024\'s build to 0.003064 / 0.007486 within 0.0002 nats',
         _the_a30_arm_reproduces),
        ('pred_b_the_optimum_is_still_30',
         'and the optimum is still alpha 0.30 at 5,419, within one grid step', _the_optimum_is_still_30),
        ('pred_c_the_axis_is_still_flat_here',
         'and the three points around it span under 0.5 milli-nats -- §1967\'s stopping rule',
         _the_axis_is_still_flat_here),
        ('pred_d_both_coverages_agree',
         'and both coverages put the optimum within one grid step of each other', _both_coverages_agree),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[('a24', 'a30'), ('a36', 'a30'), ('a30', SHIPPED), ('a26', 'a30'), ('a34', 'a30')],
)
