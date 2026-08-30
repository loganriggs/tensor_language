# ARE THE TABLE-RANK KNEE AND THE MAP KNEE THE SAME BOUNDARY?
#
# §2021 put the table-rank knee at MLP layer 10: untruncating layers 10-17 pays, layers 8-9 do not
# (0.3x). §2024 put the map knee at layer 8: cutting the map to rank 256 through layer 7 is free, layers
# 8-9 are not (2.85x). Two knees, two layers apart, both separating "late" sites from "early" ones, and
# Codex's exact certificate (§2023) says neither has a native coefficient explanation -- the rank-768
# tails are smooth across both, MLP10/MLP9 = 1.0113x.
#
# If they are one boundary seen through two parameters, moving either to the other's layer should cost
# little and moving both to a common layer should cost least. If they are two boundaries, each parameter
# is already at its own optimum and every move loses.
#
# ARMS. §2024's build (tables from 10, map cut through 7); the table knee moved to 8; the map knee moved
# to 10; both met at 9; a fallback variant of the shipped build for the inert half of the control; and one
# differing-table-rank arm for the other half. Prices are stated per arm because the moves are not
# price-neutral: a table site costs 2.52M, a map site releases 0.885M.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2024's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
RICH, POOR = 'mix30m640', 'mix30m256'
SHIPPED = 'shipped'

# value of the parameters each arm holds relative to §2024's build, at §1947's 0.010 per 100M
# a table site untruncated costs 2.52M (0.00025); a map site cut releases 0.885M (0.000089)
DELTA_PRICE = {'tab8': -0.00050, 'map10': +0.000177, 'meet9': -0.00025 + 0.000089}


def _tables(lo):
    return {('mlp', L): 1152 for L in range(lo, 18)}


def _map(n):
    poor = ','.join(f'mlp{L}' for L in range(n))
    rich = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < n))
    return f'{POOR}@{poor}+{RICH}@{rich}'


PLAN = [(RICH, BASE, SHIPPED, None),
        (_map(8), {**BASE, **_tables(10)}, 'best_2024', B.SITES),
        (_map(8), {**BASE, **_tables(8)}, 'tab8', B.SITES),
        (_map(10), {**BASE, **_tables(10)}, 'map10', B.SITES),
        (_map(9), {**BASE, **_tables(9)}, 'meet9', B.SITES),
        ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites, other fallback: the INERT pair
        (RICH, A256, 'rank_control', None)]                # differing table rank: the other half


def _gain(x, cov, lab):
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _net(x, cov, lab):
    """CE gain over §2024's build plus the value of the parameters the move releases (negative if it
    spends them) -- the only comparison that is fair when the moves are not price-neutral"""
    return _gain(x, cov, lab) - _gain(x, cov, 'best_2024') + DELTA_PRICE[lab]


def _the_2024_build_reproduces(x):
    """§2024's build rebuilds within 0.0002 nats of its own recorded pooled gain at both coverages --
    0.003064 at 5,419 and 0.007486 at 16,110. Every net below is a difference of tenths of a milli-nat"""
    want = {LO: 0.003064, HI: 0.007486}
    return all(abs(_gain(x, c, 'best_2024') - v) < 0.0002 for c, v in want.items())


def _no_move_beats_the_current_build(x):
    """and no single move is net-positive at the deployed 5,419 coverage: each parameter is already at its
    own optimum. If TRUE the two knees are two boundaries and §2024's build stands; if FALSE one of them
    was mis-placed and the build moves again"""
    return all(_net(x, LO, lab) < 0 for lab in DELTA_PRICE)


def _meeting_at_nine_is_not_the_answer(x):
    """and meeting both knees at layer 9 is not better than leaving them apart -- net negative at 5,419.
    A single shared boundary would show up here most clearly, because it is the only arm that moves both
    parameters to the same layer"""
    return _net(x, LO, 'meet9') < 0


def _the_moves_agree_across_coverages(x):
    """and no move that loses at 5,419 wins at 16,110 -- §2024 already found the map knee coverage-
    dependent, so this asks whether that dependence is large enough to reverse a build decision"""
    return all((_net(x, LO, lab) < 0) == (_net(x, HI, lab) < 0) for lab in DELTA_PRICE)


B.run(
    name='are_the_two_knees_one_boundary',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_2024_build_reproduces',
         '§2024\'s build rebuilds to 0.003064 / 0.007486 pooled within 0.0002 nats',
         _the_2024_build_reproduces),
        ('pred_b_no_move_beats_the_current_build',
         'and no single knee move is net-positive at 5,419 -- two boundaries, not one',
         _no_move_beats_the_current_build),
        ('pred_c_meeting_at_nine_is_not_the_answer',
         'and meeting both knees at layer 9 is net-negative too', _meeting_at_nine_is_not_the_answer),
        ('pred_d_the_moves_agree_across_coverages',
         'and no move that loses at 5,419 wins at 16,110', _the_moves_agree_across_coverages),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[('best_2024', SHIPPED), ('tab8', SHIPPED), ('map10', SHIPPED), ('meet9', SHIPPED)],
)
