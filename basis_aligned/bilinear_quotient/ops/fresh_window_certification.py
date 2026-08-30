# BACKLOG RUNG 6: FRESH-WINDOW CERTIFICATION OF THE BUILD OF RECORD.
#
# The standing benchmark backlog requires fresh-window certification for every new winner. §2028 recorded
# it as unsatisfiable with the caches on disk. THAT WAS WRONG. bilin18_eval_tokens_large.pt holds 512 rows
# x 513 tokens with 510 distinct 24-token prefixes and, MEASURED 2026-08-30, ZERO overlap with any of the
# three eval roles or either fit set. It is a genuine held-out window that was simply never wired in.
#
# This matters more than the usual replication because the build was SELECTED on the three published
# roles across fifteen sections (§2016-§2027). Every margin in that arc is in-sample with respect to the
# roles that chose it. §2028's out-of-selection evidence was the 16,110 fit set, which is a different FIT
# set on the SAME eval rows -- this is different eval rows entirely.
#
# ARMS. §1789's deployed design; §1959's build; the converged build of record. A fallback variant of the
# shipped build for the inert half of the control, and one differing-table-rank arm for the other half.
#
# ROLES. 'fresh' ONLY -- 512 rows, positions >=64, about 229,888 scored positions against the three
# published roles' 92,160 combined.
#
# DISCOVERY ONLY. Rung 4 -- BENCHMARK_BACKLOG rung 6, and a correction to §2028's claim that it could not
# be done.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
CUT = 8                                                  # §2024
DEPLOYED, S1959, CONVERGED = 'deployed_1789', 'build_1959', 'converged'
FRESH = ('fresh',)

# §2028 on the three published roles, pooled, at 5,419 -- the in-sample figures this certifies against
S1959_OVER_DEPLOYED = 0.069238
CONVERGED_OVER_DEPLOYED = 0.072302
CONVERGED_OVER_S1959 = 0.003064


def _cut_arm():
    early = ','.join(f'mlp{L}' for L in range(CUT))
    late = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < CUT))
    return f'mix30m256@{early}+mix30m640@{late}'


PLAN = [('map64', None, DEPLOYED, None),
        ('mix30m640', BASE, S1959, None),
        (_cut_arm(), {**BASE, **TABLES}, CONVERGED, B.SITES),
        ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites: the INERT pair
        ('mix30m640', A256, 'rank_control', None)]         # differing table rank: the other half


def _beats(x, a, b):
    """nats by which a beats b on the fresh window"""
    return -x.tpool_full(LO, a, b)['mean']


def _the_converged_build_wins_fresh(x):
    """the converged build beats §1789's deployed design on rows it was never selected on. If FALSE the
    whole §2013-§2035 arc is in-sample and the build of record does not generalise"""
    return _beats(x, CONVERGED, DEPLOYED) > 0


def _it_still_beats_the_1959_build(x):
    """and it still beats §1959's build on the fresh window -- the 3.064 milli-nats the arc bought were
    chosen by looking at the three published roles, and this is the first evidence that they are not an
    artefact of that selection"""
    return _beats(x, CONVERGED, S1959) > 0


def _the_1789_anchor_holds_fresh(x):
    """and §1970's anchor survives the window change: §1959's build beats the deployed design by within
    a factor of two of its published 69.238 milli-nats. A margin this large should transport; if it does
    not, the fresh rows differ from the published ones in some way that invalidates the comparison rather
    than testing it"""
    m = _beats(x, S1959, DEPLOYED)
    return 0.5 * S1959_OVER_DEPLOYED < m < 2.0 * S1959_OVER_DEPLOYED


def _the_arc_margin_is_not_mostly_selection(x):
    """and the converged build's margin over §1959 retains at least half its in-sample size -- more than
    1.532 milli-nats against the 3.064 recorded on the selecting roles. Registered directionally: fifteen
    sections of parameter choices made against three roles could plausibly lose most of their margin on
    rows that had no say, and half is the bar for calling the gain real rather than fitted"""
    return _beats(x, CONVERGED, S1959) > 0.5 * CONVERGED_OVER_S1959


B.run(
    name='fresh_window_certification',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_converged_build_wins_fresh',
         'the converged build beats §1789\'s deployed design on the fresh window',
         _the_converged_build_wins_fresh),
        ('pred_b_it_still_beats_the_1959_build',
         'and it still beats §1959\'s build there', _it_still_beats_the_1959_build),
        ('pred_c_the_1789_anchor_holds_fresh',
         'and §1970\'s 69.238 milli-nat anchor transports within a factor of two',
         _the_1789_anchor_holds_fresh),
        ('pred_d_the_arc_margin_is_not_mostly_selection',
         'and the arc\'s margin keeps more than half its in-sample 3.064 milli-nats',
         _the_arc_margin_is_not_mostly_selection),
    ],
    paired_pairs=[(S1959, DEPLOYED), (CONVERGED, DEPLOYED), (CONVERGED, S1959)],
)
