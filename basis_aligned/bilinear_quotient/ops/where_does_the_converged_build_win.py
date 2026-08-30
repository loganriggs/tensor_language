# DOES THE CONVERGED BUILD WIN EVERYWHERE, OR ONLY ON AVERAGE?
#
# §2028 recorded the converged build as beating §1789's deployed design by 72.302 milli-nats pooled and
# §1959's build by 3.064, at 202.6M values against 230.087M. Every number in the fifteen-section arc that
# produced it was a POOLED AVERAGE over 92,160 positions.
#
# A build that wins on average can lose where it matters. Both instruments to check that are in every
# artifact already: §1789's target-frequency buckets and §1936's covered/uncovered input axis. §2015
# showed the gains come from the two deepest MLPs, whose tables serve the most frequent targets most --
# so a win concentrated in the frequent buckets, with the rare and unseen ones flat or losing, is the
# specific failure mode to look for, and nothing in the arc would have caught it.
#
# ARMS. §1959's build and the converged build -- the comparison the 3.064 milli-nats describes -- plus
# §1789's deployed design as the outer anchor; a fallback variant of the shipped build for the inert half
# of the control; and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2028's implicit open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
CUT = 8                                                  # §2024
DEPLOYED, S1959, CONVERGED = 'deployed_1789', 'build_1959', 'converged'


def _converged_arm():
    early = ','.join(f'mlp{L}' for L in range(CUT))
    late = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < CUT))
    return f'mix30m256@{early}+mix30m640@{late}'


PLAN = [('map64', None, DEPLOYED, None),
        ('mix30m640', BASE, S1959, None),
        (_converged_arm(), {**BASE, **TABLES}, CONVERGED, B.SITES),
        ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites: the INERT pair
        ('mix30m640', A256, 'rank_control', None)]         # differing table rank: the other half


def _win(x, role, cls='pooled', bucket='overall'):
    """nats by which the converged build beats §1959's, on one cell, at 5,419"""
    return (x.ce(LO, role, S1959, cls, bucket) - x.ce(LO, role, CONVERGED, cls, bucket))


def _it_wins_in_every_bucket(x):
    """the converged build beats §1959's in ALL FIVE target-frequency buckets, on >=2 roles. §2015 put its
    gains at the two deepest MLPs, which serve frequent targets most, so a win concentrated there with the
    rare buckets flat or negative is the failure mode this asks about"""
    return sum(1 for r in x.roles if all(_win(x, r, 'pooled', b) > 0 for b in x.buckets)) >= 2


def _it_wins_on_the_unseen_bucket(x):
    """and specifically on the unseen-target bucket, on all three roles -- the cell the fallback exists
    for, and the one §2024's map cut touched most directly by making the shallow map poorer"""
    return all(_win(x, r, 'pooled', x.bot) > 0 for r in x.roles)


def _it_wins_at_uncovered_inputs(x):
    """and at uncovered inputs, on all three roles. §2024 cut the map rank at MLP layers 0-7 and the map
    acts ONLY on uncovered rows, so if the build lost anywhere it should lose here"""
    return all(_win(x, r, 'uncovered_input') > 0 for r in x.roles)


def _the_win_is_not_only_frequent_targets(x):
    """and the unseen bucket's share of the win is at least a fifth of the most frequent bucket's, on
    >=2 roles -- a build whose entire margin sat in the frequent cells would be a worse object than the
    average says, even while winning everywhere"""
    return sum(1 for r in x.roles
               if _win(x, r, 'pooled', x.bot) >= 0.2 * _win(x, r, 'pooled', x.top)) >= 2


B.run(
    name='where_does_the_converged_build_win',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_it_wins_in_every_bucket',
         'the converged build beats §1959\'s in all five frequency buckets (>=2 roles)',
         _it_wins_in_every_bucket),
        ('pred_b_it_wins_on_the_unseen_bucket',
         'and on the unseen-target bucket, on all three roles', _it_wins_on_the_unseen_bucket),
        ('pred_c_it_wins_at_uncovered_inputs',
         'and at uncovered inputs, on all three roles -- where §2024 made the map poorer',
         _it_wins_at_uncovered_inputs),
        ('pred_d_the_win_is_not_only_frequent_targets',
         'and the unseen bucket carries at least a fifth of the frequent bucket\'s win (>=2 roles)',
         _the_win_is_not_only_frequent_targets),
    ],
    refs=[(S1959, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(CONVERGED, S1959), (CONVERGED, DEPLOYED), (S1959, DEPLOYED)],
)
