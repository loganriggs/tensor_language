# HOW FEW DIMENSIONS DOES ATTENTION 6'S CONTENT ACTUALLY NEED?
#
# §1996 swept attention 6's table rank from 16 to 384 and found the curve smooth and shallow: rank 16
# recovers 72% of what rank 384 does. The bottom has never been measured, and it is the one number that
# decides whether a head-level account of attention 6 is worth building. Nine heads of 128 dimensions
# each; if rank 2 or rank 4 already recovers most of the content, the signal is small enough that "which
# heads" is the obvious next question rather than a speculation.
#
# §2007 also showed attention 5's lever is entirely intact at rank 16 (+0.001), so the same sweep at
# attention 5 costs nothing extra to include and gives the comparison a floor.
#
# ARMS. mlp2 + attention 5,6 with attention 6's table at rank 1 / 2 / 4 / 8 / 16 / 384; the same at
# attention 5's rank 1 for the floor comparison; the arm omitting attention 6 as the ceiling of this
# comparison; the full 36-site program with a fallback variant for the inert half of the control; and one
# differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2008's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
SITES2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
RANKS = (1, 2, 4, 8, 16, 384)
NO6 = 'mlp2_a5'

# §1996 at 5,419, skip7000: attention 6 at rank 16 costs 2.137 and at rank 384 costs 1.971; the arm
# omitting attention 6 costs 2.556, so rank 16 recovers 72% of attention 6's whole contribution.
FULL_RANK = (1.971, 2.090, 1.952)


def _spec6(r):
    return {'mlp': 768, 'attn': 384, ('attn', 6): r}


PLAN = [(ARM, _spec6(r), f'a6r{r}', SITES2) for r in RANKS] + [
    (ARM, {'mlp': 768, 'attn': 384, ('attn', 5): 1}, 'a5r1', SITES2),
    (ARM, BASE, NO6, [('mlp', 2), ('attn', 5)]),        # §1992: 2.556 / 2.730 / 2.558
    (ARM, BASE, 'full_program', None),                  # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),          # all 36 sites: the INERT pair
    (ARM, A256, 'rank_control', None)]                  # differing rank: other half


def _recovered(x, role, lab):
    """share of attention 6's whole contribution that `lab` recovers, measured from the arm omitting it"""
    hi, lo = x.penalty(C, role, NO6), x.penalty(C, role, 'a6r384')
    return (hi - x.penalty(C, role, lab)) / (hi - lo)


def _full_rank_reproduces(x):
    """the rank-384 arm reproduces §1992's mlp2_a56 to 0.005 nats on all three roles -- every share below
    is measured against it"""
    return all(abs(x.penalty(C, r, 'a6r384') - v) < 0.005 for r, v in zip(x.roles, FULL_RANK))


def _rank_four_recovers_most_of_it(x):
    """and rank 4 at attention 6 already recovers at least half of its whole contribution, on >=2 roles --
    rank 16 recovered 72% (§1996). If TRUE the signal is small enough that a head-level account is the
    obvious next instrument; if FALSE the content is spread over many dimensions and 'which heads' is the
    wrong question"""
    return sum(1 for r in x.roles if _recovered(x, r, 'a6r4') >= 0.50) >= 2


def _attention_five_stays_free_at_rank_one(x):
    """and attention 5 is still intact at RANK 1 -- within 0.02 nats of the full-rank arm on all three
    roles. §2007 measured +0.001 at rank 16, and rank 1 is the strongest form of the presence-only claim:
    a single direction, the same for every token"""
    return all(abs(x.penalty(C, r, 'a5r1') - x.penalty(C, r, 'a6r384')) < 0.02 for r in x.roles)


B.run(
    name='the_bottom_of_attention_sixs_curve',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_rank_reproduces',
         'the rank-384 arm reproduces §1992\'s mlp2_a56 to 0.005 nats on 3/3 roles', _full_rank_reproduces),
        ('pred_b_rank_four_recovers_most_of_it',
         'and rank 4 at attention 6 recovers at least half its whole contribution (>=2 roles)',
         _rank_four_recovers_most_of_it),
        ('pred_c_attention_five_stays_free_at_rank_one',
         'and attention 5 is intact at RANK 1, within 0.02 nats, on 3/3 roles',
         _attention_five_stays_free_at_rank_one),
    ],
    refs=[(NO6, B.PT + 'ops/where_the_threshold_gap_lives_results.json', 'mlp2_a5', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('a6r1', 'a6r384'), ('a6r4', 'a6r384'), ('a5r1', 'a6r384')],
)
