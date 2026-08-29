# THE TWO MLPs BELOW THE BOUNDARY THAT HAVE NEVER BEEN MEASURED.
#
# §2001 gave the threshold a domain: compiling attention 5 and 6 costs a flat ~0.65 nats and buys back an
# amount that grows with the compiled MLP's lone damage. It loses below ~2.0 (mlp5 −0.24, mlp6 −0.66,
# mlp7 −0.64) and pays above ~4.8 (mlp2 +2.84, mlp3 +4.58). The crossing is bracketed and not located,
# because no measured MLP has a lone damage between 2.02 and 4.81.
#
# §1988's depth profile runs mlp2 4.81 · mlp3 6.57 · mlp4 10.67 · mlp5 2.02 · mlp6 0.20 · mlp7 0.07, and
# mlp0 and mlp1 have never been measured at all. They are the only remaining MLPs below the boundary, they
# extend the profile downward, and either could land in the gap.
#
# ARMS. mlp0 and mlp1 alone and each with attention 5,6; mlp2 and mlp5 alone and fixed as the §2001
# anchors; the full 36-site program with a fallback variant for the inert half of the control; and one
# differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2001's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
NEW, ANCHORS = (0, 1), (2, 5)
PLATEAU = 0.65      # §2001: the flat standalone cost of compiling attention 5 and 6, 0.62-0.70 nats


def _lone(L):
    return [('mlp', L)]


def _fixed(L):
    return [('mlp', L), ('attn', 5), ('attn', 6)]


PLAN = [(ARM, BASE, f'm{L}', _lone(L)) for L in NEW + ANCHORS] + \
       [(ARM, BASE, f'm{L}_fix', _fixed(L)) for L in NEW + ANCHORS] + [
    (ARM, BASE, 'full_program', None),                 # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),         # all 36 sites, other fallback: the INERT pair
    (ARM, A256, 'rank_control', None)]                 # differing table rank: the other half


def _gain(x, role, L):
    return x.penalty(C, role, f'm{L}') - x.penalty(C, role, f'm{L}_fix')


def _anchors_reproduce(x):
    """mlp2 and mlp5 rebuild to §2001's gains within 0.005 nats on all three roles -- +2.842/+3.200/+3.006
    and -0.241/-0.239/-0.194. If they do not, the two new sites cannot be placed on that curve"""
    want = {2: (2.842, 3.200, 3.006), 5: (-0.241, -0.239, -0.194)}
    return all(abs(_gain(x, r, L) - v) < 0.005
               for L, vs in want.items() for r, v in zip(x.roles, vs))


def _both_are_damaged_below_the_boundary(x):
    """and both new sites are genuinely damaged by lone compilation -- over 1 nat on >=2 roles each. Every
    MLP below attention 6 measured so far costs at least 2.02, and layers 0-1 are the furthest from it.
    If FALSE the damage does not extend to the bottom of the model and §1988's profile has another end"""
    return all(sum(1 for r in x.roles if x.penalty(C, r, f'm{L}') > 1.0) >= 2 for L in NEW)


def _the_gain_curve_predicts_them(x):
    """and §2001's account places them: for each new site the fix's gain exceeds -PLATEAU, i.e. the loss
    never exceeds the flat 0.65-nat standalone cost of attention 5 and 6, on all three roles at both
    sites. That is the one quantitative commitment §2001 makes about sites it did not measure"""
    return all(_gain(x, r, L) > -PLATEAU for r in x.roles for L in NEW)


B.run(
    name='the_last_two_sites_below',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_anchors_reproduce',
         'mlp2 and mlp5 rebuild to §2001\'s gains within 0.005 nats on 3/3 roles', _anchors_reproduce),
        ('pred_b_both_are_damaged_below_the_boundary',
         'and lone mlp0 and mlp1 each cost more than 1 nat (>=2 roles)',
         _both_are_damaged_below_the_boundary),
        ('pred_c_the_gain_curve_predicts_them',
         'and neither loses more than §2001\'s flat 0.65-nat plateau, on 3/3 roles',
         _the_gain_curve_predicts_them),
    ],
    refs=[('m2', B.PT + 'ops/where_the_fix_stops_paying_results.json', 'm2', C, 0.0005),
          ('m5_fix', B.PT + 'ops/where_the_fix_stops_paying_results.json', 'm5_fix', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('m0_fix', 'm0'), ('m1_fix', 'm1'), ('m0', 'm1')],
)
