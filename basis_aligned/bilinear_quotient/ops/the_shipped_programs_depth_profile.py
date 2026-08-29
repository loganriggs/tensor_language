# THE SHIPPED PROGRAM'S OWN DEPTH PROFILE.
#
# §2014 found that replacing one site's table with a mean row inside the shipped 36-site program costs
# -0.0001 at mlp2 and +0.027 at mlp12 -- a hundredfold difference -- and no profile of that quantity
# exists. Every depth profile in this line (§1988, §2002) measured LONE-COMPILATION DAMAGE WITH ATTENTION
# LIVE, which §2013 showed is a different regime: there the interface dominates, here nothing downstream
# reads anything and each site contributes only its own residual term.
#
# This is the shipped program's profile, and unlike everything since §1983 it is directly about the
# artifact we would deploy: a site whose table content is worth nothing is a site whose rank is being
# bought for nothing.
#
# ARMS. the shipped 36-site program, and the same with a MEAN ROW at mlp0, 2, 4, 6, 8, 10, 12, 14, 16 and
# 17 in turn; a fallback variant of the full program for the inert half of the control; and one differing-
# table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2014's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
FULL = 'full_program'
LAYERS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 17)
SHALLOW, DEEP = (0, 2, 4), (14, 16, 17)

# §2014 at 5,419, skip7000 / skip11000 / skip1200
ANCHORS = {2: (-0.00012, 0.00002, -0.00028), 12: (0.02692, 0.02912, 0.02687)}


def _mean_arm(L):
    others = ','.join(f'{k}M' .replace('M', str(m)) for k in ('mlp', 'attn') for m in range(18)
                      if not (k == 'mlp' and m == L))
    return f'meanrow@mlp{L}+{ARM}@{others}'


PLAN = [(ARM, BASE, FULL, None)] + \
       [(_mean_arm(L), BASE, f'm{L}', B.SITES) for L in LAYERS] + [
    ('map512', BASE, 'full_fb_control', None),          # all 36 sites, other fallback: the INERT pair
    (ARM, A256, 'rank_control', None)]                  # differing table rank: the other half


def _d(x, role, L):
    return x.penalty(C, role, f'm{L}') - x.penalty(C, role, FULL)


def _anchors_reproduce(x):
    """§2014's mlp2 and mlp12 figures rebuild within 0.0005 nats on all three roles -- the profile is a
    set of differences of a few thousandths and needs both ends anchored"""
    return all(abs(_d(x, r, L) - v) < 0.0005
               for L, vs in ANCHORS.items() for r, v in zip(x.roles, vs))


def _deep_sites_matter_more(x):
    """and the profile rises with depth: every one of mlp14, 16, 17 costs more than every one of mlp0, 2,
    4, on >=2 roles. §2014 saw mlp12 at a hundred times mlp2 and this asks whether that is a trend"""
    return sum(1 for r in x.roles
               if min(_d(x, r, L) for L in DEEP) > max(_d(x, r, L) for L in SHALLOW)) >= 2


def _the_shallow_half_is_worth_nothing(x):
    """and every one of mlp0, 2, 4 is worth under 0.002 nats on all three roles -- these are sites whose
    rank-768 tables the build pays for and whose content a constant row replaces for free. If TRUE the
    per-site allocation §2013 raised has a concrete target; if FALSE the shallow sites do carry content
    and only mlp2 was unusual"""
    return all(abs(_d(x, r, L)) < 0.002 for r in x.roles for L in SHALLOW)


B.run(
    name='the_shipped_programs_depth_profile',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_anchors_reproduce',
         '§2014\'s mlp2 and mlp12 figures rebuild within 0.0005 nats on 3/3 roles', _anchors_reproduce),
        ('pred_b_deep_sites_matter_more',
         'and every deep site (14/16/17) costs more than every shallow one (0/2/4), on >=2 roles',
         _deep_sites_matter_more),
        ('pred_c_the_shallow_half_is_worth_nothing',
         'and mlp0, 2 and 4 are each worth under 0.002 nats, on 3/3 roles',
         _the_shallow_half_is_worth_nothing),
    ],
    refs=[(FULL, B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('m0', FULL), ('m12', FULL), ('m17', FULL)],
)
