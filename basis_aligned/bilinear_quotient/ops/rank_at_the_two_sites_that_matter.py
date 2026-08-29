# THE RANK QUESTION AT THE TWO SITES THAT ACTUALLY CARRY THE PROGRAM.
#
# §2015 measured the shipped program's per-site table content: nothing at mlp0/2/6, thousandths at 4 and
# 8, hundredths at 10-14, and 0.354 and 0.815 nats at mlp16 and mlp17. Two of thirty-six tables carry 96%
# of the sampled total.
#
# §2013 priced rank 128 against rank 768 at mlp2 -- a site whose whole content is worth 0.0001 nats -- and
# found the truncation marginally justified under §1947's 0.010-per-100M rule. That is the least
# informative site in the program to have asked at. The same question at mlp16 and mlp17 is where a rank
# decision actually moves anything, and it has never been asked.
#
# Each MLP site's rank-r table costs r x (NCOV + D) + 2D = r x 6571 + 2304 values at 5,419 coverage, so
# 768 -> 128 saves 4.205M per site and is worth 0.00042 nats at the price rule.
#
# ARMS. the shipped 36-site program; the same with mlp16 at rank 128 / 384, mlp17 at rank 128 / 384, and
# both at 128; a fallback variant of the full program for the inert half of the control; and one
# differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2015's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
FULL = 'full_program'
PRICE_PER_SITE = 0.00042        # §1947: 4.205M values saved by 768 -> 128, at 0.010 nats per 100M

FULL_PROGRAM = (2.80750, 2.97891, 2.70217)      # §2013 / §2014 / §2015


PLAN = [(ARM, BASE, FULL, None),
        (ARM, {**BASE, ('mlp', 16): 128}, 'm16_r128', None),
        (ARM, {**BASE, ('mlp', 16): 384}, 'm16_r384', None),
        (ARM, {**BASE, ('mlp', 17): 128}, 'm17_r128', None),
        (ARM, {**BASE, ('mlp', 17): 384}, 'm17_r384', None),
        (ARM, {**BASE, ('mlp', 16): 128, ('mlp', 17): 128}, 'both_r128', None),
        ('map512', BASE, 'full_fb_control', None),      # all 36 sites, other fallback: the INERT pair
        (ARM, A256, 'rank_control', None)]              # differing table rank: the other half


def _d(x, role, lab):
    return x.penalty(C, role, lab) - x.penalty(C, role, FULL)


def _full_program_reproduces(x):
    """the shipped program rebuilds to 2.80750 / 2.97891 / 2.70217 within 0.0005 nats on all three roles"""
    return all(abs(x.penalty(C, r, FULL) - v) < 0.0005 for r, v in zip(x.roles, FULL_PROGRAM))


def _truncation_costs_more_than_it_saves_here(x):
    """and at these two sites rank 128 costs MORE than the 0.00042 nats its 4.205M saved values are worth
    -- on all three roles at both sites. §2013 found the opposite at mlp2, where the whole content is
    0.0001 nats. If FALSE the truncation is justified even where the content is largest, and §1959's
    uniform rank 768 is over-bought everywhere"""
    return all(_d(x, r, lab) > PRICE_PER_SITE
               for r in x.roles for lab in ('m16_r128', 'm17_r128'))


def _rank_384_is_the_better_buy(x):
    """and rank 384 is cheaper per value saved than rank 128 at both sites: it costs less than half what
    rank 128 costs, on >=2 roles each, while saving 2.52M against 4.21M. If TRUE the curve is convex and a
    middle rank is the right purchase; if FALSE the loss is roughly linear in rank and there is no knee"""
    return all(sum(1 for r in x.roles
                   if _d(x, r, f'm{L}_r384') < 0.5 * _d(x, r, f'm{L}_r128')) >= 2
               for L in (16, 17))


def _the_two_sites_compound(x):
    """and truncating both costs more than the sum of truncating each, on >=2 roles -- §2014 measured the
    shipped program as super-additive in loss by 22%, and this is that prediction applied to the only two
    sites where the quantity is large enough to see"""
    return sum(1 for r in x.roles
               if _d(x, r, 'both_r128') > _d(x, r, 'm16_r128') + _d(x, r, 'm17_r128')) >= 2


B.run(
    name='rank_at_the_two_sites_that_matter',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_program_reproduces',
         'the shipped program rebuilds to 2.80750/2.97891/2.70217 within 0.0005 nats on 3/3 roles',
         _full_program_reproduces),
        ('pred_b_truncation_costs_more_than_it_saves_here',
         'and rank 128 at mlp16 and mlp17 each cost more than §1947\'s 0.00042 price, on 3/3 roles',
         _truncation_costs_more_than_it_saves_here),
        ('pred_c_rank_384_is_the_better_buy',
         'and rank 384 costs under half what rank 128 does at both sites (>=2 roles each)',
         _rank_384_is_the_better_buy),
        ('pred_d_the_two_sites_compound',
         'and truncating both costs more than the sum of each -- §2014\'s super-additivity (>=2 roles)',
         _the_two_sites_compound),
    ],
    refs=[(FULL, B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('m16_r128', FULL), ('m17_r128', FULL), ('both_r128', FULL)],
)
