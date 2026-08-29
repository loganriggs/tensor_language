# IS THE SHIPPED RANK 768 LEAVING CONTENT ON THE TABLE AT THE TWO SITES THAT MATTER?
#
# §2016 showed every truncation BELOW 768 loses money at mlp16 and mlp17, by 5x to 24x §1947's price. Every
# rank this line has tested is below 768, and 768 is itself a truncation: at 5,419 coverage a site's table
# is 5,419 x 1152, so its untruncated rank is 1152. §1959 chose 768 by sweeping the rank UNIFORMLY across
# all eighteen MLP sites, where the marginal buyer is the average site -- and §2015 showed the average site
# carries almost nothing while mlp16 and mlp17 carry 96% of the content.
#
# So the uniform sweep could not see whether these two sites want more rank than the average one. Going
# 768 -> 1152 at one site adds 384 x 6571 = 2.52M values, worth 0.00025 nats at §1947's 0.010-per-100M
# price. If either site buys more than that, the shipped allocation is under-bought where it matters and
# the fix is cheap and local.
#
# ARMS. the shipped 36-site program; mlp16 at rank 1152, mlp17 at rank 1152, and both; a fallback variant
# of the full program for the inert half of the control; and one differing-table-rank arm for the other
# half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2016's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
FULL = 'full_program'
UNTRUNCATED = 1152              # min(NCOV, D) at 5,419 coverage: the table's own rank
PRICE = 0.00025                 # §1947: 768 -> 1152 adds 2.52M values per site

FULL_PROGRAM = (2.80750, 2.97891, 2.70217)      # §2013 / §2014 / §2015 / §2016


def _gain(x, role, lab):
    """nats bought by raising the rank: positive means the shipped 768 was leaving content on the table"""
    return x.penalty(C, role, FULL) - x.penalty(C, role, lab)


PLAN = [(ARM, BASE, FULL, None),
        (ARM, {**BASE, ('mlp', 16): UNTRUNCATED}, 'm16_full', None),
        (ARM, {**BASE, ('mlp', 17): UNTRUNCATED}, 'm17_full', None),
        (ARM, {**BASE, ('mlp', 16): UNTRUNCATED, ('mlp', 17): UNTRUNCATED}, 'both_full', None),
        ('map512', BASE, 'full_fb_control', None),      # all 36 sites, other fallback: the INERT pair
        (ARM, A256, 'rank_control', None)]              # differing table rank: the other half


def _full_program_reproduces(x):
    """the shipped program rebuilds to 2.80750 / 2.97891 / 2.70217 within 0.0005 nats on all three roles --
    every gain below is a difference of a few thousandths and needs the anchor"""
    return all(abs(x.penalty(C, r, FULL) - v) < 0.0005 for r, v in zip(x.roles, FULL_PROGRAM))


def _raising_the_rank_buys_something(x):
    """and untruncating either site buys strictly positive CE on all three roles -- if it does not, rank
    768 already captures everything these tables have and the axis is closed above as §2016 closed it
    below"""
    return all(_gain(x, r, lab) > 0 for r in x.roles for lab in ('m16_full', 'm17_full'))


def _it_buys_more_than_it_costs(x):
    """and at at least one of the two sites it buys more than §1947's 0.00025 price, on >=2 roles. If TRUE
    the shipped allocation is UNDER-bought where the program lives and the fix is one site's rank; if
    FALSE 768 is the right purchase in both directions and §1959's uniform choice happens to be correct at
    the sites its sweep could not see"""
    return any(sum(1 for r in x.roles if _gain(x, r, lab) > PRICE) >= 2
               for lab in ('m16_full', 'm17_full'))


def _the_two_sites_compound_again(x):
    """and raising both buys MORE than the sum of raising each, on >=2 roles -- §2014 measured the shipped
    program super-additive in loss by 22% and §2016 by 24%, both for DEGRADATIONS. The same nonlinearity
    should make improvements compound too, and it has never been tested in that direction"""
    return sum(1 for r in x.roles
               if _gain(x, r, 'both_full') > _gain(x, r, 'm16_full') + _gain(x, r, 'm17_full')) >= 2


B.run(
    name='is_768_leaving_content_on_the_table',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_program_reproduces',
         'the shipped program rebuilds to 2.80750/2.97891/2.70217 within 0.0005 nats on 3/3 roles',
         _full_program_reproduces),
        ('pred_b_raising_the_rank_buys_something',
         'and untruncating mlp16 or mlp17 buys strictly positive CE, on 3/3 roles',
         _raising_the_rank_buys_something),
        ('pred_c_it_buys_more_than_it_costs',
         'and at one of them it buys more than §1947\'s 0.00025 price (>=2 roles)',
         _it_buys_more_than_it_costs),
        ('pred_d_the_two_sites_compound_again',
         'and raising both buys more than the sum of raising each -- super-additivity in the other direction',
         _the_two_sites_compound_again),
    ],
    refs=[(FULL, B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('m16_full', FULL), ('m17_full', FULL), ('both_full', FULL)],
)
