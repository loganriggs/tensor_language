# THE SAME QUESTION ON THE ATTENTION AXIS.
#
# §2016-§2021 found the MLP rank axis had an unexplored region above 768: untruncating layers 10-17 buys
# 3.30 milli-nats pooled at 5,419 for 20.2M values, and the marginal ladder crosses §1947's price between
# layers 10 and 8. The reason it went unfound is that §1947 and §1959 set the MLP rank by sweeping it
# UNIFORMLY across eighteen sites, and a uniform sweep cannot find a knee it averages over.
#
# §1959 set ATTENTION at 384 by exactly the same uniform sweep, over eighteen attention sites whose lone
# contributions §1989 measured as differing by 6x (attn3 0.113 to attn4 0.288). The axis above 384 has
# never been tested per-site. This is the same instrument, the same failure mode, and one arm away.
#
# Each attention site's rank-r table costs r x (NCOV + D) + 2D = r x 6571 + 2304 values, so 384 -> 1152
# adds 5.05M per site, worth 0.00050 nats at §1947's 0.010-per-100M price -- twice the MLP step's price
# per site, so it needs twice the gain.
#
# ARMS. §2020's best-known build; the same with attention 14-17, 10-17 and 0-17 untruncated to 1152; a
# fallback variant of the shipped build for the inert half of the control; and one differing-table-rank
# arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2021's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
BEST = {('mlp', L): 1152 for L in range(10, 18)}        # §2020's build
SHIPPED, B2020 = 'shipped', 'best_2020'
PRICE_PER_SITE = 0.00050                                # attention 384 -> 1152 adds 5.05M values

BEST_POOLED = {LO: 0.003300, HI: 0.007621}              # §2020 / §2021


def _attn(lo):
    return {('attn', L): 1152 for L in range(lo, 18)}


PLAN = [(ARM, BASE, SHIPPED, None),
        (ARM, {**BASE, **BEST}, B2020, None),                          # §2020: 0.003300 / 0.007621
        (ARM, {**BASE, **BEST, **_attn(14)}, 'attn14', None),          # +4 sites, 20.2M, price 0.00202
        (ARM, {**BASE, **BEST, **_attn(10)}, 'attn10', None),          # +8 sites, 40.4M, price 0.00404
        (ARM, {**BASE, **BEST, **_attn(0)}, 'attn_all', None),         # +18 sites, 90.9M, price 0.00909
        ('map512', BASE, 'shipped_fb_control', None),                  # all 36 sites: the INERT pair
        (ARM, A256, 'rank_control', None)]                             # differing table rank: other half


def _gain(x, cov, lab):
    """nats bought over the SHIPPED build, pooled across all three roles (LESSON 101)"""
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _the_best_known_reproduces(x):
    """§2020's build rebuilds to 0.003300 pooled at 5,419 and 0.007621 at 16,110 within 0.0002 nats --
    every attention increment below is measured over it"""
    return all(abs(_gain(x, c, B2020) - v) < 0.0002 for c, v in BEST_POOLED.items())


def _attention_rank_buys_something(x):
    """and untruncating the late four attention sites buys strictly positive CE over §2020's build, at
    both coverages. If it does not, attention 384 is already past its useful point and the axis is closed
    without a costing question"""
    return all(_gain(x, c, 'attn14') > _gain(x, c, B2020) for c in (LO, HI))


def _the_late_four_clear_their_price(x):
    """and those four sites clear their 0.00202 price at the deployed 5,419 coverage. The MLP analogue
    (layers 14-17) returned 2.1x. If TRUE the attention axis has the same unexplored region and the build
    moves again; if FALSE attention 384 was correctly set and the MLP knee is specific to MLPs"""
    return _gain(x, LO, 'attn14') - _gain(x, LO, B2020) > 0.00202


def _the_shallow_attention_sites_do_not(x):
    """and the ten sites below layer 8 do not pay: going from attention 10-17 to all eighteen buys under
    its 0.00505 price at 5,419. §1989 measured lone attention contributions as small and flat below layer
    6, so the shallow attention block should behave like the shallow MLP block did in §2020"""
    return _gain(x, LO, 'attn_all') - _gain(x, LO, 'attn10') < 0.00505


B.run(
    name='does_attention_want_rank_too',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_best_known_reproduces',
         '§2020\'s build rebuilds to 0.003300 / 0.007621 pooled within 0.0002 nats',
         _the_best_known_reproduces),
        ('pred_b_attention_rank_buys_something',
         'and untruncating attention 14-17 buys strictly positive CE over it, at both coverages',
         _attention_rank_buys_something),
        ('pred_c_the_late_four_clear_their_price',
         'and those four sites clear their 0.00202 price at 5,419', _the_late_four_clear_their_price),
        ('pred_d_the_shallow_attention_sites_do_not',
         'and the ten sites below layer 10 do not clear their 0.00505 price',
         _the_shallow_attention_sites_do_not),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(B2020, SHIPPED), ('attn14', SHIPPED), ('attn10', SHIPPED), ('attn_all', SHIPPED)],
)
