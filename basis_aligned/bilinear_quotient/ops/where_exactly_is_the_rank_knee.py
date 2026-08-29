# LOCATE THE RANK KNEE, AND ASK WHY IT IS SHARP WHERE THE CONTENT PROFILE IS SMOOTH.
#
# §2020 found untruncating MLP layers 10-17 buys 3.30 milli-nats pooled at 5,419 for 20.2M values (1.63x
# §1947's price) while adding layers 0-9 on top buys 0.124 for 25.2M more (0.05x). The knee is bracketed
# between layers 9 and 10 by a single arm.
#
# §2015's content profile gives no reason for a sharp knee there: replacing a site's table with a mean row
# costs 0.00138 at mlp8, 0.01065 at mlp10, 0.02692 at mlp12, 0.01258 at mlp14 -- a smooth rise with no
# feature at 9/10. A knee that is sharp where the content profile is smooth is a fact about what EXTRA
# RANK buys, not about what the tables carry, and those are different quantities.
#
# ARMS. the shipped build; layers 14-17, 12-17, 10-17 and 8-17 untruncated to rank 1152; a fallback
# variant of the shipped build for the inert half of the control; and one differing-table-rank arm for the
# other half. Each step adds exactly two sites and 5.05M values, worth 0.00050 nats at §1947's price, so
# every increment is scored against the same bar.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2020's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
SHIPPED = 'shipped'
STEP_PRICE = 0.00050          # §1947: two extra sites, 5.05M values
BEST_POOLED = {LO: 0.003300, HI: 0.007621}      # §2020, layers 10-17


def _from(lo):
    return {('mlp', L): 1152 for L in range(lo, 18)}


PLAN = [(ARM, BASE, SHIPPED, None),
        (ARM, {**BASE, **_from(14)}, 'from14', None),
        (ARM, {**BASE, **_from(12)}, 'from12', None),
        (ARM, {**BASE, **_from(10)}, 'from10', None),      # §2020: 0.003300 / 0.007621
        (ARM, {**BASE, **_from(8)}, 'from8', None),
        ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites, other fallback: the INERT pair
        (ARM, A256, 'rank_control', None)]                 # differing table rank: the other half


def _gain(x, cov, lab):
    """nats bought over the shipped build, pooled across all three roles (LESSON 101)"""
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _the_knee_arm_reproduces(x):
    """§2020's layers 10-17 build rebuilds to 0.003300 pooled at 5,419 and 0.007621 at 16,110, within
    0.0002 nats -- every increment below is measured against it"""
    return all(abs(_gain(x, c, 'from10') - v) < 0.0002 for c, v in BEST_POOLED.items())


def _sites_8_and_9_do_not_pay(x):
    """and adding layers 8 and 9 on top of 10-17 buys under their 0.00050 price at the deployed 5,419
    coverage -- §2020 measured the whole shallow block at 0.05x and this asks whether the two sites
    immediately below the knee are any different"""
    return _gain(x, LO, 'from8') - _gain(x, LO, 'from10') < STEP_PRICE


def _sites_10_and_11_do_pay(x):
    """and layers 10 and 11 DO clear the same 0.00050 bar, at both coverages -- if they do not, the knee
    is above layer 10 and §2020's build is buying two sites it should not"""
    return all(_gain(x, c, 'from10') - _gain(x, c, 'from12') > STEP_PRICE for c in (LO, HI))


def _the_knee_is_a_step_not_a_slope(x):
    """and the step across it is sharp: layers 10-11 buy more than three times what layers 8-9 buy, at
    5,419. §2015's content profile rises smoothly through this range (0.00138 at mlp8, 0.01065 at mlp10),
    so a threefold step in what EXTRA RANK buys would be a feature the content profile does not have"""
    below = _gain(x, LO, 'from8') - _gain(x, LO, 'from10')
    at = _gain(x, LO, 'from10') - _gain(x, LO, 'from12')
    return at > 3.0 * max(below, 1e-6)


B.run(
    name='where_exactly_is_the_rank_knee',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_knee_arm_reproduces',
         '§2020\'s layers 10-17 build rebuilds to 0.003300 / 0.007621 pooled within 0.0002 nats',
         _the_knee_arm_reproduces),
        ('pred_b_sites_8_and_9_do_not_pay',
         'and adding layers 8 and 9 buys under their 0.00050 price at 5,419', _sites_8_and_9_do_not_pay),
        ('pred_c_sites_10_and_11_do_pay',
         'and layers 10 and 11 clear the same 0.00050 bar, at both coverages', _sites_10_and_11_do_pay),
        ('pred_d_the_knee_is_a_step_not_a_slope',
         'and layers 10-11 buy more than three times what layers 8-9 buy, at 5,419',
         _the_knee_is_a_step_not_a_slope),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[('from10', SHIPPED), ('from12', SHIPPED), ('from8', SHIPPED), ('from14', SHIPPED)],
)
