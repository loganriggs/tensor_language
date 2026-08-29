# THE ONLY REMAINING UNTESTED ALLOCATION IN THIS FAMILY.
#
# §2018 recorded mlp17 untruncated to rank 1152 as the best-known build: +0.47 milli-nats pooled at 5,419
# and +1.34 at 16,110, against §1947's 0.25 milli-nat price for the 2.52M values it adds. mlp16 was
# measured and NOT adopted -- it clears the price on 2 of 3 roles at 5,419 (one role -0.00008) but on 3/3
# at 16,110, so it is a coverage-dependent purchase and the deployed coverage is 5,419.
#
# That leaves one allocation untested: BOTH sites untruncated, scored as a build. §2017 measured
# improvements as SUB-additive by 10-28% where degradations are super-additive by 22-24%, so the pair
# should buy less than the sum -- the question is whether it still buys more than the 0.50 milli-nats its
# 5.05M extra values cost.
#
# ARMS. the shipped build; §2018's candidate (mlp17 at 1152); mlp16 at 1152; both at 1152; a fallback
# variant of the shipped build for the inert half of the control; and one differing-table-rank arm for the
# other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2018's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
SHIPPED, M17, M16, BOTH = 'shipped', 'mlp17_1152', 'mlp16_1152', 'both_1152'
PRICE_ONE = 0.00025          # §1947: +2.52M values at one site
PRICE_BOTH = 0.00050         # +5.05M values at two

# §2018, skip7000 / skip11000 / skip1200
M17_LO = (0.00063, 0.00030, 0.00051)
M17_HI = (0.00164, 0.00082, 0.00178)

PLAN = [(ARM, BASE, SHIPPED, None),
        (ARM, {**BASE, ('mlp', 17): 1152}, M17, None),
        (ARM, {**BASE, ('mlp', 16): 1152}, M16, None),
        (ARM, {**BASE, ('mlp', 16): 1152, ('mlp', 17): 1152}, BOTH, None),
        ('map512', BASE, 'shipped_fb_control', None),   # all 36 sites, other fallback: the INERT pair
        (ARM, A256, 'rank_control', None)]              # differing table rank: the other half


def _gain(x, cov, role, lab):
    return x.penalty(cov, role, SHIPPED) - x.penalty(cov, role, lab)


def _the_candidate_reproduces(x):
    """§2018's mlp17 gains rebuild at BOTH coverages within 0.0002 nats on all three roles -- the pair's
    margin over it is a difference of quarter-milli-nats and needs the anchor at both"""
    return all(abs(_gain(x, c, r, M17) - v) < 0.0002
               for c, vs in ((LO, M17_LO), (HI, M17_HI)) for r, v in zip(x.roles, vs))


def _the_pair_is_subadditive(x):
    """and the pair buys LESS than the sum of the two single-site gains, at both coverages on >=2 roles
    each -- §2017 measured improvements sub-additive by 10-28% and this is that at build scale"""
    n = {c: sum(1 for r in x.roles
                if _gain(x, c, r, BOTH) < _gain(x, c, r, M17) + _gain(x, c, r, M16))
         for c in (LO, HI)}
    return n[LO] >= 2 and n[HI] >= 2


def _the_pair_beats_its_price_at_high_coverage(x):
    """and at 16,110 the pair still clears its own 0.00050 price on >=2 roles -- both sites qualify singly
    there, and if sub-additivity does not eat the margin the pair is the right build at that coverage"""
    return sum(1 for r in x.roles if _gain(x, HI, r, BOTH) > PRICE_BOTH) >= 2


def _mlp16_stays_unjustified_at_the_deployed_coverage(x):
    """and at 5,419 adding mlp16 to §2018's build does NOT pay for itself: the pair beats mlp17-alone by
    less than the 0.00025 its extra 2.52M values cost, on >=2 roles. If FALSE the deployed build should
    take both sites and §2018 under-bought"""
    return sum(1 for r in x.roles
               if _gain(x, LO, r, BOTH) - _gain(x, LO, r, M17) < PRICE_ONE) >= 2


B.run(
    name='the_pair_as_a_build',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_candidate_reproduces',
         '§2018\'s mlp17 gains rebuild at both coverages within 0.0002 nats on 3/3 roles',
         _the_candidate_reproduces),
        ('pred_b_the_pair_is_subadditive',
         'and the pair buys less than the sum of the two single-site gains, at both coverages (>=2 roles)',
         _the_pair_is_subadditive),
        ('pred_c_the_pair_beats_its_price_at_high_coverage',
         'and at 16,110 the pair still clears its own 0.00050 price (>=2 roles)',
         _the_pair_beats_its_price_at_high_coverage),
        ('pred_d_mlp16_stays_unjustified_at_the_deployed_coverage',
         'and at 5,419 adding mlp16 does not pay for itself over §2018\'s build (>=2 roles)',
         _mlp16_stays_unjustified_at_the_deployed_coverage),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[(BOTH, M17), (BOTH, SHIPPED), (M16, SHIPPED)],
)
