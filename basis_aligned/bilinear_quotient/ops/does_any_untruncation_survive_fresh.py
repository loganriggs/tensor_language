# DOES ANY DEPTH OF UNTRUNCATION SURVIVE FRESH ROWS?
#
# §2038 showed §2020's table raise -- MLP layers 10-17 untruncated to rank 1152 -- costs -11.578 milli-nats
# on 98,304 fresh positions against +3.300 in-sample, accounting for 98.4% of the retracted build's
# deficit. That was measured at ONE depth, the eight-site one the arc converged on.
#
# §2021 built a marginal ladder in-sample: 16,17 then 14,15 then 12,13 then 10,11 each cleared §1947's
# price, and 8,9 did not. The whole ladder was scored on the three selecting roles. If the harm grows with
# depth, some shallow untruncation may still be neutral or positive on fresh rows; if every depth is
# harmful, the axis is not a matter of degree and rank 768 is simply correct.
#
# This runs §2021's ladder on the fresh window. It is the cheapest way to know whether §2038 falsified a
# CHOICE OF DEPTH or an entire axis.
#
# ARMS. §1959's build; untruncation from layer 16, 14, 12 and 10 upward. A fallback variant of the shipped
# build for the inert half of the control, and one differing-table-rank arm for the other half.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2038's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
S1959 = 'build_1959'
FRESH = ('fresh',)
FROM10_FRESH = -0.011578        # §2038, nats: layers 10-17 untruncated, fresh window

# §2021 in-sample, pooled over the three selecting roles, nats over §1959's build
IN_SAMPLE = {'from16': 0.000962, 'from14': 0.002132, 'from12': 0.002768, 'from10': 0.003300}


def _from(lo):
    return {('mlp', L): 1152 for L in range(lo, 18)}


PLAN = [('mix30m640', BASE, S1959, None)] + \
       [('mix30m640', {**BASE, **_from(n)}, f'from{n}', None) for n in (16, 14, 12, 10)] + [
    ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites: the INERT pair
    ('mix30m640', A256, 'rank_control', None)]         # differing table rank: the other half


def _beats(x, lab):
    """nats by which `lab` beats §1959's build on the fresh window"""
    return -x.tpool_full(LO, lab, S1959)['mean']


def _the_2038_anchor_reproduces(x):
    """the eight-site arm rebuilds to §2038's -11.578 milli-nats within 0.5. Every shallower depth is
    measured on the same rows and needs that anchor"""
    return abs(_beats(x, 'from10') - FROM10_FRESH) < 0.0005


def _every_depth_is_harmful_fresh(x):
    """and EVERY depth is negative on fresh rows, including the two-site one that gained +0.962
    milli-nats in-sample. If TRUE the axis itself does not travel and rank 768 is simply correct; if
    FALSE some shallow untruncation survives and §2038 falsified a choice of depth rather than an axis"""
    return all(_beats(x, f'from{n}') < 0 for n in (16, 14, 12, 10))


def _the_harm_grows_with_depth(x):
    """and the harm deepens monotonically with depth -- from16 shallower than from14 than from12 than
    from10. In-sample the gain grew the same way (+0.962, +2.132, +2.768, +3.300), so a monotone
    reversal means every site contributes harm rather than one site dominating"""
    return (_beats(x, 'from16') > _beats(x, 'from14') > _beats(x, 'from12')
            > _beats(x, 'from10'))


def _the_reversal_is_not_a_small_shift(x):
    """and the two-site arm alone reverses by more than its whole in-sample gain: its fresh figure is
    below -0.000962 nats, the +0.962 milli-nats §2021 measured. A margin that merely shrank to zero would
    be consistent with noise; one that reverses past its own size is not"""
    return _beats(x, 'from16') < -IN_SAMPLE['from16']


B.run(
    name='does_any_untruncation_survive_fresh',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_2038_anchor_reproduces',
         '§2038\'s -11.578 milli-nat eight-site figure rebuilds within 0.5 milli-nats',
         _the_2038_anchor_reproduces),
        ('pred_b_every_depth_is_harmful_fresh',
         'and every depth of untruncation is negative on fresh rows, including the two-site one',
         _every_depth_is_harmful_fresh),
        ('pred_c_the_harm_grows_with_depth',
         'and the harm deepens monotonically with depth, as the in-sample gain did',
         _the_harm_grows_with_depth),
        ('pred_d_the_reversal_is_not_a_small_shift',
         'and the two-site arm reverses past its own +0.962 milli-nat in-sample gain',
         _the_reversal_is_not_a_small_shift),
    ],
    paired_pairs=[('from16', S1959), ('from14', S1959), ('from12', S1959), ('from10', S1959)],
)
