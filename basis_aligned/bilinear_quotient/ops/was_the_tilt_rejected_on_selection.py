# WAS THE PER-TOKEN TILT REJECTED ON EVIDENCE THAT TRANSFERS?
#
# §2046's audit found thirteen in-sample selections still standing in the registry, and the largest
# cluster is the tilt-and-blend family: §1964 and §1967 rejected the per-token alpha tilt on margins of
# 1.92 to 2.94 milli-nats, and §1970 settled the blend at finer resolution on 1.6. All three decided a
# parameter of the build of record, and all three sit in the 1-3 milli-nat regime §2037 showed does not
# transfer between row sets.
#
# The shipped build uses a FLAT alpha as a result. If the fresh window prefers a tilt, that decision was
# selection and the build has a free parameter set the wrong way. If it also rejects the tilt, the
# decision is validated on rows that did not make it -- and §1964/§1967 become the first small-margin
# decisions in the tilt family confirmed out-of-sample.
#
# §2043 is the precedent for the third outcome: the two row sets may disagree with significance in both
# directions, in which case the tilt is unidentifiable and flat stands as incumbent, not as demonstrated.
#
# ARMS. §1967's own grid on §1959's build -- flat alpha 0.30, and tilts 28->32, 25->35, 22->38, 20->40,
# 15->45, 10->50. A fallback variant for the inert half of the control, and one differing-table-rank arm
# for the other half.
#
# ROLES. 'fresh' ONLY. DISCOVERY ONLY. Rung 3 -- §2046's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
FLAT = 'flat'
FRESH = ('fresh',)

# §1967's grid, narrowest to widest; the tilt span in points of alpha
GRID = [('flat', 'mix30m640'), ('t28_32', 'pat28_32m640'), ('t25_35', 'pat25_35m640'),
        ('t22_38', 'pat22_38m640'), ('t20_40', 'pat20_40m640'), ('t15_45', 'pat15_45m640'),
        ('t10_50', 'pat10_50m640')]
LAB = [g[0] for g in GRID]
TILTS = [l for l in LAB if l != FLAT]

# §1964/§1967 on the three published roles: the tilt cost 1.92 to 2.94 milli-nats and was rejected
IN_SAMPLE_TILT_COST = 0.00192

PLAN = [(arm, BASE, lab, None) for lab, arm in GRID] + [
    ('map512', BASE, 'shipped_fb_control', None),                       # all 36 sites: the INERT pair
    ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _beats(x, a, b):
    """nats by which arm a beats arm b on the fresh window"""
    if a == b:
        return 0.0
    return -x.tpool_full(LO, a, b)['mean']


def _best_tilt(x):
    return max(TILTS, key=lambda l: _beats(x, l, FLAT))


def _flat_still_wins(x):
    """flat alpha still beats every tilt on the fresh window. §1964 and §1967 rejected the tilt on the
    three published roles at 1.92 to 2.94 milli-nats; if flat wins here too, that rejection is confirmed
    on rows that did not make it. If FALSE the build's blend shape was chosen by selection"""
    return all(_beats(x, FLAT, l) > 0 for l in TILTS)


def _the_margin_is_the_same_order(x):
    """and the best tilt's deficit is within a factor of five of the 1.92 milli-nats §1967 measured --
    between 0.384 and 9.6. A rejection that transports should transport in size as well as sign, and
    §2044 showed the fresh window can amplify an effect several-fold"""
    m = -_beats(x, _best_tilt(x), FLAT)
    return 0.2 * IN_SAMPLE_TILT_COST < m < 5.0 * IN_SAMPLE_TILT_COST


def _the_widest_tilt_is_worst(x):
    """and the widest tilt, 10->50, is the worst of the six here as it was in-sample -- §1967's curve rose
    monotonically with tilt span. If the ordering scrambles, the axis behaves on fresh rows the way §2039
    found the untruncation depths behaving, and no tilt conclusion transports"""
    return all(_beats(x, 't10_50', l) < 0 for l in TILTS if l != 't10_50')


def _no_tilt_is_within_noise_of_flat(x):
    """and no tilt comes within a third of a milli-nat of flat -- if one did, the axis would be flat in
    the §2043 sense and the rejection would be a choice between indistinguishable options rather than a
    finding. Registered because that is the outcome §2043 found for the blend VALUE and it would be the
    honest answer here too"""
    return all(abs(_beats(x, l, FLAT)) > 0.00033 for l in TILTS)


B.run(
    name='was_the_tilt_rejected_on_selection',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419)],
    roles=FRESH,
    predicates=[
        ('pred_a_flat_still_wins',
         'flat alpha still beats every tilt on the fresh window', _flat_still_wins),
        ('pred_b_the_margin_is_the_same_order',
         'and the best tilt\'s deficit is within a factor of five of §1967\'s 1.92 milli-nats',
         _the_margin_is_the_same_order),
        ('pred_c_the_widest_tilt_is_worst',
         'and the widest tilt 10->50 is the worst of the six, as it was in-sample',
         _the_widest_tilt_is_worst),
        ('pred_d_no_tilt_is_within_noise_of_flat',
         'and no tilt comes within a third of a milli-nat of flat', _no_tilt_is_within_noise_of_flat),
    ],
    paired_pairs=[(l, FLAT) for l in TILTS],
)
