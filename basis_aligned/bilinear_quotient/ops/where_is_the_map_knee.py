# LOCATE THE MAP KNEE, WHICH IS BRACKETED BETWEEN LAYERS 5 AND 9.
#
# §2023 found the map over-bought at the shallow MLP sites: cutting it from rank 640 to 256 at layers 0-5
# costs 0.24x what the released parameters are worth, while the increment out to layer 9 costs 1.71x. The
# knee is bracketed by a step that averages four sites.
#
# §2015's content profile is an order of magnitude apart inside that bracket -- mlp6 at 0.00012 and mlp8 at
# 0.00138 -- so the knee may sit within it. Layers 0-7 halves the bracket, and layers 0-4 and 0-6 make the
# ladder two sites at a time, which is the resolution §2021 used on the table-rank axis.
#
# Each site's map at rank 256 instead of 640 releases 0.885M values, worth 0.00089 nats per ten sites at
# §1947's price -- so a two-site step releases 1.77M, worth 0.000177 nats, and that is the bar each step
# is scored against.
#
# ARMS. §2023's build (map 256 at MLP 0-5); the same cut extended to layers 0-4, 0-6, 0-7 and 0-9; a
# fallback variant of the shipped build for the inert half of the control; and one differing-table-rank
# arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2023's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
BEST = {('mlp', L): 1152 for L in range(10, 18)}        # §2020's table ranks
RICH, POOR = 'mix30m640', 'mix30m256'
SHIPPED = 'shipped'
STEP_WORTH = 0.000177          # §1947: two sites' map cut releases 1.77M values

# §2023, pooled over all three roles, against the shipped build
CUT6 = {LO: 0.003175, HI: 0.007575}


def _mixed(n):
    """rank-256 map at MLP layers 0..n-1, rank-640 everywhere else"""
    poor = ','.join(f'mlp{L}' for L in range(n))
    rich = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < n))
    return f'{POOR}@{poor}+{RICH}@{rich}'


PLAN = [(RICH, BASE, SHIPPED, None)] + \
       [(_mixed(n), {**BASE, **BEST}, f'cut{n}', B.SITES) for n in (4, 6, 7, 8, 10)] + [
    ('map512', BASE, 'shipped_fb_control', None),      # all 36 sites, other fallback: the INERT pair
    (RICH, A256, 'rank_control', None)]                # differing table rank: the other half


def _gain(x, cov, lab):
    return -x.tpool_full(cov, lab, SHIPPED)['mean']


def _the_cut6_anchor_reproduces(x):
    """§2023's layers 0-5 build rebuilds to 0.003175 pooled at 5,419 and 0.007575 at 16,110, within
    0.0002 nats -- every step below is measured against it"""
    return all(abs(_gain(x, c, 'cut6') - v) < 0.0002 for c, v in CUT6.items())


def _the_step_to_layer_7_is_still_free(x):
    """and extending the cut to layers 6-7 still releases more than it costs at 5,419: under 0.000177
    nats for two sites. §2023 measured the four-site step out to layer 9 at 1.71x, so if this half of it
    is free the knee is at layer 8, not 6"""
    return _gain(x, LO, 'cut6') - _gain(x, LO, 'cut8') < STEP_WORTH


def _the_step_to_layer_9_is_not(x):
    """but the next two sites, layers 8-9, cost more than their 0.000177 -- that is where §2023's 1.71x
    would then be concentrated. If both steps are free the bracket was mis-read and the cut should run to
    layer 9 after all"""
    return _gain(x, LO, 'cut8') - _gain(x, LO, 'cut10') > STEP_WORTH


def _the_knee_is_in_the_same_place_at_both_coverages(x):
    """and the same step is the first to fail at 16,110 -- §2023 measured every shallow cut as CHEAPER at
    the higher coverage, so the knee should sit at the same layer or later, never earlier"""
    def first_fail(cov):
        prev = 'cut4'
        for lab in ('cut6', 'cut7', 'cut8', 'cut10'):
            n = int(lab[3:]) - int(prev[3:])                    # sites added by this step
            per_two = (_gain(x, cov, prev) - _gain(x, cov, lab)) * 2.0 / n
            if per_two > STEP_WORTH:
                return lab
            prev = lab
        return 'none'

    return first_fail(LO) == first_fail(HI)


B.run(
    name='where_is_the_map_knee',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_cut6_anchor_reproduces',
         '§2023\'s layers 0-5 build rebuilds to 0.003175 / 0.007575 pooled within 0.0002 nats',
         _the_cut6_anchor_reproduces),
        ('pred_b_the_step_to_layer_7_is_still_free',
         'and extending the cut to layers 6-7 costs under its 0.000177 worth at 5,419',
         _the_step_to_layer_7_is_still_free),
        ('pred_c_the_step_to_layer_9_is_not',
         'but layers 8-9 cost more than theirs', _the_step_to_layer_9_is_not),
        ('pred_d_the_knee_is_in_the_same_place_at_both_coverages',
         'and the first failing step is the same layer at 16,110',
         _the_knee_is_in_the_same_place_at_both_coverages),
    ],
    refs=[(SHIPPED, B.PT + 'ops/the_minimal_path_results.json', 'full_program', LO, 0.0005)],
    paired_pairs=[('cut6', SHIPPED), ('cut8', SHIPPED), ('cut10', SHIPPED), ('cut4', SHIPPED)],
)
