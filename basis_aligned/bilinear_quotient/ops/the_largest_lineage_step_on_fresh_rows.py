# DOES THE LARGEST AT-RISK STEP OF THE BUILD LINEAGE SURVIVE FRESH ROWS?
#
# §2049 found 53 of 111 certified entries in the selection-noise regime, dominated by the §1870-§1959
# chain that built the build of record -- margins of 0.00001 to 0.0244 nats. The chain's NET effect is
# validated: §2037 measured §1959's build beating §1789's deployed design by +127.889 milli-nats on fresh
# rows, and §2044 confirmed +98.768 at the other coverage. But no individual STEP has been tested.
#
# The largest at-risk step is §1941's: that the nn75m512 fallback dominates the deployed design, at 0.0244
# nats. It is expressible in the current arm grammar (nn<P>m<R>: route the top P% of uncovered types by
# cosine to the output-NN neighbour, the rest to a rank-R map) and it is one arm.
#
# Registered expectation: it holds. §2037's +127.889 is the sum of this chain and it transports with room
# to spare, so a 24-milli-nat step inside it reversing would be surprising -- but that is exactly the
# reasoning §2013-§2035 used about its own steps, and it was wrong there.
#
# ARMS. §1789's deployed design; §1941's nn75m512 fallback; §1959's build as the endpoint. A fallback
# variant for the inert half of the control, and one differing-table-rank arm for the other half.
#
# ROLES. 'fresh' ONLY, both coverages. DISCOVERY ONLY. Rung 3 -- §2049's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
DEPLOYED, S1941, S1959 = 'deployed_1789', 'nn75m512', 'build_1959'
FRESH = ('fresh',)

S1941_IN_SAMPLE = 0.0244          # §1941: nn75m512 over the deployed design, nats
S1959_FRESH_LO = 0.127889         # §2037: §1959's build over the deployed design, fresh, 5,419

PLAN = [('map64', None, DEPLOYED, None),
        ('nn75m512', BASE, S1941, None),
        ('mix30m640', BASE, S1959, None),
        ('map512', BASE, 'shipped_fb_control', None),                       # all 36 sites: INERT pair
        ('mix30m640', {'mlp': 512, 'attn': 384}, 'rank_control', None)]     # differing table rank


def _beats(x, cov, a, b):
    if a == b:
        return 0.0
    return -x.tpool_full(cov, a, b)['mean']


def _the_endpoint_reproduces(x):
    """§2037's +127.889 milli-nat endpoint rebuilds within 2 milli-nats at 5,419. The step below is
    measured against the same baseline on the same rows and needs that anchor"""
    return abs(_beats(x, LO, S1959, DEPLOYED) - S1959_FRESH_LO) < 0.002


def _the_step_still_dominates(x):
    """and §1941's nn75m512 still beats the deployed design on fresh rows, at both coverages. This is the
    largest at-risk step of the lineage; if it reverses while the chain's net holds, the ledger's
    per-section claims must be read differently from its build claims"""
    return all(_beats(x, c, S1941, DEPLOYED) > 0 for c in (LO, HI))


def _the_step_is_the_same_order(x):
    """and its margin is within a factor of five of §1941's 0.0244 nats at 5,419 -- between 0.00488 and
    0.122. §2044 showed the fresh window can amplify an effect 2.8x, so a wide band is the honest one"""
    m = _beats(x, LO, S1941, DEPLOYED)
    return 0.2 * S1941_IN_SAMPLE < m < 5.0 * S1941_IN_SAMPLE


def _the_endpoint_still_beats_the_step(x):
    """and §1959's build still beats §1941's fallback at both coverages -- the chain kept improving after
    this step, and if it did not, the later supersessions were selection even though the endpoint
    transports"""
    return all(_beats(x, c, S1959, S1941) > 0 for c in (LO, HI))


B.run(
    name='the_largest_lineage_step_on_fresh_rows',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    roles=FRESH,
    predicates=[
        ('pred_a_the_endpoint_reproduces',
         '§2037\'s +127.889 milli-nat endpoint rebuilds within 2 milli-nats at 5,419',
         _the_endpoint_reproduces),
        ('pred_b_the_step_still_dominates',
         'and §1941\'s nn75m512 still beats the deployed design at both coverages',
         _the_step_still_dominates),
        ('pred_c_the_step_is_the_same_order',
         'and its margin is within a factor of five of §1941\'s 0.0244 nats', _the_step_is_the_same_order),
        ('pred_d_the_endpoint_still_beats_the_step',
         'and §1959\'s build still beats §1941\'s fallback at both coverages',
         _the_endpoint_still_beats_the_step),
    ],
    paired_pairs=[(S1941, DEPLOYED), (S1959, DEPLOYED), (S1959, S1941)],
)
