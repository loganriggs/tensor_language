# THE CEILING AND THE LEVER AT THE OTHER COVERAGE.
#
# §2004–§2007 built a quantitative account at 5,419 types: a damage ceiling near 10.7 nats that one
# compiled MLP already reaches, attention 5 as the only site that moves it (2.65 nats, attention 6 live),
# attention 4 worth exactly nothing beneath it, and attention 5 presence-only in every role.
#
# §1993 replicated the threshold at 16,110 and found it unmoved. The ceiling has never been checked there.
# §1963 and §1965 are the standing reminders that the second coverage has reversed 5,419 claims twice, and
# it halves the uncovered arm, so any part of this account that is really about §1870's map rather than
# about the computation should move.
#
# ARMS. mlp4 alone (the ceiling); mlp2+mlp3+mlp4 (three sites, also at it); mlp4 + attention 4 (worth
# nothing); mlp4 + attention 5 (the lever); mlp4 + attention 5 with a mean row (presence-only); mlp4 +
# attention 5,6 (repaired); the full 36-site program with a fallback variant for the inert half of the
# control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §2007's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
M4 = [('mlp', 4)]
CEIL, TRIO, A4, A5, MEAN5, FIXED = 'm4', 'trio', 'm4_a4', 'm4_a5', 'm4_a5_mean', 'm4_a56'

PLAN = [(ARM, BASE, CEIL, M4),                                            # §1985: 10.669 at 5,419
        (ARM, BASE, TRIO, [('mlp', L) for L in (2, 3, 4)]),               # §2004: 10.701
        (ARM, BASE, A4, M4 + [('attn', 4)]),                              # §2006: 10.669
        (ARM, BASE, A5, M4 + [('attn', 5)]),                              # §1990:  8.021
        ('meanrow@attn5+mix30m640@mlp4', BASE, MEAN5, M4 + [('attn', 5)]),  # §2007: 7.845
        (ARM, BASE, FIXED, M4 + [('attn', 5), ('attn', 6)]),              # §1990:  1.555
        (ARM, BASE, 'full_program', None),                                # §1985:  2.808
        ('map512', BASE, 'full_fb_control', None),                        # all 36 sites: the INERT pair
        (ARM, A256, 'rank_control', None)]                                # differing rank: other half


def _the_ceiling_holds_at_high_coverage(x):
    """at 16,110 the ceiling still holds: mlp4 alone, three compiled MLPs, and mlp4 + attention 4 all lie
    within 0.4 nats of one another on all three roles, as they did at 5,419 (spread 0.343 there)"""
    def spread(r):
        v = [x.penalty(HI, r, a) for a in (CEIL, TRIO, A4)]
        return max(v) - min(v)
    return all(spread(r) < 0.4 for r in x.roles)


def _the_lever_survives_the_coverage_change(x):
    """and attention 5 still takes more than 2.0 nats off the ceiling at 16,110, on all three roles -- it
    was 2.65 at 5,419. The second coverage halves the uncovered arm, so a lever that were really about
    §1870's map rather than the computation would shrink"""
    return all(x.penalty(HI, r, CEIL) - x.penalty(HI, r, A5) > 2.0 for r in x.roles)


def _presence_only_survives_too(x):
    """and attention 5's content is still worthless at 16,110: the mean row is no worse than its compiled
    table plus 0.01 nats, on all three roles. Registered directionally -- §1998 and §2007 both measured
    the mean row as strictly BETTER, so the prediction has a sign"""
    return all(x.penalty(HI, r, MEAN5) <= x.penalty(HI, r, A5) + 0.01 for r in x.roles)


B.run(
    name='the_ceiling_at_both_coverages',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_ceiling_holds_at_high_coverage',
         'at 16,110 the three ceiling arms lie within 0.4 nats of one another, on 3/3 roles',
         _the_ceiling_holds_at_high_coverage),
        ('pred_b_the_lever_survives_the_coverage_change',
         'and attention 5 still takes more than 2.0 nats off the ceiling there, on 3/3 roles',
         _the_lever_survives_the_coverage_change),
        ('pred_c_presence_only_survives_too',
         'and a mean row at attention 5 is still no worse than its table there, on 3/3 roles',
         _presence_only_survives_too),
    ],
    refs=[(CEIL, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', LO, 0.0005),
          (A5, B.PT + 'ops/is_the_lever_also_presence_only_results.json', 'a5_table', LO, 0.0005),
          (FIXED, B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', LO, 0.0005)],
    paired_pairs=[(A5, CEIL), (MEAN5, A5), (TRIO, CEIL)],
)
