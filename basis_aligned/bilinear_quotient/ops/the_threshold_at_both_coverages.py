# THE ONE SURVIVING STATEMENT, AT THE OTHER COVERAGE.
#
# §1986, §1990 and §1992 each proposed a rule for which sites must be compiled together, and each was
# falsified within an hour by an arm that deleted a member. What survived all three is a threshold rather
# than a rule: across fifteen arms at 5,419, every configuration compiling BOTH attention 5 and attention
# 6 costs 1.50–2.15 nats, and every one omitting either costs 2.56–10.94. No exceptions.
#
# That statement is worth more than the rules were, so it should be replicated before anything is built on
# it. The second coverage is the standing instrument: 16,110 types halves the uncovered arm, and it has
# broken claims that held at 5,419 before (§1963, §1965).
#
# ARMS. four on the good side of the threshold (mlp4+a5,6; mlp2+a5,6; mlp2+a4,5,6; mlp3+a5,6) and three on
# the bad side (mlp2+a5; mlp2+a6; mlp4 alone), plus the full 36-site program and one fallback variant of it
# so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §1992's threshold at the other coverage.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
LO, HI = 'c5419', 'c16110'
A56 = [('attn', 5), ('attn', 6)]
GOOD = ('mlp4_a56', 'mlp2_a56', 'mlp2_a456', 'mlp3_a56')
BAD = ('mlp2_a5', 'mlp2_a6', 'mlp4')
FULL = 'full_program'

PLAN = [(ARM, A384, 'mlp4_a56', [('mlp', 4)] + A56),                  # §1990: 1.555 / 1.640 / 1.498
        (ARM, A384, 'mlp2_a56', [('mlp', 2)] + A56),                  # §1992: 1.971 / 2.090 / 1.952
        (ARM, A384, 'mlp2_a456', [('mlp', 2), ('attn', 4)] + A56),    # §1991: 1.845 / 1.959 / 1.807
        (ARM, A384, 'mlp3_a56', [('mlp', 3)] + A56),                  # §1992: 1.996 / 2.152 / 1.933
        (ARM, A384, 'mlp2_a5', [('mlp', 2), ('attn', 5)]),            # §1992: 2.556 / 2.730 / 2.558
        (ARM, A384, 'mlp2_a6', [('mlp', 2), ('attn', 6)]),            # §1992: 4.876 / 5.355 / 5.016
        (ARM, A384, 'mlp4', [('mlp', 4)]),                            # §1985: 10.669 / 10.937 / 10.580
        (ARM, A384, FULL),                                            # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]                          # other fallback: INERT


def _threshold_holds_at_high_coverage(x):
    """the threshold separates the two groups at 16,110 with no exceptions: every arm compiling both
    attention 5 and 6 costs less than every arm omitting either, on all three roles"""
    return all(max(x.penalty(HI, r, a) for a in GOOD) < min(x.penalty(HI, r, a) for a in BAD)
               for r in x.roles)


def _the_gap_is_not_marginal(x):
    """and it separates by more than 0.3 nats at 16,110 on all three roles -- a threshold that only just
    holds is a coincidence of ordering, not a fact about the model"""
    return all(min(x.penalty(HI, r, a) for a in BAD) - max(x.penalty(HI, r, a) for a in GOOD) > 0.3
               for r in x.roles)


def _coverage_does_not_reorder_them(x):
    """and coverage does not reorder the good side: the four good arms rank the same at both coverages on
    >=2 roles. §1963 and §1965 both found a claim that held at 5,419 and reversed at 16,110, so this asks
    the question that instrument exists to ask"""
    def order(c, r):
        return tuple(sorted(GOOD, key=lambda a: x.penalty(c, r, a)))
    return sum(1 for r in x.roles if order(LO, r) == order(HI, r)) >= 2


B.run(
    name='the_threshold_at_both_coverages',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_threshold_holds_at_high_coverage',
         'every arm compiling both attention 5 and 6 beats every arm omitting either, at 16,110, 3/3 roles',
         _threshold_holds_at_high_coverage),
        ('pred_b_the_gap_is_not_marginal',
         'and the two groups are separated by more than 0.3 nats at 16,110 on all three roles',
         _the_gap_is_not_marginal),
        ('pred_c_coverage_does_not_reorder_them',
         'and the four good arms rank the same at both coverages (>=2 roles)',
         _coverage_does_not_reorder_them),
    ],
    refs=[('mlp4_a56', B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', LO, 0.0005),
          ('mlp2_a56', B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a56', LO, 0.0005),
          (FULL, B.PT + 'ops/the_minimal_path_results.json', FULL, LO, 0.0005)],
    paired_pairs=[('mlp2_a56', 'mlp2_a5'), ('mlp4_a56', 'mlp2_a56'), ('mlp2_a456', 'mlp2_a56')],
)
