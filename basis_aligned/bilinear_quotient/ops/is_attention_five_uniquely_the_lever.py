# IS ATTENTION 5 UNIQUELY THE LEVER BENEATH THE CEILING?
#
# §2005 confirmed §2004's ceiling holds to about 3% and killed the mechanism I had proposed for §1981's
# below_all figure: compiling attention 0–3 beneath a compiled mlp4 is worth 0.018 nats, not the 1.4 that
# would explain 9.266. The ledger already names the alternative — §1990 measured mlp4 + attention 5 at
# 8.021, 2.65 nats under the ceiling with attention 6 still live, and §1985 measured mlp4 + attention 0–5
# at 8.560, WORSE than attention 5 alone.
#
# Attention 4 is the one layer between those two facts that has never been tested on its own. If it sits
# at the ceiling like attention 0–3 did, attention 5 is uniquely the lever and every remaining number in
# this line is about two adjacent attention layers.
#
# ARMS. mlp4 with attention 4 alone, attention 5 alone, attention 4 and 5, and attention 0–3, each with
# attention 6 live; mlp4 alone as the ceiling anchor; the full 36-site program with a fallback variant for
# the inert half of the control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2005's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
M4 = [('mlp', 4)]
CEIL, A4, A5, A45, A03 = 'm4', 'm4_a4', 'm4_a5', 'm4_a45', 'm4_a03'

PLAN = [(ARM, BASE, CEIL, M4),                                            # §1985: 10.669/10.937/10.580
        (ARM, BASE, A4, M4 + [('attn', 4)]),
        (ARM, BASE, A5, M4 + [('attn', 5)]),                              # §1990:  8.021/ 8.300/ 7.962
        (ARM, BASE, A45, M4 + [('attn', 4), ('attn', 5)]),
        (ARM, BASE, A03, M4 + [('attn', L) for L in range(4)]),           # §2005: 10.651/10.922/10.563
        (ARM, BASE, 'full_program', None),                                # §1985:  2.808/ 2.979/ 2.702
        ('map512', BASE, 'full_fb_control', None),                        # all 36 sites: the INERT pair
        (ARM, A256, 'rank_control', None)]                                # differing rank: other half


def _attention_five_anchor_reproduces(x):
    """§1990's mlp4 + attention 5 rebuilds to 8.021 / 8.300 / 7.962 within 0.005 nats on all three roles.
    Every comparison below is measured against it"""
    want = (8.021, 8.300, 7.962)
    return all(abs(x.penalty(C, r, A5) - v) < 0.005 for r, v in zip(x.roles, want))


def _attention_four_is_not_a_lever(x):
    """and attention 4 alone leaves mlp4 at the ceiling -- within 0.3 nats of mlp4 alone on all three
    roles, as attention 0-3 was within 0.018. If FALSE attention 4 is a second lever and the sub-ceiling
    effect is not attention 5's alone"""
    return all(abs(x.penalty(C, r, A4) - x.penalty(C, r, CEIL)) < 0.3 for r in x.roles)


def _adding_four_to_five_does_not_help(x):
    """and adding attention 4 on top of attention 5 buys nothing: mlp4_a45 is no better than mlp4_a5 minus
    0.01 nats, on >=2 roles. Registered directionally -- §1986 and §1990 both measured sites below the
    compiled MLP as costing rather than helping, so the prediction has a sign"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, A45) >= x.penalty(C, r, A5) - 0.01) >= 2


B.run(
    name='is_attention_five_uniquely_the_lever',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_attention_five_anchor_reproduces',
         '§1990\'s mlp4 + attention 5 rebuilds to 8.021/8.300/7.962 within 0.005 nats on 3/3 roles',
         _attention_five_anchor_reproduces),
        ('pred_b_attention_four_is_not_a_lever',
         'and attention 4 alone leaves mlp4 at the ceiling (within 0.3 nats, 3/3 roles)',
         _attention_four_is_not_a_lever),
        ('pred_c_adding_four_to_five_does_not_help',
         'and adding attention 4 on top of attention 5 buys nothing (>=2 roles)',
         _adding_four_to_five_does_not_help),
    ],
    refs=[(CEIL, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005),
          (A03, B.PT + 'ops/what_sits_at_the_ceiling_results.json', 'm4_a03', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[(A4, CEIL), (A45, A5), (A5, CEIL)],
)
