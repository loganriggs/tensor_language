# DOES RANK 1 CARRY 60% OF ATTENTION 6'S CONTENT AT A SECOND COMPILED MLP?
#
# §2009 found that a single direction recovers 59-62% of attention 6's whole contribution when the
# compiled MLP is mlp2. §2000 measured that contribution at 0.212 for mlp2 and 0.095 for mlp4 — a factor
# of 2.2 — and §1999 established that the SIZE of attention 6's content is a fact about the pair rather
# than the layer.
#
# The rank-1 FRACTION is a different quantity from the size, and it has not been checked anywhere else.
# If 60% transfers to mlp4 despite the size halving, the direction is a property of attention 6; if the
# fraction moves with the pair, then even the shape of attention 6's content is pair-specific and there is
# no single direction to name.
#
# ARMS. mlp4 + attention 5,6 with attention 6's table at rank 1 / 4 / 16 / 384, and the arm omitting
# attention 6; the mlp2 rank-1 and rank-384 arms rebuilt here so both fractions are computed in-run; the
# full 36-site program with a fallback variant for the inert half of the control; and one differing-rank
# arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2 -- §2009 second-class confirmed at mlp4.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
S4 = [('mlp', 4), ('attn', 5), ('attn', 6)]
S2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
NO6_4, NO6_2 = 'm4_a5', 'm2_a5'

# §2009 at 5,419: the rank-1 fraction of attention 6's contribution with a compiled mlp2
FRAC_MLP2 = (0.592, 0.611, 0.618)


def _spec6(r):
    return {'mlp': 768, 'attn': 384, ('attn', 6): r}


PLAN = [(ARM, _spec6(r), f'm4_r{r}', S4) for r in (1, 4, 16, 384)] + [
    (ARM, _spec6(1), 'm2_r1', S2),                       # §2009: 2.210 / 2.339 / 2.183
    (ARM, BASE, 'm2_r384', S2),                          # §1992: 1.971 / 2.090 / 1.952
    (ARM, BASE, NO6_4, [('mlp', 4), ('attn', 5)]),       # §1990: 8.021 / 8.300 / 7.962
    (ARM, BASE, NO6_2, [('mlp', 2), ('attn', 5)]),       # §1992: 2.556 / 2.730 / 2.558
    (ARM, BASE, 'full_program', None),                   # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),           # all 36 sites: the INERT pair
    (ARM, A256, 'rank_control', None)]                   # differing rank: other half


def _frac(x, role, r1, full, no6):
    hi, lo = x.penalty(C, role, no6), x.penalty(C, role, full)
    return (hi - x.penalty(C, role, r1)) / (hi - lo)


def _the_mlp2_fraction_reproduces(x):
    """§2009's rank-1 fraction at mlp2 rebuilds to 0.592 / 0.611 / 0.618 within 0.01 on all three roles --
    the quantity the mlp4 fraction is compared against"""
    return all(abs(_frac(x, r, 'm2_r1', 'm2_r384', NO6_2) - v) < 0.01
               for r, v in zip(x.roles, FRAC_MLP2))


def _the_fraction_transfers_to_mlp4(x):
    """and the rank-1 fraction at mlp4 is within 0.10 of the mlp2 one on >=2 roles, though §2000 measured
    the CONTENT ITSELF as 2.2x smaller there. If TRUE the direction is a property of attention 6 and worth
    naming; if FALSE even the shape of its content is pair-specific and there is no single direction"""
    return sum(1 for r in x.roles
               if abs(_frac(x, r, 'm4_r1', 'm4_r384', NO6_4)
                      - _frac(x, r, 'm2_r1', 'm2_r384', NO6_2)) < 0.10) >= 2


def _the_curve_has_the_same_shape(x):
    """and the curve is flat between rank 1 and rank 16 at mlp4 as it was at mlp2 -- rank 16 recovers less
    than 0.20 more of the contribution than rank 1 does, on >=2 roles. At mlp2 that step was 0.12"""
    def step(r):
        return (_frac(x, r, 'm4_r16', 'm4_r384', NO6_4)
                - _frac(x, r, 'm4_r1', 'm4_r384', NO6_4))
    return sum(1 for r in x.roles if step(r) < 0.20) >= 2


B.run(
    name='is_the_rank_one_direction_the_same',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_the_mlp2_fraction_reproduces',
         '§2009\'s rank-1 fraction at mlp2 rebuilds to 0.592/0.611/0.618 within 0.01 on 3/3 roles',
         _the_mlp2_fraction_reproduces),
        ('pred_b_the_fraction_transfers_to_mlp4',
         'and the rank-1 fraction at mlp4 is within 0.10 of it (>=2 roles), though the content is 2.2x smaller',
         _the_fraction_transfers_to_mlp4),
        ('pred_c_the_curve_has_the_same_shape',
         'and rank 16 adds less than 0.20 over rank 1 at mlp4, as it added 0.12 at mlp2 (>=2 roles)',
         _the_curve_has_the_same_shape),
    ],
    refs=[(NO6_4, B.PT + 'ops/is_the_lever_also_presence_only_results.json', 'a5_table', C, 0.0005),
          (NO6_2, B.PT + 'ops/where_the_threshold_gap_lives_results.json', 'mlp2_a5', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('m4_r1', 'm4_r384'), ('m2_r1', 'm2_r384'), ('m4_r16', 'm4_r1')],
)
