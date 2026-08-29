# IS THE FULLY COMPILED PROGRAM ADDITIVE OVER SITES?
#
# §2013 found mlp2's table content inside the shipped 36-site program worth about 0.0003 nats, against
# 1.408 in the three-site repaired arm, and gave the structural reason: with every module's output a fixed
# per-token row, NOTHING DOWNSTREAM READS mlp2's output. It contributes only its own additive term to the
# residual stream.
#
# That mechanism makes a sharp prediction never tested. §1981 measured compilation cost as badly
# non-additive over sites — six compiled MLPs worse than all thirty-six — but every arm there had LIVE
# attention. In the shipped frame there is no path from one site to another, so degrading two sites should
# cost exactly the sum of degrading each.
#
# The instrument is the mean row: replacing a site's table with a constant is a degradation whose size
# §2013 measured at 0.00012 nats for mlp2. Two such degradations, and their sum, is the whole test.
#
# ARMS. the full 36-site program; the same with a MEAN ROW at mlp2, at mlp3, at mlp12, and at all three;
# a fallback variant of the full program for the inert half of the control; and one differing-table-rank
# arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2013's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
FULL = 'full_program'
ONES = ((2, 'mean_m2'), (3, 'mean_m3'), (12, 'mean_m12'))
ALL3 = 'mean_all3'

FULL_PROGRAM = (2.80750, 2.97891, 2.70217)     # §2013's rank-768 arm, which is the shipped program


def _mean_arm(*layers):
    """meanrow at the named MLPs, the shipped table everywhere else"""
    named = ','.join(f'mlp{L}' for L in layers)
    others = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                      if not (k == 'mlp' and L in layers))
    return f'meanrow@{named}+{ARM}@{others}'


PLAN = [(ARM, BASE, FULL, None)] + \
       [(_mean_arm(L), BASE, lab, B.SITES) for L, lab in ONES] + [
    (_mean_arm(2, 3, 12), BASE, ALL3, B.SITES),
    ('map512', BASE, 'full_fb_control', None),          # all 36 sites, other fallback: the INERT pair
    (ARM, A256, 'rank_control', None)]                  # differing table rank: the other half


def _d(x, role, lab):
    return x.penalty(C, role, lab) - x.penalty(C, role, FULL)


def _full_program_reproduces(x):
    """the shipped program rebuilds to §2013's 2.80750 / 2.97891 / 2.70217 within 0.0005 nats on all three
    roles -- every degradation below is a difference of a few ten-thousandths and needs that anchor"""
    return all(abs(x.penalty(C, r, FULL) - v) < 0.0005 for r, v in zip(x.roles, FULL_PROGRAM))


def _each_degradation_is_tiny(x):
    """and each single-site mean row costs under 0.005 nats on all three roles -- §2013 measured mlp2 at
    0.00012, and a site whose degradation were large would mean something downstream does read it"""
    return all(abs(_d(x, r, lab)) < 0.005 for r in x.roles for _L, lab in ONES)


def _the_three_are_additive(x):
    """and degrading all three costs the sum of degrading each, within 0.002 nats on >=2 roles. This is
    the structural prediction of §1765 + §2013: no path between compiled sites, so no interaction. If
    FALSE something does couple compiled sites in the shipped frame and §2013's mechanism is incomplete"""
    return sum(1 for r in x.roles
               if abs(_d(x, r, ALL3) - sum(_d(x, r, lab) for _L, lab in ONES)) < 0.002) >= 2


B.run(
    name='is_the_shipped_program_additive',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_program_reproduces',
         'the shipped program rebuilds to §2013\'s 2.80750/2.97891/2.70217 within 0.0005 nats on 3/3 roles',
         _full_program_reproduces),
        ('pred_b_each_degradation_is_tiny',
         'and each single-site mean row costs under 0.005 nats, on 3/3 roles', _each_degradation_is_tiny),
        ('pred_c_the_three_are_additive',
         'and degrading all three costs the sum of degrading each, within 0.002 nats (>=2 roles)',
         _the_three_are_additive),
    ],
    refs=[(FULL, B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('mean_m2', FULL), (ALL3, FULL), (ALL3, 'mean_m2')],
)
