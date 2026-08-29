# DOES THE REPAIR'S SUPER-ADDITIVITY GROW WITH THE NUMBER OF COMPILED MLPs, OR SATURATE?
#
# §2003 found the two signs that make §1981's single-interface account quantitative: compiling mlp2 and
# mlp3 together is SUB-additive in damage (10.542 against a sum of 11.388) while the threshold's repair is
# SUPER-additive in value (+0.528 / +0.166 / +0.299 over the sum of single-site gains with the 0.65 flat
# cost charged once instead of twice).
#
# Both readings of that are alive. If two broken MLPs feed one interface and repairing it serves both,
# a THIRD should extend the pattern -- more damage folded into the same interface, more value from the
# same 0.65. If instead the interface has limited capacity, the excess should stop growing. §2003's own
# margin is the reason to ask: it ranges 0.166 to 0.528 across roles, a factor of three, and skip11000's
# 0.166 is the smallest quantity this line has rested a claim on.
#
# ARMS. mlp2+mlp3+mlp4 alone and with attention 5,6; mlp2+mlp3 alone and fixed as the §2003 anchors; the
# three single sites alone and fixed; the full 36-site program with a fallback variant for the inert half
# of the control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2003's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
A56 = [('attn', 5), ('attn', 6)]
SINGLES = (2, 3, 4)
PAIR = [('mlp', 2), ('mlp', 3)]
TRIO = [('mlp', L) for L in SINGLES]
PLATEAU = 0.65      # §2001: the flat standalone cost of compiling attention 5 and 6

# §2003, skip7000 / skip11000 / skip1200 -- the two-site excess over the naive prediction
EXCESS_2 = (0.528, 0.166, 0.299)

PLAN = [(ARM, BASE, f'm{L}', [('mlp', L)]) for L in SINGLES] + \
       [(ARM, BASE, f'm{L}_fix', [('mlp', L)] + A56) for L in SINGLES] + [
    (ARM, BASE, 'pair', PAIR),                         # §2003: 10.542 / 10.826 / 10.476
    (ARM, BASE, 'pair_fix', PAIR + A56),               # §2003:  1.944 /  2.068 /  1.887
    (ARM, BASE, 'trio', TRIO),
    (ARM, BASE, 'trio_fix', TRIO + A56),
    (ARM, BASE, 'full_program', None),                 # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),         # all 36 sites, other fallback: the INERT pair
    (ARM, A256, 'rank_control', None)]                 # differing table rank: the other half


def _gain(x, role, lone, fix):
    return x.penalty(C, role, lone) - x.penalty(C, role, fix)


def _excess(x, role, lone, fix, sites):
    """how much the fix beats the naive per-site prediction, with the flat 0.65 charged once"""
    naive = sum(_gain(x, role, f'm{L}', f'm{L}_fix') for L in sites) + PLATEAU * (len(sites) - 1)
    return _gain(x, role, lone, fix) - naive


def _pair_anchor_reproduces(x):
    """§2003's two-site excess rebuilds to +0.528 / +0.166 / +0.299 within 0.01 nats on all three roles.
    The three-site comparison below is a difference of two small quantities, so if the anchor does not
    reproduce, the comparison means nothing"""
    return all(abs(_excess(x, r, 'pair', 'pair_fix', (2, 3)) - v) < 0.01
               for r, v in zip(x.roles, EXCESS_2))


def _three_site_damage_is_subadditive_too(x):
    """and three compiled MLPs cost less than the sum of their three lone damages, on all three roles --
    §2003 measured this at two sites and the single-interface account requires it at three"""
    return all(x.penalty(C, r, 'trio') < sum(x.penalty(C, r, f'm{L}') for L in SINGLES)
               for r in x.roles)


def _the_excess_grows_with_the_third_site(x):
    """and the repair's super-additivity GROWS: the three-site excess exceeds the two-site excess on >=2
    roles. If TRUE more damage folded into the same interface yields more value from the same 0.65; if
    FALSE the interface has limited capacity and the excess saturates -- which is the more interesting
    answer and the one §2003's three-fold spread across roles hints at"""
    return sum(1 for r in x.roles
               if _excess(x, r, 'trio', 'trio_fix', SINGLES)
               > _excess(x, r, 'pair', 'pair_fix', (2, 3))) >= 2


B.run(
    name='does_the_repair_saturate',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_pair_anchor_reproduces',
         '§2003\'s two-site excess rebuilds to +0.528/+0.166/+0.299 within 0.01 nats on 3/3 roles',
         _pair_anchor_reproduces),
        ('pred_b_three_site_damage_is_subadditive_too',
         'and three compiled MLPs cost less than the sum of their lone damages, on 3/3 roles',
         _three_site_damage_is_subadditive_too),
        ('pred_c_the_excess_grows_with_the_third_site',
         'and the repair\'s super-additivity grows rather than saturating (>=2 roles)',
         _the_excess_grows_with_the_third_site),
    ],
    refs=[('pair', B.PT + 'ops/does_the_fix_add_across_sites_results.json', 'both', C, 0.0005),
          ('pair_fix', B.PT + 'ops/does_the_fix_add_across_sites_results.json', 'both_fix', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('trio_fix', 'trio'), ('trio_fix', 'pair_fix'), ('trio', 'pair')],
)
