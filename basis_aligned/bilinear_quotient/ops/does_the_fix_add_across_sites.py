# THE ACCOUNT'S FIRST PREDICTION: IS THE FIX'S VALUE ADDITIVE ACROSS COMPILED MLPs?
#
# §2001 and §2002 made the threshold quantitative: compiling attention 5 and 6 costs a flat ~0.65 nats and
# recovers a share of the compiled MLP's lone damage, crossing at ~2.15. Every measurement behind that used
# ONE compiled MLP, and the account has never been asked to predict anything.
#
# Two compiled MLPs is the cheapest configuration it has not seen. §1981 established that compilation COST
# is badly non-additive — six compiled MLPs cost 3.9x the full 36-site program. Whether the fix's VALUE is
# additive is a separate question, and it has a clean form: if the pair rescues each site independently,
# the gain on {mlp2, mlp3} should be near the sum of the single-site gains minus the 0.65 it is charged
# only once. If the sites interfere, it will not be.
#
# ARMS. mlp2+mlp3 alone and with attention 5,6; mlp2 and mlp3 alone and fixed as the §2001 anchors; the
# full 36-site program with a fallback variant for the inert half of the control; and one differing-rank
# arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2002's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
A56 = [('attn', 5), ('attn', 6)]
PAIR = [('mlp', 2), ('mlp', 3)]
PLATEAU = 0.65      # §2001: the flat standalone cost of compiling attention 5 and 6

PLAN = [(ARM, BASE, 'both', PAIR),
        (ARM, BASE, 'both_fix', PAIR + A56),
        (ARM, BASE, 'm2', [('mlp', 2)]),                   # §2001: 4.813 / 5.291 / 4.958
        (ARM, BASE, 'm2_fix', [('mlp', 2)] + A56),         # §1992: 1.971 / 2.090 / 1.952
        (ARM, BASE, 'm3', [('mlp', 3)]),                   # §1988: 6.574 / 6.894 / 6.567
        (ARM, BASE, 'm3_fix', [('mlp', 3)] + A56),         # §1992: 1.996 / 2.152 / 1.933
        (ARM, BASE, 'full_program', None),                 # §1985: 2.808 / 2.979 / 2.702
        ('map512', BASE, 'full_fb_control', None),         # all 36 sites, other fallback: the INERT pair
        (ARM, A256, 'rank_control', None)]                 # differing table rank: the other half


def _gain(x, role, lone, fix):
    return x.penalty(C, role, lone) - x.penalty(C, role, fix)


def _lone_cost_is_superadditive(x):
    """compiling mlp2 and mlp3 together costs MORE than the sum of their lone damages, on >=2 roles --
    §1981's non-additivity at the smallest scale it can be tested. This is the premise the prediction is
    made against, not the prediction itself"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, 'both') > x.penalty(C, r, 'm2') + x.penalty(C, r, 'm3')) >= 2


def _the_fix_still_pays_on_the_pair(x):
    """and the fix still pays on the two-MLP configuration -- gain strictly positive on all three roles.
    Both sites are far above the 2.15-nat crossing, so §2001's account requires this"""
    return all(_gain(x, r, 'both', 'both_fix') > 0 for r in x.roles)


def _the_gain_is_not_merely_additive(x):
    """and the gain on the pair EXCEEDS the sum of the single-site gains plus the 0.65 charged once
    instead of twice, on >=2 roles. That is what a fix repairing a shared interface -- rather than two
    sites independently -- would look like. If FALSE the fix's value is additive or sub-additive, and the
    account should be stated per-site"""
    def naive(r):
        return _gain(x, r, 'm2', 'm2_fix') + _gain(x, r, 'm3', 'm3_fix') + PLATEAU
    return sum(1 for r in x.roles if _gain(x, r, 'both', 'both_fix') > naive(r)) >= 2


B.run(
    name='does_the_fix_add_across_sites',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_lone_cost_is_superadditive',
         'compiling mlp2 and mlp3 together costs more than the sum of their lone damages (>=2 roles)',
         _lone_cost_is_superadditive),
        ('pred_b_the_fix_still_pays_on_the_pair',
         'and attention 5,6 still pays on the two-MLP configuration, on 3/3 roles',
         _the_fix_still_pays_on_the_pair),
        ('pred_c_the_gain_is_not_merely_additive',
         'and the gain exceeds the sum of the single-site gains plus the 0.65 charged once (>=2 roles)',
         _the_gain_is_not_merely_additive),
    ],
    refs=[('m2', B.PT + 'ops/where_the_fix_stops_paying_results.json', 'm2', C, 0.0005),
          ('m3', B.PT + 'ops/where_the_fix_stops_paying_results.json', 'm3', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('both_fix', 'both'), ('both_fix', 'm2_fix'), ('both', 'm3')],
)
