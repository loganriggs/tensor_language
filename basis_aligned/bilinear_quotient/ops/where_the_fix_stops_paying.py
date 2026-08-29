# WHERE DOES THE THRESHOLD'S FIX STOP PAYING?
#
# §2000 found the first site the pair does not rescue: compiling mlp5 with attention 5 and 6 costs 2.263
# nats where compiling mlp5 alone costs 2.022. The arithmetic is plain — a lone attention 5 costs 0.241
# and a lone attention 6 about the same (§1989) — so for a site whose own damage is already small the fix
# adds more than it removes.
#
# That gives the threshold a DOMAIN, and this line has implied one since §1985 without ever quoting it.
# mlp6 and mlp7 sit at lone damages of 0.202 and 0.072, an order of magnitude below mlp5's 2.022, so the
# fix should be a much larger net loss there. mlp3 at 6.574 and mlp2 at 4.813 are rescued. The crossing
# point lies between mlp5 and mlp3 in damage terms, and these five sites bracket it.
#
# ARMS. mlp3, mlp5, mlp6, mlp7 alone and each with attention 5,6; mlp2 alone and with attention 5,6 as the
# §1987/§1992 anchors; the full 36-site program with a fallback variant for the inert half of the control;
# and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2000's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
MLPS = (2, 3, 5, 6, 7)
RESCUED, LOST = (2, 3), (6, 7)


def _lone(L):
    return [('mlp', L)]


def _fixed(L):
    return [('mlp', L), ('attn', 5), ('attn', 6)]


PLAN = [(ARM, BASE, f'm{L}', _lone(L)) for L in MLPS] + \
       [(ARM, BASE, f'm{L}_fix', _fixed(L)) for L in MLPS] + [
    (ARM, BASE, 'full_program', None),                 # §1985: 2.808 / 2.979 / 2.702
    ('map512', BASE, 'full_fb_control', None),         # all 36 sites, other fallback: the INERT pair
    (ARM, A256, 'rank_control', None)]                 # differing table rank: the other half


def _gain(x, role, L):
    """what compiling attention 5 and 6 alongside mlp_L is worth: positive means the fix pays"""
    return x.penalty(C, role, f'm{L}') - x.penalty(C, role, f'm{L}_fix')


def _the_fix_pays_for_damaged_sites(x):
    """the fix still pays at mlp2 and mlp3, whose lone damages are 4.813 and 6.574 -- gain strictly
    positive on all three roles at both"""
    return all(_gain(x, r, L) > 0 for r in x.roles for L in RESCUED)


def _the_fix_is_a_loss_for_cheap_sites(x):
    """and it is a strict LOSS at mlp6 and mlp7, whose lone damages are 0.202 and 0.072 -- gain negative
    on all three roles at both. §2000 found mlp5 losing by 0.241; these should lose by more. If FALSE the
    fix is not simply additive at cheap sites and something else is happening"""
    return all(_gain(x, r, L) < 0 for r in x.roles for L in LOST)


def _the_loss_grows_as_damage_falls(x):
    """and the loss deepens as the site gets cheaper: gain at mlp7 < gain at mlp6 < gain at mlp5 < 0, on
    >=2 roles. That is what a fixed additive cost against a shrinking benefit looks like, and it makes the
    crossing point a real number rather than an ordering accident"""
    return sum(1 for r in x.roles
               if _gain(x, r, 7) < _gain(x, r, 6) < _gain(x, r, 5) < 0) >= 2


B.run(
    name='where_the_fix_stops_paying',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_the_fix_pays_for_damaged_sites',
         'compiling attention 5 and 6 still pays at mlp2 and mlp3, on all three roles',
         _the_fix_pays_for_damaged_sites),
        ('pred_b_the_fix_is_a_loss_for_cheap_sites',
         'and it is a strict loss at mlp6 and mlp7, on all three roles',
         _the_fix_is_a_loss_for_cheap_sites),
        ('pred_c_the_loss_grows_as_damage_falls',
         'and the loss deepens monotonically from mlp5 to mlp6 to mlp7 (>=2 roles)',
         _the_loss_grows_as_damage_falls),
    ],
    refs=[('m2', B.PT + 'ops/is_layer_six_the_boundary_for_all_results.json', 'mlp2', C, 0.0005),
          ('m2_fix', B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a56', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('m5_fix', 'm5'), ('m6_fix', 'm6'), ('m7_fix', 'm7')],
)
