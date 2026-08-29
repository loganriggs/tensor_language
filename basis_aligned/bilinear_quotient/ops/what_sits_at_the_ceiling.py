# WHAT SITS AT THE CEILING, AND WHAT GETS UNDER IT WITHOUT TOUCHING ATTENTION 5 OR 6?
#
# §2004 found a damage ceiling near 10.7 nats: one compiled MLP (mlp4) 10.669, two 10.542, three 10.701,
# and §1981's six-MLP below_mlp arm 10.688 — four configurations from one to six sites within 0.16 nats.
# The account is that context-free rows arriving at a live attention 6 cost about 10.7 however many sites
# produce them.
#
# That account makes two testable commitments it has not been asked for. Any set containing mlp4 and no
# attention 5 or 6 should sit AT the ceiling, whatever else is in it — including all eighteen MLPs, which
# is three times §1981's largest. And the one thing already known to move the number without touching
# attention 5 or 6 is compiling attention layers BELOW: §1981's below_all arm (18 MLPs + attention 0–5)
# measured 9.266, a nat and a half under the ceiling and unexplained by it.
#
# ARMS. mlp4 alone; mlp4 + mlp12 (a site above the boundary); all eighteen MLPs; mlp4 + attention 0–3;
# §1981's below_all rebuilt for a second-class confirm; the full 36-site program with a fallback variant
# for the inert half of the control; and one differing-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2004's open question, and rung 2 for
# §1981's below_all triple.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
ALLMLP = [('mlp', L) for L in range(18)]
CEILING = 'm4'
AT = ('m4', 'm4_m12', 'all_mlps')

PLAN = [(ARM, BASE, 'm4', [('mlp', 4)]),                                    # §1985: 10.669/10.937/10.580
        (ARM, BASE, 'm4_m12', [('mlp', 4), ('mlp', 12)]),
        (ARM, BASE, 'all_mlps', ALLMLP),
        (ARM, BASE, 'm4_a03', [('mlp', 4)] + [('attn', L) for L in range(4)]),
        (ARM, BASE, 'below_all', ALLMLP + [('attn', L) for L in range(6)]),  # §1981: 9.266/9.531/9.141
        (ARM, BASE, 'full_program', None),                                   # §1985: 2.808/2.979/2.702
        ('map512', BASE, 'full_fb_control', None),                           # all 36 sites: the INERT pair
        (ARM, A256, 'rank_control', None)]                                   # differing rank: other half


def _below_all_replicates(x):
    """§1981's below_all triple rebuilds to 9.266 / 9.531 / 9.141 within 0.05 nats on all three roles --
    a second-class confirm of the one published number that sits between the ceiling and the fixed arms"""
    want = (9.266, 9.531, 9.141)
    return all(abs(x.penalty(C, r, 'below_all') - v) < 0.05 for r, v in zip(x.roles, want))


def _everything_with_mlp4_sits_at_the_ceiling(x):
    """and every set containing mlp4 with no attention 5 or 6 is within 0.3 nats of mlp4 alone, on all
    three roles -- including ALL EIGHTEEN MLPs, three times §1981's largest such arm. If FALSE the ceiling
    is not a property of the interface but of the particular sets measured so far"""
    return all(abs(x.penalty(C, r, a) - x.penalty(C, r, CEILING)) < 0.3
               for r in x.roles for a in AT)


def _attention_below_gets_under_it(x):
    """and compiling attention 0-3 beneath mlp4 drops it more than 0.3 nats below the ceiling on >=2
    roles -- the only mechanism known to move the number without touching attention 5 or 6, and the one
    §1981's below_all shows exists. If FALSE that 9.266 comes from something else entirely"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, CEILING) - x.penalty(C, r, 'm4_a03') > 0.3) >= 2


B.run(
    name='what_sits_at_the_ceiling',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_below_all_replicates',
         '§1981\'s below_all rebuilds to 9.266/9.531/9.141 within 0.05 nats on 3/3 roles',
         _below_all_replicates),
        ('pred_b_everything_with_mlp4_sits_at_the_ceiling',
         'and every mlp4-containing set without attention 5 or 6 is within 0.3 nats of mlp4 alone, 3/3 roles',
         _everything_with_mlp4_sits_at_the_ceiling),
        ('pred_c_attention_below_gets_under_it',
         'and compiling attention 0-3 beneath mlp4 drops it more than 0.3 nats below the ceiling (>=2 roles)',
         _attention_below_gets_under_it),
    ],
    refs=[('m4', B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005),
          ('full_program', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('all_mlps', 'm4'), ('m4_a03', 'm4'), ('below_all', 'all_mlps')],
)
