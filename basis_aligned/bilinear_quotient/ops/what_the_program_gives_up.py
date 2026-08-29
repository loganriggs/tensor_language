# WHAT DOES THE COMPILED PROGRAM GIVE UP?  -- pricing S1765's premise site by site.
#
# §1977: the 36-site substitution costs 2.63-2.98 nats, and the whole §1946-§1976 arc recovered 0.069 --
# 2.4% of it. The price sits in the COVERED arm (+2.57 to +2.89 nats even where the program has an exact
# table), so it is not the fallback's: having the right table is worth ~0.3 nats, having attention ~2.7.
# Every section since §1746 has optimised inside that premise and none has priced it.
#
# This restores context to subsets of sites and measures what each buys back. It is the first experiment
# in thirty sections that asks what the program is GIVING UP rather than how cheaply it can give it up.
#
# ARMS, all on the converged build's rows at {mlp 768, attn 384} with the rank-640 blend fallback:
#   all36     substitute everything -- the compiled program (§1765)
#   mlp_only  substitute the 18 MLP sites; the 18 attention sites run LIVE
#   attn_only substitute the 18 attention sites; the 18 MLPs run live
#   late_half substitute layers 9-17 only, both kinds -- the early half runs live
#   early_half substitute layers 0-8 only
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419. Rung 3 -- §1977's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
C = 'c5419'
ARM = 'mix30m640'
MLPS = [('mlp', L) for L in range(18)]
ATTNS = [('attn', L) for L in range(18)]
LATE = [(k, L) for k in ('mlp', 'attn') for L in range(9, 18)]
EARLY = [(k, L) for k in ('mlp', 'attn') for L in range(0, 9)]

PLAN = [(ARM, A384, 'all36', None),
        (ARM, A384, 'mlp_only', MLPS),
        (ARM, A384, 'attn_only', ATTNS),
        (ARM, A384, 'late_half', LATE),
        (ARM, A384, 'early_half', EARLY),
        ('map512', A384, 'spec_partner', None)]      # same spec as all36: keeps the control two-sided


def _attention_carries_the_price(x):
    """substituting only the ATTENTION sites costs more than substituting only the MLPs -- §1977 inferred
    that attention is worth ~2.7 nats and the tables ~0.3, and this measures it directly. If FALSE the
    inference from the covered/uncovered split was wrong about which component the price belongs to"""
    return x.count(lambda c, r: x.ce(c, r, 'attn_only') > x.ce(c, r, 'mlp_only'))[C] >= 2


def _partials_are_far_cheaper_than_the_whole(x):
    """and every partial substitution costs less than half of the full one -- if the 36-site cost were
    additive and uniform, each half would cost about half. Anything much below half means the sites
    interact and the last ones substituted are the expensive ones"""
    base = 'all36'
    def frac(c, r, a):
        live = x.res[c][r][a]['pooled']['overall']['ce_live']
        return (x.ce(c, r, a) - live) / max(x.ce(c, r, base) - live, 1e-9)
    return x.count(lambda c, r: max(frac(c, r, a) for a in
                                    ('mlp_only', 'attn_only', 'late_half', 'early_half')) < 0.5)[C] >= 2


def _late_costs_more_than_early(x):
    """and substituting the LATE half costs more than the early half -- §1891/§1908 found the tracking
    that the program destroys is late attention, so removing context late should hurt more"""
    return x.count(lambda c, r: x.ce(c, r, 'late_half') > x.ce(c, r, 'early_half'))[C] >= 2


B.run(
    name='what_the_program_gives_up',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_attention_carries_the_price',
         'substituting attention alone costs more than substituting the MLPs alone',
         _attention_carries_the_price),
        ('pred_b_partials_are_sublinear',
         'and every partial substitution costs under half the full 36-site price',
         _partials_are_far_cheaper_than_the_whole),
        ('pred_c_late_costs_more',
         'and the late half costs more than the early half, as §1891/§1908 imply',
         _late_costs_more_than_early),
    ],
    refs=[('all36', B.PT + 'ops/what_we_built_results.json', 'converged', C, 0.0005)],
    paired_pairs=[('mlp_only', 'all36'), ('attn_only', 'all36')],
)
