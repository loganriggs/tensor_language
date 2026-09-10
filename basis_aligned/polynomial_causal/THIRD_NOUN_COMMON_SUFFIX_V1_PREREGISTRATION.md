# Common mature positions before cross-layout interchange

The causal role audit proves that the first possible mixed write occurs at the
attractor in the original layout, but at the object in the fronted layout. Swapping
those writes by noun role can inject information before it is available. The two
shared mature roles are `to` and the final action word. Their positions are fixed
from token roles and causal support, before outcomes; no position search occurs.

Use all16 original and all16 fronted worlds from THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.
At layer9 capture the frozen local-value-removal write difference D and calculate
Dm=Qoh D as before. Split Dm=C+N, where C retains the `to` and action positions
and N retains every other position. C is the candidate common suffix interface;
N contains the last-evidence noun position. No across-layout transfer is done yet.

Run baseline/local-value capture, then three full native runs subtracting Dm, C,
or N at attention output. All positions are represented, QK/sharedfirstV retained,
native downstream model recomputed.10forwards/world×32=320forwards5120sequence
instances, no fits,600s cap. No fresh text or OOD claim: this is a new causal
position partition on opened original/fronted cases.

A instrument: bound inputs, finite output, restored hooks, exact320/5120counts;
baseline and full-Dm three-reader grids replay each world's previous factorial
or structure result with BOTH maxabs<=1e-3 and relativeFrobenius<=1e-5. Inherited
value controls (weight commutation<=1e-4, early support<=1e-6, unchangedQK/firstV)
pass. C+N=Dm and both are pure mixed, relative FP64 errors<=1e-10; installed
full-Dm impurity<=1e-4. Incoming patched attention outputs match baseline bitwise.
Zero scientific effect fails the relevant gate, not instrument validity.

All scientific gates require EVERY world separately, with layout summaries:
- B common-interface fidelity: ||Q(E_Dm-E_C)||/||Q E_Dm||<=.10 for the correct
  answer margin. A passing B licenses a later common-position interchange test,
  not independent extraction or semantic invariance.
- C last-evidence-noun fidelity: the same relative bound for E_N. This is an
  independent alternative hypothesis; do not select by whichever error is smaller.
- D composition: ||E_Dm-E_C-E_N||/||E_Dm||<=.10 for both correct-margin and
  centered three-reader full tables.
- E common-interface factor selectivity: ||(I-Q)E_C||/||QE_C||<=.25.
- F common-interface gender: ||Q deltaG||/||Q deltaN_number||<=.25, where
  G=z_himself-z_herself, N_number=z_themselves-(z_himself+z_herself)/2.
Zero denominators fail. Report signed projection on native interaction as well;
no branch is promoted as explaining native behavior simply by passing relative
fidelity to the partial parent effect.

If B fails, common suffix alone does not preserve the full component under the
frozen fidelity bar, so do not proceed with an unqualified suffix-only swap. If B
passes but E/F fail, the interface can still support a diagnostic interchange,
with those semantic limitations explicit. D failure requires joint interpretation
of the two position branches. No altered positions, role mappings, gains or rows.
All545902902weights and native four-corner input production remain; savings0.
