# Middle-entity matching in the two directional join writers

The head/port experiment passed all4 predicates. L2H1 writes the forward value
(B2 later); L2H2 is the primary backward-key writer (B3 later), with a smaller
H3 contribution preserved in the results. V-only restoration recovers the forward
effect; K1+K2 restoration recovers the backward effect. All-port query replay is
bit-exact. This still does not show how the writing heads recognize composable facts.

Test the shared middle entity in B2=(b,c), B3=(c,d), with B1=(a,b) and query(a,3).
Choose a foil t away from the first3 predecessor/successor steps around c. Swap
c↔t independently in the binding-key column and the binding-value column:

| Case | B2 value | B3 key | Old pair joins? |
|---|---|---|---|
| base |c|c|yes|
| keys |c|t|no|
| values |t|c|no|
| both |t|t|yes|

Each swap applies globally within that column, so keys and values remain unique;
all four inputs describe valid permutations. Both swaps together are a global
entity renaming (conjugation), leaving a,b,d and the correct answer unchanged.
Each single swap gives the same different correct answer f(t). Excluding the nearby
cycle neighbors makes the modified cycles length>=4; this is a predefined structural
domain, not native-capability filtering. Other occurrences of c/t are changed too,
so report this as a valid permutation intervention, not an isolated two-token edit.

Fresh16 IID24-cycle worlds seed14909 and16 two12-cycle worlds seed14910. Same
counterbalanced6 chain orders at slots3/11/19; each order has all4 rename cases,
hop3 cold query only.32 independent worlds,768 correlated variants. Freeze the
original S→J edge mask and orientation across the four cases. Use L2H1 forward,
L2H2 backward, fixed from the preceding experiment; no new head selection.

A: parser checks guarantee permutation validity, conjugation, expected answers,
unchanged query and source positions, and source-mask identity. Native correctness
>=.8 in each population/orientation/case; record all examples without filtering.
B: each mismatched case's RMS native **joint product-attention score** over the
four S→J cells is<=.10 times the smaller matched-case RMS, in each population/
orientation. Both matched RMS values must exceed1e-6. No individual QK-factor
semantics are assumed, and no normalization or RoPE factor is removed.
C: the same selectivity holds for the centered full29-way query-logit change
caused by removing those selected-head S→J edges: each mismatch RMS<=.10 times
the smaller matched RMS, with both matched effects live (>1e-6). This makes the
score gate causally meaningful rather than relying on attention visualization.
D: both matched cases have selected-route mean gold-probability loss>=.25 in
both populations/orientations. This is a nontrivial-use gate on the new equality
intervention, separate from the prior>=.5 head-localization test, whose result stays
unchanged. Preserve signed effects and all per-row native/cut logits and score cells.

If B/C fail, the fixed middle-equality account is not established: do not rename
native factors, relax margins, or sweep token classes. If they pass, extract a
joint read-route-write implementation and test replacement fidelity, rather than
counting another localization as a compact program. All400640 native parameters
remain charged. Batch4 FP64,1800s,<256MiB/tensor, GPU only through managed enqueue.
