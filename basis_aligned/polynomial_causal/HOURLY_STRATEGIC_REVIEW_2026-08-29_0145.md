# Hourly strategic review — 2026-08-29 01:45 UTC

## Bottom line

The honest whole-model ledgers have not moved:

| Question | Currently established | Remaining gap |
|---|---:|---:|
| Structural sites represented | 36/36 | formulas exist, but most are not semantic explanations |
| Whole-program storage certified removable | 5.3481% | 94.6519% not certified removable |
| Strict named causal CE recovery | 10.923% | 4.72714 nats, or 89.077%, remains unnamed |
| Final extraction/removal/OOD actions closed | 0/68 | semantic reducer and final consequence replay remain open |

Here **CE** is next-token cross-entropy.  “Strict named causal recovery” measures how
much of a fixed loss increase is restored by named interventions.  It is not the same
denominator as storage removal, local reconstruction, or descriptive behavioral
coverage, so these percentages cannot be added.

The best current mechanistic unit is not an isolated late MLP.  Attention and its
following bilinear MLP interact and compensate, so the smallest justified local
compiler unit is the coupled attention--MLP block with its residual/RMSNorm interface.

## What changed

### The cost baseline improved, but explanation coverage did not

The independent bilinear-quotient stream completed an equal-storage comparison.  At a
roughly 230.087-million-value budget, lower-rank tables with broader token coverage beat
the deployed full-rank 5,419-type table by about 0.042--0.080 CE nat across three roles.
The gain is already nearly saturated around 11,954 covered types: that point lies within
0.0022--0.0061 nat of the 14,405-type point.  This gives future compiled programs a more
honest cost-matched baseline.  It does not name a circuit or move a causal ledger.

### The next mechanistic experiment now charges executed products

The prior typed reduced-rank regression was demoted to a diagnostic control.  Although
it can reduce a coefficient matrix, it still evaluates all 4,608 native bilinear
products and retains the full native `Down` bank.  It is therefore not a simpler
executable MLP under the cost we actually care about.

The replacement now being frozen selects one shared subset of native product gates at
Block 3.  For the exact RMS-polarized inputs

\[
u=\gamma(h+a)h,\qquad v=\gamma(h+a)a,
\]

the four diagnostic terms are

\[
W_{pq}=D[(Lp)\odot(Rq)],\qquad(p,q)\in\{u,v\}^2.
\]

One gate set of size \(K\in\{256,512\}\) and one fitted decoder are shared across all
four terms.  The deployable all-term program does **not** execute four banks.  It first
forms \(u+v\) and evaluates

\[
\widehat D_S[(L_S(u+v))\odot(R_S(u+v))]+b,
\]

which costs exactly \(K\) products.  The four-bank form is used only to ask which typed
interaction fails under intervention and receives no deployment-cost credit.

The literal program price is

\[
3\cdot1152\cdot K+1152
\]

float32 values plus \(K\) integer indices, versus the same expression with
\(K=4608\) for native MLP3.  Reports also include literal bytes, products per token,
and linear multiplies per token.

### Why this is more than a local reconstruction fit

The fit uses 92,160 positions from 480 documents, with disjoint 192-document validation
and final roles.  The same selected gates must explain all four typed writes.  The
later evaluation will run all 16 native/replacement term masks, matched omission
stakes, a sign-mirrored error, final KL/CE/top-1 effects, and document bootstrap bounds.
This distinguishes:

1. a locally accurate, composable smaller port;
2. a locally inaccurate error that downstream blocks happen to cancel;
3. one-sided compensation that should not be called a safe removal;
4. failure of only this native-gate-subset grammar.

The collector and deterministic fitter were adversarially reviewed before launch.  The
first review found real blockers: float64 programs that could not consume float32 model
states, incomplete dependency closure, metadata-only upstream integrity, and missing
late-drift tests.  These are fixed.  Exact committed blobs, checkpoint and row receipts,
embedding-plus-Blocks-0--3 tensor content, MLP3 factors, and physical call counts are
bound before and after collection.  Create-only, receipt-last publication is tested
against late source, row, checkpoint, and payload drift.  The current focused suite is
35/35 and the independent final verdict is GO.

## Largest remaining gaps and confusing evidence

1. **Dense coefficients, possibly sparse usage.**  Raw Block-3 coefficient HOSVD needs
   rank 630 for 95% energy and rank 512 still has 0.312 relative error.  This does not
   tell us whether natural text uses only a small shared subset.  The present assay is
   the first direct test of that distinction.
2. **Average CE is simpler than conditional editing.**  About 98% of the completed
   five-action cube's CE variance is degree one or two, but the small high-order tail
   contains the conditional compatibility effect.  A compressor can look excellent in
   average CE while deleting the circuit one wants to extract or remove.
3. **Downstream cancellation is real.**  MLP2 and later attention can compensate for an
   early replacement.  Low final CE alone is insufficient; local port error, mirror
   error, mask recovery, and composition must be reported together.
4. **The 68 final actions remain scientifically unopened.**  Physical capture exists,
   but objective comparators, uncertainty semantics, and terminal replay still need a
   reviewed producer.  Until then no extraction/removal/OOD claim is closed.
5. **Current “OOD” roles are held-out documents, not a new distribution.**  They test
   document transport and doubling-data stability, but not genre, language, or
   algorithmic distribution shift.

## Pruned moves

- Raw coefficient HOSVD/CP is not next: its registered low-rank gate failed.
- More scalar norm balancing is not next: it changed the weighted norm by only 0.024%.
- Typed reduced-rank regression is not a deployable simplifier because it retains all
  products; it remains a nonpromotive diagnostic.
- Independent late-MLP replacement is contradicted by attention compensation.
- Generic SAE/dictionary learning on raw weight columns is not yet justified.  A sparse
  dictionary becomes meaningful only when charged for execution and tested through a
  shared producer-consumer consequence interface.
- Global worst-case Lipschitz certificates are likely vacuous before a candidate passes;
  local finite-horizon certificates come afterward.

## Ranked top five

1. **Run the audited Block-3 shared native-gate subset assay.**  Highest information
   gain: it directly tests whether activation use makes a dense tensor program sparse,
   charges real products, has an exact composable port, and can fail sharply at moderate
   GPU cost.
2. **Close the 68-action semantic reducer.**  Highest direct relevance to extraction,
   selective removal, and transport, but more engineering-heavy and partly downstream
   of having candidate programs worth comparing.
3. **Prospectively replay the frozen sparse action spectrum at an adjacent cut.**  Cheap,
   falsifiable test of whether the low-degree action law composes rather than merely
   describing one completed cube.
4. **Fit a joint downstream-weighted MLP0/MLP1/MLP2 dictionary or DAG.**  This is the
   proper place for sparse lexical parts and shared hierarchical parents, but it should
   use the consequence/cost lessons from the Block-3 assay rather than optimize raw
   weight reconstruction alone.
5. **Add local incremental-quadratic finite-horizon certificates.**  Potentially gives
   certified composition and safe edits after a replacement passes; it cannot discover
   the replacement itself.

Ranking uses expected information gain, causal relevance, whole-model composability,
falsifiability, GPU cost, and redundancy with completed work.  Priority 1 dominates
because it tests the missing simplicity principle rather than another local error
curve.

## Action executed this hour

The full native-gate protocol, collector, deterministic fitter, executable float32
program, pricing, provenance guards, and adversarial lifecycle tests were implemented
and independently re-audited.  The audit returned GO to commit, push, and collect.
The shared GPU was still occupied by the independent low-budget coverage run at the
time of writing; source freezing and publication proceed now, and collection starts as
soon as that job releases the device.  No global ledger receives credit before held-out
evaluation.
