# Plain-English plan — 2026-09-01 01:54 UTC

(Damage is extra next-token cross-entropy above the native model; **lower is better**.)

## Our goal

We are compiling the 546M-parameter bilin18 model into a much smaller, explicit tensor program.  A
successful program must satisfy four requirements at once:

1. **Predictive:** it matches the native model on held-out census rows, fresh documents, and genuinely
   different text distributions.
2. **Composable:** its replacements still work when installed together; local wins cannot hide destructive
   interactions.
3. **Manipulable:** named removals, swaps, and edits produce the same signed causal changes as in the native
   model.
4. **Literally simpler:** stored values, executed multiplies, state, edges, tables, and routing are counted.
   A large lookup table or hidden downstream correction is not an explanation.

The current milestone is to choose a faithful adoption base on the corrected attention path, then make its
prediction, causal behavior, and exact price survive broader falsification.  The end goal remains a small
glass-box program whose failures are predicted by its stated parts.

## Why the next move changed

The old `+0.055` attention floor was mostly an invalid primitive: a context-blind token table, `a1v`, was
standing in for block 1's context-dependent native value map.  Removing only that table made the full-rank
replacement path effectively exact.  The corrected ct96 program then reached:

- census damage `+0.0034195`;
- `56/62` circuit certificates;
- zero to four printed decimals on eight fresh windows;
- near-native signed m16 intervention effects;
- approximately 154.4M stored values after replacing the 57.9M-value table by a 1.33M-value matrix.

That changes the central question.  We no longer need a learned repair for an alleged global score-map
mode.  We need to rebuild the Pareto frontier using the faithful primitive and determine how much genuine
QK rank, causal fidelity, and compute the program needs.

## The executing plan

### 1. Test the corrected mixed-spectrum point

The old best-value point used rank-96 plus the smallest eight score directions throughout the replaced
attention stack.  It cost about 180M values and measured `+0.0573`, but it carried the same `+0.0520`
`a1v` error.  Removing that instrument predicts

$$
0.0573-0.0520 \approx 0.0053
$$

damage and removes `56,568,960` stored values, suggesting an approximately 123.4M-value program.  Rung 290
is now executing this configuration cleanly rather than subtracting old receipts.  It is preregistered to
pass only if census damage is at most `0.012`, at least `50/62` certificates survive, and all eight fresh
windows are at most `0.020`.  A physical rank-shape tripwire verifies that all intended QK maps really use
rank 96.  Damage at least `0.040` is the opposing result: a genuine pattern-compression floor remains.

This test has the highest immediate value because a success would be both *smaller* and almost as accurate
as corrected ct96.  A failure would cleanly falsify aggregate path subtraction and keep ct96 as the base.

### 2. Turn an approximate price into an exact bill

The `123.4M` and `154.4M` totals inherit rounded historical anchors.  Before adoption, every component will
be counted from shapes and multiplicities: embedding tables, native matrices, factored QK maps, retained
MLP factors, routing state, and any class/dictionary tensors.  Storage and executed compute will be separate
ledgers.  The candidate must dominate under the exact bill, not merely under a rounded headline.

### 3. Stress prediction and composition

The current fresh-window report rounds to four decimals, so the winner gets a high-precision fresh/OOD
receipt on disjoint documents and at least one shifted corpus.  We will also install the established
per-token MLP top-k rule on the corrected base.  The old evidence predicts an approximately `+0.016`
compute-sparsity surcharge; the clean run tests whether that additivity survives after removing `a1v`.
This yields an explicit storage/compute choice rather than conflating the two prices.

### 4. Use attention 16 as the causal falsifier

The corrected ct96 program already passed a strong signed m16 test, but m16 is only one intervention.
Attention 16 is the best next falsifier because it was the lone outlier in the earlier six-component
battery: its collateral ranking transferred much worse than the others.  We will compare the direct signed
effect vectors

$$
\Delta_{a16}^{\mathrm{compiled}}
=\mathrm{CE}(g+\mathrm{KO}_{a16})-\mathrm{CE}(g)
$$

and the corresponding native vector, rather than subtracting unsigned summaries.  If the corrected base
restores a16, the old anomaly was another consequence of the bad path.  If it does not, it is a genuine
causal limitation and becomes the target for a tangent-aware compiler.  Only after this discriminating test
do we expand to the full signed intervention battery.

## Ranked alternative directions

The immediate corrected-frontier route is cheapest and most informative, but it is not the only route.
These are materially different objects, not variants of a rank sweep.

1. **Tangent/Sobolev compiler.** Match native outputs *and* derivatives along registered removal and swap
   directions.  This becomes first priority if a16 or the broader signed battery fails despite low ordinary
   CE.  Kill criterion: held-out intervention magnitudes do not improve at equal prediction damage, or the
   derivative fit fails when replacements compose.
2. **Causal-response coordinates.** Order directions by preserved signed circuit/logit response per stored
   value, rather than by weight singular value.  This becomes attractive if corrected rank-96 prediction is
   cheap but specific certificates remain the bottleneck.  Kill criterion: no Pareto gain over ordinary SVD
   on held-out circuits and documents at equal literal price.
3. **Predictive-state causal quotient.** Stop imitating each native module and identify hidden states only up
   to their future logits and controlled interventions.  Prefix/continuation Hankel rank provides the lower
   bound and candidate state dimension.  This is the most radical route and can bypass redundant native
   coordinates.  Kill criterion: numerical rank grows with sample size, shifts across corpora, or fails under
   controlled interventions.
4. **Exactness contracts and lower bounds.** Treat every replacement as carrying an executable error
   contract, propagate signed bounds through the tensor program, and compare the empirical frontier with
   information-theoretic or Hankel-rank lower bounds.  This can tell us whether another 2x compression is
   structurally plausible before launching a search.  Kill criterion: propagated bounds are too loose to
   predict certificate or CE failures, or the lower bound already meets the current bill.

The former shared-invariant/rank-one-floor route is retired unless new clean-path evidence revives it.  The
previously proposed downstream output-state patch remains parked because it would conceal an exact, smaller
primitive fix.

## Decision order

The sequence is therefore:

1. corrected mixed-spectrum prediction/certificate run;
2. exact component bill and precise fresh/OOD receipt;
3. signed a16 falsifier;
4. corrected compute-sparse composition and broader signed battery;
5. adoption of the best faithful point, or a documented falsification followed immediately by the highest-
   ranked alternative above.

At each elapsed hour the research driver steps back, restates the full four-part goal, audits instrument and
composition confounds, re-ranks these alternatives, records material direction changes, and then takes the
next concrete step.  A completed rung is a decision input, not a reason for the research process to stop.

## First two results

The first corrected run measured `+0.00853845` census, `52/62` certificates, and eight fresh-window damages
between `-0.0061` and `+0.0052`.  A subsequent code audit caught a label mismatch before adoption: the
physical factors were contiguous top 96, whereas the historical mixed-spectrum point used top 96 plus the
smallest eight directions.  The prediction/certificate/fresh receipt remains valid for the top-96 program,
but its scalar comparison to the mixed receipt and the 123.4M mixed-anchor headline are void.  The true
104-direction companion is now registered with an exact index-set tripwire and no subtraction bar.

The signed a16 falsifier then held decisively: effect cosine `0.992605`, normalized error `0.133050`,
collateral circuit Spearman `0.997347`, and own-family magnitude ratio `1.053832`.  Those causal numbers are
for the physical top-96 program and remain valid; the old a16 anomaly was path-contaminated.  The active
compute rung composes per-token top-1152 MLP execution with top96, while the true mixed companion, exact
component pricing, and shifted-corpus OOD remain required before final adoption.
