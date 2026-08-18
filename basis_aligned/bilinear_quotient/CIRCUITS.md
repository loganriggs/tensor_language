# The circuit program (standing loop)

The benchmark's second strategy, run as a continual loop rather than a
finite arc. Layer-by-layer replacement found its ceiling (BENCHMARK.md);
circuit-by-circuit has no ceiling: every well-predicted token slice the
model owns is a candidate circuit, and every certified circuit is amortized
knowledge that cheapens the next one (sparse edges between understood
components make neighbors easier to read).

## The loop (each iteration = one circuit)

1. **SLICE**: pick a set of tokens the model predicts well (low base CE)
   that cluster together in damage space — they are hurt by the same small
   set of components. Discovery instrument: cluster the per-token columns
   of the fingerprint atlas (36 components x 16k tokens).
2. **LOCALIZE**: name the owning components (top damage share) and the
   edges between them (kinship margin + interchange, causal numbers only).
3. **HYPOTHESIZE**: write the plain-language story — what the slice has in
   common, what each component contributes, in what order. Register it
   with predictions before any scoring run.
4. **CERTIFY**: Track-1 score on the slice (fingerprint match in declared
   regime, floors and nulls per BENCHMARK.md); Track-2 leg where possible
   (replace the named components with the hypothesized simple function, or
   replace everything else and keep the circuit, measured on the slice).
5. **FOLD IN**: certified circuit -> BENCHMARK.md scoreboard; its
   components' stories become priors for neighboring circuits and for the
   layer-track's stand-ins.

## Progress metrics (reported each iteration)

- certified circuits (count) and their slice sizes
- COVERAGE: fraction of well-predicted tokens lying in a slice with a
  certified circuit (the number that should climb wake over wake)
- refuted/floor-grade stories (kept on the scoreboard -- honest misses)

## Rules inherited from the ledger

Registered predictions with controls and nulls; measure floors before
bars; pooled aggregation; margin-not-rank for edges; causal numbers for
certification; two instruments before a verb; every certification gets a
fresh-data leg before entering the scoreboard.
