# Fresh CrossFirst two-stage Hessian composition test V1

## Frozen candidate

The opened-panel per-layer causal-Hessian experiment identified a stable
exploratory pair: the local mixed Hessians at **MLP10** and **attention17**.
Their summed downstream contractions reduced the child-relative composition
residual below 10% in all four opened families, and leave-one-family-out energy
ranking selected the same pair four times.  Freeze that pair now, before any
model outcome on `CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ROWS.json` is evaluated.

For each row, use the existing exact CrossFirst child and remainder scalar
fields and fixed head9 writer.  At the post-attention9 boundary set
`a=-child*w` and `b=-remainder*w`.  Propagate both first tangents through the
native suffix.  Compute every stage's nested-JVP mixed term for audit, but the
candidate correction is only the sum of the MLP10 and attention17 terms after
their exact native suffix pullbacks.  Predict the physical joint-removal effect
as the sum of separately measured child and remainder effects plus this frozen
two-stage correction.

The 48 score-blind rows contain four new templates and four city pairs and had
zero full-prefix overlap with every repository `*ROWS.json` at construction.
The six spelling endpoints are existing.  This is construction OOD, not corpus
OOD, and native upstream states and the complete suffix remain charged ports.

## Frozen comparisons

Four two-stage nulls are fixed independently of outcomes:

- `attention11 + mlp13`;
- `mlp9 + mlp12`;
- `mlp15 + mlp16`;
- `attention13 + readout`.

No stage, coefficient, sign, or scale may be changed after opening the panel.

## Predictions

- **A -- instrument:** all 24 native British/American pairs have the expected
  contrast sign; baseline replay and the complete 18-stage Hessian allocation
  close within `2e-6` and `2e-5` relative error respectively.
- **B -- fresh composition:** in every 12-row family, the frozen two-stage
  prediction leaves at most `0.10` of the child-effect norm.
- **C -- finite-interaction fidelity:** in every family, the two-stage term has
  cosine at least `0.90` with the physical finite interaction and relative L2
  error at most `0.75`.
- **D -- stage specificity:** the candidate's worst family child-relative
  residual is at least `0.02` lower than the median null pair's worst-family
  residual.
- **E -- context and control audit:** serialize every prompt/stage/reader term
  and both physical readers; report target and unrelated-control errors without
  dropping small denominators or sign exceptions.

Only A--E together license the frozen two-stage Hessian correction as a fresh
composition rule.  Passing does not make MLP10 or attention17 standalone
semantic circuits: the rule still depends on the native child/remainder input
ports, per-context derivatives, and native suffix.  Failure is preserved and
does not invalidate the exact full-Hessian allocation on the opened panel.

## Price

One checkpoint load; 48 prefixes; one native upstream trace per prefix; four
physical suffix arms per prefix; one actual-writer 18-stage nested-JVP
allocation per prefix; no fitted coefficients, optimization, backward-to-weight
gradients, parameter updates, or fresh-panel selection.
