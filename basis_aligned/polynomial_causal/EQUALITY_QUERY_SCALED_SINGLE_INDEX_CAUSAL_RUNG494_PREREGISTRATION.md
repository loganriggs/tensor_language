# Rung494 preregistration: causal scaled-intervention test of the single-index composition law

Status: frozen before rung494 model outcomes.

## Decision and circuit target

The three equality-query MLP components at layers8,9,12 have non-additive joint effects. A CPU leave-one-pair-out
screen recorded before this run found that, at each selected query occurrence, a monotone function of the sum of the
three singleton effects reduced held-out pair-effect median error by22–39% relative to direct addition in all six
window×source cells. That result used only the seven already observed all-or-nothing subset interventions and may be
interpolation over eight points.

Rung494 asks whether the same fitted function predicts genuinely new physical interventions in which one component's
query-product change is applied at half strength or1.5× strength. A pass advances computational specification and
composition: the three components would act through one scalar index followed by an input-dependent monotone readout.
It is not a rank, sparsity, reconstruction, compression, or adoption test.

## Frozen inputs and scope

- Reuse rung474's three windows: code_validation, natural_wave0, natural_wave1; sources N and H; the frozen selected
  query positions; sites m8,m9,m12; and the subtractive product coordinate.
- Recompute every baseline and all seven unit-strength subset effects inside this process because §2599 forbids
  sub-.1-nat bridges to old processes.
- The new test arms are each singleton site at `lambda in {0.5,1.5}`. At a selected query position the physical hook
  replaces product `p` by `p-lambda*delta_i`, where `delta_i` is the in-process intact-source minus absent-source
  product change for site i.
- No validation/odd-root/SEALED family is opened. This is new intervention-type evidence at already selected query
  occurrences, not held-out documents, new-corpus OOD evidence, or a semantic circuit label.

## Frozen predictor

For one query occurrence and source, let `e_i` be the measured unit-strength singleton CE damage for site i. For each
of the eight subsets S, including the empty subset, define

`x_S = sum_{i in S} e_i`, and `y_S = measured CE damage of the physical subset S`.

Fit one non-decreasing isotonic regression `h_t` to the eight `(x_S,y_S)` points with equal weights. Duplicate x
values use the library's deterministic aggregation. Prediction outside the fitted x range is clipped to the nearest
endpoint (`sklearn.isotonic.IsotonicRegression(increasing=True,out_of_bounds="clip")`). For a new scaled singleton
arm `(i,lambda)`:

- single-index prediction: `h_t(lambda*e_i)`;
- additive baseline: `lambda*e_i`;
- position-permuted control: apply the isotonic function belonging to the query occurrence shifted by each of the16
  frozen document-position offsets, while retaining the current occurrence's `lambda*e_i` input.

The primary error is the median absolute per-token difference from the newly measured physical scaled effect. The
same computation is reported separately for every window, source, scale, and48-document half. Pearson correlation,
RMS, in-range fraction, and absolute errors are descriptive companions, not replacements for the frozen median bars.

## Frozen predictions

### A. Exact, live scaled-intervention instrument

All must hold:

1. checkpoint and pinned parent/review hashes match;
2. native replay relative squared error≤1e-12, factor reconstruction≤1e-10, and an empty query mask changes logits by
   exactly0;
3. the newly implemented `lambda=1` singleton path agrees with the ordinary in-run unit singleton effect at maximum
   absolute CE error≤3e-5 in every cell;
4. every half-strength and1.5× site arm has nonzero RMS, and the two scales differ by RMS≥1e-4 for every site/cell;
5. exact4,266 full-model forwards and5,634 product-hook calls occur. The extra18 forwards are one independently
   implemented unit-strength bridge per window×source×site on the first batch only.

### B. Half-strength causal interpolation

For every one of the six window×source cells at `lambda=.5`:

1. the additive baseline's median absolute error is at least1e-4 nat, so a relative improvement is measurable;
2. `median_error(single_index) ≤ .85 * median_error(additive)`; and
3. `median_error(single_index) ≤ .90 * q05(median_error(position-permuted controls))`.

### C. One-and-a-half-strength causal transfer

The identical three clauses hold in every cell at `lambda=1.5`. This is scored separately so a B-true/C-false result
is explicitly an interpolation-only law rather than a general causal composition rule.

### D. Document-half stability

In both fixed48-document halves of every window×source×scale cell, single-index median error must be≤.90 times
additive median error. This does not create a new held-out-data claim; it prevents one half of the selected positions
from carrying the pooled result.

### E. Conditional interpretation-outcome validation

E is true iff A–D are all true. No additional outcomes are opened. It licenses only the statement that a locally
fitted scalar readout predicts two new intervention strengths at the same query occurrences. It does not license a
shared readout across tokens, OOD generalization, extraction, semantic naming, or compression.

## Opposing outcomes and routing

- **A–D true:** the single-index law survives a physical causal test. Next fit a shared, simpler readout on one
  document/corpus split and predict the other corpus and the existing62 circuit signatures; require selective
  composition and unrelated-circuit preservation before extraction language.
- **B true, C false:** the law is interpolation-only at this grain. Retain it as a local response-chart fact and move
  to exact attention1 QK1×QK2×OV grouping by downstream use.
- **B false:** the eight-point result does not predict new physical interventions. Close the single-index route at
  this grain and move to the exact attention1 decomposition.
- **A false:** instrument failure only. Repair only the failed instrument clause without reading science as evidence.

Strong null: A is false, B is false, C is false, or D is false. Validation/SEALED remains closed in every case. Frozen
thresholds are not relaxed and failed cells are not dropped.

## Literal price

Per batch: native+replay+absent=`3`; per source, source+empty plus two selected-position slots times
`(7 unit subsets + 6 scaled singletons)`=`28`; two sources give`59` forwards. Across72 batches this is4,248, plus18
first-batch unit-strength bridge forwards, for exactly4,266 full-model forwards. Product-hook calls are78 per batch
plus18 bridge calls,5,634 total. Deployed values saved/added=`0/0`.
