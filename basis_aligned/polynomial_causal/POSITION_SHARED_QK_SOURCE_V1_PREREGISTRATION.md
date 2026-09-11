# One QK source space across positions — 11 September 2026

Select one rank17 source space per attention17 head from weights, holding the
source-reader identity fixed across relative positions. This extends the prior
two-distance source spaces rather than proposing a new semantic head label.

Query position511. Discovery source positions0,2,...,510 (256 odd distances).
Validation sources1,3,...,509 (255 even distances), excluded from primary frame
selection. All are distinct query/source positions; no self-position calculation.
Use the actual BF16-rounded RoPE tables in the coefficient maps. No text data.

For each position, form the exact source influence S with trace equal to that
position's joint QK numerator energy. Average S/trace(S) over discovery positions,
then take17leading eigenvectors in raw source coordinates. This optimizes the
mean influence surrogate exactly, not the nonlinear touch objective. Report
the associated mean-touch upper bound, true inside/mixed/outside energies on
both splits, and the smallest validation touch.

Position-specific comparison: select the17leading influence directions separately
at32sources given by rounded linspace(0,510,32). Compare mean exact touch on this
fixed grid with the frozen common source space. This is a spectral comparator,
not a certified globally optimal per-position touch baseline.

After scoring the primary frozen frame, independently select a frame from the
validation-position influence mean for a split-stability diagnostic. This second
frame never replaces the discovery frame in the primary validation scores.

Predictions:

- A: existing dense/analytic controls pass; batch energies, mean-influence trace,
  basis orthogonality and selected direct contractions agree within1e-10.
- B: for every head, the common frame retains at least90%of the separate-frame
  mean touch on the32fixed positions.
- C: for every head, validation mean touch>=.45 and minimum touch>=.30.
- D: every discovery/validation frame pair has mean squared principal cosine>=.95.

Any miss is scoped to these fixed spaces, positions and numerator metric. Inspect
head-resolved results and compare shared versus separate frames before attributing
it to absent structure. The influence-selected frame is not necessarily the
best true-touch frame. Mixed dependencies and all query/key norm factors remain
explicit; no normalized routing or whole-model replacement is inferred.

Price:0model-body forwards, no corpus,9heads,511source positions, rank17. Managed
GPU lane1, FP64, TF32off; source spaces saved in /dev/shm. No iterative solver or
convergence claim. Fixed source storage9*1152*17numbers; query-position operations,
normalizer and value ports remain additional required computation. This tests
position reuse, not the four-property behavioral goal by itself.
