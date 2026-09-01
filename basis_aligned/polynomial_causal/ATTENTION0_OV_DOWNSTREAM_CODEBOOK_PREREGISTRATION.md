# Attention0 OV downstream-codebook preregistration

Date: 2026-09-01 18:50 UTC

## Claim and why this is not another head-clustering run

This is a **gauge-invariant sub-head identification screen**, not compression or semantic certification. The unit being
modeled is the exact output-projected token payload

`P_h(t) = O_h V_h RMSNorm(embedding(t))`,

not an architectural head's complete context-dependent output. The head label remains only a producer/provenance
index. One code atom may group payloads from several heads and several tokens.

The registered question is: after restricting to attention0's already-validated task-useful rank-16 routed-write
space and measuring distance by the exact finite responses of MLP0 and attention1 readers, is there one shared sparse
payload codebook across heads, or does every head require a private codebook?

No result here licenses a semantic feature name, source equivalence, physical program, or compression claim.

## Dossier constraints

The assay must not duplicate or contradict these established facts:

1. attention0 raw values are exactly token-indexed on vocabulary support;
2. substituting one all-head `O V` product is invalid because heads have different QK routing patterns;
3. the honest routed `c_proj` input has a task-useful rank-16 output surrogate costing about `+0.0943 nat`, whereas
   matched random rank-16 directions cost about `+1.33` to `+1.41 nat`;
4. recursive spectral clustering of the whole routed write found no subdivision;
5. attention0 context causally drives a named MLP0 article computation, and attention1 can reread that evidence.

Accordingly, all payloads remain per-head until after their own QK scalar has acted. This first rung studies the OV
payload vocabulary; the next rung may multiply it by the already folded double-QK scores.

## Frozen data separation

- `FIT`: FineWeb documents `[0,96)` from the existing diverse census state.
- `SELECT`: FineWeb documents `[96,192)`.
- `FINAL`: remains unopened.
- Vocabulary fitting tokens: token IDs with `id mod 5 != 4`.
- Vocabulary evaluation tokens: token IDs with `id mod 5 == 4`.
- Natural positions: positions `16,32,...,240` in each document.

All rank bases, response scales, consumer normalizers, code centers, and per-head center allocations use FIT only.

## Exact objects and arithmetic checks

For every token and head, compute `P_h(t)` directly from the checkpoint in float64 chunks. Verify against live
attention0 value and output-projection slices on at least 256 frozen token/head cases, maximum absolute error
`<=1e-10` before model-dtype rounding.

For natural attention0 edges, retain the exact head-specific score

`g_h(q,t,delta) = s_h,1(q,t,delta) * s_h,2(q,t,delta)`

and edge write `g_h P_h(t)`. The sum of all causal edge writes plus the explicitly measured numerical remainder must
reproduce the live attention0 residual write with relative squared error `<=1e-12` in float64 accounting and maximum
model-dtype replay error `<=2e-5`.

## Frozen rank-16 task interface

Re-execute the old honest `c_proj` A-SVD recipe on FIT routed inputs only. Orthonormalize the range of its first 16
output factors to obtain `U16 [1152,16]`. This is not refit to payload tables.

Required calibration:

- full-rank `c_proj` replay changes CE by `<1e-3 nat`;
- the rank-16 routed-write surrogate costs `<=0.12 nat` on SELECT;
- three seed-fixed Haar rank-16 controls each cost `>=0.80 nat` on SELECT;
- `U16^T U16` maximum absolute error is `<=2e-5`.

Failure of any calibration voids content and permits only an instrument repair without changing data or bars.

## Interaction-determined response metric

Let `a_h(t)=U16^T P_h(t)`. Raw Euclidean distance between these 16-number codes is only a control.

At each FIT natural position, measure centered finite responses to each standardized mode. Let

`sigma_j = RMS_FIT(U16_j^T attention0_write)`

with a floor of `1e-8` used only to declare an inactive mode. For every active `j`, run both
`+sigma_j U16_j` and `-sigma_j U16_j` at the pre-MLP0 residual boundary. Concatenate the centered changes in:

1. MLP0's 1,152-number write;
2. attention1's four normalized Q/K tensors, each 1,152 numbers across heads;
3. attention1's fresh-value tensor, 1,152 numbers.

Each of these six consumer blocks is divided by its FIT response Frobenius RMS and receives equal total weight. If
`R_x [consumer_coordinates,16]` is the finite-response matrix at natural state `x`, define

`G_FIT = mean_x R_x^T R_x`.

Construct `G_SELECT` independently with the frozen scales and normalizers. Report normalized Frobenius cosine,
eigenvalue spectra, rank retaining 90% trace, and principal-subspace overlap. No consumer may be removed after seeing
its result.

The response-weighted payload coordinate is any square-root factor `L` satisfying `L^T L=G`; all claims use distances
`(a-b)^T G (a-b)`, which are invariant to the chosen square-root gauge.

## One-sparse shared codebook versus private codebooks

Fit a deterministic k-means codebook of `K=256` centers to all FIT vocabulary payloads `L a_h(t)` jointly. Each
payload receives exactly one center: this is a one-sparse code, not top-k routing.

The matched private control has the same total 256 centers distributed across nine heads. Choose the integer
allocation `K_h>=1, sum K_h=256` by dynamic programming to minimize FIT distortion from precomputed per-head k-means
curves. Thus the private control receives the best FIT allocation rather than an arbitrary equal split.

Additional controls:

- the same global/private comparison in raw Euclidean U16 coordinates;
- seed419 token-row permutation independently within each head before applying the learned response metric;
- three seed-fixed Haar output interfaces with their own FIT response metrics but the same center budget.

All k-means initializations, tie breaks, iteration caps, and seeds are fixed in code before SELECT is read.

### Execution freeze before implementation

To obtain the complete private-center allocation curve without fitting thousands of unrelated initializations, use
deterministic **bisecting k-means** for both the global and private arms. Begin with the exact mean. At every step,
split the occupied cluster with greatest total squared error; ties choose the smaller cluster ID. Initialize its two
children at the mean plus/minus the leading covariance eigenvector scaled by one half standard deviation, perform 12
two-center Lloyd updates, keep the old ID for the child with lexicographically smaller center, and assign the other
the next ID. Save the distortion after every split. The global arm stops at256. Each private-head curve runs through
248 centers, the maximum possible under `K_h>=1` for the other eight heads, and the registered dynamic program chooses
the exact minimum-sum allocation. Empty children are repaired by moving the point with largest current residual;
lexicographic index breaks ties. All accumulation is float64; assignments may be computed in float32 chunks.

Seed419 controls only token-row ordering and Haar matrices; the bisecting construction itself is deterministic. The
phrase “raw Euclidean geometry alone is not sufficient” in prediction C means the response-metric global advantage
must exceed the raw-U16 global advantage by at least 5 percentage points. This sentence resolves the computational
procedure and an otherwise qualitative clause before any implementation or SELECT measurement; it changes no arm,
budget, or previously numeric bar.

## Natural routed-edge transport

On SELECT natural positions, replace each projected payload coordinate `a_h(t)` by its frozen center while retaining:

- the native head-specific double-QK scalar `g_h`;
- the native component orthogonal to `U16` for the screen arm;
- every residual, normalization, bias, and numerical remainder.

This isolates codebook error without charging the already-known rank-16 tail deletion. Compare global and private
codebooks on:

1. relative MSE of the summed routed `U16` attention write;
2. relative MSE for each of the six named downstream consumers;
3. CE difference from native and from the no-quantization `U16 + native-tail` identity arm;
4. both disjoint 48-document SELECT waves.

The retained native tail means this is an identification intervention, not a compressed artifact.

## Frozen predictions

### A. Exactness and interface calibration

All exact object, replay, split, live-hook, rank-16, and Haar checks above hold.

### B. The downstream metric is stable and nontrivial

- normalized Frobenius cosine between `G_FIT` and `G_SELECT` is `>=0.85`;
- their rank-90 values differ by at most 2;
- at least one response-metric eigenvalue ratio to the median is `>=4`, so the metric is not merely a rescaled
  Euclidean norm;
- every named consumer has nonzero response RMS and the two sign arms differ from identity.

### C. A cross-head shared payload vocabulary exists

On held-out vocabulary tokens, the global codebook has at least 15% lower response-weighted distortion than the
optimally allocated private codebooks. At least 25% of occupied global centers receive at least 100 held-out tokens
from each of three or more distinct heads. The result must hold under both vocabulary halves formed by even/odd
held-out token quotient; raw Euclidean geometry alone is not sufficient.

### D. The shared vocabulary transports through real QK routing

On natural SELECT edges, the global codebook:

- explains at least 70% of summed routed-U16 squared magnitude;
- explains at least 60% of every named consumer's squared response change;
- beats the matched private codebook by at least 5 percentage points in routed-write R2 and in the mean consumer R2;
- has CE damage no more than `0.01 nat` worse than the private codebook in either SELECT wave.

## Strong null and decisions

The strong null `no discrete shared downstream OV vocabulary at K=256` fires if A fails, or if either:

- global held-out distortion improves on private by `<2%`;
- fewer than 10% of occupied centers have support from at least three heads;
- mean natural consumer R2 is `<=0.30`;
- the global codebook loses to private on both natural routed metrics.

If A-D hold, the result identifies a finite shared payload vocabulary only at this interface. The next rung may join
its atoms with the folded double-QK query/source factors and perform atom-swap/removal CE tests. If C holds but D
fails, the vocabulary is geometric but not a routed computation. If the strong null fires, do not tune K: move to a
continuous block-term factorization of the complete QK-times-OV edge tensor, because the shared structure may be
continuous rather than codebook-like. In all cases, do not return to whole-head clustering.
