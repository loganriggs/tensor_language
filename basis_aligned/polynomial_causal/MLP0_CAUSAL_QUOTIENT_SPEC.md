# MLP0 causal-response quotient: preregistered first discriminator

## Question

Does the token-conditioned part of MLP0 admit a coarse downstream-equivalence
quotient, or is the smallest useful interface continuous and token-resolved?

This experiment is a necessary-condition test for the current class-cluster story.
It does not ask whether class is decodable or whether class centroids are separated.
It asks whether replacing one token-conditioned MLP0 value by another value assigned
to the same proposed state is behaviorally negligible for every registered consumer
over multiple live residual backgrounds.

## Frozen interpretation

Three claims are distinct and must not be substituted for one another:

1. **Geometric organization:** class labels correlate with MLP0 output geometry.
2. **Readable information:** a downstream map can decode or use class information.
3. **Causal quotient:** states merged by a proposed partition are interchangeable for
   every registered consumer within a declared tolerance.

The first two are already partly supported. Only the third licenses a discrete
replacement interface, extraction, or independently editable class component.

## Stage 0: cheapest global worst-cell screen

Do not learn a new quotient or run donor swaps first. Reuse the frozen construction
in `mlp0_downstream_clusters.py`: fit-row token means, the five block-1 reader maps,
seed-13 random projections, frequency-weighted k-means, and K=64. Compare five arms
under global deployment:

1. `O`: original live MLP0;
2. `T`: exact frozen token-mean table;
3. `Q64`: reader-defined K=64 centroid table;
4. `A64`: activation-k-means K=64 at matched assignment/centroid price;
5. `M`: global mean, as an assay-sensitivity control.

K=256 is an optional, explicitly higher-price diagnostic. It cannot replace the
matched-price K=64 comparison.

The confirmatory evaluation window must be newly frozen and outcome-unexposed. Repo
search currently finds no prior `skip=17000, N=192` use, but the row authority must
still verify availability plus document/full-row/token-prefix disjointness against
all recorded roles before binding it. This specification alone does not bind or load
those rows.

Every effect is retained by document and by the full 2x2x2x2 background grid:

- first versus second sequence half;
- fit-frozen below/above-median token frequency among covered positions;
- fixed previous-token punctuation/boundary versus other;
- fit-frozen below/above-median pre-MLP0 residual norm.

The primary non-cancelling output metric is per-position `KL(p_T || p_arm)`. Also
retain paired `CE_arm - CE_T`, block-1 attention-output nRMSE, and block-1 MLP-output
nRMSE. Direct-output RMS scales and both median boundaries are frozen on fit rows.
Unseen token types receive a sentinel and are excluded from the quotient claim;
coverage must be reported and reach 90%. Every cell needs at least 30 documents.

For every contrast, resample documents jointly across every consumer and cell. In
each bootstrap replicate recompute cell effects and

```
M* = max_{consumer, cell} effect* / margin_consumer
```

The simultaneous one-sided 95% upper confidence bound of `M*` must be below 1.
Missing or underpowered cells fail closed as inconclusive. Frozen practical margins
are KL <= .01 nat/token, CE harm <= .0075 nat/token, and direct attention/MLP nRMSE
<= .05.

Required gates are:

1. `T` versus `O` passes first; otherwise this is only a quotient of the static-token
   approximation, not live MLP0.
2. Every `Q64` versus `T` consumer/cell passes the simultaneous equivalence gate.
3. `Q64` is no worse than `A64` in every cell and strictly lowers the maximum
   standardized point distortion.
4. `M` versus `T` exceeds a margin in every consumer family, establishing assay
   sensitivity.

Any one registered cell over budget falsifies compositional equivalence for Q64.
Passing licenses only a finite-grid claim and the Stage-1 local interchange test; it
does not establish arbitrary-background causal equivalence.

## Stage 1 intervention object (only if Stage 0 survives)

Let `z(i, s)` be MLP0 output at a position with current token `i` and live background
`s`. On fit rows, estimate the token-conditioned mean `mu_i` and the global mean
`mu`. The primary intervention at a target position is

```
z'(i, s; j) = z(i, s) - mu_i + mu_j
```

for donor token `j`. This preserves the target position's observed contextual
deviation while changing the token-conditioned value. Full-output donor swaps are a
secondary stress test and cannot rescue a failure of the primary test.

Donors are paired within strata of token frequency, output-delta norm, absolute
position, and clean target CE. Same-token resampling is the empirical intervention
floor. Same-class and different-class donors use the same pairing procedure; unmatched
pairs are excluded before any outcome is read.

## Registered consumers and backgrounds

Response signatures retain separate axes for live background and consumer:

- block-1 attention `q`, `k`, `q2`, `k2`, and `v` preactivations;
- block-1 MLP left and right preactivations and MLP output;
- final pre-softcap logits on the target token plus class-mass logits;
- final KL and target-token CE.

At least eight live target backgrounds per evaluated donor value are required. A
consumer absent from an arm is a failed arm, not a zero. Consumer responses are
normalized only by fit-row scales frozen before quotient scoring.

## Candidate interfaces

The later interchange run compares:

1. deterministic human surface classes;
2. Euclidean activation k-means at matched `K`;
3. clustering under the stacked downstream-response metric at matched `K`;
4. token identity as a fidelity/price ceiling, not a nontrivial quotient candidate;
5. continuous stacked-response PCA at a separately reported bit/rank price.

A later shared-plus-private dictionary is licensed only after this discriminator.
No candidate may use final-row responses to choose labels, rank, thresholds, or
price.

## Primary causal-quotient gate

For each consumer independently, compute response distance for every within-state
donor pair in every live background. Report the mean for diagnosis, but never use it
for acceptance. A discrete quotient passes only if all conditions hold:

1. the 95th percentile of within-state pairwise worst-background distance is at most
   the frozen consumer-specific `epsilon_q95`;
2. the maximum within-state distance is at most `epsilon_max`;
3. every consumer passes separately (no averaging or free-riding);
4. at least 90% of evaluated states belong to non-singleton cells;
5. median matched between-state distance exceeds within-state q95 by at least 2x;
6. the same verdict holds on the untouched final rows and no consumer's normalized
   q95 worsens by more than 25%;
7. composed two-swap responses do not exceed the sum of registered single-swap
   bounds by more than 25%.

The tolerances are fixed from same-token resampling and validation rows before final
access. Exact `epsilon=0` is not the target: language modeling may legitimately retain
lexical identity. The output is a rate-distortion/description-length frontier, not a
claim that a privileged number of classes exists.

## Simplicity price

Every arm supplies an external, auditable price record. The scorer does not infer
that a human label is free or that a tensor rank equals encoded bits. Price includes:

- producer/classifier program and constants;
- lookup assignments when not generated by the program;
- shared response atoms once;
- private consumer wiring once per consumer;
- residual/backoff tables;
- declared coefficient precision and online operations.

Selection is minimum total bits among arms passing the causal gate, with FLOPs and
maximum final-row distortion retained as separate Pareto coordinates. An invertible
latent gauge change must leave the total price comparison invariant or be explicitly
canonicalized by fit-row whitening.

## Registered outcomes

- **Coarse quotient supported:** the human or response-metric partition passes every
  consumer/background/final/composition gate at lower price than token identity.
- **Hierarchical interface supported:** a coarse partition passes only after adding a
  priced within-class continuous residual.
- **Continuous interface supported:** no nontrivial discrete partition passes, while
  a continuous low-rank response code reaches the same distortion at lower price.
- **Token-resolved interface supported:** no nontrivial quotient passes and a program
  that generates token-specific coordinates is required to meet the distortion bar.
- **Inconclusive:** empirical floors are unstable, pairing coverage is below 90%, or
  a registered consumer/background is missing.

Geometric separation, downstream probe accuracy, or donor-directed movement averaged
over backgrounds cannot override a failed causal-quotient gate.

## Authority and execution state

This specification and its CPU scorer do not authorize model loading or final-row
access. The first implementation action is CPU-only. A separate collector authority
must bind row identities, model/source hashes, response scales, donor matching, and
the final-access lock before any GPU run. The sealed compiler-v2.1 roles and artifacts
are out of scope and must not be reused.
