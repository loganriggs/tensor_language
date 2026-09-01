# Attention0 realized-edge continuous block-term preregistration

Date: 2026-09-01 19:35 UTC

## Decision and claim boundary

Rungs 416–420 reject whole-head equivalence, discrete cross-head QK/OV atoms, and one rank-24 global carrier, while
rung 421 independently confirms that the downstream U16 output interface transports through native QK routing.
Rung 422 asks the next changed-object question: can the **complete realized edge product** be represented by a small
set of continuous score and payload modes whose coupling matters?

For natural edge `n`, architectural head `h`, and downstream output coordinate `c`, define

`e[n,h,c] = s1[n,h] * s2[n,h] * p[n,h,c]`,

where `s1` and `s2` are the exact two rotary QK score halves and `p` is the exact output-projected source-token
payload in rung 419's task-useful U16 interface. The head index is a contraction index, not the proposed feature
basis.

This is an identification screen. Every arm still computes the native score and payload generators before
projecting them. It therefore saves zero model values and cannot support compression or adoption.

## Why this differs from closed prior work

- The July sparse-core work fitted per-head unigram-average `(key1,key2,value)` CP atoms. This rung uses natural
  query/source/offset edges, couples heads, and evaluates downstream response.
- Ticks 207–208 fitted naked composed tensors in weight space and tied their corrected null. This rung works in the
  realized natural-edge measure and the independently validated downstream U16 quotient.
- Rung 416 fitted covariance modes of already-summed output writes. This rung preserves the two score halves and
  the source payload as separate modes until their exact product.
- Rungs 419–420 tested a one-sparse vocabulary and one global token-function carrier. This rung uses continuous
  projectors and permits a graded dense core.

## Frozen data and edge population

- `FIT`: the existing 96 FineWeb FIT documents used by rungs 419–420.
- `SELECT`: the existing disjoint 96 FineWeb SELECT documents.
- `FINAL`: unopened.
- Query positions: `16,32,...,240` in every 256-token document.
- For each query position, include every causal source position from 0 through the query position. No edge sampling
  is used for evaluation.
- Joint optimization minibatches may sample these FIT edges with seed 422 only.
- SELECT is not read until all parameters, restarts, and the winning FIT restart are frozen.

## Frozen task interface and metric

Rebuild rung 419's task U16 using the same A-SVD algebra on routed attention0 inputs from FIT. Rebuild its six-block
finite-response Gram from MLP0, attention1 Q1/K1/Q2/K2, and attention1 fresh value. Required reproduction:

- U16 orthogonality maximum error `<=2e-5`;
- full-rank attention output replay CE change `<1e-3 nat`;
- rank-16 damage `<=0.12 nat` and all three Haar-rank16 damages `>=0.80 nat`;
- response-Gram FIT values and task payload code table reproduce rung 419 at relative error `<=1e-6`;
- exact unprojected edge sum plus the measured numerical remainder reproduces native U16 attention with relative
  squared error `<=1e-12` and post-remainder maximum error `<=2e-5`.

The response metric is represented by any factor `L` with `L^T L=G`; all edge/output errors use `x L^T`, so the
choice of square-root gauge is irrelevant.

## Three equal-parameter continuous arms

All modes are affine: subtract the FIT mean, apply an orthogonal projector, then restore the mean.

- Score-half 1 rank: `r1=6` in the nine-head coordinate.
- Score-half 2 rank: `r2=6`.
- Payload rank: `rv=32` after flattening head×U16 to 144 coordinates.

### Marginal PCA

Fit each projector separately by ordinary covariance: top six FIT modes of `s1`, top six of `s2`, and top 32 of
the edge-frequency-weighted flattened payload. This is optimal for the three marginal squared-error objectives but
does not see their product.

### Joint block term

Initialize the same three projectors from marginal PCA. Optimize only their orthonormal bases on FIT for 512 Adam
steps at learning rate `0.02`, minibatch 4096, with seeds 422 and 423. QR-orthonormalize inside every forward pass.
Choose the restart with lower complete FIT objective before SELECT.

The objective is

`0.5 * individual-edge response-SSE / individual-edge target energy`

`+ 0.5 * head-summed-edge response-SSE / head-summed-edge target energy`.

This defines a continuous block-term model: substituting the three projected factors produces an induced dense core
whose entries are products of the score-head loadings and payload head/output loadings. It permits one score mode to
combine with several other-score and output modes.

### Head-deranged coupling control

Use the frozen joint bases and reconstructions, but cyclically map reconstructed score-half-2 head `h` to
`(h+4) mod 9` before multiplication. This preserves every mode's values, rank, spectrum, and literal parameter
count while breaking which two score halves and payload are contracted together.

## Literal screen price

Materializing the screen requires:

- U16 interface: `1152*16 = 18,432` values;
- two score means and bases: `2*(9 + 9*6) = 126` values;
- payload mean and basis: `144 + 144*32 = 4,752` values;
- total: `23,310` values.

The native Q/K/V/O generators remain required, so net model saving is exactly zero. Marginal, joint, and deranged
arms have identical stored shapes and price. A pass licenses a later physical latent-generator factorization with a
new literal price; it does not make this screen adoptable.

## Held-out measurements

On every SELECT edge report response-weighted relative MSE for individual head edges and head-summed edges. On full
SELECT documents, reconstruct every causal source edge, sum it, retain the native U16-orthogonal attention tail and
measured numerical remainder, and report:

1. zero-origin R2 of the routed U16 attention write;
2. R2 of the U16-induced response for MLP0 and each of the five attention1 Q/K/value consumers;
3. CE damage from native, pooled and in two disjoint 48-document waves;
4. marginal-versus-joint and deranged-versus-joint differences;
5. FIT-half projector overlap for independently optimized joint bases.

CE is cross-entropy added above the native model in natural-log units; lower is better.

## Frozen predictions

### A. Instrument validity

All split, source hash, U16, response metric, exact edge, native-forward, call-count, finite-value, orthogonality,
rank, and price checks above hold. `FINAL_opened=0`.

### B. The complete edge product contains a joint continuous block

On SELECT, the joint arm has head-summed edge relative MSE `<=0.45`, reduces that error by at least 20% relative to
marginal PCA, and reduces it by at least 20% relative to the head-deranged control. Individual-edge relative MSE
must not be more than 0.05 worse than marginal PCA.

### C. The block transports through native routing and downstream readers

The joint arm has routed-U16 R2 `>=0.60`, every named consumer R2 `>=0.50`, and improves both routed R2 and mean
consumer R2 by at least 0.05 over marginal PCA. In neither SELECT wave may joint CE damage exceed marginal damage
by more than `0.005 nat`.

### D. The coupling is reproducible and specific

- FIT-half independently optimized joint projector overlaps are `>=0.70` for both score halves and `>=0.50` for
  payload;
- the deranged arm loses at least 0.10 routed R2 and at least 0.02 mean-consumer R2 relative to joint;
- joint beats deranged CE by at least `0.005 nat` pooled and in both waves;
- every projector moves by a nonzero amount from initialization and receives nonzero gradient during fitting.

## Strong null and decisions

The strong null fires on any instrument failure, or if joint routed R2 is `<=0.30`, joint improves held-out summed
edge error over marginal by `<2%`, deranged routed R2 is within 0.02 of joint, or joint loses CE to marginal in both
waves.

- A–D hold, null clear: identify a realized continuous edge block and next factor its latent score/payload generators
  physically against the current adoption frontier.
- B holds but C fails: record a geometric edge factorization with no downstream mechanism claim.
- C holds but D fails: record predictive projection without shared-coupling identification.
- Strong null: close low-rank head-service coupling at these frozen ranks and move to MLP0's nonlinear token `Q` or
  attention1's stronger copy/retrieval prior. Do not tune ranks, K, heads, or thresholds.
