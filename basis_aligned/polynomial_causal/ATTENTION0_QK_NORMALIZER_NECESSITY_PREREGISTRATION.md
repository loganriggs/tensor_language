# Rung 433 preregistration — are attention0's token/head RMS denominators the missing computation?

Date: 2026-09-01 21:13 UTC

Status: frozen before implementation or result inspection. Diagnostic screen only. It licenses no compression,
semantic atom, circuit extraction, or adoption claim.

## Decision

Rung 431 showed that six direct linear-bilinear score generators fail catastrophically even though six mixtures of
the already-computed native scores are nearly lossless. The proposed explanation is that a mixture of headwise
RMS-normalized scores is not an ordinary bilinear form. That explanation is algebraically valid but not yet causal:
the 36 token/head normalization functions might vary too little to explain the observed failure.

This rung changes only those denominators while preserving every native numerator. It asks two questions in order:

1. Does token-dependent normalization materially affect natural attention0 score products and language-model loss?
2. If it does, do the 36 denominator functions share a stable low-dimensional token vocabulary?

Only a positive answer to the first question justifies spending a later rung on a joint quadratic-form generator.

## Exact computation and dimensions

There are four layer-0 maps `m in {q1,k1,q2,k2}`, nine heads `h`, residual width `D=1152`, and head width `H=128`.
For real vocabulary token `t in {0,...,50256}`, the exact state entering attention0 is

`x_t = RMSNorm((lambda_0 + lambda_1) RMSNorm(embedding_t)) in R^1152`.

Let `W_mh in R^(128 x 1152)` be one head slice. Before rotary position encoding, its denominator is

`d_mh(t) = sqrt(mean_a((W_mh x_t)_a^2) + eps)`.

Rotary encoding is orthogonal, so it does not change this norm. For branch `b`, head `h`, query token `t`, source
token `u`, and relative offset `delta`, the exact score is

`s_bh(t,u,delta) = x_t^T B_bh(delta) x_u / (128 d_qbh(t) d_kbh(u))`,

where `B_bh(delta)=W_qbh^T R_delta W_kbh`. The attention pattern is `s_1h * s_2h`. Thus each denominator squared is
the positive-semidefinite quadratic function

`d_mh(t)^2 - eps = x_t^T A_mh x_t`, with `A_mh=W_mh^T W_mh/128`.

The full partially symmetric tensor `A[map,head,input,input]` has shape `36 x 1152 x 1152`. This rung does not fit
that tensor; it first tests whether its function is causally worth factoring.

## Frozen data roles

- Token-function FIT: all real token IDs with `id mod 5 != 4`.
- Token-function SELECT: all real token IDs with `id mod 5 == 4`.
- Physical SELECT: the exact 96-document SELECT authority and positions `16,32,...,240` used by rung 431, in two
  fixed 48-document waves. FINAL remains closed.
- Constants, centering, map-space bases, and thresholds use only token-function FIT. No document target, next-token
  ID, loss, consumer response, or SELECT statistic enters them.

The script must hash the token IDs and physical rows, reproduce the rung-431 SELECT hash, and assert zero overlap of
the token FIT/SELECT roles.

## Arms

1. `NATIVE`: unchanged model.
2. `EXACT_TABLE`: form every raw native Q/K vector, multiply by the precomputed exact `1/d_mh(token)` table, then
   apply rotary encoding and the native dot product. This must reproduce `NATIVE`; it is the instrument arm.
3. `CONSTANT`: replace each `d_mh(t)` by
   `bar_d_mh = sqrt(mean_FIT_token d_mh(t)^2)`. All 36 constants are frozen before physical SELECT. Raw native Q/K
   vectors, rotary maps, both score branches, values, output projection, first-value broadcast, MLP0, block1
   consumers, and suffix remain exact.
4. `TOKEN_PERMUTED_TABLE`: independently permute FIT token rows within each of the 36 exact log-denominator columns,
   using seed 4331, and use the resulting map-space statistics only as the shared-structure null. It is not a
   physical candidate.

The physical constant arm may be implemented equivalently by multiplying the native score by
`d_q(t)d_k(u)/(bar_d_q bar_d_k)`; an in-run direct raw-vector comparison must show this identity to relative squared
error at most `1e-10` in float64 before the shortcut is used.

## Measurements

### Instrument and liveness

- state-table maximum absolute error against a live block-0 pre-attention capture;
- exact denominator-table maximum absolute and relative error against live raw projections;
- `EXACT_TABLE` branch-score, product, full-write, consumer, logit, and CE differences from `NATIVE`;
- all 36 constant-to-exact denominator ratios, with a live-range assertion that at least one ratio differs from one
  by `>=1e-3` on SELECT;
- exact hashes, call counts, and dtype/epsilon fields.

### Causal importance of token-dependent denominators

Using the same natural edges and denominators as rung 431, report for `CONSTANT`:

- relative squared error for each score branch and their product;
- relative squared error of the full attention0 write;
- routed-U16 and six-consumer `R^2` values;
- mean and each-wave CE added above `NATIVE` (lower is better).

### Shared token structure

Form `L[t,j] = log d_j(t)` for `j=1,...,36`. Subtract the FIT mean of each column. Learn only the right singular
vectors in 36-dimensional map/head space on FIT tokens; project SELECT rows without refitting. Report the rank-1,
rank-4, rank-8, rank-16, and full singular spectra for description, but the rank-8 bar alone is binding.

Split the 36 columns deterministically into alternating columns after lexicographic `(map,head)` ordering. For each
18-column half, compute its top-eight left token subspace on FIT. Report normalized projector overlap between the two
token subspaces. Repeat after independently permuting token rows within each column for 64 fixed seeds `433100+k`;
the binding null is the empirical 99th percentile, not a guessed absolute control window.

This is a finite-token screen. Even a low-rank `L` table does not show that later-layer continuous states share the
same structure or that its singular vectors have semantics.

## Literal prices and non-claims

- native layer-0 Q/K maps: `4*1152*1152 = 5,308,416` scalar values;
- exact 36-scalar denominator table over real tokens: `50,257*36 = 1,809,252` scalar values;
- constants: 36 scalar values;
- descriptive rank-8 log table, if stored literally: `50,257*8 + 36*8 + 36 = 402,380` scalar values.

These prices cover denominators only. Every physical arm in this rung retains the native numerator maps, so none is
a Q/K compression. A later common squared-feature model

`A_j approximately sum_r c_jr u_r u_r^T`, `c_jr >= 0`,

would cost `R*(1152+36)` scalar values for its quadratic forms, before reciprocal square root and every numerator,
rotary, decoder, index, and precision cost. This formula is a prospective bill, not a result.

## Frozen predictions

### Prediction A — exact-table instrument

All of the following must hold:

- state-table and raw-denominator relative errors are at most `1e-6`;
- float64 direct-score versus score-rescaling identity error is at most `1e-10`;
- `EXACT_TABLE` product and full-write relative squared errors are at most `1e-10`;
- maximum absolute logit difference is at most `2e-5`;
- absolute mean CE difference is at most `1e-6` nat;
- all physical paths and the constant liveness assertion fire.

Failure makes the instrument invalid and withholds all content.

### Prediction B — token-dependent denominators are causally material

`CONSTANT` must have product relative squared error at least `.20`, full-write relative squared error at least `.10`,
and mean CE damage at least `+.010` nat, with positive damage in both waves. These are deliberately one-sided: the
claim is that removing token dependence causes material damage, not that constants reproduce rung 431's exact
failure magnitude.

### Prediction C — the 36 functions share a stable token vocabulary

The FIT-derived rank-8 map-space basis must explain at least `.90` of centered SELECT `L` energy. The two map-half
top-eight token subspaces must have normalized overlap at least `.60` and exceed the token-permuted 99th percentile
by at least `.20`.

### Prediction D — the effect is not one exceptional map

On SELECT, at least three of the four map families must have a median across-head coefficient of variation of
`d_mh(t)` at least `.02`, and both branch scores must have relative squared error at least `.15` under `CONSTANT`.

## Strong null and routing

The strong null fires if either:

- `CONSTANT` has product relative squared error at most `.05` and mean CE damage at most `+.002` nat; or
- rank-8 explains at most `.60` of centered SELECT log-denominator energy and split overlap is within `.05` of the
  token-permuted 99th percentile.

If the first null fires, the rung-431 normalizer diagnosis is rejected: do not build a quadratic-normalizer
generator, and return to numerator parameterization, optimization, or the already successful shared-QK family. If
Prediction B holds but Prediction C fails, normalization matters but is head-private; compare an exact token table
and independently factored forms against shared-QK at complete price. Only if B and C both hold does the next rung
fit the gauge-canonical partially symmetric quadratic tensor, with matched private-form and permutation controls,
before reconnecting it to head-indexed numerators and physical scores.
