# Block-3 native-Down behavioral port v1 — prospective design

**Status:** designed after seeing the Family-F fit result and before freezing or loading
any new evaluation role.  No validation authority exists.  The Family-F result is the
selection event; this experiment requires registry-fresh whole documents and cannot
reuse Family A/F validation or the repeatedly exposed compiler roles.

## Why this exists

The Family-F K512 support with the exact native Down columns obtains fit-teacher KL
`0.05772`, whereas the registered local decoder refit obtains `0.08476`.  The native-
Down program uses 512 native products rather than 4,608:

- 1,770,624 float values plus 512 int64 indices;
- 7,086,592 total bytes;
- 512 products and 1,769,472 linear multiplies per token;
- 88.9% fewer product gates and factor floats than native MLP3.

Its summed-write NRMSE is `0.86957`, so it is not a physical-write reconstruction.
The fit KL was also used to notice this arm.  The only prospective question is whether
it is a restricted *behavioral* substitute and finite-edit port on new documents, or
whether its low one-sided KL comes from nonlinear downstream compensation.

## Frozen candidate and controls

The authority must hash-bind the exact K512 support, balanced Left/Right rows, native
Down columns, and native bias reconstructed from the spent Family-F program artifact
(file SHA256
`d4af5bfbae03f8df9be8127e2e06c6f1a66b189be180ce72e5c74b6c7ac7a038`).
No parameter is refitted.

Programs:

1. exact native MLP3;
2. zero-write denominator;
3. Family-F K512/native-Down candidate;
4. seed-2026082907 matched-random K512 support with its native Down columns;
5. same Family-F support with Down columns cyclically shifted by one gate, preserving
   price and the multiset of column norms while breaking feature/decoder pairing;
6. exact-native replay control.

The candidate must execute its 512-product path with zero native-MLP3 calls.  Teacher
and denominators may use native MLP3 under an exact call ledger.

## Fit-derived edit directions

Before any fresh row is loaded, use only the old n480_skip80 fit role to compute

$$
e_{rt}=C_{rt}-N_{rt},
$$

where (N) is the native MLP3 write and (C) the frozen candidate write.  Center the
fit errors, take the top four covariance eigenvectors, orient each by making its
largest-absolute coordinate positive, and scale each to unit RMS in physical residual
coordinates.  Hash and publish these four directions before the fresh role opens.
No fresh-row PCA, rotation, sign choice, magnitude choice, or direction selection is
allowed.

## Fresh role

Freeze one approximately 192-document, 257-token FineWeb-v2 role with the registry-wide
row freezer.  Exclude every document identity in prior fit, validation, final,
compiler-discovery, tangent, de-alias, and response receipts.  Score target positions
64--255.  Aggregate rows inside source document and bootstrap source documents, not
tokens.  The role selects no model variant: all arms, signs, amplitudes, and gates are
frozen first.

If this role is later used to choose a direction or program for extraction, a second
fresh replication role is mandatory.

## Experiment 1: ordinary behavioral substitution

Measure candidate, random, decoder-shift, zero, and native output distributions,
ground-truth CE, centered-logit error/cosine, top-1 agreement, and per-document
uncertainty.  Normalize candidate KL by the full-zero native KL stake.

Required ordinary gate:

- point candidate KL/full-zero KL at most `0.20`;
- simultaneous one-sided q95 document-bootstrap upper bound at most `0.35`;
- one-sided q95 CE-difference upper bound at most `0.01` nat;
- candidate beats each matched control by a paired simultaneous lower confidence
  bound greater than a frozen 5% relative improvement;
- exact-native replay passes and every denominator is positive and finite.

## Experiment 2: two-sided contextual-error secants

For every fresh row, suffix-run the five writes

$$
N,\qquad N\pm\tfrac12 e,\qquad N\pm e,
$$

where fresh-row (e=C-N).  `N+e` is the ordinary candidate and `N-e=2N-C` is its
mirror.  Both signs and both amplitudes must pass separately.  This tests whether the
observed error direction is downstream-null in a two-sided neighborhood rather than
one-sided compensation.

For each sign/amplitude require KL/full-zero point at most `0.20`, simultaneous q95 at
most `0.35`, and centered-logit response cosine q05 at least `0.90`.  No averaging
across signs or amplitudes is allowed.  A candidate-only pass with mirror failure is
explicitly classified as nonlinear compensation.

## Experiment 3: finite physical edit-port transport

For each of the four frozen directions $u_i$, use amplitudes
$\epsilon\in\{0.5,1.0\}$ times the fit error RMS.  Apply the same physical write edit to
native and candidate at both signs and compare suffix response secants:

$$
\Delta_{i,\pm,\epsilon}=
[G(C\pm\epsilon u_i)-G(C)]-[G(N\pm\epsilon u_i)-G(N)].
$$

Every native edit must be material: its KL effect is at least 5% of the full-zero KL
stake.  For every material direction, sign, and amplitude require:

- centered-logit secant NRMSE simultaneous q95 at most `0.35`;
- response cosine simultaneous q05 at least `0.90`;
- response-norm ratio simultaneous interval contained in `[0.8, 1.2]`;
- finite CE and KL, with no ordinary-baseline regression hidden by an edit score.

No signs, directions, or amplitudes may be dropped or averaged into a pass.

## Interpretation table

| Ordinary | Error secants | Edit port | Allowed conclusion |
|---|---|---|---|
| fail | any | any | fit-only candidate does not transfer |
| pass | one-sided only | any | downstream compensation, not a null |
| pass | two-sided pass | fail | restricted downstream-null replacement, not extraction |
| pass | two-sided pass | pass | restricted behavioral equivalence and editable finite port |

Even the strongest outcome does not establish local-write fidelity, atom semantics,
arbitrary-direction edits, or broad distributional generalization.

## Why not jointly optimize the decoder yet

With fixed balanced product features, correlated gates and the suffix/softmax nullspace
make multiple decoders behaviorally equivalent.  Freeing both scores and decoder also
restores the scale gauge in `score_i * decoder_i`, and the suffix objective is
nonconvex.  A fixed-support downstream decoder with native-proximity/minimum-norm
regularization would define an operational compiler, not identifiable columns.

Only if the native-Down candidate passes the prospective test should a later decoder
experiment be considered.  It must freeze optimizer trajectory and regularization,
report multiple restarts and the empirical quotient Hessian rank/conditioning, and
claim the deployed function rather than semantic uniqueness of decoder columns.

## Lifecycle

Required order: commit/push runner, tests, row freezer, this document, all transitive
sources and exact parent pins; independent outcome-blind audit; create-only authority;
freeze/hash fit-derived directions; publish a final-attempt token; load the fresh role
once; run the exact arm/call ledger; publish result and semantic manifest; receipt last.
Any partial execution writes failure and spends v1.  No threshold or arm changes after
fresh rows load.
