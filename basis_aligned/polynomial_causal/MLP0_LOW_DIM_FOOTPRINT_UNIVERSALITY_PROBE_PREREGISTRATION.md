# Parallel probe: does MLP10's low-dim reliable-footprint finding generalize to MLP0? ("look elsewhere")

**Status:** prospectively frozen after the MLP10 estimation chapter (§2658 reliable ~3-dim shared subspace,
§2660 no residual, §2666 top-3 covers 0.76 of reliable variance vs 0.58 noise floor), before applying the same
noise-unbiased instrument to MLP0. CPU-only, zero forwards, zero deployed parameters. Owner: Claude parallel
lane. Serves Logan's "look elsewhere / discover details" direction — a universality check of the low-dim law on
a second module. Not a frontier/certificate claim (§2135 unused).

## The question and the honest caveat

MLP10's reliable circuit-effect footprint is a low-dim source-shared summary (§2658/§2666). Is a DIFFERENT module
(MLP0) also low-dim by the same noise-unbiased cross-half instrument? CAVEAT (stated up front): the MLP0 R519
object is the 49 exact bilinear INTERACTION TERMS of ONE documented source (`H4.DISTANT_SAME`), whereas the MLP10
R520 object is 22 source-STARS across all sources — different decompositions, so this is NOT a numeric
apples-to-apples coverage comparison. It cleanly tests only: pooling MLP0's 49 term-effects, is the reliable
circuit-effect a low-dimensional shared object above the noise floor? A yes generalizes the low-dim LAW; the
coverage numbers are reported side-by-side as indicative, not as a strict cross-module equality.

## Object (frozen, from the rung519 discovery bundle)

Bundle `mlp0_one_circuit_interaction_atlas_rung519_bundle.pt` (`54a4ce1c…`), key `discovery_effects/circuit` =
`A[t,h,c]`, float64, shape `(49 terms, 2 halves, 32 circuits)`. `M0=A[:,0,:]`, `M1=A[:,1,:]` (49x32), circuit
columns mean-centred over the 49 terms. Noise-unbiased cross-half cross-covariance `S=(M0^T M1 + M1^T M0)/2`
(32x32); eigenvalues `w_1>=...>=w_32`. Total reliable variance `T = sum max(w_i,0)`; top-3 captured `C3`;
coverage `f = C3/T`. `whole_circuit[.,8]` is the `r.2.0.2` whole-source effect (`0.003909/0.004190`), the
instrument-reproduction anchor (matches §2654/§2655).

Term bootstrap: 400 hash-fixed resamples (seeds 13000+) of the 49 terms with replacement -> `f_boot` CI.
Term-permutation null: 400 shuffles of M1 rows (E[S]=0) -> `w1_null`, `f_null` (pure-noise baselines).

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `54a4ce1c465b6b953b54d2fa4e104c055f5446f39f3ac5167f7aae12b320bd8a`;
  `A` is `(49,2,32)`; `|whole_circuit[0,8]-0.003909140586171755|<1e-6` and `|whole_circuit[1,8]-0.004190039411971824|<1e-6`;
  `T>0`.

- **B — MLP0 has a reliable shared circuit-effect direction (pooling terms).** `w_1 > q95` of the
  term-permutation null's top eigenvalue. (If false, MLP0's pooled term-effects carry no reliable shared
  structure at this N and the low-dim LAW does not obviously extend to MLP0's term decomposition.)

- **C — and that reliable structure is LOW-DIMENSIONAL with majority coverage.** The top-3 coverage
  `f >= 0.50` at the point estimate AND `f` exceeds the pure-noise coverage baseline `f_null` q95. (Mirrors
  §2666; a yes says MLP0, like MLP10, concentrates its reliable footprint in a few directions above noise.)

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction/pin clause.
- A true, B false: MLP0's pooled 49-term effects have no reliable shared direction — the low-dim law seen at
  MLP10 does NOT extend to this MLP0 decomposition at current N; report and stop (the term decomposition may be
  the wrong object, consistent with §2655's per-term noise).
- A,B true, C false: MLP0 has a reliable shared direction but it is not majority-covering / not above noise —
  MLP0's reliable footprint is higher-dim than MLP10's; report the dimensionality difference.
- A,B,C true: the low-dim reliable-footprint finding GENERALIZES to MLP0 — a second module whose reliable
  circuit-effect footprint is a few directions above noise. Report MLP0's `f`, CI, noise floor, and eig count
  beside MLP10's (§2666: f=0.76, floor 0.58) as indicative cross-module evidence for the "MLPs read context as
  low-dim summaries" thesis (§2649/§2652/§2658), obtained here by an independent noise-unbiased route.

Assumptions that may fail: the 49 terms are one source's interactions (within-source, not across-source like
MLP10) — comparison is indicative only; positive-eigenvalue truncation slightly upward-biases T (pred_c noise
baseline controls interpretation); effect space is a lossy readout.

## Literal price

Zero forwards, zero backwards, zero deployed parameters. One eig + 800 resample/permutation eigs; CPU, < 2 s.

## Frozen inputs

- rung519 bundle SHA256: `54a4ce1c465b6b953b54d2fa4e104c055f5446f39f3ac5167f7aae12b320bd8a`
