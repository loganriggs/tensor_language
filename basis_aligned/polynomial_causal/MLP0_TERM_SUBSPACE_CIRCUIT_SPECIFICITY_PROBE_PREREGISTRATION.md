# Parallel probe: is a low-dim TERM SUBSPACE a circuit-specific MLP0 unit where single terms are not?

**Status:** prospectively frozen after rung519 (§2654) returned its strong null (no single exact MLP0 source
interaction term is circuit-specific for the documented `r.2.0.2` attention8-family circuit) and BEFORE any
subspace metric on the rung519 bundle is computed. CPU-only, zero model forwards, zero deployed parameters.
Owner: Claude parallel lane. This is a red-team of the §2654 null and a cheap scout for Logan's finer-grain /
DAS / reusable-decomposition direction; it does NOT relax or re-score rung519 — it is a separate registered
analysis with its own held-out split and null.

## Motivation

Rung519 scored each of 49 exact bilinear interaction terms of the R518-selected `H4.DISTANT_SAME` source by
whether that single term ranks top-4 among 32 circuits and exceeds 2x the circuit median, in BOTH document
halves. Zero terms passed. But §2649/§2652 already showed MLP0 reads context as an essentially rank-1 / diffuse
low-dim summary — exactly the regime where NO single coordinate is the unit but a LINEAR COMBINATION (subspace)
can be. Logan's standing direction: find the specific subspace (DAS-style) rather than the specific term, and
seek reusable components across circuits. This probe tests the cheap CPU analog on already-published data before
any GPU DAS rung is priced.

## Object (frozen, from the rung519 bundle)

Bundle `mlp0_one_circuit_interaction_atlas_rung519_bundle.pt`, key `discovery_effects/circuit`:
`A[t,h,c]`, float64, shape `(49 terms, 2 document halves, 32 circuits)` — term t's finite cross-entropy effect
on circuit c in half h. `discovery_effects/whole_circuit` `W[h,c]` shape `(2,32)` is the whole-source effect.
Target circuit is the documented `r.2.0.2` at index `j=8` (`W[0,8]=0.003909140586171755`,
`W[1,8]=0.004190039411971824`, matching §2654). Half 0 fits; half 1 tests. No refitting on half 1.

`A0 = A[:,0,:]` (49x32), `A1 = A[:,1,:]` (49x32). For a term-weight vector `w in R^49`, the combined circuit
profile is `p = A^T w in R^32`. Selectivity of a profile for index `k`:
`S(p,k) = |p[k]| / median_{m != k} |p[m]|`.

## Fitted object (frozen)

For circuit index `k`, the minimum-L2-norm term-weight vector producing a pure unit response on that circuit
in half 0 is `w_k = pinv(A0^T) e_k` (least-norm exact solution; A0^T is 32x49, generically full row rank so
`A0^T w_k = e_k` exactly). `w_k` is a low-dim combination of the 49 terms — the term-space analog of a DAS
subspace direction. It is frozen from half 0 and never refit. The out-of-sample test profile is `q_k = A1^T w_k`
and its out-of-sample selectivity is `S1_k = S(q_k, k)`.

Diagnostics reported (not scored): single-term baseline `S1_single = max_t S(A1[t,:], j)` for the target (the
best any SINGLE term achieves out-of-sample — rung519's failed object), the cross-half term-effect stability
`corr(A0[:,j], A1[:,j])`, and a rank-8 SVD-restricted variant of `w_j` for robustness.

## Frozen predictions (with measured bars)

- **A — instrument exactness.** Frozen bundle SHA256 = `54a4ce1c465b6b953b54d2fa4e104c055f5446f39f3ac5167f7aae12b320bd8a`
  and result SHA256 = `3eb5188fa65a746a987d4bee851aaed46b08d7ba905b596dd091d01bd29386f6`; `A` is `(49,2,32)`,
  `W` is `(2,32)`; target index 8 is tag `r.2.0.2` with `|W[0,8]-0.003909140586171755|<1e-6` and
  `|W[1,8]-0.004190039411971824|<1e-6`; and the half-0 fit is exact, `max_k ||A0^T w_k - e_k||_inf < 1e-6`.

- **B — the target's term-subspace generalizes as circuit-specific.** The frozen `w_8` produces an
  out-of-sample profile that is selective for the target: `S1_8 >= 2.0` (rung519's own bar, so this is an
  apples-to-apples "does a subspace pass where every single term failed") AND `argmax_m |q_8[m]| == 8`.

- **C — subspace localization is a GENERAL property (a reusable term-subspace decomposition exists).** A strict
  majority of circuits localize out-of-sample: `#{ k : S1_k >= 2.0 and argmax_m|q_k[m]|==k } >= 17` of 32.

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the named instrument clause; no interpretation.
- A true, B false: even a min-norm term combination fails to localize the documented circuit out of sample —
  the redundancy is genuinely diffuse in term space; the only remaining finer-grain route is DAS on the live
  ACTIVATION subspace (gradient-trained interchange), not on exact-term combinations. Report to Codex as the
  decisive reason to spend GPU on activation-DAS rather than more term enumeration.
- A,B true, C false: the target is subspace-localizable but it is not a general property — term-subspaces are
  bespoke, not a clean reusable decomposition; report which circuits localize and which do not.
- A,B,C true: low-dim term subspaces ARE circuit-specific units where single terms were not, and the property
  is general — a term-subspace decomposition exists. Hand Codex the target `w_8` loading and the per-circuit
  selectivity table as the seed for a GPU physical-substitution + reuse rung.

No outcome licenses lowering rung519's single-term bars, calling reconstruction a circuit without physical
substitution (that is Codex's GPU step), a rank sweep, an SAE, or a quantization/compression claim.

## Literal price

Zero model forwards, zero backwards, zero deployed parameters. One 49x32 pseudo-inverse per document half and
32 min-norm solves; pure NumPy float64, CPU, < 1 second.

## Frozen inputs

- rung519 bundle SHA256: `54a4ce1c465b6b953b54d2fa4e104c055f5446f39f3ac5167f7aae12b320bd8a`
- rung519 result SHA256: `3eb5188fa65a746a987d4bee851aaed46b08d7ba905b596dd091d01bd29386f6`
