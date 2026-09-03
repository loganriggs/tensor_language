# Rung 524 implementation receipt

**Implementation choices frozen before execution:** 2026-09-03 09:30 UTC

This receipt fills in mechanical choices that the committed preregistration left implicit. No outcome has been
computed at this point.

- FIT has 96 member inputs and 96 control inputs per target/map.
- VALIDATION has 192 of each per target/map.
- OOD has 192 of each per target/map and remains sealed until predictions A and B pass.
- Planted frame/readout seed: `524000`.
- FIT, VALIDATION, and OOD data seeds: `524100`, `524200`, and `524300`.
- Initial frame seeds: `524400..524404`, reused across the three omitted targets to give 15 fits.
- All arithmetic is CPU float64.
- Gaussian FIT/VALIDATION member inputs use unit-variance four-dimensional planted coefficients plus orthogonal
  nuisance with scale `0.25`; controls are orthogonal Gaussian inputs with scale `1.0`.
- OOD planted coefficients are variance-normalized Student-t with three degrees of freedom. OOD nuisance/control
  coordinates are multiplied by a deterministic diagonal scale ranging from `0.5` to `1.5`, then projected back
  into the planted orthogonal complement.
- Each `12 x 64` target/map readout has iid Gaussian entries divided by `sqrt(64)`.
- The result and frame archive are create-only.

## Pre-execution code seal

Eight focused CPU tests pass, covering tangent geometry, orthonormal retraction, basis-invariant recovery scores,
fail-closed OOD access, deterministic split construction, exact planted responses, and objective invariance. The
hash-only dry run also passes. Exact SHA-256 values before the first scientific execution are:

- math: `b38c3551ac537940c8c8b72b95e37db4e22389185e6fae34cb5cc25c1d9b4072`
- math tests: `92b1e531fd55e907df58352a730b7ae888e08e6205f61a48acf6e5113c01b9b3`
- runner: `729a23c818640f0537f4982554290f5509f181eac02d0168b30a8180d0d4c0b3`
- runner tests: `4cd31660b3abbd3261306786b6a0016f23b9f1a07a07946225ac92b489606e88`
