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
- The result and frame archive are create-only. Exact code hashes and test receipts will be appended before the
  first scientific execution.
