# Rung 526 implementation and pre-model receipt

**Recorded:** 2026-09-03 10:09 UTC  
**Status:** CPU gates pass; managed GPU smoke is the next execution boundary  
**Claim level:** instrument validation only

## Frozen code

- Preregistration SHA: `53be05fdd22ff9153066bb680a9d67ab170319ecb2a335c1760f224449b3fc22`.
- Exact contraction math SHA: `126917d791282df56dc2a27a62750759c6f58bec57f8e4518da49a7409eaf6af`.
- Runner SHA: `12adf336e70f219fa7c12a1e896fc46349ce8983db43a2c43fe2de976022c23e`.
- Runner commit: `29f0cd5e2`.

## CPU checks

- Nine focused tests pass.
- The repository experiment gate passes with no findings.
- The no-model dry run passes and confirms that D0 selects, D1 gates, and the validation circuit families remain
  closed until the discovery predicate passes.
- The planted 32-class quotient selects the correct hidden class for 100% of receivers. Token-specific scrambling
  lowers this to 3.846%. Candidate/raw distance ratios on the three unseen response banks are
  `2.91e-5`, `2.75e-5`, and `3.01e-5`, all below the frozen `0.20` bar.
- The independent differentiable toy matches the explicit circuit directional derivative with relative squared
  error `1.09e-31`.

These checks validate the algebra and known-answer grouping code. They do not establish that real MLP0 operators
form circuit-equivalent token groups. The GPU smoke tests the identity-leaf/autograd path on two real circuit tags;
only a clean smoke can place the full frozen run in the managed queue.
