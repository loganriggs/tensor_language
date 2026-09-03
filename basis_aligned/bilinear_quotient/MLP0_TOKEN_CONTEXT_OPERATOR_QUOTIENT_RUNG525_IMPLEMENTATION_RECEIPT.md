# Rung 525 implementation and planted-preflight receipt

**Recorded:** 2026-09-03 09:52 UTC  
**Status:** planted toy passed; real model run eligible for the managed GPU queue  
**Claim level:** instrument validation only, not circuit identification

## Frozen objects

- Preregistration: `../polynomial_causal/MLP0_TOKEN_CONTEXT_OPERATOR_QUOTIENT_RUNG525_PREREGISTRATION.md`
  (`sha256 fdc1575846a97e43c4834e4caa0d2081fea5e5b2ab5d73f5c36b180a3de5f683`).
- Exact operator math: `ops/mlp0_token_context_operator_quotient_rung525_math.py`
  (`sha256 b65e875855d4dd0a65afb140c73e60568af590dfbde3b6d411f30fb921353729`).
- Runner: `ops/mlp0_token_context_operator_quotient_rung525_run.py`
  (`sha256 8fa1d3c2022f5a3ee8aca4b4f79c64a21fa7f0940bc3207fa7295f828e838b8a`).
- Runner commit: `26265c003`.

## Checks run before model loading

- Focused unit tests: `9 passed`.
- Repository experiment gate: `PASS`, with no findings.
- no-model dry run: `DRYRUN OK`; it confirmed toy-before-model ordering, exact `K_t` sketches,
  bank-A selection/bank-B scoring, and no downstream calls.
- Planted 32-class toy: passed. The selected donor had the correct hidden operator class for
  `100%` of receivers, versus `3.846%` after independently scrambling each token's coordinates.
  The held-out-bank candidate/raw distance ratio was `0.0` because same-class planted operators
  were exactly equal. The receipt explicitly reports `model_loaded: false`.

This planted result only establishes that the code can recover known shared operators and that
the negative control destroys that relation. It does not establish that real MLP0 tokens form
stable groups or that any group is used as one circuit downstream. Those are the registered real
model and successor causal questions.
