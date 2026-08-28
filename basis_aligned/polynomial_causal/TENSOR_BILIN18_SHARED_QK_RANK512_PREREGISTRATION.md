# Preregistration: shared-QK rank-512 causal-transport discriminator

Date: 2026-08-28

Status: one preregistered follow-up on the same opened roles and frozen prefix fixture.
It opens no new selection or OOD authority.

## Question

Shared-QK-384 saves 10.21% and has only about 0.02 nat CE harm, but fails causal
transport with recovery 0.84682 and cosine 0.92111. Does increasing the shared routing
rank to 512 cross the causal gate, or is the activation-weighted basis itself misaligned
with causal transport?

## Frozen candidate and predictions

The program is identical to the rank-384 complete standalone candidate except that each
shared QK encoder and four decoders have rank 512. Exact stored price is 503,436,726
values, saving 42,467,328 values (7.7793%) from dense.

1. Complete ownership, total support, zero fitted lookup tables/calls/native references,
   model-object collection, and exact price must pass.
2. Covered and all-position CE harm must each be at most 0.025 nat on both roles, and no
   worse than rank 384 on the matching role by more than 0.001 nat.
3. Context-delta recovery must be at least 0.90 and cosine at least 0.95.
4. For evidence that rank was limiting, recovery must exceed rank384 by at least 0.03
   and cosine by at least 0.02.
5. Covered CE harm must replicate across roles within 0.01 nat.

If gates 3--4 pass, rank is the immediate limiting resource and rank512 becomes the
first admitted compressed complete point on these roles. If either fails, further
ordinary activation-weighted rank increases are deprioritized in favor of a
context-weighted routing objective and sitewise causal localization.
