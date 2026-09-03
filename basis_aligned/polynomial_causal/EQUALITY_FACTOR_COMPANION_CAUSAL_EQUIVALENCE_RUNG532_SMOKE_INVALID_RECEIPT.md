# Rung 532 first managed smoke: instrument invalid

**Completed:** 2026-09-03 12:28 UTC

The first 21-forward managed smoke exited 0 and opened no scientific outcomes, but it is invalid. Native replay was
exact and every edit was live. The product diagnostic reported a maximum difference of `0.0009231567` because the
new diagnostic converted each BF16 score factor to FP32 before multiplication, while deployed attention multiplies
the BF16 factors first and converts the product afterward. This tested a different numerical computation.

The wrapper also printed `smoke_passed` without combining its diagnostics into a fail-closed predicate. Both defects
are corrected before any circuit outcome: native product checking now preserves deployed operation order, and smoke
status must satisfy every instrument clause or exit nonzero.

- frozen invalid core SHA-256: `877453b5471b167cf7b47a88219b7405b824fce349b69f435b035b8ebaa23f0b`
- invalid wrapper SHA-256: `994d375dd9b5091d342e76070e6827260f0fe597f0c652b8bee9dcb0c6e89066`
- invalid log SHA-256: `377a92d053f3fdb4a8b46ce06742b471b4ac50d4959caabb761f8d2b3c504a1b`

The full 2,625-forward run remains sealed pending a separately named v2 smoke.
