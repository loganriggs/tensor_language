# Rung 524 terminal receipt — direct-subspace planted falsifier

**Scored:** 2026-09-03 09:34 UTC  
**Status:** `direct_subspace_instrument_falsified`  
**Claim level:** optimizer-instrument evidence only; no model or circuit evidence

## Result

The frozen direct Grassmann optimizer failed the deliberately easy problem with a known four-dimensional answer.
Zero of 15 leave-one-target-out fits passed. Fourteen fits exhausted all 16 allowed backtracks before completing 200
updates; the remaining fit completed 200 but still failed validation and subspace recovery. Final validation losses
were 15.8%--59.9% of initialization, far above the registered 5% ceiling. Relative projector errors were
0.511--1.004 versus the required 0.10, and minimum principal cosines were 0.006--0.701 versus the required 0.995.

The OOD split correctly remained unopened. This is not evidence against a selective attention8 circuit. It shows
that the proposed direct-subspace optimizer is not trustworthy enough to test for one.

## Independent audit

The CPU audit reloaded all 15 saved frames, verified their tensor hashes, recomputed projector errors and principal
cosines, reconstructed every frozen pretest gate and final decision, and reconciled the execution ledger from the
per-update records. It passed exactly:

- 2,095 accepted updates;
- 14 failed line searches;
- 2,109 gradient evaluations;
- 21,044 line-search objective evaluations;
- 30 initial/final FIT and 30 initial/final VALIDATION evaluations;
- zero model forwards, model backwards, GPU calls, or OOD evaluations.

Artifact SHA-256 values:

- result: `0d25569e155a150ef34532f38bba3f0f7bb0f0784f543bdbacde2bf3caa312b2`
- frames: `00fc1c03ea3ee2a330d12204e9ed88cf03c1c1c04879ea76da4453c3addb69fa`
- terminal audit: `e3836847ee258ea8fa88a29c9fb99ddef8ca6c07d154daa36304afa2a5af60da`

## Frozen consequence

Close this attention8 optimizer route. Do not tune the step size, rank, toy dimensions, or loss weights. The next
object is MLP0's exact token-only, token-by-context, and context-only computation, using the known token inventory and
downstream causal equivalence to merge token groups or split context-dependent roles. That route directly targets a
human-readable computational specification and selective manipulation rather than a lower-rank coordinate system.
