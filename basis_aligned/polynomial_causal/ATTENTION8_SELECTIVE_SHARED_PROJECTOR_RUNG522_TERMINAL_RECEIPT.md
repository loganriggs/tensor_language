# Rung 522 terminal receipt: optimizer invalid before TEST

**Terminal:** 2026-09-03 09:03:42 UTC<br>
**Managed process exit:** 0<br>
**Scientific status:** invalid optimizer instrument; no circuit conclusion

## Registered terminal state

The create-only result reports `terminal_pretest_validation_failure`. TEST was never opened, no pre-TEST manifest
was created, Predictions C and D were not scored, and no removal intervention ran. The process completed the full
registered pre-TEST census rather than stopping after the first unhealthy fit.

The independent terminal auditor reloaded the frame archive and checked every result health entry, frame hash,
failure list, call-ledger bucket, and forbidden TEST-stage field. It passed.

## Optimizer health

- real leave-one-target-out: 0/15 healthy;
- recovery-only: 1/15 healthy;
- single-target oracle: 2/20 healthy;
- randomized circuit labels: 4/48 healthy;
- all-three-target: 1/5 healthy;
- total: 8/103 healthy.

Across all fits, 77 failed the first-20-versus-last-20 training-loss check and 65 failed to improve the fixed
VALIDATION objective over initialization. The separately archived trajectory analysis found 84/103 fits with a
loss above 100, 59/103 above 1,000, and a maximum loss of 270,121,856. Projector orthonormality and movement passed,
so this is optimization instability rather than a malformed four-dimensional space.

The stricter scheduler audit found that exact spike recurrence has almost no cross-seed power: only 43/13,668 exact
target/map/member/control patterns occur under two seeds, and only two spike-producing patterns have a cross-seed
comparison opportunity. Thus the post-hoc archive does not identify normalization versus step size. Rung 523 tests
those causes prospectively.

## Exact execution price

- optimization forwards: 20,600;
- optimization backwards: 20,600;
- inference-only forwards: 5,029;
- removal forwards: 0;
- wall-clock runtime: 4,981.154 seconds (`83.019` minutes).

The inference total is exactly the registered sum of native capture/replay, self-donor checks, complete-attention8
FIT/VALIDATION responses, 206 health evaluations, leave-one-out comparisons, recovery-only comparisons, random
projectors, and all-three VALIDATION selection measurements.

## Immutable artifacts

- terminal result SHA-256:
  `0fd7380230d19ea0cb28140bfdb4c03c7761427fb561a8a8455b5a0daa8977da`
- independent audit SHA-256:
  `3129614a9a74c8873e00f952da7222df6abffe8da151cf5b04b4016ea27fe645`
- frame-archive file SHA-256:
  `2b8d3709714903890c4ae935a07da7284ac3253b7b2242d055023b33adeca2bb`
- frame-archive canonical content SHA-256:
  `20a9f857ccc8c13733d8ec892b18c3dc79b6cfa60fa7bc520ce4cdca08a5f51f`

## Consequence

Rung 522 provides no evidence for or against a shared selective attention8 computation because the prerequisite
optimizer health gates failed. Its scientific thresholds, rank, controls, and nulls remain frozen and unused. The
managed runner immediately started the already-preregistered rung-523 FIT/VALIDATION-only optimizer diagnosis, which
cannot access TEST or score a circuit claim.
