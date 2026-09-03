# Pairwise Fisher subset-price probe — v2 addendum (bug-fix rerun as the physical control)

Registered 2026-09-03 20:25Z (box clock; a first draft line said 20:26Z, corrected before the script hash was frozen), after the v1 receipt (§2708, sha 2d0dc207…) was read and BEFORE the v2 script runs.
Lane 2 (CPU-only, `# BQLANE: cpu`). SIGN CONVENTION (§2135): measured numbers are CE ADDED ABOVE THE REAL MODEL on docs 64–95 —
LOWER IS BETTER; ratio = measured / J.

## What changed and why
v1's pred_a(ii) (in-code pairwise identity, 1e-6 relative) FAILED on every set containing both mlp16 and mlp17. Root cause,
verified from the receipt: the pair (mlp16, mlp17) is both the 91st pair and nested set A1, so the certificate loop accumulated
its key twice per batch, doubling that pair certificate (.2420 = 2 × .1210) and inflating J(A) by exactly .1210 on all eight
supersets (J_used − direct = .1210 on each). v2 makes ONE change: the certificate set list is de-duplicated
(`joint_sets = list(dict.fromkeys(...))`). Same preregistration (36a777cb…), bars, nulls, seeds, split, k=32, S=4.

## What is at stake (stated before running)
- pred_a must now be TRUE (identity ≤ 1e-6 on all 15 direct sets; (i) and (iii) were already met). If it is still false the bug
  diagnosis in §2708 is wrong and must be retracted.
- The post-hoc corrected numbers in §2708 (A1 1.13, A2 1.19, A3 1.34, A4 1.17, R6 1.18, R9 1.27, R11 1.30, R12 1.17) must be
  reproduced within the MC-sample tolerance (~0.01 in CE; the RNG stream is pinned by seed 0, so exact agreement is expected
  for the measured column and for the certificate given identical batch order). If they are not reproduced the correction is
  retracted.
- BEST7 is re-chosen from the corrected J; it may differ from v1's. pred_e is scored on the v2 design. If v2's pred_e fails,
  §2708's design claim (ii) is withdrawn.
- pred_c is expected to stay FALSE (77 of 91; the bug touched only one pair, whose sign is positive either way).

## Price
Same as v1: ~1,350 CPU document-forward equivalents, ~17 min on lane 2 at 4 threads. Output
pairwise_fisher_subset_price_probe_v2_results.json (v1's receipt is preserved unchanged and frozen as a prior).
