# Mathematical review — 2026-09-02 10:10 UTC (three-hourly)

Context: 480 in-flight (101 min, healthy). The fresh mathematical material
this cycle is the reproducibility breach itself (§2599) — treated here not
as an ops nuisance but as a measurable property of the bilin18 forward map.

## Ranked moves

### 1. Float-perturbation Lyapunov analysis of the squared-bilinear forward
### — EXECUTED (fit below), predictions frozen before the queued data lands
**Object.** Per-process variation of single-token intervention damages on
the hook-laden 472/474 pathway (the ~.03–.08 nat breach).
**Theory.** bilin18's attention squares scores (s1·s2) and its MLPs are
bilinear: a float-level perturbation δ injected at layer L is multiplied,
per subsequent layer, by a factor tied to the local Jacobian norm — the
numerical-analysis view (Higham, *Accuracy and Stability of Numerical
Algorithms*) of a positive Lyapunov exponent. Kernel-selection differences
between processes inject δ ~ 1e-6–1e-5; eighteen layers of multiplicative
growth turn that into ~1e-1 at the logits. Prediction: log(noise floor) is
LINEAR in depth-to-readout of the intervention site.
**Measured fit (run1's three singleton sites, depths 9/8/5 layers to
readout):** slope **0.1970/layer** ⇒ amplification **1.218×/layer**;
residuals ≤ 4e-4 nat on all three points:
  m8 depth 9: actual .08406, fit .08440; m9 depth 8: actual .06968,
  fit .06931; m12 depth 5: actual .03833, fit .03838.
Three points is thin; the law earns belief only by out-of-sample hits.
**Frozen predictions (data arrives free from the already-queued runs):**
(P1) probe run2 and the b-variant's vs-bundle per-site diagnostics preserve
the ordering m8 > m9 > m12; (P2) their log-slope vs depth lies in
0.20 ± 0.06/layer; (P3) extrapolation — an intervention at layer 17 (depth
0-1) has cross-process noise ≤ ~.015, i.e., the old CUDA-wobble tolerance
was correct FOR SHALLOW READOUTS and unsound for deep ones. Consequence
beyond reconstruction: principled, depth-dependent tolerance schedule for
every future cross-process bar, and a numerical-stability certificate
dimension for the compiled program (a compiled circuit that reduces
effective depth also reduces noise floor — executable-cost AND
certification win). Cheapest falsifier: the queued receipts themselves.

### 2. Deterministic-kernel cure probe — EXECUTED (registered + enqueued)
**Object.** The breach mechanism (H-A) and its cure.
**Operational theory.** cuBLAS heuristic kernel/workspace selection is the
canonical source of run-to-run float differences on identical inputs
(NVIDIA cuBLAS reproducibility guarantees: results are bitwise reproducible
for fixed workspace + fixed algorithm on the same architecture); pinning
CUBLAS_WORKSPACE_CONFIG=":4096:8" removes the workspace degree of freedom.
**ops/det_replication_probe.py** (queued behind the b-variant): two fresh
pinned child processes re-run the 474-own-code probe; pred_b (the cure) =
the pinned pair agrees exactly; pred_c (the mechanism signature) = the
unpinned run2 varies ≥.001 from run1 or from the pinned arm; null = pinning
does not cure ⇒ H-A weakens toward allocation-history dependence. If the
cure holds, the ops recommendation writes itself: pinned workspace for
every collector whose receipts feed cross-session comparisons.

### 3. Behavior-level causal abstraction for the compiled equality program
**Object.** The equality circuit's compile target, given the four-grain
representational individuation theorem (§2595–§2598).
**Theory.** Approximate bisimulation with metrics (Girard & Pappas 2007):
define each module by its intervention-response morphism over the frozen
intervention family, not by weight-space structure; module equality =
ε-bisimilarity of response profiles; composition certified at interfaces by
the measured laws (far+/near−, T×G, register-conditioning). The day's nulls
say this is the ONLY level at which the trio composes — so the compiled
program should store response-profile contracts, and its correctness
certificate is behavioral, not representational.
**May fail.** ε-bisimilarity is task-family-relative; transport beyond the
measured register family is exactly where 468/470 showed 7× weakening.
**Falsifier.** Already partially in hand (matcher transplant 459/460); the
next handle is whether 480's slab (if it passes) admits the same
behavioral-contract treatment on the odd-root validation families.
**Status: proposal** — becomes concrete at the 480 verdict; no rung now.

## Pruned this cycle
Exact/compensated-summation repair of the bridge (noise enters inside conv
kernels, not accumulation); another commutant/gauge pass (479 closed it);
Hankel/automata, info bottleneck, archetypal (standing prunes).

## Citations
Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM, 2e 2002);
NVIDIA cuBLAS documentation §Reproducibility (workspace config and bitwise
guarantees); PyTorch Reproducibility notes (use_deterministic_algorithms,
CUBLAS_WORKSPACE_CONFIG); Girard & Pappas, "Approximate bisimulation
relations for constrained linear systems" (Automatica 2007); Oseledets'
multiplicative ergodic theorem (Lyapunov exponents, framing only).
