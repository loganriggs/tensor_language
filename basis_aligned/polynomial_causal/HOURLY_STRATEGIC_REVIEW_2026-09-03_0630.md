# Hourly strategic review — 2026-09-03 06:30 UTC (Claude)

Sign convention §2135: frontier L2 = CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER; frontier §312 norm-2304 at
+2.6735. Role split: Codex leads direction + owns GPU (rung522 building); Claude red-teams + CPU probes + ops.

## Where the program stands

Strict explained fraction unchanged: 5.348% / 10.923% / 4.727 nat / 0 of 68. Recent arc is mechanistic
understanding of MLP10, now with two self-corrections landed (§2664: the shared subspace is NOT block-6-specific,
that was base rate; §2665: rung521's failure is a reproducible BROAD effect, not an N-limit — Codex's read was
right). Net validated picture at MLP10: a reproducible ~3-dim SOURCE-SHARED circuit-effect summary exists
(§2658/§2661, existence+reproducibility stand), no reliable per-source/private residual at current N (§2660), and
the whole-a8 response is reliable but broad/non-selective (§2665). rung522 (Codex, GPU) now tests whether a
selective rank-4 sub-projector lives inside that reliable broad response; its instrument smoke passed 06:10,
science still sealed (~20 min GPU idle while the gradient entrypoint is finalised — registration latency).

## Largest gaps
1. Tail dictionaries / COVERAGE CREDIT — the +2.6735 price is tail-dominated, and validated structural results
   (the §2658/§2661 shared subspace) earn 0 strict credit. The coverage-credit accounting is unresolved and is
   the one named gap I can advance CPU-side today.
2. m16 remainder — CPU-blocked (no committed bundle on the non-persistent disk).
3. attn5's write price cliff — CPU-blocked; off Logan's circuits-not-compression steering.

## Candidate moves (pruned)
- **Coverage fraction of the validated shared subspace.** How much of MLP10's TOTAL reliable circuit-effect
  variance does the 3-dim summary capture? A concrete coverage-credit input; §2658 gave dimensionality and §2660
  "no reliable residual" but neither computed the fraction. CPU, new, non-colliding. [program-structure/estimation]
- **Post-hoc red-team of rung522 science** — highest value, GATED (not landed).
- **CPU preview of rung522 selectivity** — data-blocked (rung521 saved summary stats, not per-target vectors).
- **Frontier price-decomposition (tail/m16/attn5)** — DATA-BLOCKED + off-steering.
- **Raise document count** — GPU/Codex lane; and §2665 shows it does NOT bind rung521 (broadness does), so lower
  priority than it seemed.

## Ranked top five
1. **Coverage fraction of the shared subspace — EXECUTE (CPU, this wake).** Advances the coverage-credit named
   gap with a concrete number + bootstrap CI; new; falsifiable; 0 GPU; non-colliding.
2. **Post-hoc rung522 red-team** — execute when its science lands (waiter armed).
3. **Coverage-credit accounting proposal** — turn the fraction into a credit rule; needs Codex sign-off.
4. **Raise document count for per-source resolution** — GPU/Codex lane; propose.
5. **Frontier price-decomposition** — highest leverage, DATA-BLOCKED.

## Executed
Move 1: preregistration MLP10_SHARED_SUBSPACE_COVERAGE_FRACTION_PROBE + CPU run + enqueue this wake. It computes
the fraction of total reliable (positive-eigenvalue) circuit-effect variance captured by the top-3 shared
subspace of the noise-unbiased cross-half cross-covariance, with a node-bootstrap CI (pred_b), and checks the
fraction exceeds the pure-noise (node-permutation) baseline (pred_c). Combined with §2660, a high fraction states
precisely: at current N, MLP10's reliable causal footprint IS one low-dim source-shared summary.
