# Three-hourly mathematical review — 2026-09-03 07:08 UTC (Claude)

Grounded in the completed MLP10 estimation chapter (§2657-§2667) and rung521/522 (Codex, GPU; rung522 building,
no science yet). Sign convention §2135 (CE added above native, lower better) — not used here (estimation/coding
statistics, not frontier claims).

## The state the mathematics must address

Explained fraction stuck at 5.348% / 10.923% / 4.727 nat / 0 of 68. The recent arc established: MLP10's reliable
causal footprint is a low-dim (~3) SOURCE-SHARED subspace (§2658) capturing ~76% of reliable variance vs a 58%
noise floor (§2666), with no reliable per-source residual (§2660); this is an ACROSS-SOURCE property (§2667);
and the whole-a8 response is reliable but broad/non-selective (§2665). Two of my own claims were corrected this
window (§2664 block-6 base-rate, §2665 broad-not-N) — a signal that the effect-variance metrics I have been
using are soft. The mathematics needed now is (a) a RIGOROUS, noise-robust measure of how much structure the
validated low-dim objects actually capture (to make coverage credit real and to red-team §2666), and (b) the
compositional/cross-module machinery that turns per-module summaries into a reusable decomposition.

## Ranked moves

### 1. Prequential / MDL cross-validated rank — EXECUTE (CPU)
**Object.** The 83x32 node-by-circuit effect matrix (R520), split by document half. **Theory.** Rissanen MDL /
Dawid prequential coding: fit the top-r circuit subspace on half0, code half1's effects with it; the held-out
residual energy has NO noise-floor inflation (independent-half noise does not transfer), so the description
length `DL(r) = (n/2)ln(||M1(I-P_r^{h0})||^2/n) + (r*32/2)ln n` (BIC form) selects the true effective rank and
yields a bits-saved coverage number that penalises overfitting automatically. **Assumption that may fail.**
Gaussian residual coding is an approximation; the effects are signed CE deltas, roughly Gaussian pooled — the
prequential (held-out) evaluation guards against mis-specification. **Consequence beyond reconstruction.** A
noise-robust coverage number (cross-validated captured fraction `g(r*)`, free of §2666's 58% floor), the
MDL-optimal effective rank (does it confirm §2658's 3?), and a bits-saved figure that COMPOSES additively across
validated results — the rigorous foundation the coverage-credit named gap lacks, and a red-team of §2666: if the
low-rank does not save bits prequentially, §2666's 76% was a soft-metric artifact. **Cheapest falsifier.**
Registered below; CPU, 0 forwards.

### 2. Coupled / shared-dictionary factorization across modules (JIVE) — PROPOSED (data-blocked)
**Object.** Per-module node-by-circuit effect matrices for several MLPs (only MLP10's R520 exists on disk).
**Theory.** Joint-and-Individual Variation Explained (Lock et al. 2013) / coupled matrix factorization decomposes
several matrices into a SHARED low-dim subspace plus per-module individual parts — the "reused parts are our
decomposition" object Logan asked for. **Consequence.** A cross-module reusable dictionary (shared) vs
module-specific (individual), certifiable by held-out interchange. **Falsifier.** Needs an across-source
source-star effect bundle for a 2nd module (the R520 analog for e.g. MLP9) — a GPU re-measure = Codex lane;
§2667 already showed within-source terms are the wrong granularity, so the ask is precise. Proposed, not
executable CPU-side.

### 3. Causal-abstraction consistency certificate (Beckers-Halpern / Geiger IIA) — PROPOSED (overlaps rung522)
**Object.** The 3-dim shared subspace as an abstraction map tau. **Theory.** A constructive abstraction must
COMMUTE with interventions on HELD-OUT intervention sets, not just fitted ones; interchange-intervention accuracy
on unseen sources/documents is the certificate that the subspace is an editable abstraction. **Consequence.** A
pass turns the shared subspace into a certified editable smaller object. **Falsifier.** This is essentially
rung522's held-out target-prediction test in activation space — Codex's active lane; proposed as the formal
framing of their success criterion rather than executed here.

## Pruned
Effect-variance fractions in the soft form (§2666, now to be superseded by the MDL number); activation-Hankel /
minimal realization (RMSNorm non-composability, unchanged); gauge-quotient of the bilinear CP tensor (removes
parameterisation dof but not storage — does not yield a smaller program); another per-source term slice (§2667
closed the within-source object as noise-limited). Literature: Rissanen 1978/1983 MDL; Dawid 1984 prequential;
Lock, Hoadley, Marron, Nobel 2013 JIVE; Beckers-Halpern 2019; Geiger et al. 2021 IIA. No fetch needed; classical.

## Top three, ranked
1. **Prequential/MDL cross-validated rank — DONE this wake (CPU):** rigorous noise-robust coverage + effective
   rank + bits-saved; red-teams §2666.
2. **Cross-module JIVE shared dictionary — propose to Codex:** the reusable-decomposition object; needs a 2nd
   module's source-star bundle (GPU).
3. **Causal-abstraction consistency certificate — propose:** formal framing of rung522's held-out criterion.

## Executed
Move 1: preregistration MLP10_PREQUENTIAL_MDL_RANK_PROBE + CPU run + enqueue this wake. It computes the
held-out (cross-validated) captured-energy curve and the BIC/MDL-optimal rank over the R520 effect matrix, and
the bits saved vs the mean-only baseline, as the noise-robust coverage-credit number and a red-team of §2666.
