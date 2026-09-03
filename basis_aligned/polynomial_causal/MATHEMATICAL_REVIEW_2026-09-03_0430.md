# Three-hourly mathematical review — 2026-09-03 04:30 UTC (Claude)

Grounded in the R506–R520 grouping-null wall (§2653–§2656), my §2655/§2657 CPU red-teams, and Logan's live
direction (relayed 04:02: finer-grain via DAS / important interaction terms, build compositional REUSE of
found components, narrow below an MLP block). Sign convention §2135 (CE added above native, lower better) —
not used here; these are effect-covariance/estimation statistics, not frontier claims.

## The state the mathematics must address

R510–R520 proved that no exact interaction term, source, head-by-source pair, term subspace, or grouped
22-term source star is a portable/reusable circuit unit under held-out physical bars. My §2657 then showed WHY
the tests can't see structure even if present: the per-node 32-circuit causal fingerprint has cross-half
reliability ~0.016 (below a 0.077 permutation null). By classical attenuation, a cross-node cosine is bounded
by the geometric mean of the two reliabilities, so EVERY grouping test is attenuation-capped near zero — the
nulls are estimation-limited, not necessarily structure-absent. The mathematics needed is therefore about
ESTIMATION under low per-unit reliability, and about defining the reusable object at a granularity that is
actually identifiable at this N.

## Ranked moves

### 1. Noise-unbiased shared-subspace estimation via cross-half cross-covariance — EXECUTED (CPU, §2658)
**Object.** The 83x32 material node-by-circuit finite-effect matrices `M0, M1` (rung520 discovery halves,
disjoint docs 500:624 / 624:748). **Theory.** Split-half signal-subspace estimation / generalizability theory:
with independent half-noise, `E[M0^T M1] = Sigma_signal`, so the eigenvalues of the symmetrized cross-half
cross-covariance are noise-UNBIASED estimates of shared circuit-effect variance (noise adds zero-mean scatter,
not positive bias). Pooling 83 nodes estimates `Sigma_signal` even though each fingerprint's reliability is ~0.
**Assumption that may fail.** The 83 nodes are 4 actions x 22 sources, not independent — handled by a preregistered
WITHIN-ACTION permutation null as a secondary bar. **Consequence beyond reconstruction.** Names the
dimensionality of the reusable circuit-effect subspace (extraction/reuse/certification target) robustly.
**Result (§2658).** Registered A/B/C true, NOT null: a reliable ~3-dim shared subspace exists (lambda1 0.00933 >
node-null q95 0.00437; 3 eigs clear it). BUT lambda1 does NOT beat the within-action null (0.00942), so the
shared subspace is SOURCE-SHARED (all 22 sources collinear across circuits within an action), not
source-specific — the §2649/§2652 low-dim context-summary law re-derived at MLP10 by a noise-unbiased route.
The reusable object is a single ~3-dim circuit-effect subspace many sources feed, not per-source units.

### 2. Reliability-corrected required-N (attenuation inversion / effective sample size) — PROPOSED (CPU-ready)
**Object.** The per-node fingerprint reliability rho=0.016 (§2657) and its dependence on document count.
**Theory.** Spearman disattenuation `r_true = r_obs / sqrt(rho_xx rho_yy)` plus the split-half reliability's
Spearman–Brown scaling in N: to raise a node's reliability from rho0 to a target rho*, documents scale by
`(rho*/(1-rho*)) / (rho0/(1-rho0))`. With rho0=0.016, reaching rho*=0.5 needs ~61x documents; rho*=0.3 needs
~26x. **Assumption that may fail.** Spearman–Brown assumes homogeneous noise across the added documents (plausible
for iid corpus draws). **Consequence beyond reconstruction.** Converts §2657's null into a CONCRETE document
budget for Codex's next grouping/DAS instrument, and predicts, before any GPU spend, whether a source-SPECIFIC
test can ever clear the within-action null at a feasible N. **Cheapest falsifier.** CPU: estimate rho as a
function of a sub-sampled document fraction on the R520 bundle's per-document task effects (the bundle stores
`task` at (4,23,248,6) per-document), fit the Spearman–Brown curve, and read off the N where rho crosses 0.3/0.5.
Zero forwards. This is the natural §2659 and it gates move-1's source-specific follow-up.

### 3. Optimal singular-value shrinkage for the pooled effect matrix — PROPOSED (Codex/GPU-adjacent)
**Object.** The full node-by-circuit effect tensor as a low-rank signal + heteroscedastic noise. **Theory.**
Gavish–Donoho (2014) optimal SVHT and Nadler's spiked-covariance corrections give the minimax-optimal denoiser
and a hard rank threshold from the noise level; whitening each circuit column by its sampling SE first handles
heteroscedasticity. **Consequence.** A denoised low-rank factorization whose right subspace IS move-1's shared
subspace and whose left subspace assigns nodes — the executable reuse map. **Assumption that may fail.** Needs a
per-entry noise-level estimate; the cross-half difference `(M0-M1)/sqrt2` supplies it CPU-side, but a clean
GPU re-measure at higher N is the honest instrument. **Falsifier.** Overlaps Codex's DAS lane (the denoised
subspace is what DAS should localize), so PROPOSED for their lane rather than executed here; move-1 already gives
the CPU existence result it would build on.

## Pruned
Correlating raw per-node fingerprints in any form (attenuation-capped, §2657); latent dictionaries (509 closed);
activation-Hankel (RMSNorm non-composability, unchanged); another write-space grouping slice (R510–R520
exhaustive); literature fetch skipped — the binding objects are in-house and the classical cites suffice
(Spearman 1904 disattenuation; Brown 1910 / Spearman 1910 split-half; Gavish–Donoho 2014 SVHT; Nadler 2008
spiked covariance; Cranmer/Cook sufficient dimension).

## Top three, ranked
1. **Cross-half cross-covariance shared-subspace estimator — DONE (§2658):** a reliable ~3-dim source-shared
   circuit-effect subspace exists at current N; the reusable object is that pooled subspace, not per-source units.
2. **Reliability-corrected required-N — execute next (CPU):** turn §2657's null into a document budget and decide
   whether source-specific structure is ever reachable; gates any higher-N grouping/DAS rung.
3. **Optimal SVHT denoiser — propose to Codex:** the executable reuse map DAS should localize, built on move 1.

## Executed
Move 1 this wake: preregistration `MLP10_SHARED_SUBSPACE_CROSS_HALF_COVARIANCE_PROBE_PREREGISTRATION.md`
(`f75e6aa1…`) + CPU run, ledgered §2658, result `1e8ade7c…`. Move 2 is the registered next CPU analysis.
