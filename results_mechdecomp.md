# Mechanism decomposition: progress log

> ⚠️ **This is a chronological log. Several claims in it were later RETRACTED.**
> For the current, accurate position read **`mechdecomp_findings.md`** instead —
> it states what is established, what was retracted and why, the proposed spec
> amendments, and the open problems. Reading this file linearly WILL mislead you.


Spec: mechanism_decomposition_spec.md. Plan (Logan 2026-07-09): Tier 0 → Tier 1 →
"Tier 1.5" our bilinear models (tiny attn2/block3 as circuit ground truth — the
contraction matrix must recover the causally-verified L0H3→L1H2 edge in both K
branches; OWT comparison models on disk: block2-dense-seed0, attn2-s120k-dense-seed0)
→ Pythia → Gemma-2B OV subset.

## Tier 0.1 — theorem verification: ALL PASS (float64)
- M* = [YX†X]_r X† matches the closed-form optimal value at r ∈ {1,4,10,32}
  (float32 was 2e-4 off at r=32 from XXᵀ conditioning — tests pinned to float64).
- 5-seed Adam on (A,B) never beats it (529.5 vs best 1038.0 at r=6, 3k steps).
- Problem 1 with r = m_true recovers W P_S on the feature span (rel err 2e-10).
- Problem 2 (counterfactual edit targets) matches optimum.

## Tier 0.2 — negative result verified (float64, k_max=20 DGP)
| |S| | diag mass (pred 1/|S|) | cos to W·P_S (pred 1/√|S|) |
|---|---|---|
| 1 | 1.000 (1.000) | 1.000 (1.000) |
| 2 | 0.582 (0.500) | 0.709 (0.707) |
| 5 | 0.246 (0.200) | 0.437 (0.447) |
| 10 | 0.136 (0.100) | 0.302 (0.316) |
Compositional point: cos 0.48 to each parent (pred ~0.5). Pairs sharing one of three
features: cos 0.08 → naive cos-sim clustering degrades exactly as §1.2 predicts.
(Slight excess over 1/|S| comes from α~U[0.5,1.5] weight variance.)

## Tier 0.3 — recovery: IN PROGRESS (gate not yet met; diagnosis clean)
Decisive diagnostic: with truth-init, E-STEP ALONE is PERFECT (min-cos 1.0000, F1
1.000, R² 0.994) → objective + Hadamard-Gram lasso are correct. The Adam M-step
DESTROYS truth-init (min-cos → 0.907 in 12 rounds; scale-invariant steps walk along
noise-floor gradients) → replaced with the spec-preferred EXACT per-atom fixed-point
update (β-freeze → W d = Σβr/Σβ² via lstsq). Result from random init: mean cos 0.990,
ablation ratios 32-64×, but min-cos 0.925 and support-F1 low (imperfect atoms recruit
corrector atoms → L0 4.3 vs true 3). Dedup+drop of duplicate atoms added (final-solve
resplit bug fixed). NEXT (spec init (c), predicted best): k-means init + closed-form
per-cluster extractor, then refine. λ sweep so far: 0.05 best; larger λ hurts.

## Tier 0 — ALL GATES PASS ✓
Final recipe (λ=0.05, m=40, N=10k): k-means init (spec (c)) → exact per-atom M-step ↔
Hadamard-Gram E-step → dedup (cos>0.95 merge) → BACKWARD-SELECTION PRUNE (drop atoms
whose removal costs <2e-3 R² after re-coding). Results: min max-cos 0.9994 (>0.99 ✓),
support F1 1.000 (>0.95 ✓), ablation ratios 32-64× active/inactive ✓, R² 0.996.
Two failure modes found and solved en route (both will matter at scale):
1. Adam M-step walks off the optimum along noise-floor gradients (scale-invariance);
   exact per-atom fixed-point updates replace it.
2. COMPOSITE-ATOM SHORTCUT: for co-occurring features, one composite atom is L1-cheaper
   than two pure codes (precision 1.0 / recall 0.3 signature); a single λ can't fix
   both this and corrector-density. Dictionary-level backward selection with an R²
   guard removes composites (pure atoms are protected by singleton datapoints).
Remaining spec item: λ-sensitivity table with the final recipe (quick, next tick),
then Tier 1 adversarial toys.

## λ sensitivity (final recipe): robust over 10×
λ 0.01/0.02/0.05/0.1 → exactly 10 atoms, F1 1.000, min-cos ≥0.9964; λ 0.2 over-prunes
(14 atoms, F1 0.845). Report the 0.01-0.1 plateau as the operating range.

## Tier 1 first run: DGP design flaw found (not a method failure)
With INDEPENDENT coefficients α0,α1, a ρ=1.0 co-occurring pair is still separable —
the nonneg-L1 geometry prefers the true basis, and the method recovered pure e0/e1
atoms at ρ=1 (hence OOD singletons reconstruct fine; novelty AUROC ~chance 0.41 —
consistent, not broken). Spec's 1.1 merge case requires a TIED coefficient
(x-variation only along e0+e1). Spec's 1.2 "mechanism-side rescue" needs its geometry
derived before a fair test: with free codes, any orthobasis of the pair-plane
reconstructs both x and Wx exactly; the separating signal must enter via L1/nonneg ×
gain structure. QUESTION FOR LOGAN (also in LOG.md): the independent-α result is
arguably a POSITIVE surprise — mechanism dictionaries separate always-co-occurring
features from coefficient fluctuation alone; do we adopt tied-α as the canonical
"truly merged" case (my plan), and is there a sharper operationalization of 1.2 you
had in mind?

## Tier 1.1 PASS / Tier 1.2 CLAIM FALSIFIED (as constructed)

1.1 (tied-coefficient merge case, the corrected DGP): merged direction learned
(cos 0.85), and the NOVELTY DETECTOR works: decorrelated singletons spike 32× in
reconstruction error, AUROC 0.994. §1.4 design stance validated. (Correction to the
previous entry: the independent-α run had recovered an ARBITRARY pair-plane basis —
degenerate optimum — not pure atoms; tied-α is the true merge case.)

1.2 (anisotropic-W rescue, 3 seeds): NO rescue. Mechanism best-cos on the high-gain
feature 0.60-0.67 ≈ SAE 0.63-0.84 (SAE better on one seed). Analysis: with free codes,
exact reconstruction of both x and Wx is basis-invariant on the co-occurring plane
(Σ c_j (d_jᵀx) W d_j = Wx for ANY orthobasis with c=1); λ-shrinkage asymmetry from
gain anisotropy is too weak to break the degeneracy in practice. ⇒ Spec §1.4 claim (b)
("W treating features differently rescues identifiability") is NOT SUPPORTED for
rank-1 masked-projector dictionaries under gain anisotropy. FLAG FOR LOGAN: this
affects the headline "advantage over SAEs"; possible salvage constructions worth
trying — (i) sparsity on OUTPUT usage / gain-weighted L1 (penalize c·||W d|| so the
low-gain direction's code truly costs), (ii) nonlinearity downstream of W making
cross-basis reconstructions unequal, (iii) rank-1 output-side separation (different
output DIRECTIONS may need output-side dictionary structure, not input-side).

### Tier 1.2 salvage (i) gain-weighted L1: FAILED too
Weighted penalty λ·||W d_j|| has the right objective-level sign (aligned basis costs
4.25λ vs rotated 5.68λ) but 3-seed results unchanged (0.63-0.66 ≈ SAE). The preference
is too weak vs local optimization dynamics. VERDICT stands: no mechanism-side rescue
for co-occurring features in the rank-1 masked-projector family as optimized here.
Remaining salvages (downstream nonlinearity; output-side dictionaries) deferred —
not blocking; the method's honest scope is: separability requires features to occur
separately somewhere (which real data mostly provides), plus the 1.1 novelty detector
for the rest.

## Tier 1.3 PASS (strong) + Tier 1.4 PASS
- 1.3 superposition: 100 random features in d=64 (1.56× overcomplete), m=200 atoms →
  pruned to EXACTLY 100, all recovered at cos>0.9 (mean max-cos 1.000), R² 0.992.
  The mechanism dictionary handles superposition perfectly at this scale.
- 1.4 cheating check: 5 seeds, ZERO support-matching-but-unaligned atoms (min-cos
  ≥0.9993, F1 1.000 every seed) — the pinned parameterization removes the APD cheating
  channel as claimed; recovery is also perfectly seed-stable.
Tier 1 summary: 1.1 ✓ (novelty AUROC .994), 1.2 ✗ falsified (documented + salvages
tried), 1.3 ✓ (perfect superposition recovery), 1.4 ✓ (no cheating). 1.5/1.6 optional,
skipped for now. NEXT: Tier 1.5 (our bilinear models — contraction-matrix circuit
recovery vs the causally-verified L0H3→L1H2 ground truth).

## Tier 1.5 run 1: NOT a verdict — decomposition failed to fit (config regime)
First real-model attempt (attn2-seed0, 4× L0-OV + L1H2 K1/K2, 10k activation points):
OV R² only 0.06-0.46 — the toy-tuned prune guard (absolute 2e-3) ate the dictionary
(1-12 atoms; codes fire on a minority of points). Real RMSNorm'd residuals are not
toy-sparse. Contraction gate FAILED but is uninformative at this R² — reconstruction
before circuit claims. Also fixed en route: solve_codes built an autograd graph when W
comes from live model parameters (9.5 GB leak; W now detached + solver @torch.no_grad —
critical for Pythia/Gemma tiers). Rerunning at λ=0.01, m=128, prune_tol=1e-4.

## Tier 1.5 verdict: PARTIAL — K2 recovers L0H3, K1 does not; the gap is DATA-CONDITIONING

Retuned decomposition fits the keys well (K1 R² 0.99, K2 0.93); OV maps fit modestly
(0.10-0.59, rank-limited by d_head). Contraction gate: K2 PASS (L0H3 dominant), K1 FAIL
(L0H0 dominant). Dictionary-FREE confirmation (project each head's OV output on real
activations onto L1H2's K row spaces, weight by write magnitude): SAME result —
K1 max is L0H0 (0.54), K2 max is L0H3 (0.24). So this is a property of the weights on
GENERIC activations, not a method artifact.

Reconciliation with the causal ground truth (which said ONLY L0H3@src matters, .434→.031):
the causal test conditioned on INDUCTION datapoints (source positions carrying the
matched token). The L0H3→L1H2 induction pathway is a DATA-CONDITIONED MINORITY direction;
on generic val tokens L0H0's larger generic writing into K1 dominates the aggregate. This
is exactly the regime the method is *designed* for (data-conditioned decomposition) — but
Tier 1.5 fed it generic activations, so it measured the generic geometry, correctly.

FIX (next tick): condition the decomposition/contraction on the induction datapoints
(the gated_depth2 source positions), matching the causal test's conditioning. Prediction:
L0H3 dominance appears in BOTH branches once the data matches. This is the sharper claim
anyway — the method should recover the circuit that is causal ON THE RELEVANT DATA.
QUESTION FOR LOGAN (LOG.md): is data-conditioned contraction (decompose on the datapoints
where the circuit fires) the intended reading of §1.5 for circuit discovery? I believe yes.

## Tier 1.5 RESOLVED (conceptually): contraction magnitude ≠ causal circuit; SELECTIVITY does

Three contraction variants all rank L0H0 > L0H3 into K1 (data-conditioned on source
positions did NOT change it — so it's magnitude, not data): raw usage-weighted, write-
magnitude, and query-aligned mean match-contribution (K1: H0 −3.44 ≫ H3 −0.28).
Yet the causal retention table is unambiguous: zeroing L0H0@src leaves L1H2's match at
−0.41 (base −0.434); only L0H3 collapses it (−0.031).

RECONCILIATION (the real finding): L0H0 contributes large but POSITION-CONSTANT match
signal (a baseline that doesn't discriminate the correct source from others — previous-
token content is everywhere); L0H3 contributes the token-IDENTITY signal that SELECTS
the right source. Induction strength = selectivity (Δ match at correct vs other sources),
which is what both the causal ablation and the classic induction-score measure. Mean-
magnitude contraction is the wrong read-off; the circuit lives in the VARIANCE of the
contraction across candidate sources.

⇒ §1.5 contraction, to recover circuits, must be scored by DISCRIMINABILITY (variance of
d_kᵀ W d_j · a across datapoints / candidate keys), not mean |contraction|. This is a
substantive correction to the spec's "circuit strength = read a matrix entry" claim: for
SELECTION circuits (induction, retrieval) the matrix entry must be a variance/contrast,
not a norm. NEXT: implement selectivity-scored contraction (contribution at matched vs
shuffled source), predict L0H3 dominance in both branches. Also: the earlier directional
composition through embed(" j") worked precisely because a single token-identity input
exposes the selective direction — consistent with this account.
QUESTION FOR LOGAN (LOG.md): endorse selectivity/variance contraction as the §1.5
circuit read-off for selection circuits? It's the fix that reconciles method with ground
truth.

## Tier 1.5 COMPLETE: selectivity-scored contraction recovers the causal circuit

SELECTIVITY score (each L0 head's contribution to L1H2's match at the CORRECT source
minus at a shuffled source, averaged over induction queries):
  L0H0 −0.88 | L0H1 −0.62 | L0H2 −0.45 | L0H3 −0.29  → L0H3 most selective, GATE PASS.
(L0H3 best preserves the match specifically at the correct source; the others degrade it
more when misdirected — i.e. their large writes are non-discriminative.) This inverts the
magnitude ranking (L0H0 dominant) and matches the causal retention ground truth (only
L0H3@src collapses the match). Proxy is noisy (linear branch-sum stand-in for the bilinear
product; high sd) — directionally correct, and the causal retention table remains the
clean gold measure.

TIER 1.5 VERDICT: the contraction structure of §1.5 recovers our causally-verified
induction circuit ON THE RELEVANT DATA WITH A SELECTIVITY (variance/contrast) SCORE, not
a magnitude one. Net methodological deliverables for the real-model tiers:
  (1) reconstruction gates circuit claims (run-1 lesson),
  (2) detach W + no_grad solver (leak fix),
  (3) selection circuits need discriminability-scored contraction, not |matrix entry|.
Ready for Tier 2 (Pythia / Gemma OV) — with the caveat that the "circuit = matrix entry"
read-off is magnitude-valid but selection-circuits need the variance form.

## Tier 2 (block2 OWT L0-OV): SOLVER/INIT BLOCKER found — not a data or parameterization limit

Real OWT L0-input residuals are DENSE/high-rank (activation PCA: top-1 5%, top-20 38%,
top-40 55% — no sparse feature structure). On them:
- kmeans-init + sparse alternating solve: R² 0.21-0.30, L0 ~2 (λ 0.005-0.1, m 128-256).
- DENSE codes (λ=0, signed, 256 atoms): R² 0.074 — WORSE, and clearly broken.

Decisive check: the parameterization is COMPLETE — with D = identity (standard basis)
and c≡1, Ŷ = Σ_j x[j]·(W e_j) = W x EXACTLY (R²=1). So R²<0.1 is a SOLVER/INIT failure,
not a method or data limit: the kmeans+lasso alternation cannot find the complete-basis
solution from a random/kmeans start on dense high-rank input (the toys succeeded only
because their data was sparse & low-rank, where kmeans lands near features).

FIX (spec init (b), deferred to next tick): initialize D from the top right-singular
vectors of W X (or identity/PCA of X), which is at/near the exact solution; refine from
there. The toy pipeline used init (c) kmeans (predicted best FOR SPARSE data) — wrong
regime for dense real activations. This is a clean, expected Tier-2 lesson and the 4th
deliverable for real-model tiers: DENSE activations need SVD/identity init, not kmeans.
QUESTION FOR LOGAN (LOG.md): for real dense-activation layers, is the intended use a
sparse OVERCOMPLETE dictionary (many atoms, low L0) or a near-complete basis? R²/L0
frontier only becomes meaningful once init is fixed; flagging before I pick.

## Tier 2 RESOLVED: the blocker is SITE, not init or solver

SVD-init added (spec init (b)). Results:
- identity init + dense codes → R² 1.0000 (parameterization complete, confirmed again).
- SVD-init + sparse solve → still R² 0.17-0.30 at L0 ~1-2, at every m/λ.
Conclusion: NOT init and NOT solver. For DENSE data, sparse codes CANNOT reconstruct
(Wx needs ~effective-rank active atoms; L0=2 caps R² at ~0.3). The site we chose — the
INPUT to L0 = token embeddings — is dense (PCA top-40 = 55% energy) with no sparse
feature structure, so there is no sparse mechanism decomposition to find. This is
consistent, not a failure: the method's premise is sparse features, which embedding-level
activations lack.

KEY LESSON (5th deliverable): SITE SELECTION matters — decompose where features are
sparse (mid-network residual stream / MLP HIDDEN activations, exactly where SAEs operate
via superposition), not the embedding layer. Our tiny models have only 2-6 layers and
128-dim; the sparse-feature regime barely exists in them. ⇒ Tier 2 proper needs a real
LLM with a genuine MLP hidden layer (Pythia-70m/160m or Gemma), where superposition
creates the sparse structure the method exploits. Our OWT bilinear models are good for
the CIRCUIT/contraction test (Tier 1.5, done) but NOT for the sparse-reconstruction test.
NEXT: skip to Pythia-70m, decompose an MLP down-proj input (its hidden activations),
where R²/L0 becomes a meaningful frontier. QUESTION FOR LOGAN (LOG.md): confirm the site
pivot — sparse-reconstruction Tier 2 belongs on Pythia/Gemma MLP-hidden, not our tiny
models (whose value was the Tier 1.5 circuit ground truth, already delivered).

## Tier 2 Pythia-70m (running): correcting the sparsity proxy
Self-correction: raw-activation near-zero fraction is the WRONG sparsity proxy. GELU
MLP-hidden is dense by that measure (0.4% near-zero) — but the method needs FEATURE-
sparsity (data on a union of low-dim subspaces / sparse in an overcomplete dictionary),
which is what SAEs exploit and is NOT visible as near-zero raw entries. So: don't
pre-judge by raw sparsity; run the decomposition and read the R²/L0 frontier directly.
Launched: Pythia-70m L3 MLP down-proj (W: 512×2048, X = post-GELU hidden, 30k Pile
tokens), SVD init, overcomplete m∈{1024,2048}, + top-activating tokens per atom.

## Tier 2 Pythia: CP-DEGENERACY on wide W (spec §1.5 hazard, realized)
First Pythia run diverged: R² → −1e15, codes exploding. Root cause (concrete): the MLP
down-proj W is WIDE (512×2048) with a 1536-dim NULL SPACE. The exact per-atom M-step
(lstsq solving W d = rbar) is underdetermined — atoms wander in null(W), and atoms with
collinear W d (distinct d, same output) grow with cancelling codes = textbook CP
degeneracy. The toys never exposed this (square full-rank W, trivial null space). 6th
real-model deliverable: WIDE maps need degeneracy guards.
FIXES applied: (a) elastic-net ridge on the E-step Gram diagonal (tames WD-collinearity
code blowup); (b) row-space-regularized M-step (W d = rbar via (WWᵀ+εI)⁻¹, projecting the
atom into row(W) so null-space components can't wander). Guarded rerun in progress.

## Tier 2 Pythia-70m VERDICT: method does NOT beat low-rank or yield interpretable atoms here

Stable after degeneracy guards. Numbers (L3 MLP down-proj, W 512×2048, 20k Pile tokens):
| method | R² | L0 |
|---|---|---|
| closed-form rank-21 (dense, no sparsity) | 0.551 | — |
| closed-form rank-50 (dense) | 0.724 | — |
| our masked-projector (m=1024) | 0.489 | 66 |
The method uses 66 effective dims/point yet reconstructs WORSE than a 21-dim DENSE map
(0.49 < 0.55). And atoms are NOT interpretable — top-activating tokens per atom are
incoherent grab-bags ('from'/'Thus'/'87'/'and'; 'Friday'/'('/'Three'/'I'/'Monday'). So
on this site the method fails BOTH Tier-2 soft criteria (competitive reconstruction;
interpretable ablatable atoms).

HONEST DIAGNOSIS (3 candidates, not yet distinguished):
1. SITE: X = pre-down-proj GELU hidden may lack sparse-feature structure; SAEs usually
   decompose the RESIDUAL STREAM / MLP OUTPUT, not the down-proj input. Wrong X.
2. MODEL: Pythia-70m (6 layers, d512) is tiny — weak/absent clean features regardless
   of method. Gemmascope proves good features exist in Gemma-2-2B; Pythia-70m has no
   such guarantee.
3. METHOD: rank-1 input-gated mechanisms may genuinely not capture MLP computation
   better than low-rank.
The spec's ACTUAL Tier-2 target is Gemma-2-2B with Gemmascope SAE comparison —
precisely because good features are known there. Pythia-70m was my cheaper warm-up and
it under-delivered, but that is CONFOUNDED with model size.

QUESTION FOR LOGAN (LOG.md, decision needed): three options —
  (a) jump to Gemma-2-2B MLP-down / OV subset (spec target; ~5GB model, feasible here);
  (b) first retry Pythia on the RESIDUAL STREAM (where SAEs are trained) to test the
      site hypothesis cheaply before Gemma;
  (c) Pythia-160m/410m as a middle step.
My recommendation: (b) then (a) — cheap site-hypothesis test, then the real target. The
method is validated on structured data (Tier 0/1) and circuits (Tier 1.5); the open
question is whether real LM activations have the sparse-mechanism structure it needs.

## Tier 2 site hypothesis CONFIRMED: OV + residual stream → R² 0.97 (vs MLP-down 0.49)

Same Pythia-70m, but decomposing the ATTENTION OV map (rank-64) on the RESIDUAL STREAM
(where SAE features live), not the wide MLP-down on GELU hidden:
| site | R² | L0 |
|---|---|---|
| MLP down-proj (wide 512×2048), GELU-hidden X | 0.49 | 66 |
| OV (rank-64), residual-stream X | **0.969** | 23.6 |
RECONSTRUCTION IS NOW STRONG: 23 sparse atoms reconstruct 97% of the OV action. The
Tier-2 blocker was SITE + wide-W degeneracy, NOT the method — confirmed. Low-rank maps
(OV) on the residual stream are the method's regime; wide maps on dense hidden are not.

REMAINING GAP: interpretability. Atoms are still token-incoherent even here
('ust'/'et'/'so'/'conditional'; 'the'/'PE'/'Those'/'of'). But reconstruction working
now isolates the cause: this is almost certainly Pythia-70m's WEAK FEATURES (6 layers,
d512 — its residual directions are dominated by positional/frequency structure, not
clean semantic features), the confound the spec resolves by targeting Gemma-2-2B where
Gemmascope proves good features exist.

DECISION (resolving the LOG.md question myself given the strong reconstruction result):
reconstruction is validated on the right site → proceed to Gemma-2-2B OV subset for the
INTERPRETABILITY + Gemmascope-SAE comparison (the spec's actual Tier-2 target and the
only place the interp question can be answered against ground-truth-good features).
Gemma-2-2B fp16 ≈ 5GB, fits the 16GB GPU. Deliverables banked so far (7): reconstruction
gates claims; detach+no_grad; selectivity-scored contraction for selection circuits;
dense data needs SVD init; SITE matters (low-rank map + residual stream); wide W needs
degeneracy guards (E-step ridge + row-space M-step); OV+residual is the working regime.

## Gemma tier BLOCKED (gated) — bridging with Pythia-410m
google/gemma-2-2b requires HF auth (401). Flagged for Logan (HF_TOKEN or non-gated
substitute). Bridging test running: Pythia-410m (24 layers, d1024, non-gated) OV+residual
decomposition — if atom interpretability improves vs 70m, the interp gap is model-size
(as hypothesized), and Gemma should deliver clean atoms once access is sorted.

## Pythia-410m bridge: reconstruction scale-robust (R² 0.99), interp NOT fixed by scale

410m OV+residual: R² 0.9897 at L0 47 (70m was 0.969) — reconstruction is site-correct and
scale-robust, confirmed. But atoms are STILL mostly token-incoherent at 6× scale; only a
few are clean (newline/positional, repeated 'Category', 'the'). So the "weak features =
small model" hypothesis is WEAKENED — scale alone didn't fix interp.

Two live explanations now:
1. WEAK PROBE (likely a big part): I'm eyeballing top single BPE tokens by code magnitude.
   Real SAE auto-interp uses MAX-ACTIVATING CONTEXT WINDOWS + the actual token attended/
   copied. For an OV circuit the meaningful content is the SOURCE token's features, not
   the query position's BPE piece. Single-subword-token decoding ('ph','ir','ry') is
   inherently noisy. → upgrade the probe to context windows.
2. GENUINE polysemanticity of OV-on-residual atoms in Pythia.
Distinguishing needs a STRONGER interp measure than eyeballing: (a) max-activating context
windows, (b) the ABLATION-FAITHFULNESS test (spec Tier-2 (iv): remove atom from OV, loss
delta concentrated on atom-active contexts) — doesn't need Gemma, and (c) Gemmascope
max-cos comparison (needs Gemma access).
NEXT (no Gemma needed): ablation-faithfulness test on 410m OV atoms + context-window probe.
This is the real "do the atoms mean something" test; top-token eyeballing is too weak to
conclude either way.

## STABILITY BUG: method intermittently diverges on low-rank W (guards insufficient)
The faithfulness run DIVERGED (R² −3.7e10) — SAME code that gave R² 0.99 two runs earlier.
So those numbers were invalid (garbage atoms). Root cause: OV is rank-64 in 1024-dim →
960-dim INPUT null space; atoms wander there and CP-degenerate. The E-step ridge +
row-space-lstsq M-step guards only INTERMITTENTLY prevent it (stochastic via resample_dead
reinit). This retroactively casts doubt on the "lucky" 0.97/0.99 runs — the method is not
reliably stable on low-rank maps as-implemented.
PRINCIPLED FIX: project atoms onto ROW SPACE of W (top singular vectors) after each M-step
— null-space wandering becomes impossible by construction, not penalized. Added
rowspace_basis + rowspace= arg to update_dictionary. Testing across 3 seeds for stability
(the real gate now: does it diverge on ANY seed?).
NOTE: all "faithfulness/interp" conclusions on hold until reconstruction is STABLE across
seeds. This is the correct gate ordering (reconstruction before claims), re-applied.

## STABILITY FIX CONFIRMED: rowspace projection → 3/3 seeds R² 0.99
Row-space-projected M-step: seed 0/1/2 R² 0.9907/0.9913/0.9914, L0 ~46, ZERO divergence.
The intermittent CP-degeneracy is fixed by constraining atoms to row(W) (64-dim for this
OV) by construction. 8th deliverable: low-rank/wide maps REQUIRE rowspace-constrained
atoms, not just ridge penalties. Now re-running ablation-faithfulness on a TRUSTWORTHY
(stable, R² 0.99) decomposition — the prior faithfulness numbers were on a diverged run
and are discarded.

## Tier 2 Pythia FINAL VERDICT (stable, trustworthy): strong reconstruction, WEAK atoms

On a stable R² 0.9911 decomposition (rowspace fix, no divergence):
- Ablation faithfulness: median 2.6× (active/inactive ||ΔOV·x||), only 18% of atoms >3×,
  4% >5×, max 15×. Compare Tier-0 toys: 32-64×. Real atoms are WEAKLY localized.
- Context-window interp: even highest-faithfulness atoms are incoherent.

Honest program status — the method, thoroughly de-bugged (8 fixes), behaves as:
| regime | reconstruction | atom quality |
|---|---|---|
| Tier 0/1 structured toys | R² 0.99, F1 1.0 | ablation 32-64×, perfectly interpretable |
| Tier 1.5 known circuit | (contraction) | recovers L0H3 via SELECTIVITY score |
| Tier 2 Pythia OV (real) | **R² 0.99, stable** | **ablation 2.6× (weak), not interpretable** |

The gap is REAL and now un-confounded (stability/site/degeneracy all fixed): on real
Pythia-410m the OV map reconstructs sparsely & stably, but its atoms are heavily
OVERLAPPING (each ablation hits many contexts) — the clean sparse-mechanism structure of
the toys is largely ABSENT in real small-LM activations. Two possibilities remain, and
ONLY Gemma+Gemmascope distinguishes them:
  (A) Pythia's features are genuinely weak/polysemantic (→ Gemma atoms would be clean);
  (B) the method's rank-1 input-gated mechanisms don't isolate real features regardless
      (→ Gemma atoms also weak; a real limitation of the approach).
This is the crux the whole tier plan was built to reach. It NEEDS Gemma access.

### DECISION / BLOCKER for Logan (top of queue)
Gemma-2-2b is gated. To finish the program's central question I need ONE of:
  (1) an HF_TOKEN with Gemma access → echo 'HF_TOKEN=hf_...' >> ${WORKSPACE}/.env
  (2) approval to use a non-gated model WITH published good SAEs for the comparison —
      best option: GPT-2-small (many SAE suites) or Pythia-410m/2.8b + EleutherAI SAEs.
  (3) accept the Pythia verdict as-is (strong recon, weak atoms) and conclude the method
      is validated for reconstruction/circuits but not yet for real-feature interp.
Until then, mechdecomp is at a natural stopping point — all cheaper tiers done, the crux
identified, blocked on model access.

## Resolving A-vs-B WITHOUT Gemma: scale+rank-matched synthetic control (running)
Instead of waiting on Gemma access, a decisive control: run the STABLE pipeline on toy
data at real-model scale (d=1024) with a rank-64 W (matching the OV map) and KNOWN clean
sparse features (100 features, |S|≤5). Logic:
- if it recovers cleanly (feature cos>0.9, ablation >5-10×) → the method works at this
  scale/rank, so Pythia's weak atoms are because REAL features aren't sparse-in-basis
  (rules out method-limitation B; the answer is A, and Gemma should show clean atoms).
- if it fails even with KNOWN clean features → method limitation on low-rank W at scale (B),
  and Gemma wouldn't rescue it.
This isolates method-vs-data with zero external dependencies — the smart way around the
access block. Result pending.

## Control run 1 REINTERPRETED: mis-designed, not a method verdict
First control (100 RANDOM features in d=1024, rank-64 W): R² 0.9997 but 0/100 recovery.
This is NOT "method limitation" (my hasty verdict line was WRONG). It's correct behavior:
960 of 1024 input dims are in null(W); features living mostly there are MATHEMATICALLY
INVISIBLE to W (W maps them to ~0), and the rowspace projection correctly refuses to
represent them. This is the method's DESIGNED behavior (spec §1.3: "feature discovery
weighted by mechanistic relevance to this layer" — irrelevant features are ignored).
The fair test puts features IN row(W). Corrected control running (40 features spanning
W's 64-dim row space). This also reframes the PYTHIA result: a single rank-64 OV head
reveals only ~64-dim of feature structure; per-atom recovery of arbitrary features SHOULD
be weak — decomposing one low-rank head is the wrong expectation for broad feature
discovery. Implication if corrected control passes: to discover many features, decompose
a HIGH-rank map (MLP up/down, full rank on d_model) or sum over heads, not a single OV.

## REGRESSION FOUND + FIXED: my stability M-step ridge broke Tier 0 recovery
Regression check (correct discipline after code changes): Tier 0 had DROPPED to min-cos
0.945 (gate 0.99). Cause: the M-step ridge W.T(WWᵀ+0.01I)⁻¹ I added for wide-W stability
PERTURBS the exact solution on full-rank W. Fix: reverted M-step to clean lstsq — the
ROWSPACE PROJECTION (not the M-step ridge) is the real low-rank stability fix. Tier 0
back to PASS (min-cos 0.9994, F1 1.0). This ALSO invalidates the two prior "corrected
control" runs (they used the broken M-step). Re-running the A-vs-B control with Tier 0
verified-clean. Lesson (9th): re-run earlier gates after EVERY solver change — the
stability patch silently regressed recovery.

## A-vs-B RESOLVED: it's the SITE (map rank), not a method limitation. Actionable fix.

Unified pinv M-step (exact full-rank, stable any-rank; Tier 0 re-verified PASS 0.9994).
Controls at real-model scale, all R²≈1.0, numerically sound:
| W | features | recovery @cos>0.9 | mean cos |
|---|---|---|---|
| rank-64 (like 1 OV head), d=1024 | 40 in row(W) | **0/40** | 0.54 |
| FULL-RANK, d=256 | 60 in full space | **60/60** | 0.988 |
DECISIVE: a full-rank map recovers features perfectly at scale; a rank-64 map recovers
none. So Pythia's weak OV atoms are because a SINGLE rank-64 OV HEAD is the wrong target —
its 64-dim row space can't separate features — NOT a method limitation and NOT (only)
weak Pythia features. Answer is "SITE/rank", resolving A-vs-B cleanly.

ACTIONABLE PRESCRIPTION (the real deliverable): decompose HIGH-RANK maps —
  - the FULL attention output (concat all heads' OV, rank up to d_model), not one head;
  - or the MLP as a whole;
where features have room to be separable. This is testable on Pythia NOW (no Gemma):
decompose the full layer-6 attention output map on the residual stream.
10 deliverables banked. This is the method's validated operating regime: full-rank (or
high-rank) maps, rowspace-free, pinv M-step. NEXT: Pythia full-attention-output decomp to
confirm interpretable atoms appear at high rank.

## FULL RESOLUTION: the method needs SPARSE-FEATURE DATA, which real Pythia lacks

Full attention-output map (rank 1016, near-full): R² 0.35 (WORSE than single-head 0.99),
faithfulness median 1.1× (worse than 2.6×), atoms incoherent. This unifies everything:

| map | rank | reconstruction | atom separation |
|---|---|---|---|
| 1 OV head | 64 (low) | R² 0.99 (easy, small output) | POOR — 64-dim can't separate features |
| full attn | 1016 (high) | R² 0.35 (sparse codes can't fit DENSE data) | moot |
| synthetic full-rank | full | R² 0.99 | 60/60 — but DATA was sparse by construction |

THE UNIFYING FINDING: the method needs the DATA to be sparse in a feature basis aligned
with W's structure. Toys satisfy this (sparse by construction) → perfect recovery. Real
Pythia residual/hidden activations DON'T (dense, high-rank: top-40 PC=55%) → either the
map is low-rank (reconstructs but can't separate) or high-rank (can't sparsely reconstruct
dense data). There is no single-map site on Pythia that gives both.

This is the deep answer, and it RE-MOTIVATES GEMMA precisely: Gemma-2-2b's residual stream
is KNOWN (via Gemmascope SAEs) to be sparse in an overcomplete feature basis — the exact
data property the method requires and Pythia apparently lacks. So Gemma is not just
"bigger"; it is the one model whose activations are validated to have the structure the
method needs. If the method produces clean atoms on Gemma residual stream → validated on
real data. If not → a genuine limitation vs SAEs (which decompose activations directly,
x≈Σc·f, rather than W's action).

CONCEPTUAL NOTE: this method decomposes W's ACTION (Wx≈Σc_j(d_j·x)W d_j — features WEIGHTED
BY mechanistic relevance to W), fundamentally different from SAEs (x≈Σc_j f_j — features in
x directly). The method's premise (data sparse in a W-aligned basis) is stronger than the
SAE premise (data sparse in some basis). Pythia may satisfy the SAE premise but not the
W-aligned one at these sites.

### PROGRAM STATUS: envelope fully characterized; decisive test = Gemma residual stream.
10 deliverables. Method VALIDATED on: structured/sparse data (Tier 0/1 exact), known
circuits (Tier 1.5 via selectivity contraction), reconstruction of low-rank maps. OPEN &
DECISIVE: does it produce clean atoms on Gemma's SAE-validated-sparse residual stream?
NEEDS the HF token (LOG.md). This is a clean handoff point — the autonomous exploration
has mapped the method's operating envelope and identified the single decisive experiment.

## SAE baseline (matched compute): also fails to sparsify Pythia-410m residual
Vanilla ReLU SAE, 3k steps, m=4096 on L6 residual: R² 0.999 but L0 **1918** (features fire
on ~half of tokens — NOT sparse) and features incoherent, same as my method's atoms.
Honest read: with matched MODEST compute, neither the method nor a quick SAE finds clean
sparse features here. NOT a clean method-vs-SAE verdict (published SAEs use 100M+ tokens +
tuned sparsity) — but it shows Pythia-410m residual resists CHEAP sparse decomposition,
consistent with "needs a well-trained SAE or a stronger-feature model (Gemma)".

---

# PROGRAM SUMMARY — mechanism decomposition (autonomous run, for Logan)

**What the method is:** unsupervised decomposition of a weight map W into rank-1
mechanisms W d_j d_jᵀ, discovered SAE-style (shared dictionary + sparse codes) but
decomposing W's ACTION (Wx≈Σ c_j(d_j·x)W d_j) rather than activations. Spec:
mechanism_decomposition_spec.md.

**VALIDATED (all gates pass):**
- Tier 0: closed-form theorem exact; recovery cos>0.999, F1 1.0; ablation 32-64×. (float64)
- Tier 1.1: novelty detector AUROC 0.994 (tied-coeff merge case).
- Tier 1.3: 100 superposed features in d=64 recovered perfectly.
- Tier 1.4: zero cheating atoms (pinned parameterization works).
- Tier 1.5: recovers the causally-verified L0H3→L1H2 induction circuit — via a SELECTIVITY-
  scored contraction (magnitude fails; a spec §1.5 refinement).
- Scale: full-rank map at d=256 recovers 60/60 features.

**FALSIFIED / LIMITS (honest):**
- Tier 1.2: NO mechanism-side rescue of co-occurring features (2 salvages tried). The
  method's identifiability is not better than SAEs for co-occurring features.
- Real Pythia (Tier 2): no clean interpretable atoms. Root cause CHARACTERIZED: the method
  needs data sparse in a W-aligned basis; real dense activations don't provide it — low-rank
  maps can't separate features (64-dim), high-rank maps can't sparsely reconstruct dense
  data (R² 0.35). ⚠️ RETRACTED: "a matched-compute SAE also fails to sparsify" — my Pythia
  SAE was UNDER-TRAINED (L0 1918, i.e. never sparse); a properly-trained SAE
  (gpt2-small-res-jb) hits L0 56 @ R² 0.99 on a real residual stream. That run established
  NOTHING about Pythia's data sparsity. See mechdecomp_pythia_postmortem.md §2 Run 5.

**10 methodological deliverables** (corrections the spec needs): float64 for theorem;
composite-atom shortcut → backward-selection prune; SVD init for dense data; detach W +
no_grad (autograd leak); selectivity-scored contraction for selection circuits; wide/low-
rank W → rowspace-projected atoms + pinv M-step (both degeneracy AND regression fixed);
re-run gates after every solver change; SITE/rank determines recoverability; the method
needs W-aligned sparse data.

**THE ONE OPEN QUESTION (decision for Logan):** does the method produce clean atoms on
data KNOWN to be sparse-in-feature-basis? Only Gemma-2-2b residual (Gemmascope-validated
sparse) tests this definitively. Needs: HF_TOKEN for gemma-2-2b, OR approve GPT-2-small
(has SAEs) as substitute, OR accept the current envelope characterization as the result.
Until a decision, the program is complete for everything reachable without model access.

## Tuned SAE also can't sparsify in-budget (L0 1891) — but clean features DO exist
Annealed-penalty SAE (12k steps, unit-norm decoder): R² 0.9997 still at L0 1891 — quick
SAE training can't reach low L0 here (known finicky; real SAEs use ghost-grads/long runs).
BUT several CLEAN monosemantic features emerged: whitespace, numbers, '(' , ':' , '/' ,
punctuation detectors. So Pythia-410m residual DOES contain clean features; the barrier is
that neither my method NOR a quick SAE sparsely isolates them at this compute.

## PROGRAM CONCLUSION (autonomous run complete for reachable scope)
Two independent baselines (my method; two SAE attempts) all fail to produce a clean SPARSE
interpretable decomposition of Pythia-410m residual in-budget, while all succeed on
structured toy data. The decisive real-data test genuinely requires INFRASTRUCTURE we don't
have quick access to: pretrained good SAEs (Gemmascope → Gemma, gated) OR a proper multi-
hour SAE training run with the right techniques. Quick autonomous experiments have reached
diminishing returns; further quick variants won't resolve the crux.

STOPPING aggressive experimentation. Program deliverables are complete and documented:
method validated on toys/circuits (10 deliverables), real-data envelope characterized,
decisive test identified and its requirements specified. Awaiting Logan's decision (LOG.md):
HF token for Gemma, GPT-2-small substitute, a budgeted proper-SAE run, or accept as-is.
The heartbeat will idle-check rather than launch speculative runs until then.

## UNBLOCKED via sae_lens: GPT-2-small + pretrained res-jb SAE (non-gated!)
Installed sae_lens; loaded gpt2-small-res-jb SAE (24576 known-good interpretable features
at blocks.6.hook_resid_pre). This provides the "data KNOWN sparse-in-feature-basis" I
claimed only Gemma could — non-gated. The decisive Tier-2(iii) test is now runnable:
(1) verify GPT-2 L6 residual is sparse in the SAE basis (SAE L0/R²),
(2) decompose a map at that site with my method,
(3) max-cos comparison: do my atoms align with the SAE's validated features?
Setup running (SAE sparsity check + activation cache). This RESOLVES the crux without
Gemma access.

## DECISIVE TEST (GPT-2-small + res-jb SAE, non-gated): atoms do NOT align with SAE features

Data confirmed sparse-in-feature-basis (res-jb SAE: R² 0.99, L0 56 on TL-processed L6
resid). Decomposed L6 MLP-in map (rank 767), compared 512 atoms to the SAE's 24,576
validated features by max-cos:
  atoms→SAE: median 0.16, frac>0.5 = 0% | random baseline: median 0.15.
Atoms are indistinguishable from RANDOM directions w.r.t. the known-good feature basis.
Caveat: this run's gradient M-step DEGRADED (R² 0.997 init → 0.91, L0 294 not sparse), so
it's on a sub-optimal decomposition — but even the svd-init (R² 0.997) gives PCA-like
principal directions, which are known not to align with sparse SAE features. Both readings
point the same way.

## HONEST FINAL CONCLUSION of the mechanism-decomposition program

The method is mathematically sound and clean on STRUCTURED data (Tier 0/1 exact; circuits
recovered via selectivity contraction), but has NOT been shown to recover real
interpretable features:
1. Every real-model run (block2, Pythia-70m/410m, GPT-2) hit a DISTINCT solver instability
   the toy tiers never exposed (dtype, scale/normalization, wide-W CP-degeneracy,
   low-rank null-space wandering, a stability patch that regressed recovery, per-atom
   M-step non-scaling, gradient M-step divergence). ~13 engineering issues total.
2. Where decompositions completed on data with KNOWN-GOOD sparse features (GPT-2 + res-jb),
   atoms do NOT align with the validated feature basis (≈ random).
3. The method decomposes W's ACTION and its atoms tend toward the map's principal/PCA-like
   directions, which are fundamentally different from (orthogonal to the character of)
   sparse SAE features.

VERDICT: the approach is a valid, exact tool for structured/toy settings and for
circuit-contraction analysis, but its real-model feature-discovery claim is UNSUPPORTED as
implemented — the optimization is fragile on real high-dim activations AND, where it works,
it doesn't find the sparse feature basis (it finds map-principal directions). Making it work
on real models needs (a) a robust production solver and (b) likely a reformulation that
biases atoms toward sparse-feature (not principal) directions — substantial research, not a
quick experiment.

This is where the autonomous run honestly ends. The spec's Tier-2/3 "competitive with SAEs,
interpretable atoms" bar is NOT met on real models by the current method+implementation. All
20+ experiments, fixes, and this verdict are documented. Recommend to Logan: treat the
positive results (exact toy recovery, circuit contraction) as the validated contribution;
treat real-model feature discovery as an open problem needing solver + objective work.

## VERIFICATION of the final verdict (the load-bearing claim, now MEASURED not asserted)
I claimed "even the clean svd-init gives PCA-like directions that don't align with SAE
features." Measured directly (GPT-2 L6, MLP-in map, vs 24,576 res-jb features):
| atoms | median max-cos to SAE | frac>0.5 | random baseline |
|---|---|---|---|
| svd-init top-64 (clean, R²≈1) | 0.229 | **0%** | 0.150 |
| svd-init top-256 | 0.163 | 0% | 0.149 |
| svd-init top-512 | 0.153 | 0% | 0.149 |
| trained atoms (512) | 0.162 | 0% | 0.148 |
Scale reference: the SAE's OWN features have median inter-feature overlap 0.539 — a
genuine feature match would read ≥0.5. NOTHING from the method reaches it; everything sits
at the random-direction floor.

CONCLUSION CONFIRMED (not a solver artifact): the map's principal/action directions are a
fundamentally different object from the sparse feature basis, and the masked-projector
objective — which ties atoms to reconstructing Wx — gravitates to the former. This is
structural, not an optimization failure. The honest final verdict stands:
  • VALIDATED: exact closed-form theory, toy recovery, superposition, no cheating,
    circuit recovery via selectivity-scored contraction.
  • REFUTED for real models: the atoms are not the interpretable sparse features; the
    method as specified does not do SAE-style feature discovery on real activations.
  • Making it work needs an objective reformulation that biases atoms toward sparse-feature
    (not principal) directions — a research problem, clearly scoped.
PROGRAM CLOSED. No Gemma needed; the question was answerable and is answered.

## Detailed post-mortem written
`mechdecomp_pythia_postmortem.md` — module-by-module account of the Pythia Tier-2 runs:
exact W/X/hook sites, where each run broke (M-step CP-degeneracy vs the later structural
failure), the failure taxonomy (5 numerical bugs, all fixed, vs 1 structural limit), and a
RETRACTION of the "matched-compute SAE also fails to sparsify" claim (my SAE was
under-trained; it proves nothing about Pythia's data).

---

# ⚠️ VERDICT RETRACTED (Logan review, 2026-07-09). Program REOPENED.

The "structural failure, program closed" conclusion is **over-concluded**. Logan's review
identified four errors, all of which I accept:

1. **Metric drift.** The spec listed Gemmascope max-cos as ONE of five Tier-2 metrics and
   explicitly framed *disagreement* as an interesting output (the method is *designed* to
   ignore features W treats as noise). Blocked from Gemma, I promoted max-cos-vs-a-pretrained-
   SAE into the single decisive pass/fail oracle. SAE features are not ground truth (splitting,
   absorption, their own prior), and non-alignment is *partially predicted by the method's own
   premise*.
2. **SITE CONFOUND (unflagged).** My atoms lived in `W_in`'s input space (post-ln2, AFTER the
   attention block writes); res-jb features live at `resid_pre`. I compared directions across an
   attention block AND a LayerNorm — this mechanically deflates max-cos. The "decisive" test
   used a questionable ground truth at a mismatched site.
3. **The load-bearing measurement was taken on a FAILED OPTIMIZATION.** The GPT-2 run's
   gradient M-step *degraded* (R² 0.997 → 0.91) and ended at **L0 ≈ 294** — never in the sparse
   regime. Every toy success used the careful recipe (exact M-step, dedup, backward-selection
   prune, L0 near truth); the GPT-2 run had none of it. I concluded "structural, not
   optimization" from evidence collected on a broken optimization.
4. **The svd-init argument is vacuous.** It is PCA of WX evaluated with *dense* codes; the
   completeness check already told us dense codes give R²≈1 trivially. No one hypothesized PCA
   directions would be features.

**The feasibility argument that kills the verdict (Logan).** Set `d_j` = SAE decoder directions
and `c_ij = s_ij / (f_jᵀ x_i)`. That is an approximately feasible masked-projector solution at
**L0 ≈ 56, high R²** — strictly dominating the L0 294 / R² 0.91 my solver found. When a known
feasible point beats the found solution by that margin, one cannot conclude the objective
"structurally gravitates to principal directions"; one has shown **the optimizer cannot find the
feature basin from SVD/random init on real data.** That is an initialization/optimization result,
and an unsurprising one — sparse coding is exactly as non-convex as SAE training, which itself
needs 100M+ tokens and tricks.

**Also: the original plan was silently replaced.** The literal plan (sparse-dictionary
reconstruction of the per-datapoint rank-1 maps `M_i = (Wx_i)x_iᵀ`) was replaced by the
masked-projector objective. These are NOT equivalent: the masked projector reconstructs only the
output vector `Wx_i`, so the input side enters solely through scalar gates that free codes can
compensate. An SAE on the `M_i` reconstructs the **outer product**, pinning input-side directions
in the target. Tier 1.2's basis degeneracy and the drift-to-principal-directions are both symptoms
of the masked projector's **weak input-side constraint**. The literal SAE-on-M_i was never run, so
the verdict never covered it.

**One genuine structural quirk survives, and it's a one-line fix.** The L1 penalty is on `c`, but
the effective coefficient is `c·(dᵀx)`. The sparsity geometry is *gate-warped*: atoms with small
gates pay extra for the same output contribution — a real bias against feature directions with
modest projections. Implemented as `solve_codes(..., gate_weighted_l1=True)`.

## The adjudicating experiments (running: `mechdecomp/feasibility.py`)
Site-corrected: work in **ln1-normalized space** (the literal input to attention's W_V; TL folds
the LN scale into W_V, leaving `ln1(x) = P x / std(x)` with P a fixed centering projector and
1/std absorbed into the free codes). Feature directions map linearly as `normalize(P f_j)`.
W = **full attention OV** (all heads, high rank), which reads exactly that space.
  (0) sanity: SAE at its own site hits published L0/R².
  (1) **explicit feasible point** `c_ij = s_ij/(d_jᵀx̃_i)` — no solving. If R² is high at L0≈56,
      the feature basis IS a near-optimal point of this objective and the structural claim is DEAD.
  (2) **pinned-D E-step**: dictionary fixed to features, codes solved. Does the lasso find it?
  (3) baselines: relevance-blind random SAE features, svd-init principal dirs, random dirs.
  (4) **gate-warp fix**: plain L1 on c vs L1 on the effective coefficient c·|dᵀx|.
Then: release D and check whether refinement walks away from the feature basis AND whether the
loss actually improves (distinguishes "features aren't optimal" from "optimizer wanders").
Queued: the literal **SAE-on-M_i** control on the same 20k points (the original plan).

**KEPT WITHOUT DISPUTE:** closed-form theorem verification; toy + superposition recovery;
no-cheating result; selectivity-scored contraction for the induction circuit (a real spec
correction); solver deliverables (rowspace projection, pinv M-step, normalization, and the
re-run-gates-after-solver-changes discipline).

**STATUS: not closed. The objective has a feasibility question one pinned-dictionary run answers.**

## Feasibility run v1: MY construction was buggy (not evidence against the argument)
`feasibility.py` gave the "explicit feasible point" R² = **−2.38** at L0 30. That is an
implementation error on my side, not a refutation of Logan's argument. The identity
Σ_j c_ij (d_jᵀx̃_i)·W d_j = W x̃_i requires THREE terms I dropped:
  (i) the decoder norms n_j = ‖P W_dec_j‖ (I unit-normalized atoms but never rescaled codes),
  (ii) the per-datapoint 1/std_i introduced by LayerNorm,
  (iii) the decoder bias b_dec (the masked projector has no bias; it needs a bias atom).
Correct point: **c_ij = s_ij·n_j / (std_i·(d_jᵀ x̃_i))**, plus a bias atom for W·P·b_dec/std_i.
Rerunning as `feasibility2.py`, which also reports a parameterization-free UPPER BOUND —
push the SAE reconstruction through the true map, W·(P x̂/std) vs W·x̃ — since that bounds what
ANY feature-basis solution can achieve at the SAE's L0.
Other v1 readings (valid): pinned SAE-feature dict, solved codes → R² 0.969 @ L0 565;
random SAE features → 0.993 @ L0 722; svd-init → 0.9994 @ L0 990. All are DENSE-ish; none is
in the L0≈56 regime, which is exactly the question the corrected feasible point answers.

## Feasibility v2 (site-correct): partial, and it reframes the question
```
(0) SAE at its own site            R2 0.9924   L0 56.2   ✓ sanity
(A) UPPER BOUND  W·(P x̂/std)      R2 0.7756   L0 56.2   ← what ANY feature-basis point can do
(B) feasible pt, masked-projector  R2 0.343/0.386 (no bias / +bias)  L0 40  ← still buggy, see below
(C) pinned SAE-feature dict, λ.02  R2 0.969    L0 565
    random SAE features            R2 0.993    L0 722
    svd-init principal dirs        R2 0.9994   L0 990
```
**(B) is still not exact:** algebraically (B)+bias MUST equal (A) (both are W·(P x̂/std)). It
doesn't, because I formed c = s·n/(std·a) with a clamp on near-zero gates a, so c·a ≠ s·n/std,
and I truncated to K=4096 features. The objective only uses the PRODUCT c·a, so v3 sets that
product directly and uses all features. **Two of my three feasibility implementations have now
had bugs — the algebra is unforgiving here; every future claim gets the (A)-vs-(B) consistency
check as a gate.**

**Two real findings already:**
1. **(A) = 0.776, not ~0.99.** Pushing the SAE's reconstruction through the true map costs a
   lot: the SAE is 99% accurate on x, but W amplifies exactly the directions it misses. So a
   *feature-basis* solution at L0 56 tops out near R² 0.78 on Wx̃. This is a genuine, previously
   unmeasured quantity — and it is the honest ceiling for "features as mechanisms" at this site.
2. **No solved solution is anywhere near L0 56.** Every dictionary — including the pinned
   feature dictionary — lands at L0 565-990. The λ range I used never enters the sparse regime.
   So the earlier GPT-2 comparison (L0 294) was not just "a failed optimization"; the *whole
   sweep* was off the sparsity scale where the feature basis lives.

**⇒ The adjudicating comparison is MATCHED-L0, which I had not run.** 0.969 @ L0 565 vs 0.776 @
L0 56 are different points on a tradeoff curve, not a comparison. v3 sweeps λ so every dictionary
is evaluated at L0 ≈ 56/120/300 and asks: at matched sparsity, does the feature basis DOMINATE
what the solver finds from svd/random init? If yes → the basin exists and the optimizer misses it
(Logan's diagnosis, structural claim dead). If no → features are genuinely not optimal for THIS
objective (which would then indict the masked projector's weak input-side constraint, not features).

## Feasibility v3: identity check PASSES; the feasible point is confirmed
```
(0) SAE at own site                    R2 0.9923  L0 56.3
(A) UPPER BOUND W·(P x̂/std) vs W·x̃    R2 0.7754  L0 56.3
(B) EXACT feasible point (product c·a set directly, all 24576 features)
      no bias atom                     R2 0.7294  L0 56.3
      + bias atom                      R2 0.7754  L0 57.3   ← EQUALS (A) ✓ identity gate passes
```
So **Logan's feasible point is real and now verified**: the feature basis is a masked-projector
solution at **L0 57.3 with R² 0.7754**. My v1/v2 numbers (−2.38, 0.386) were construction bugs.

**But the λ sweep never reached that sparsity.** Even at λ=0.5:
```
(C)  SAE-feature dict (pinned)   L0 311.6  R2 0.9297
     svd-init principal dirs     L0 369.0  R2 0.9850
```
Two consequences:
1. Every prior "found solution" (incl. the L0 294 GPT-2 run that carried the structural verdict)
   lived at L0 ≳ 300. **The solver was never once evaluated in the L0≈57 regime.** The old verdict
   compared a dense solution against a sparse-basis hypothesis. That comparison was meaningless.
2. At L0 ~312-369 the solver beats the feature point on R² (0.93/0.985 vs 0.775) — but that is a
   *denser* solution, so it says nothing yet.

**The decisive number is R² of the solver AT L0≈57**, which requires λ ≫ 0.5. Rerunning with
λ up to 20 and printing the full R²(L0) frontier for each dictionary.
  - If solver@L0 57 < 0.775 → the feature basis DOMINATES at matched sparsity ⇒ a better sparse
    optimum exists that the optimizer never approaches ⇒ Logan's diagnosis confirmed, "structural"
    is dead, and SAE-init-then-refine is the production recipe.
  - If solver@L0 57 > 0.775 → features are genuinely not optimal for THIS objective, which
    indicts the masked projector's weak input-side constraint (→ the literal SAE-on-M_i control),
    NOT a vindication of the old verdict.
Note an expected asymmetry to keep honest: PCA is *optimal* for L2 reconstruction, so principal
directions having an edge on Wx̃ reconstruction is unsurprising and is precisely why "reconstruct
Wx" may be the wrong discovery signal — the spec's own premise (§1.3) is that mechanistic
relevance, not variance explanation, should select the atoms.

## ★ DECISIVE: matched-L0 comparison. The feasible point DOMINATES. Structural claim is DEAD.

GPT-2 L6, W = full attention OV, X = ln1(resid_pre) (site-correct), N=6k, K=1024.
Identity gate passed, so the feasible point is trustworthy.

| solution | L0 | R² on W·x̃ |
|---|---|---|
| **feature basis + SAE's own codes (Logan's feasible point)** | **56.0** | **0.7767** |
| svd-init principal dirs + lasso | 56.8 | 0.6618 |
| SAE-feature dict + lasso | 59.3 | 0.6120 |
| random dirs + lasso | 59.4 | 0.5524 |

**Conclusions, in order of importance:**
1. **A sparse solution at L0≈56 with R² 0.777 EXISTS, and no solver run ever came near it.**
   Every solution the optimizer found at matched L0 is strictly worse (0.55-0.66). The claim
   "the objective structurally gravitates to principal directions" is REFUTED: at the sparsity
   the argument is about, the FEATURE basis beats the principal basis (0.777 vs 0.662).
2. **My old verdict was taken on the wrong side of a crossover.** In the dense regime (L0
   300-990) principal dirs win (0.985-0.999); in the sparse regime (L0≈56) features win. Every
   number in the "structural failure" writeup — including the L0 294 GPT-2 run — lived in the
   dense regime. The regime was never the one the argument concerned.
3. **The failure localizes to the E-step, not just the init.** Even handed the CORRECT
   dictionary, the lasso reaches only 0.612 vs the SAE codes' 0.777 on the same atoms. So
   "SAE-init then refine" is insufficient as stated: the CODES must be seeded too, and the
   code-solve itself is deficient. Two candidate mechanisms, both testable:
     (a) **lasso shrinkage bias** — L1 shrinks surviving coefficients, degrading R² at fixed
         support. Standard fix: debias (OLS refit on the selected support).
     (b) **gate-warp** (Logan) — the penalty is on c but the effective coefficient is c·(dᵀx),
         so small-gate atoms are over-penalized. Fix already implemented: gate_weighted_l1.
Running both now.

**Status: the program's central negative claim is retracted and refuted by measurement.** What
survives: the objective is *feasible* on real data at real sparsity with a feature dictionary;
the optimizer (init AND code-solve) is what fails.

## E-step diagnosis: shrinkage is only ⅓ of the gap; SUPPORT SELECTION is the rest
On the pinned SAE-feature dictionary at matched L0:
```
plain L1                       L0 59.3  R2 0.6120
plain L1 + DEBIAS (OLS on support)  L0 59.3  R2 0.6525   ← shrinkage removed
feasible point (SAE's own codes)    L0 56.0  R2 0.7767   ← target
```
Debiasing recovers ~0.04 of the 0.165 gap. The remaining ~0.12 means **the lasso selects a
WORSE SUPPORT than the SAE does** — even though the lasso is optimizing Wx-reconstruction and
the SAE's support was chosen for x-reconstruction. This is the classic **L1-vs-L0 gap**: the
lasso solution is the global optimum of the *L1-penalized* problem (it's convex per datapoint),
but L1-optimal supports are not best-subset supports at matched cardinality.

⇒ The E-step's L1 relaxation is the wrong support selector for this objective. Running:
  (1) OLS on the SAE's own support (best coefficients on those atoms — upper bound for it),
  (2) OLS on the lasso's support (isolates the support-selection gap),
  (3) OMP (greedy, k=56) — a direct L0-targeting selector.
If OMP approaches 0.777, the production recipe is: SAE-feature dictionary + an L0-targeting
code solver (OMP / IHT / seeded-from-SAE), NOT L1 as specified in §3.

## Support diagnosis v1: my own bug again (dictionary truncation)
"OLS on SAE support → R² 0.357 @ L0 20.7" is impossible: OLS on a support must beat ANY specific
coefficients on that support (the feasible point, 0.779). Cause: I restricted the dictionary to
K=1024 atoms, so the SAE's ~56-feature support was TRUNCATED to ~21 features. Not a result.
(Third construction bug in this line of work. The identity/consistency gates are earning their
keep — every feasibility-style number now gets a "must-dominate" sanity check before it is read.)
Rerunning with the FULL 24,576-atom dictionary for the support tests, plus a fixed OMP.

## ★★ CONSTRUCTIVE RESULT: the method WORKS with an L0-targeting solver. The spec's L1 E-step is the failure.

Corrected support diagnosis (full 24,576-atom dictionary; sanity gate OLS-on-support ≥ feasible ✓):

| solution at L0 ≈ 56 (GPT-2 L6, W = full attn OV, site-correct) | R² on W·x̃ |
|---|---|
| **OMP greedy (L0-targeting; candidates = SAE features)** | **0.9017** |
| OLS on the SAE's own support | 0.8686 |
| feasible point (SAE's own codes) — Logan's construction | 0.7852 |
| lasso (spec §3 E-step) + OLS debias | 0.6653 |
| lasso as specified | ≈0.61 |

**What this establishes:**
1. **The masked-projector objective is fine.** At L0 56 it admits solutions with R² ≈ 0.90 whose
   atoms ARE interpretable SAE features (OMP selects among them). "Feature discovery weighted by
   mechanistic relevance" is achieved: OMP picks, per datapoint, the features W actually acts on.
2. **The spec's §3 E-step (L1 lasso) is the primary failure**, costing ~0.29 R² at matched
   sparsity. This is the classic L1-vs-L0 gap: the per-datapoint lasso is the exact global optimum
   of the *L1-penalized* problem (convex), but L1-optimal supports are not best-subset supports.
   Debiasing (OLS refit) recovers only ~⅓ of the loss; the rest is bad support SELECTION.
3. **OMP even beats the SAE's own support** (0.902 vs 0.869) — a W-aware selector finds a better
   subset of features than x-reconstruction does. That is exactly the spec's premise (§1.3):
   mechanistic relevance ≠ activation-reconstruction relevance. The disagreement is the product,
   as the spec originally said, not a failure.
4. **Every claim of the retracted "structural failure" verdict is now explained**: it was measured
   in the dense regime (L0 ~294-990) with a degrading solver, at a mismatched site, against a
   metric the spec never made decisive. In the sparse regime with a proper solver, the method does
   what it was designed to do.

**REVISED PRODUCTION RECIPE (replaces spec §3's lasso E-step):**
  dictionary ← SAE decoder directions (mapped to W's input space; no dictionary learning needed
               to get here), codes ← OMP / iterative hard thresholding targeting L0 directly
               (optionally seeded from the SAE's support), + gate-weighted L1 if a convex
               relaxation is wanted (penalize c·|dᵀx|, the effective coefficient — Logan).
  Dictionary learning (M-step) then becomes a *refinement* on top of a good starting point, which
  is also the only regime where its exactness properties (Tier 0) were ever validated.

**NEXT (queued):** (a) are OMP-selected atoms a mechanistically-meaningful SUBSET of the SAE's
active features, or different ones? (the spec's "interesting disagreement" analysis);
(b) release D and check whether refinement improves the loss or wanders;
(c) the literal SAE-on-M_i control (input-side constraint), still the untested original plan.

## Disagreement analysis (spec §2(v)): the predicted "W-noise" signature is NOT there — and why that matters

400 datapoints, OMP k=56 over 4096 SAE-feature candidates, vs the SAE's own active set (~54):
```
OMP ∩ SAE-active overlap:  10.6%  of the 56 selected atoms  (≈6 atoms)

mechanistic relevance ||W d_j||   n        median
  kept   (SAE-active & OMP-picked)   2,371   2.665
  DROPPED(SAE-active, OMP rejected) 19,381   2.498   ← spec predicts these are W-noise (LOW)
  added  (OMP-picked, SAE-inactive) 20,029   2.478
  all features                               2.313
```
**The spec's §2(v) prediction fails**: features the SAE activates but the method rejects are NOT
low-relevance (2.498 vs kept 2.665, added 2.478 — all near the global median). So the
disagreement is not mechanistic filtering.

**What it actually is — and it's Logan's diagnosis again.** OMP selects atoms mostly OUTSIDE the
SAE's active set (89% of picks). It can do this because the masked projector's input side enters
only through the scalar gate d_jᵀx_i, which is nonzero for *almost every* direction. Reconstructing
Wx_i does not require the atom to be a feature that is *on* for x_i. **This is precisely the weak
input-side constraint Logan identified**: the objective cannot tell "feature j is active here" from
"direction j happens to have a usable projection here." Consequences:
  * Sparse decompositions of Wx are massively NON-UNIQUE over an overcomplete dictionary. OMP's
    0.902 and OLS-on-SAE-support's 0.869 are different supports with near-equal loss.
  * ⇒ Reconstruction quality alone does NOT identify which features W uses. A mechanistic story
    read off a reconstruction-selected support is not identified. (This is a real limitation of
    the masked projector, distinct from — and more interesting than — the retracted verdict.)

**The interpretable recipe is therefore NOT OMP.** Use the SAE's own active support per datapoint
and let W weight it:
```
  OLS on SAE-active support (mechanistic reweighting)   R² 0.8686  @ L0 54.4   ← atoms ARE the
  OMP free selection                                    R² 0.9017  @ L0 56.0     token's own features
```
Giving up 0.03 R² buys identifiability: every atom is a feature the SAE says is active on that
token, and W supplies the weighting. That *is* "feature discovery weighted by mechanistic
relevance", and it needs no dictionary learning at all.

**⇒ This makes the literal SAE-on-M_i control (the ORIGINAL plan) the key open experiment**, since
reconstructing the outer product (Wx_i)x_iᵀ pins the input-side direction in the target and would
remove exactly the degeneracy measured above. Running it next.

## SAE-on-M_i control (the ORIGINAL plan) — running. Why it's the right test.

Target the per-datapoint rank-1 map M_i = (W x_i) x_iᵀ instead of the vector W x_i. Reconstruction
Σ_j c_ij · W d_j d_jᵀ. The Frobenius loss expands into a lasso whose Gram is **datapoint-independent**:
```
  masked projector :  G_i = (a_i a_iᵀ) ⊙ G_W          b_i = a_i ⊙ (WDᵀ W x_i)
  SAE-on-M_i       :  G   = (DᵀD)     ⊙ (WD)ᵀ(WD)     b_i = same
```
Same linear term; the Gram no longer collapses the input side onto the single direction x_i.
Indeed **the masked projector IS the M_i objective evaluated only along x_i** (right-multiply the
matrix equation by x_i). That is exactly the missing input-side constraint: reconstructing the outer
product forces Σ_j c_ij d_j d_jᵀ ≈ x_i x_iᵀ — atoms must ALIGN with x_i's direction, not merely have
a usable projection onto it.

**Falsifiable prediction (registered before the result):** support∩SAE-active should be MUCH higher
than the masked projector's 10.6%. If it is, the input-side constraint fixes the identifiability
degeneracy and the original plan was right to reconstruct M_i. If it is not, the degeneracy is deeper
than the target choice.

## ★ SAE-on-M_i RESULT: the original plan fails as literally specified — and the fix is the bilinear code

Registered prediction was that the outer-product target would RAISE support∩SAE-active above 10.6%.
**It did not.** Measured (GPT-2 L6, SAE-feature dictionary, K=2048):
```
  λ 3.0  L0 244  R2(M_i) 0.1065  support∩SAE-active 6.9%
  λ 1.0  L0 262  R2(M_i) 0.1067                     6.5%
  λ 0.3  L0 269  R2(M_i) 0.1067                     6.4%     (masked projector: 10.6%)
```
**Why — and it is the spec's OWN §1.2 result, now confirmed at the M_i level.** With x = Σ_{j∈S} s_j f_j,
    x xᵀ = Σ_{j,k} s_j s_k f_j f_kᵀ
is dominated by CROSS terms. A *diagonal* dictionary Σ_j c_j d_j d_jᵀ can only represent the |S|
diagonal terms — a Frobenius mass fraction of 1/|S| ≈ 1/54 ≈ 2%. Hence R² ≈ 0.107, not ≈0.99. The
spec anticipated exactly this ("cross terms are artifacts for linear W") and that is *why* it replaced
the SAE-on-M_i plan with the masked projector. **So the replacement was justified, not an oversight.**

**But the fix is available and it is the spec's own Tier-3 device: let the code enter TWICE.**
With the bilinear decoder M̂_i = Σ_{j,k} c_ij c_ik · W d_j d_kᵀ (which equals W x̂_i x̂_iᵀ when c = SAE
codes), the cross terms are representable:
```
  BILINEAR decoder, c = SAE codes :  R2(M_i) = 0.7180   @ L0 56.5
  DIAGONAL decoder (linear in c)  :  R2(M_i) = 0.1065
```
6.7× better, at the same sparsity, with atoms that ARE SAE features.

## Synthesis of the reopened program (what is now established)
1. **The retracted "structural failure" verdict is dead**, and every plank of it is explained
   (dense-regime measurement, degrading solver, mismatched site, a metric the spec never made decisive).
2. **The masked-projector objective is feasible and good** at real sparsity: R² 0.90 @ L0 56 with
   interpretable atoms — but only with an **L0-targeting code solver**. The spec's §3 L1 lasso loses
   ~0.29 R² at matched sparsity (⅓ shrinkage, ⅔ support-selection: the L1-vs-L0 gap).
3. **Logan's input-side critique is confirmed empirically**: because the input enters only through the
   scalar gate d_jᵀx, sparse decompositions of Wx are non-unique (OMP picks 89% SAE-INACTIVE atoms at
   equal loss). ⇒ *Reconstruction quality does not identify which features W uses.*
4. **The naive input-side fix (SAE-on-M_i, diagonal) fails for the reason the spec gave in §1.2**
   (cross-term mass). **The correct fix is the bilinear/quadratic code form** (spec §3 Tier 3), which
   reconstructs M_i at R² 0.72 vs 0.11 at matched L0.
5. **Identifiable, interpretable recipe available today, no dictionary learning:** per datapoint take
   the SAE's active support and let W reweight it by OLS → R² 0.869 @ L0 54. Every atom is a feature
   the SAE says is on for that token; W supplies the mechanistic weighting.

**Open, in priority order:** (a) solve the bilinear-code M_i objective (non-convex; init from SAE codes)
and check whether its support LOCKS onto the SAE-active set — that is the real test of identifiability;
(b) release-D refinement from the SAE dictionary; (c) port the L0-solver recipe back to Tier 1.5's
circuit-contraction analysis, where the selectivity result already worked.

## Bilinear-code M_i: the objective REDUCES to W-weighted sparse coding of x (key structural fact)
Since Σ_{jk} c_j c_k W d_j d_kᵀ = W u uᵀ with u = D c, the loss is
    ‖W(u uᵀ − x xᵀ)‖_F² = ‖Wu‖²‖u‖² − 2 (Wu·Wx)(u·x) + ‖Wx‖²‖x‖²
which is minimized by driving **u = D c → x in the W-weighted metric**. I.e. the bilinear M_i
objective is *sparse coding of the INPUT x, with the reconstruction error measured through W*.

This is a clean statement of what the whole method should have been:
  * masked projector: reconstruct W x  → input enters only via the scalar gate → non-identified.
  * bilinear M_i    : reconstruct x    → but with W supplying the metric → **"feature discovery
    weighted by mechanistic relevance to this layer"**, which is the spec's §1.3 goal verbatim.
It also explains the disagreement result: only an objective that reconstructs x can be expected to
select the features that are ON for x.
Solving it (projected-gradient, non-convex) from SAE-init and cold-init; the test is whether the
support LOCKS onto the SAE-active set (vs 10.6% masked projector, 6.5% diagonal-M_i).

## Bilinear M_i solve: quartic, non-convex, c=0 is a local min — and it points to the CONVEX form
```
init-from-SAE (K=2048 truncated)   R2(M_i) 0.0515  L0 32.5     ← truncation-damaged init
SAE-init + projected gradient       R2(M_i) 0.6649  L0 801.2   support∩SAE 3.3%  ← densifies
cold-init + projected gradient      R2(M_i) 0.0000  L0   0.0   ← collapses (c=0 is a local min)
```
The bilinear objective is quartic in c with a trivial minimum at c=0; naive PGD either collapses or
densifies. **But its own algebra hands over the convex surrogate.** If u = Dc ≈ x then
    u uᵀ − x xᵀ ≈ (u−x) xᵀ + x (u−x)ᵀ,
so the M_i loss is governed by ‖W(Dc − x)‖². That is **sparse coding of x in the W-weighted metric** —
convex, same Gram machinery, and it is the spec's §1.3 goal stated exactly:
*reconstruct the INPUT, but measure the error through W* ⇒ features weighted by mechanistic relevance.

Running the three-way comparison at matched sparsity:
  (I)  W-metric sparse code of x:  min ‖W(Dc − x)‖² + λ‖c‖₁     ← proposed objective
  (II) plain sparse code of x:     min ‖Dc − x‖² + λ‖c‖₁        ← an SAE, no W
  (III) masked projector (spec §1.3 as implemented)
measuring L0, R²(Wx), R²(x), and support∩SAE-active for each.

## Three-way objective comparison + a measurement caveat (quantified)
```
objective                          L0    R2(Wx)   R2(x)    support∩SAE-active   rescaled*
(I)  W-metric sparse code of x    160    0.8499  -0.3814        7.6%              12%
(II) plain sparse code of x        40    0.1124   0.0214       24.9%              41%
(III) masked projector (spec)      77    0.1098  -0.7657       14.3%              23%
     OMP on masked projector       56    0.9017     —          10.6%              17%
```
*Caveat, quantified: only **61.4%** of each token's SAE-active features lie inside the K=2048
candidate dictionary, so the maximum attainable overlap is ~61%, not 100%. Rescaled = raw/0.614.
(Every overlap number in this program's earlier entries carries this same ceiling.)

**Reading:**
- **(I) does exactly what "mechanistic relevance" means, and that is why it fails identifiability.**
  It reconstructs Wx superbly (0.85) while *worsening* x-reconstruction (R²(x) < 0): W's spectrum
  decays fast, so small ‖W(Dc−x)‖ permits large ‖Dc−x‖ along low-gain directions. Down-weighting
  W-irrelevant directions is the design goal — but it means the code need not use the features that
  are ON for x. **Mechanistic relevance and support identifiability are in tension, by construction.**
- **(II), which ignores W entirely, has by far the best support recovery (41% rescaled).** Only an
  objective that reconstructs *x* selects the features active on x. This is the sharpest statement of
  Logan's input-side critique: any objective whose target is W-filtered cannot be expected to identify
  the active feature set, because W discards precisely the information that identifies it.
- ⇒ **A single objective cannot both (a) weight by mechanistic relevance and (b) identify the active
  features.** The spec's §1.3 goal, read literally, asks for both. The resolution is to SEPARATE them:
  identify the support with an x-reconstruction objective (an SAE), then weight/ablate it through W.
  That is exactly the recipe measured earlier: **OLS-on-SAE-active-support, R² 0.869 @ L0 54** — atoms
  are the token's own features, W supplies the mechanism. No dictionary learning, no L1 on Wx.

## ★ ABLATION (spec §2(iv), §1.3's "operative faithfulness"): the static rank-1 weight edit CANNOT localize

Atoms = SAE features (the recommended recipe). Ablate W ← W − (W d_j) d_jᵀ, so ΔW·x = (W d_j)(d_jᵀx).
```
  ablation concentration ‖ΔWx‖  active/inactive :  median 1.89×   (frac>3× 27%, >5× 2%)
  raw gate |d_jᵀx|              active/inactive :  median 1.89×   ← IDENTICAL
  contribution via the SAE CODE s_ij            :  ∞ by construction (inactive code ≡ 0)
```
The ablation ratio EQUALS the raw-gate ratio, exactly as the algebra demands: the rank-1 weight edit
removes the *direction* d_j from W's input side, and thus perturbs **every** context in proportion to
its projection d_jᵀx — never in proportion to whether feature j is *used* there. A thresholded feature
is active on ~1% of tokens but has a nonzero projection on ~100% of them.

**Consequences (two more planks of the retracted verdict fall):**
1. **Pythia's "2.6× ablation localization" was never evidence that the atoms were bad.** It is what
   this edit MUST produce for any thresholded feature — including a perfect one. (Toys reached 32-64×
   only because their features were *orthonormal*, so the projection and the activity coincide.)
2. **Spec §1.3's claim that "ablatability comes for free from the parameterization W d_j d_jᵀ" is
   FALSE for thresholded / sparse features.** The parameterization excises a *direction*, not a
   feature's *usage*. Ablatability of a feature requires a **data-conditioned edit**: remove the coded
   contribution c_ij·(W d_j), i.e. exactly the counterfactual-target machinery of §1.1 Problem 2
   (M* = [Y X†X]_r X† with Y = W X̃). The closed-form theorem — the part of the spec that was always
   correct — is the right editing tool; the static weight subtraction is not.

## CONSOLIDATED STATUS of the reopened program (supersedes the retracted "closed" verdict)
**Refuted / retracted:** "structural failure"; "atoms drift to principal directions"; "weak ablation
localization ⇒ bad atoms"; "matched-compute SAE fails ⇒ Pythia lacks features". All four were artifacts
of: dense-regime measurement, a degrading solver, a mismatched site, an oracle the spec never endorsed,
and an ablation operator that cannot localize by construction.

**Established (with measurements):**
1. The masked-projector objective is feasible and strong at real sparsity — R² 0.90 @ L0 56 with atoms
   that are SAE features — **given an L0-targeting solver**. The spec's §3 L1 lasso loses ~0.29 R²
   (⅓ shrinkage, ⅔ support selection: the L1-vs-L0 gap).
2. **Mechanistic relevance and support identifiability are in tension.** Any W-filtered target discards
   the information that identifies which features are active (W-metric: R²(Wx)=0.85 but R²(x)=−0.38).
   Only an x-reconstruction objective recovers the active support (41% vs 12-23% rescaled).
3. **The SAE-on-M_i original plan fails as literally specified** (cross-term mass, exactly spec §1.2)
   — the spec's replacement was justified. Its principled repair is the **bilinear code** (code enters
   twice): R²(M_i) 0.72 vs 0.11 at matched L0. Solving it is hard (quartic; c=0 is a local min).
4. **Ablation of a sparse feature needs a data-conditioned edit**, not a rank-1 weight subtraction.

**The working recipe (no dictionary learning):** support ← SAE-active features of the token;
coefficients ← OLS through W (mechanistic reweighting) → **R² 0.869 @ L0 54**; edits ← §1.1 Problem-2
closed form on the coded contribution. "Feature discovery weighted by mechanistic relevance" =
*discovery* (SAE) + *weighting* (W), separated — not one loss.

**Kept from the original run, unchanged:** closed-form theorem verification; toy + superposition
recovery; no-cheating; selectivity-scored contraction for the induction circuit; solver deliverables.

## Regression gates re-run (own deliverable #9) + Tier 1.2 retested with an L0 solver

**Regression (current solver: pinv M-step, ridge & gate-weighted options present):**
```
Tier 0 recovery : min-cos 0.9994 (gate >0.99)   minF1 1.000 (gate >0.95)   PASS
Tier 1.4 cheating: 0 cheating atoms / 3 seeds                              PASS
```
No regressions from the E-step/M-step changes made during the reopened investigation.

**Tier 1.2 retest — was "no mechanism-side rescue" an L1 artifact?** PARTIALLY.
```
                                   e0 / e1 recovery cos (3 seeds)        mean
  mechanism dict, L1 (old verdict)       0.60 – 0.67                     ~0.63
  mechanism dict, OMP (L0 solver)        0.646 – 0.946                    0.733
  activation SAE baseline                0.630 – 0.840                    0.723
```
**Correction, bounded:** the L1 solver *was* handicapping the mechanism dictionary — OMP moves it from
clearly-worse-than-SAE to statistically tied. **But no rescue is demonstrated** (0.733 vs 0.723, n=3,
high variance), and both methods sit far below the ~0.99 recovery these toys reach when features are
genuinely separable. So spec §1.4's claim (b) — "W treating features differently rescues
identifiability where co-occurrence cannot" — remains **unsupported**, though the earlier "mechanism
dictionary is *worse* than an SAE" reading is withdrawn as an L1 artifact.

**Discipline note (against my own narrative bias).** The pattern of this reopened arc has been
"negative result → the solver did it." That makes it tempting to attribute EVERY negative to the
solver. Tier 1.2 does not support that: with the better solver the method still fails to beat the
SAE. Recording this explicitly so the arc is not over-corrected. The accepted limitation of §1.4
(features that always co-occur and are treated isotropically will merge) stands; the *anisotropic
rescue* remains unproven.

## ★ Tier 1.5 redone with OMP: reconstruction FIXED (0.9 vs 0.06-0.46) — and it exposes a spec-level error

**Reconstruction, same maps and data, only the code-solver changed:**
```
            L1 lasso (old)   OMP (L0-targeting)
  L0H0-OV      0.06-0.46          0.9297
  L0H1-OV                         0.8875
  L0H2-OV                         0.9058
  L0H3-OV                         0.9048
  L1H2-K1                         0.9712
  L1H2-K2                         0.9193
```
Back at Tier 1.5 I wrote that the gate was "uninformative at this R²" — then tuned λ/prune and moved on
to Pythia instead of fixing the solver. The L1 E-step was the cause there too. **Every Tier-1.5
conclusion was drawn from a decomposition explaining under half of each map's action.**

**Selectivity-scored contraction, now on a good fit (gate: L0H3 dominant in BOTH K branches):**
```
  K1: L0H0 -0.680  L0H1 -0.615  L0H2 -0.622  L0H3 -0.283  → max L0H3  PASS
  K2: L0H0 -0.232  L0H1 -0.044  L0H2 +0.314  L0H3 +0.125  → max L0H2  FAIL
```
Opposite branches from the old (badly-fit) run, which had K2 pass and K1 fail. **The inconsistency is
the finding.**

**Diagnosis — a spec-level error, not a solver one.** Bilinear attention's match is a PRODUCT,
`(q₁·k₁)(q₂·k₂)/d²`. A **per-branch linear contraction is ill-posed**: neither branch's magnitude or
sign is meaningful alone (note every K1 selectivity value is negative — nonsense for a "contribution").
This is exactly the XNOR structure established in the circuit-atlas work (pattern×OV agreement, not
pattern sign). Spec §1.5's contraction coefficient `d_kᵀ W¹ d_j` is derived for **stacked linear maps**
and **does not transfer to bilinear/gated attention** without a product-form generalization. The
causally-verified ground truth (only L0H3@src collapses L1H2's match, −.434→−.031) is a statement about
the product; no linear per-branch surrogate can be expected to reproduce it.

**Consequences:**
* The Tier-1.5 "selectivity-scored contraction recovers the circuit" result — which I listed among the
  program's few survivors — is **weaker than claimed**: it held for a summed-branch linear proxy on a
  poor fit, and does not hold per-branch on a good fit. Downgraded from "validated" to "needs the
  product-form contraction". The underlying *selectivity vs magnitude* insight stands (magnitude picks
  the causally-inert L0H0); the contraction machinery does not.
* For TENSOR NETWORKS specifically (Logan's setting), circuit read-off must contract through the
  product of both bilinear branches. Deriving that is the natural next piece of theory.

**Corrected survivor list:** closed-form theorem; toy + superposition recovery; no-cheating;
solver deliverables; *selectivity > magnitude* as a scoring principle. **Not** "contraction recovers
the circuit" — that is now an open problem in the bilinear setting.

## Product-form contraction: derivation right, my write vector WRONG (4th construction bug)
Derivation (correct): removing head A's write v from the key-side residual changes the bilinear match
exactly by  Δs = a₁b₂ + a₂b₁ − b₁b₂,  a_i = Q_i x_q·K_i x_k,  b_i = Q_i x_q·K_i v. Each branch's
contraction is **cross-weighted by the other branch's score** — the factor a linear per-branch score
omits, which is why those signs were meaningless.

First test FAILED with absurd magnitudes (L0H0 Δs = −13.1, whose causal effect is ~0.02). That is a bug
signature, not a result. Two errors:
  (i) I used v = OV_h · n0[src] — the head's OV applied to the source token's OWN residual. The actual
      write at src is W_O·(Σ_k pattern_h[src,k]·W_V n0[k]) — an ATTENTION-WEIGHTED sum. For a
      previous-token head like L0H3 the write at src comes from n0[src−1], not n0[src]. Applying OV to
      the position's own residual silently assumes every head attends to itself.
  (ii) RMSNorm: n1 = norm(h1). Removing the write from h1 renormalizes n1; the perturbation is not −v.
Rerunning with the true per-head write (through the real attention pattern) and exact renormalization,
scoring Δmatch at the correct source minus at a shuffled source.

## ★ Tier 1.5 GATE PASSES — with the correct write. And two more of my claims fall.

Exact per-head contribution to the bilinear match (true attention-weighted write, RMSNorm renorm),
scored on induction queries:
```
  head   Δmatch @ correct src   @ shuffled src   selectivity
  L0H0        +0.0299              +0.0226         +0.0073
  L0H1        +0.0141              +0.0049         +0.0092
  L0H2        -0.0600              -0.0270         -0.0331
  L0H3        -0.0791              -0.0177         -0.0614   ← |max| both columns
```
**Gate PASSES**: L0H3 dominates, matching the causally-verified ground truth (only L0H3@src collapses
L1H2's match, −0.434 → −0.031).

**Correction 1 — "selectivity beats magnitude" is DOWNGRADED (it was a wrong-write artifact).**
With the correct write, the raw magnitude at the correct source ALREADY identifies L0H3 (−0.0791, the
largest |Δ|). Selectivity only widens the margin (2×). My earlier claim — that magnitude picks the
causally-inert L0H0 and selectivity is *required* — was produced by scoring ‖W_k·(OV_h·X)‖, i.e.
applying each head's OV to **the position's own residual**, which silently assumes every head attends
to itself. That is the same bug as above. (Scope check, done: the circuit ATLAS is unaffected — it never
made the selectivity claim. Its directional composition feeds OV the matched token's embedding, i.e. the
CORRECT input for a previous-token head at the source, and it explicitly warns that "weight products need
the right input direction before they reflect the circuit." The atlas anticipated this bug; the error was
confined to the Tier-1.5 scoring in this log.)

**Correction 2 — spec §1.5's "circuit discovery = reading a matrix" does NOT hold for bilinear attention.**
The correct contraction Δs = a₁b₂ + a₂b₁ − b₁b₂ depends on a₁,a₂ (the *current* branch scores) — and the
write v itself passes through the attention pattern and RMSNorm. All three are **data-dependent**. So
there is no data-free weight-space matrix entry for a gated/bilinear layer; circuit strength is an
inherently data-conditioned quantity. The spec's contraction is exact only for **stacked linear maps**.

**Net for Tier 1.5:** the circuit IS recoverable from the model, but via a data-conditioned product-form
contribution, not a weight-space matrix — and the recovery does not need the selectivity trick once the
write is computed correctly. Corrected survivor list: closed-form theorem; toy + superposition recovery;
no-cheating; solver deliverables (incl. "L1 is the wrong code solver"); **and now: the product-form,
data-conditioned contraction as the correct circuit read-off for bilinear attention.**

## ⚠️ SELF-AUDIT: the identifiability "tension table" was L0-confounded. Corrected.
The table I published (plain 41% / masked 23% / W-metric 12% overlap) compared objectives at L0 40, 77
and 160. Overlap is a precision-style fraction and falls mechanically with L0. Re-measured at matched
sparsity, with precision AND recall against the token's in-dictionary active features:
```
  objective                        L0     prec   recall
  plain sparse code of x (no W)   51.5   22.2%   36.5%
  masked projector (spec)         45.2   20.4%   30.0%
  masked projector (spec)         60.8   17.0%   33.2%
  masked projector (spec)         77.4   14.3%   35.4%
  W-metric sparse code of x       74.6   13.2%   32.2%
  plain sparse code of x (no W)   86.1   16.5%   44.5%
```
**Direction survives** (plain ≥ masked ≥ W-metric) — **magnitude does not**: a few points, not 3–4×.
Also: no objective exceeds ~45% recall / ~22% precision, so dictionary-and-threshold mismatch dominates
the absolute level for all of them, including plain sparse coding. `mechdecomp_findings.md` §2.4 updated.

**Open problem #1 status:** the bilinear-M_i quartic **cannot be solved from cold init at any λ** (c=0 is
a local minimum; L0 → 0), and from a convex init it densifies (L0 839, R²(M_i) 0.616, precision falls).
So there is no matched-L0 answer for it. The objective is not merely hard to optimize — it may be
unusable without a support-constrained or homotopy solver (e.g. continuation from the SAE's codes).

---

## 2026-07-09 — SUPPORT-RECOVERY CONTROL: the identifiability metric was never valid

Follow-up to the §2.4 matched-L0 audit. That audit fixed an L0 confound but left the
metric itself (overlap with the SAE's active set) unexamined. Two controls, `logs/support_ceiling.log`
and `logs/support_disambig.log`, on GPT-2-small L6, res-jb SAE, 400 tokens:

**Control 1 — can sparse coding over the SAE's OWN decoder recover the SAE's OWN support?**
OMP, k=56 (the SAE's L0), target `x - b_dec`, dictionary = the SAE's own decoder directions:
x-R² 0.859, **precision 22.1%, recall 28.0%**. It does not.

**Control 2 — disambiguation, one metric (per-datapoint uncentered R² on `x - b_dec`), matched L0 55.5:**

| | R² | L0 | overlap |
|---|---|---|---|
| (1) SAE encoder codes, SAE support | 0.8432 | 55.5 | — |
| (2) **OLS refit on SAE support** (best coeffs) | 0.8807 | 55.5 | 100% |
| (3) **OMP greedy support**, matched per-datapoint k | **0.9328** | 55.5 | 23.2% |

(2) and (3) both use free least-squares coefficients ⇒ apples-to-apples. **(3) > (2)**: at equal
sparsity and equal coefficient freedom, a support sharing only 23% of atoms with the SAE's
reconstructs *strictly better*.

### Conclusion
The SAE's active set **is not the reconstruction optimum at its own sparsity**. This is not
solver weakness — better, largely-disjoint supports exist and are easy to find.

Consequences:
1. **Overlap-with-SAE-features cannot adjudicate any reconstruction objective**, including the
   masked projector. The ~13–22% precision measured for masked-projector / W-metric / plain
   sparse coding alike is the signature of encoder-vs-sparse-coding mismatch, not of `W` filtering.
2. §2.4's between-objective comparison is **withdrawn entirely** (not merely rescaled, as the
   previous audit did). It measured the reference's own non-optimality.
3. "OMP reaches equal loss on 89% inactive atoms" is the *same* phenomenon, and plain `x`-coding
   does it too. It is not a finding about the masked projector.
4. It supplies the quantitative reason the older "decisive" result (atom↔feature max-cos 0.16 vs
   0.15 random) was over-read, as Logan argued on other grounds: SAE features are not the
   reconstruction optimum, so a reconstruction method has no obligation to land on them, and
   failing to is not evidence that the method failed.
5. (1)→(2) is the known SAE encoder amortization gap (0.843→0.881), reproduced incidentally.

**Caveat:** (1) uses nonneg codes while (2)/(3) use signed OLS, so (1) is not directly comparable
to the others. The load-bearing comparison is (2) vs (3), which is clean.

**Standing consequence:** no future run in this program may use SAE-feature overlap as a pass/fail
oracle. A valid identifiability test needs ground truth where the active set is *defined* (toys,
Tier 1.5 circuits), not inherited from another trained model's encoder.

## 2026-07-09 — RELEASE-D REFINEMENT TEST (Logan's prescribed experiment #2)

Setup: D initialised to site-correct centered unit-norm res-jb decoder dirs (top-4096 by usage),
W = full L6 OV, X = ln1(resid_pre), codes by OMP at **fixed k=56** (so every round is matched-L0
by construction), M-step = rowspace-projected pinv update. `logs/release_d_val.log`.

Guard G1 caught a config error before any result was read: at K=2048 round-0 R² was 0.8437, not
the known 0.9017 (that measurement used top-**4096** candidates). Re-run at K=4096 reproduced
0.8971/0.9007. *The number was never adjusted to fit; the config was.*

**In-sample** (what I nearly reported):

| round | R² | mean cos(atom, own start) | frac drifted |
|---|---|---|---|
| 0 | 0.9007 | 1.000 | 0% |
| 7 | 0.9491 | 0.034 | 100% |

Loss falls 1.55e5 → 7.95e4 while atoms leave their feature directions almost completely.
Read naively: *"features are not optimal for this objective"* (Logan's structural branch).

**Held-out** (D frozen, codes re-solved by OMP on 600 unseen tokens):

| dictionary | val R² |
|---|---|
| feature basis D0 (round 0) | **0.8931** |
| refined D (round 7) | 0.8139 |

**Refinement improves training loss and degrades generalization by 0.079.** The drift is
overfitting — 4096×768 ≈ 3.1M dictionary parameters fitted to 600×768 ≈ 0.46M target numbers.

### Verdict on experiment #2
**"The optimizer wanders."** Not "features aren't optimal." The in-sample table says the
opposite of the truth, and reporting it would have been the sixth headline error of this program.

Consequences:
- **"SAE-init then refine" is NOT the production recipe** at this data scale. SAE-init then *stop*
  (or refine with far more data / an explicit prior) is. Refinement as specified has no term
  pinning atoms to anything and will memorize.
- G2 also fired (loss rose at round 1→2): the OMP-support + pinv-M-step alternation is **not a
  descent method**, because the support is re-selected each round. Worth stating in the spec.
- Caveat, and the reason a scaled re-run is in flight: N=600 train vs K=4096 atoms is
  underdetermined, so "it overfits" may be an artifact of scale, not a property of the objective.
  `release_d_big.py` repeats at 21k train / 4k held-out before this verdict is treated as settled.

### Scaled re-run + random-init control (`logs/release_d_big.log`, `logs/release_d_rand.log`)

The N=600 overfit WAS a scale artifact. At N=5872 train / 4000 held-out, K=4096, k=56:

| dictionary | held-out R² |
|---|---|
| random init, unrefined | 0.8332 |
| random init, refined 5 rounds | 0.8861 |
| SAE feature basis, unrefined | 0.8877 |
| **SAE feature basis, refined 5 rounds** | **0.9120** |

G2 no longer fires at scale (loss monotone). Drift persists: mean cos to start 0.218.

### FINAL verdict on Logan's experiment #2
**"Features are not optimal for this objective"** — refinement improves *held-out* R² by +0.024
while leaving the feature directions. The earlier N=600 "optimizer wanders" reading was wrong,
caught only by scaling. (Both readings were reported; the first is superseded, not deleted.)

But the control adds the part that matters:
- Refining from features beats refining from random by **+0.026** — the feature basis is a real basin.
- The **unrefined** feature basis (0.8877) already matches 5 rounds of from-scratch dictionary
  learning (0.8861). The SAE gives you that for free.
- Refinement is **data-hungry**: −0.079 held-out at N=600, +0.024 at N=5872. Same code.

### Thesis (now supported by two independent measurements)
**Reconstruction quality — of `x` (support control) or of `Wx` (this run) — does not select the
feature basis.** Better-reconstructing supports (23% overlap) and better-reconstructing
dictionaries (cos 0.22) exist and are largely disjoint from features.

Therefore: a pure reconstruction objective *cannot be expected* to discover features. This is a
property of the objective class, not a defect of this method's implementation. Outputs of the
masked projector must be validated **causally**, never by R² or by overlap with another SAE.

**Production recipe (revised):** SAE-init; refine only with data ≫ K·d/d = K parameters' worth;
expect atoms to move; do not read alignment-with-SAE as correctness in either direction.

---

## 2026-07-09 (later) — ⚠ RETRACTION: today's release-D numbers came from a BROKEN M-step

While building a positive control for the release-D test I found a defect in the M-step I wrote
**inline in `release_d.py` today** (NOT in `mstep.py`, which was already correct — Tier 0 gates
re-run: theorem 4/4 PASS, recovery PASS, ablation PASS).

**The defect.** Two bugs, both in the inline update:
1. residual `E` computed **once** before the per-atom loop and never updated → all 4096 atoms
   update against a stale residual (Jacobi, not Gauss-Seidel; divergent when atoms co-adapt);
2. each atom renormalized to unit norm with its codes `β` **frozen**, destroying the
   least-squares optimality that justified the update.

**How it was caught — the positive control** (`mechdecomp/refine_power.py`, `logs/refine_power.log`).
Build data with a known generator: `X = D_true C_true + noise`, `C_true` nonneg 4-sparse, `Y = W X`.
Start refinement AT `D_true`. If the data really are generated by `D_true` + iid noise, `D_true` is
the population optimum, so held-out R² must not improve.

With the broken M-step, from `D_true` at noise 0.05:

| round | train R² | held-out R² | cos to D_true |
|---|---|---|---|
| 0 | 0.9924 | 0.9918 | 1.000 |
| 7 | 0.8363 | 0.8341 | 0.927 |

It **destroys a dictionary known to be correct** — and *train* R² falls too, so this is not
overfitting: the alternation is simply not a descent method.

**The fix**: Gauss-Seidel (residual updated in place after each atom), `β` re-alternated after each
atom move, normalization deferred to after the sweep with codes rescaled to preserve the fit.

**Post-fix positive control** (`logs/refine_power_fixed.log`) — the test has power:

| setting | held-out R² round 0 → 7 | cos to init |
|---|---|---|
| noiseless, init `D_true` | 0.9940 → 0.9939 | 0.9999 |
| noise 0.05, init `D_true` | 0.9918 → **0.9916** (Δ −0.0002) | 0.9998 |
| noise 0.15, init `D_true` | 0.9733 → **0.9723** (Δ −0.0010) | 0.9994 |
| noise 0.05, init random | 0.5948 → 0.7002 | 0.523 |

The true dictionary is a **fixed point**; random init climbs but stalls far below it (0.70 vs 0.99).
Guard: oracle codes on `D_true` give held-out R² = 1.000000 exactly (harness exact); OMP on the same
true dictionary gives 0.9940, so 0.0060 is pure **greedy-selection error**, not dictionary error.

### Corrected GPT-2 release-D (fixed M-step), held-out, D frozen

| init | unrefined | refined | Δ | cos to start |
|---|---|---|---|---|
| random | 0.8332 | 0.8979 | +0.065 | 0.273 |
| **SAE feature basis** | 0.8877 | **0.9205** | **+0.033** | 0.477 |

N=600 variant, fixed M-step: 0.8931 → 0.9036 (**+0.0105**), train R² 0.9997.

### What is RETRACTED (all from the broken M-step)
1. ~~"held-out drops 0.079 at N=600 ⇒ the optimizer wanders"~~ — false; it *gains* +0.0105.
2. ~~"refinement is data-hungry / overfits at small N; SAE-init then **stop**"~~ — false; refinement
   helps at both N. The recipe is SAE-init then **refine**.
3. ~~"+0.024 held-out, cos 0.218"~~ — superseded by +0.033, cos 0.477.
4. The whole "verdict: optimizer wanders → reversed by scaling" narrative is void. It was one bug.

### What SURVIVES (and is now backed by a control with power)
- **Logan's experiment #2 resolves to "features are not optimal for this objective."** On the toy the
  true dictionary is a fixed point (Δ ≈ 0, cos 0.9998). On GPT-2 the SAE feature basis is **not**:
  refinement gains +0.033 held-out and moves atoms to cos 0.477 (~60°).
  ⇒ SAE features are not the generators of GPT-2 activations *with respect to this objective*.
- **Features are nevertheless a real basin**: feature-init refined (0.9205) > random-init refined
  (0.8979) by +0.023, and the unrefined feature basis (0.8877) ≈ random-init refined (0.8979).
- **The support control is unaffected** (per-datapoint OMP/OLS, no M-step): the SAE's active set is
  not the reconstruction optimum at its own L0 (OMP 0.9328 vs OLS-on-SAE-support 0.8807, 23% overlap).

### Thesis (unchanged in direction, now properly supported)
Reconstruction quality — of `x` (support control) or of `Wx` (release-D) — **does not select the
feature basis**. Better-reconstructing supports (23% overlap) and better-reconstructing dictionaries
(cos 0.48) exist. A pure reconstruction objective cannot be expected to discover features; its optimum
is measurably elsewhere. Validate causally, never by R² or by overlap with another SAE.

### Standing rule added
**Every dictionary-learning update must be validated on `refine_power.py` before its results are read.**
A correct update leaves a known-good dictionary where it is. This control cost ~10 minutes and
invalidated four hours of conclusions.

---

## 2026-07-09 — TIER 1: recovery phase diagram with DEFINED ground truth

Ground truth `X = D_true C_true + noise` (nonneg k-sparse), `Y = W X`. Recovery = mean over TRUE
atoms of max |cos| to any learned atom. **Chance is measured, not assumed.** Guard: init=`D_true`
holds at recovery 0.9998 (M-step validated). `logs/tier1_recovery.log`.

| K | K/d | ktrue | recovery | chance | held-out R² |
|---|---|---|---|---|---|
| 64 | 1.0 | 2 | 0.9647 | 0.3171 | 0.9540 |
| 64 | 1.0 | 4 | 0.9670 | 0.3171 | 0.9572 |
| 128 | 2.0 | 2 | 0.9704 | 0.3490 | 0.9615 |
| 128 | 2.0 | 4 | 0.9596 | 0.3490 | 0.9548 |
| 256 | 4.0 | 2 | 0.9574 | 0.3685 | 0.9504 |
| 256 | 4.0 | 4 | 0.8069 → **0.9783** @ 60 rounds | 0.3685 | 0.7945 → 0.9683 |

**The objective recovers the true dictionary from random init**, at every overcompleteness tested.
The 4×-overcomplete row is *underconvergence*, not failure (20 rounds 0.807 → 60 rounds 0.978).
This retro-explains `refine_power`'s "random init stalls at 0.70": it ran 8 rounds.

`resample_dead` fires on **0** atoms under OMP codes (all 128 atoms used across 8000 points) — a
**no-op here**, not an ineffective rescue. Recorded so it is not mistaken for a tested negative.

### ⚠ Near-miss: "the spec's lasso E-step fails" would have been headline error #7

At fixed λ=0.02 the lasso recovers 0.6077 vs OMP's 0.9596 — an apparently decisive indictment of
spec §3. **But λ=0.02 puts the lasso at L0 28.1, seven times `ktrue`=4.** At matched sparsity:

| E-step | λ | mean L0 | recovery | held-out R² |
|---|---|---|---|---|
| OMP | — | 4.0 | 0.9596 | 0.9548 |
| lasso | 0.002 | 52.7 | 0.4033 | 0.5616 |
| lasso | 0.020 | 28.1 | 0.6077 | 0.6235 |
| lasso | 0.100 | 5.6 | **0.9660** | 0.9390 |
| lasso | 0.400 | 3.2 | **0.9577** | 0.9568 |
| lasso | 1.000 | 1.4 | 0.7876 | 0.8164 |

⇒ **The lasso E-step is not deficient; it was mis-calibrated.** At matched L0 it recovers the true
dictionary as well as OMP. Same error class as the retracted §2.4 table — caught only by applying the
matched-L0 rule. Actionable: **calibrate λ to a target L0**; do not use a fixed λ across dictionaries.

### Consequences for existing claims
- §2.3's GPT-2 rows *are* matched-L0 (OMP 56 vs lasso 60.2), so "lasso loses ~0.29 R² **on GPT-2** at
  matched sparsity" stands. But it must **not** be generalized to "L1 is a worse selector": on a clean
  toy at matched L0 it is not. The GPT-2 gap is about real-data dictionary coherence, not about L1.
- ⚠ **FLAGGED, UNVERIFIED**: the Tier-1.5 corroboration "OV-map R² 0.06–0.46 (lasso) → 0.89–0.97 (OMP)"
  (`logs/tier15_omp2.log`) **records no L0 for the lasso run**. Given the above, it is very likely the
  same λ-calibration artifact. It may not be cited until re-measured at matched L0.

---

## 2026-07-09 — TIER 1.5: the circuit gate PASSES on the joint quantity; the per-branch gate is ill-posed

`mechdecomp/tier15_contraction.py` (reproducible; the earlier `logs/tier15_omp2.log` came from an
unsaved heredoc). Model `attn2-seed0`, induction batch of 96 repeated-block sequences.

**GUARD — causal ground truth by direct ablation at `src`:**

| zeroed | L1H2 match | Δs |
|---|---|---|
| — (base) | −0.0223 | — |
| L0H0 | −0.0246 | −0.0022 |
| L0H1 | −0.0229 | −0.0005 |
| L0H2 | −0.0239 | −0.0016 |
| **L0H3** | **−0.0074** | **+0.0150** |

L0H3 dominant, 6.8× the next head. ⚠ **Caveat:** base match −0.0223 here vs −0.434 on the natural
demo sequence in the atlas — random-token inductions engage the circuit ~20× more weakly. The
**ordering** reproduces; the magnitudes are not comparable. This gate is on ordering only.

**Decomposition** (OMP k=8, 128 atoms, validated Gauss-Seidel M-step): OV-map R² 0.945–0.970.

**(F) Faithfulness / (G) joint gate** — replace head h's write at `src` by its k-sparse atom
reconstruction, then remove it:

| head | true Δs | atom Δŝ | match after faithful swap |
|---|---|---|---|
| L0H0 | −0.0022 | −0.0030 | −0.0228 |
| L0H1 | −0.0005 | −0.0005 | −0.0225 |
| L0H2 | −0.0016 | −0.0004 | −0.0233 |
| **L0H3** | **+0.0150** | **+0.0121** | −0.0193 |

**JOINT GATE: argmax |Δŝ| = L0H3, 4.07× the next head → PASS.** The atoms recover the
causally-verified L0H3 → L1H2 edge. Atom Δŝ underestimates true Δs by ~19% (reconstruction error).

**(B) The spec's §1.5 per-branch contraction `|d_kᵀ OV_h d_j|` — ill-posed, not merely failing:**

| branch | aggregation | L0H0 | L0H1 | L0H2 | L0H3 | argmax |
|---|---|---|---|---|---|---|
| K1 | mean\|G\| | **0.2099** | 0.0886 | 0.0920 | 0.1693 | L0H0 FAIL |
| K1 | usage-wt | **0.1797** | 0.0879 | 0.0918 | 0.1595 | L0H0 FAIL |
| K2 | mean\|G\| | 0.1112 | 0.0776 | 0.0816 | **0.1271** | L0H3 PASS |
| K2 | usage-wt | 0.1053 | 0.0779 | 0.0807 | **0.1241** | L0H3 PASS |

The two aggregations agree, so the *earlier* log's opposite verdict (K1 PASS / K2 FAIL) was not an
aggregation artifact — it used a **signed selectivity** score with a lasso E-step, a different
statistic. Summary of the situation:

- `|G|` (either weighting): K1 → L0H0, K2 → L0H3.
- signed selectivity + lasso: K1 → L0H3, K2 → L0H2.

**No per-branch variant passes both branches, and which head it names depends on the scoring
statistic.** This is the empirical counterpart of §2.6: `Δs = a₁b₂ + a₂b₁ − b₁b₂` cross-weights each
branch by the other's score, so a per-branch number is not the causal quantity and there is no
principled aggregation to rescue it. The joint Δŝ needs no aggregation choice and passes at 4.07×.

### Verdict
- **Tier 1.5 circuit-recovery gate: PASS**, on the joint data-conditioned quantity.
- **Spec §1.5's "read the circuit off the contraction matrix" is retired for gated layers** — not
  because the atoms are bad (they reconstruct at R² 0.95 and recover the edge), but because the
  per-branch matrix entry is not the causal object. Replacement: ablate the atom-reconstructed
  write and measure Δŝ on the downstream score.
- The heartbeat's standing gate ("contraction matrix must recover the edge **in both K branches**")
  presumed a separability the product form denies. It should be restated as the joint-Δŝ gate.

---

## 2026-07-09 — TIER 1.5 ON NATURAL TEXT: gate passes at 8.8×; the uncitable lasso claim, corrected

`mechdecomp/tier15_natural.py`, `logs/tier15_natural.log`. Induction sites **mined** from the val
corpus (tok[j]==tok[q], q>j+1, tok[q+1]==tok[j+1]) rather than synthesised: 96 sites.

**Closes caveat (8).** Base L1H2 match on mined sites **−0.1857** (random-token run: −0.0223).
Guard: run is discarded unless base < −0.1, i.e. the circuit is actually engaged. It is.
(Still shy of the atlas demo sequence's −0.434, which is a single hand-picked sequence.)

| zeroed at src | match | Δs |
|---|---|---|
| — (base) | −0.1857 | — |
| L0H0 | −0.1677 | +0.0180 |
| L0H1 | −0.1915 | −0.0058 |
| L0H2 | −0.1813 | +0.0044 |
| **L0H3** | **−0.0816** | **+0.1041** |

OV-map decompositions (OMP k=8, 128 atoms): R² 0.971–0.982.

**JOINT GATE — remove the atom-reconstructed write:**

| head | true Δs | atom Δŝ |
|---|---|---|
| L0H0 | +0.0180 | +0.0099 |
| L0H1 | −0.0058 | −0.0034 |
| L0H2 | +0.0044 | +0.0023 |
| **L0H3** | **+0.1041** | **+0.0870** |

**argmax |Δŝ| = L0H3 at 8.81×** (vs 4.07× on random tokens). Δŝ underestimates true Δs by 16%.
⇒ Tier 1.5 circuit-recovery gate PASSES on natural text, with a much larger margin.

### Closes flagged item: "OV-map R² 0.06–0.46 (lasso) → 0.89–0.97 (OMP)" — corrected

That claim recorded no L0 and is **retracted as stated**. Re-measured on L0H3-OV, same dictionary
(so this isolates the E-step), λ bisected to matched sparsity, with an OLS debias to separate L1
shrinkage from support selection:

| solver | L0 | R² | R² + OLS debias |
|---|---|---|---|
| **OMP k=8** | 8.0 | **0.9749** | — |
| lasso λ=0.030 | 8.0 | 0.8278 | 0.9131 |
| lasso λ=0.012 | 13.8 | 0.9332 | 0.9706 |
| lasso λ=0.200 | 0.9 | 0.2540 | — |
| lasso λ=0.800 | 0.0 | −0.0056 | — |

At **matched L0 = 8**, OMP beats lasso by 0.147 R²; ~58% of that gap is shrinkage bias (removed by
the debias), ~42% is worse support selection. Even given 70% more atoms (L0 13.8), lasso+debias
(0.9706) does not reach OMP at L0 8 (0.9749). The old "0.06–0.46" figures correspond to λ≥0.2, i.e.
**L0 ≤ 0.9 — a dictionary switched off**, exactly the mis-calibration the Tier-1 toy predicted.

**Consistency across tiers, now properly matched:**
- clean toy, matched L0: lasso **ties** OMP (0.958–0.966 vs 0.960).
- attn2 OV map, matched L0: OMP > lasso+debias by 0.062.
- GPT-2 L6 OV, matched L0: OMP 0.902 > lasso+debias 0.665.

⇒ L1 is not a generically worse selector. Its disadvantage grows with **dictionary coherence on real
data**, and it is absent when atoms are near-incoherent. Spec §3 should say: calibrate λ to a target
L0 per dictionary; prefer an L0-targeting solver on real, coherent dictionaries.

---

## 2026-07-09 — TESTING MY OWN EXPLANATION: "the lasso gap grows with dictionary coherence"

That sentence was written from three points with coherence never measured. Measured
(`logs/coherence.log`, `logs/coherence2.log`, `logs/collinear.log`):

| dictionary | mutual coh | mean \|cos\| | OMP−lasso gap (after debias) |
|---|---|---|---|
| toy (K=128, d=64) | 0.5870 | 0.1465 | 0.000 |
| attn2 L0H3-OV (K=128, d=64) | 0.8023 | 0.1970 | 0.062 |
| GPT-2 L6 OV / res-jb (K=2048, d=768) | 1.0000 | 0.1022 | 0.237 |

**Mutual (worst-case) coherence orders the gaps; mean |cos| does not** — GPT-2 has the *lowest* mean
coherence of the three. The explanation is right only in its worst-case form. n=3 and the dictionaries
differ in K and d, so this is consistency, not proof.

### A hypothesis of mine, refuted by its own diagnostic
I drafted "W's null space (rank 758 < 768) merges distinct features ⇒ mechanistically indistinguishable
atoms." **False.** Pairs with |cos| > 0.99 number **3 before W and 3 after**; the worst pair has
|cos(d_i, d_j)| = **1.000000 before** the map; mean atom energy in null(W) is only **2.79%**.
Mutual coherence 1.0 is a property of the **SAE decoder**, not of W.

### What is actually there: live duplicate features in res-jb
- **45 decoder pairs with |cos| > 0.999** across all 24,576 features (e.g. 979↔2039, raw
  cos = 1.000000, ‖ΔW_dec‖∞ = 1.9e-4), clustered around feature 316.
- Confound checked: my first coherence run used the first 2048 features **by index** (489 dead ones
  included). The top 2048 **by usage** (0 dead) still has mutual coherence **1.0000**, before and
  after W. So these duplicates are **live**, not dead-feature collapse. The table above stands.

### Knock-on for the support control (2026-07-09, earlier)
If duplicates are live, OMP selecting feature A over its exact duplicate B counts as a support
mismatch though the two are mechanistically identical. **Bound:** 45 duplicate pairs out of 24,576
features cannot account for a 77% support mismatch. The effect is real and negligible at that
magnitude; the control's conclusion is unaffected. Recorded so it is not rediscovered as an objection.

### Consequence for the spec
Any decomposition over a *supplied* dictionary inherits that dictionary's degeneracies. An exactly
duplicated pair is unidentifiable for **any** solver — OMP picks one arbitrarily, L1 splits the mass
between them (which is precisely where L1 loses R² at matched L0). Report dictionary mutual coherence
alongside any L0/R² claim; it is a precondition for interpreting either.

---

## 2026-07-09 — REFUTED (mine): "the OMP−lasso gap is mostly duplicate-splitting"

Stated to Logan in LOG.md item 10 without testing. Tested (`logs/dedupe_gap2.log`), GPT-2 L6 OV,
top-2048-by-usage res-jb dictionary, matched L0 = 56.0 (enforced by assertion, see below):

| dictionary | atoms | OMP R² | lasso R² | lasso+OLS | gap (OMP − lasso+OLS) | lasso L0 |
|---|---|---|---|---|---|---|
| full (with twins) | 2048 | 0.8426 | 0.6591 | 0.7208 | **+0.1219** | 56.0 |
| deduped \|cos\|>0.99 | 2039 | 0.8426 | 0.6592 | 0.7209 | **+0.1217** | 56.0 |

**Deduping changes the gap by 0.0002 — 0.2% of it.** The hypothesis is dead.

It was also dead by counting: the 45 duplicate pairs form a **clique** around feature 316, so
deduping removes only **9 of 2048 atoms**. Nine atoms cannot move R² by 0.12. I should have counted
before asserting.

⇒ The OMP−lasso gap is ordinary L1 behaviour on a coherent dictionary (shrinkage + worse support
selection), not a twin artifact. Retracted in LOG.md. What survives from the coherence work: mutual
coherence orders the gap across tiers; mean coherence does not; and res-jb does contain 45 live
duplicate decoder directions (a real fact about that SAE, just not the cause of the gap).

### A sign-flip caught by an assertion
The first attempt bisected λ over `[1e-4, 5]`. On ln1-normalized data (‖x‖≈√768≈27.7) that bracket is
far too low: λ=5 still left **L0 = 135**, so the "matched" row compared lasso at L0 135 against OMP at
L0 56 and reported a gap of **−0.0099 — i.e. "lasso beats OMP."** A sign flip, from an unchecked
bracket. The rewrite expands the bracket until L0 falls below target and then asserts
`|L0 − 56| < 4` before any R² is read. Another instance of: the comparison must be *verified* matched,
never assumed matched.

---

## 2026-07-09 — SOLVER GATE IN A NEW REGIME: wide/low-rank W, and §1.4 made quantitative

The Gauss-Seidel M-step was validated only on **square full-rank** W (64×64). Pythia's `down_proj` is
**512×2048** (null space 1536) — the regime that once produced R² = −1e15. Gated before spending.

**First reading (wrong gate):** init at `D_true`, held-out R² stays flat but cos-to-true collapses to
**0.4991** (256→64) and **0.4994** (2048→512) ⇒ "M-step NOT safe."

**The number gives it away:** `sqrt(rank/d_in)` = `sqrt(64/256)` = `sqrt(512/2048)` = **0.5**, matched to
four decimals (measured 0.4992 / 0.5000). The M-step projects atoms onto `row(W)`; a random true atom
retains exactly that fraction of its norm there. **W cannot see null(W)** — those components are
unrecoverable by *any* method, and the projected atom is the correct representative.

**Corrected gate** — compare against the rowspace-projected truth, which is what is identifiable:

| regime | cos(true, projected-true) pred / meas | held-out R² (init = projected truth) | recovery vs projected truth | fixed point? |
|---|---|---|---|---|
| square 64×64 | 1.0000 / 1.0000 | 0.9923 → 0.9915 | 1.0000 → 0.9997 | YES |
| wide 256→64 | 0.5000 / 0.4992 | 0.9975 → 0.9974 | 1.0000 → 0.9999 | YES |
| **wide 2048→512 (Pythia)** | 0.5000 / 0.5000 | 0.9975 → 0.9953 | 1.0000 → 0.9988 | **YES** |

⇒ The M-step is **safe** on wide low-rank maps. The apparent drift was spec §1.4's identifiability
statement in concrete form.

### Consequence for interpreting wide maps (belongs in the spec)
Decomposing a map `W: R^{d_in} → R^{d_out}` with `d_out < d_in` recovers each atom **only up to its
`row(W)` component**. For Pythia `down_proj` (2048→512) that is `sqrt(512/2048) = 50%` of the direction
by cosine — **75% of each atom's subspace is invisible to W**. Any interpretation of a `down_proj`
atom is a claim about its rowspace component only. For GPT-2 OV (768→768, rank 758) the loss is
negligible (cos 0.993), which is why the GPT-2 tier was unaffected.

**Standing gate updated:** the fixed-point control must compare against `normalize(P_row · D_true)`,
never against `D_true`, whenever `W` is rank-deficient.

---

## 2026-07-09 — TIER 2 (Pythia-410m), RE-RUN with the validated solver

`mechdecomp/tier2_pythia_v2.py`, `logs/tier2_pythia_v2.log`, `logs/pythia_floor.log`.
The original `tier2_pythia.py` used the un-validated M-step and a fixed λ; its numbers are **not
cited** anywhere and are superseded by this run.

`W` = `down_proj` L3 = **1024×4096**, rank 1024. K=2048 atoms, OMP k=32, 8 rounds,
16k train / 4k held-out, held-out R² always with **D frozen**.

**G1 identifiability (computed, not assumed):** cos(atom, row(W)-projection) predicted
`sqrt(1024/4096)` = 0.5000, measured **0.4998**. Only **25% of each atom's subspace is visible to W**;
any interpretation of a `down_proj` atom is a rowspace-only claim.

| dictionary | train R² | held-out R² |
|---|---|---|
| random, no refinement | 0.2688 | 0.2587 |
| **random, refined** | 0.2688 → 0.7555 | **0.6020** |
| svd-init, no refinement | 0.5388 | 0.5053 |
| svd-init, refined | 0.5388 → 0.7544 | 0.5985 |

G2 (refinement improves held-out) **PASS**. G3 (refined beats unrefined) **PASS**.
svd-init gives **no lasting advantage** (0.5985 vs 0.6020) — it only speeds the first round.
Train−held-out gap of 0.15 = real overfitting (8.4M dictionary params vs 16.4M target numbers).

### Floor check — does sparsity + learning buy anything?
A 32-component PCA of `Y` is a 32-atom dictionary with **dense** codes: matched L0, no learning.

| baseline | held-out R² | L0 |
|---|---|---|
| PCA-32 of Y (dense) | 0.3509 | 32 |
| PCA-64 of Y (dense) | 0.4231 | 64 |
| OMP-32 over PCA-64 dirs | 0.3897 | 32 |
| **learned K=2048, OMP k=32** | **0.6020** | 32 |

**The learned dictionary dominates every matched-L0 floor**: +0.25 over PCA-32, and +0.18 over PCA-64
despite PCA-64 having twice the sparsity budget. Tier 2 passes with headroom.

### Retraction this forces
The old program summary asserted the method "**finds PCA-like, not sparse, directions**" on real
models. With the validated solver that is **false on Pythia**: the learned dictionary beats PCA-32 by
0.25 R² at the same L0, and beats a sparse code over PCA directions by 0.21. That claim was produced
by the un-validated M-step and is withdrawn.

**Honest absolute picture:** 0.60 held-out R² at L0 32 is far below attn2 (0.97) and GPT-2 OV (0.84).
`down_proj` is a harder, wider map with 75% of each atom invisible. The margin over floors is what is
established; the absolute quality is mediocre and data-limited.

---

## 2026-07-09 — TIER 1.5 OWT + a correction that reaches back into Program A

**Gemma tier is BLOCKED**, not deferred: `google/gemma-2-2b` returns `GatedRepoError: 401` on even
`config.json`, and no HF token is configured. Needs Logan to accept the license / supply a token.

### Step 1 — my induction-head criterion was wrong (Logan's XNOR point, again)
`tier15_owt.py` picked the induction head as the L1 head with the **most negative** match, generalising
from tiny attn2's L1H2 (−0.434). That is a sign convention, not a definition: a bilinear head copies
when pattern and OV **agree in sign** (XNOR). block2-dense's L1H2 has match **+0.0669**, the largest
magnitude of any head, and the sign rule discarded it. Both OWT models were wrongly declared
"no induction head".

Replaced with a **behavioural, sign-agnostic** criterion (`tier15_owt_behav.py`): ablate each L1 head,
measure the drop in P(copied token). **Guard: it must reproduce L1H2 on tiny attn2** — it does
(removes 38.0% of copy probability; next head 24.5%).

| model | P(copied) | P(random tok) | induction head by ablation | dominance |
|---|---|---|---|---|
| attn2-seed0 (tiny) | 0.5902 | 0.00013 | L1H2 | 38.0% vs 24.5% |
| attn2-s120k-dense (OWT) | 0.3209 | 0.00022 | L1H0 | 22.4% vs 19.0% |
| block2-dense (OWT) | 0.3862 | 0.00039 | L1H2 | 25.5% vs 24.1% |

So the OWT models **do** copy on natural text, and the copying is **distributed** (no dominant head)
⇒ the single-edge decomposition gate is inapplicable to them, for lack of a localized ground truth.

### Step 2 — but "copies on natural text" is bigram-confounded
Mined sites require `tok[q+1]==tok[j+1]`: the repeated bigram is exactly what a bigram model predicts.
The right control is repeated **random** tokens.

**Two probes failed their positive control and were discarded** (recorded, not hidden):
uniform-random 12-token blocks, and unigram-sampled 12-token blocks. On the latter, the copy-burst
trained `mix10` models — which are *supposed* to have induction — scored lift 1.42–1.49×, the same as
the dense models (1.51×). A probe that cannot detect induction where it exists proves nothing.

**Third probe, matched to the exact training distribution** `[u[:128]; u]`, `u ~ Uniform(vocab)`
(`lm_train.py:90-96`) — passes its positive control:

| model | P(copied) | lift over uniform |
|---|---|---|
| attn2-mix10-seed0 | 0.9003 | **4609×** |
| attn2-mix10-seed1 | 0.8041 | **4117×** |
| attn2-s120k-dense-seed0 | 0.0002 | 1.0× |
| block2-dense-seed0 | 0.0002 | 1.0× |
| attn2-dense-seed0 | 0.0002 | 1.0× |
| **attn2-seed0 (tiny, Tier-1.5 ground truth)** | 0.0012 | **1.2×** |

### Step 3 — and mix10's "induction" is POSITIONAL, not content-based
Training bursts always repeat with period 128. Vary the period:

| model | P=128 (trained) | P=150 | P=100 | P=64 |
|---|---|---|---|---|
| attn2-mix10-seed0 | **0.9036** | 0.0003 | 0.0001 | 0.0002 |
| attn2-mix10-seed1 | **0.8040** | 0.0002 | 0.0002 | 0.0002 |

Copying collapses to chance at every untrained period. **It is a fixed-offset positional copier.**
Structural cause: in `[u[:128]; u]` the token at `q` always first occurred at `q−128` and the target
always sits at `q−127`, so content-matching and a constant −127 offset are **exactly equivalent** in
that distribution. The copy-burst lever *cannot* teach content matching; the period sweep is the only
disambiguator.

### RETRACTIONS forced (Program A)
1. ~~"10% copy-burst mixture installs induction 3/3 seeds"~~ → it installs a **positional copier**.
   The burst distribution's constant period is a design flaw. Fix: randomise the repeat period.
2. ~~"tiny attn2-seed0 has an induction head (L1H2)"~~ in the Elhage/Olsson sense → the model copies
   arbitrary repeated tokens at **1.2× chance**. Its L1H2 is a **repeated-bigram match-and-copy
   circuit on natural text**, not a general induction head.

### What SURVIVES — and it matters for this program
The Tier-1.5 mechdecomp gate is **unaffected**. It validated that the decomposition recovers a
*causally verified* edge (`L0H3 → L1H2`, Δŝ 8.81×, `logs/tier15_natural.log`). That causal fact is
untouched: zeroing L0H3's write at `src` still collapses L1H2's match on natural text. **Only the
circuit's name was wrong.** Everywhere this program says "induction circuit" as Tier-1.5 ground truth,
read "causally-verified repeated-bigram match-and-copy circuit."

---

## 2026-07-09 — THE COPY-BURST FIX: a probe bug, a falsified mechanism, and a lever that stops working

Follow-up to the previous entry's retraction. Three corrections, one new result.

### (a) ⚠ My own period-sweep probe was unsound below P=128
Last tick's sweep used `b = [u[:P]; u]` truncated. For `q ≥ P` the token `u[q−P]` has an earlier
occurrence **only if `q ≤ 2P−1`**; beyond that it is a fresh token appearing for the first time. So
for `P < 128` most evaluated positions were **not content-predictable** and the columns were rigged.

Replaced by a **tiled** probe, `b = [w w w …]`, `|w| = P`, where every `q ≥ P` is predictable at any
period. Re-audit of `attn2-mix10-seed0`:

| P | 128 | 110 | 100 | 96 | 64 | 48 | 43 | 150 | 180 |
|---|---|---|---|---|---|---|---|---|---|
| old `[u;u]` probe | 0.9036 | — | 0.0001 | — | **0.0002** | — | — | 0.0003 | — |
| **tiled probe** | 0.8995 | 0.0005 | 0.0003 | 0.0001 | **0.5836** | 0.0001 | 0.2144 | 0.0004 | 0.0001 |

**P=64 is 0.58, not chance.** The old columns below 128 were wrong.
**The retraction survives regardless**: P=96, 85, 150, 180 are all at chance under the sound probe,
and P≥128 was sound under both. Copying is *period-dependent* ⇒ **not** content-based induction.

### (b) ✗ My "fixed −127 offset positional copier" mechanism is FALSIFIED
It predicts copying iff `P | 128`. Tested: 128 ✓ 0.8995, 64 ✓ 0.5836, 32 ✓ 0.4820, 16 ✓ 0.4046,
96 ✓ chance, 85 ✓ chance — **but 43 → 0.2144 and 127 → 0.0586**, both non-divisors. Prediction broken.

Attention analysis (L1H1 is the copier: |attn| ≈ 16–20 vs background 1.5) shows a **mixture**:
- at P=96 it puts mass **17.4** on the fixed key `q−127`, which holds an unrelated token, and 0.16 on
  content-matched keys → copying fails. *Positional component.*
- at P=43 it puts most mass (**19.7**) on `q−kP+1`, exactly the induction-target positions → copying
  partially succeeds. *Content component.*

*(Caveat: bilinear scores are unnormalized and signed; this used |attn|, which discards the sign that
determines what the OV writes. This bounds the mechanism, it does not pin it. **Mechanism unresolved.**)*

⇒ Correct statement: the copy-burst lever installs a **period-dependent copier with both positional
and content components**, not a period-agnostic induction head. "Positional copier" was too strong.

### (c) ✗ My first random-period fix was itself broken
`[w ; u]` with random `P` leaves positions `q > 2P−1` as fresh unseen tokens — the burst is mostly
noise. The model learned nothing, and it was the construction's fault. Fixed by **tiling** the block.
Verified before training: every evaluated position has an earlier occurrence at every period.

### (d) NEW RESULT: removing the period regularity kills the lever

Matched runs, `attn2` dense, 30k steps, tiled random period `P ~ U[42,128]`:

| model | P=128 | 96 | 64 | 43 | 150 | 180 |
|---|---|---|---|---|---|---|
| mix10 **random-period** | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| mix30 **random-period** | 0.0007 | 0.0004 | 0.0005 | 0.0006 | 0.0004 | 0.0002 |
| mix10 **fixed-period** (control, 20k) | **0.7152** | 0.0001 | 0.4546 | 0.1622 | 0.0003 | 0.0003 |

Uniform baseline 0.00020. **With the period randomised, no copying is learned at all** — at 10% or
30% mixture, 30k steps — while the fixed-period control learns it in 20k.

⇒ **The copy-burst lever worked *because of* the period regularity.** Remove the shortcut and the
signal disappears within this budget. Whether content-based induction forms at all in this
architecture with a longer schedule is **untested and open**.

### Net effect on Program A
- ~~"10% copy-burst mixture installs induction 3/3 seeds"~~ stays retracted.
- The replacement is **not** "it installs a positional copier" (falsified) but: *it installs a
  period-dependent copier of unresolved mechanism, and randomising the period removes the learning
  signal entirely at 30k steps.*
- `lm_train.py` now has `--randperiod` (tiled). The fixed path's RNG draw is unchanged, so all
  existing runs remain reproducible.

**Tier-1.5 mechdecomp gate remains unaffected** — it recovers a causally-verified edge on natural
text, whatever that circuit is called.

---

## 2026-07-09 — CONTENT-BASED INDUCTION *CAN* FORM; and the fixed-period mechanism, resolved

### (1) POSITIVE CONTROL: the architecture can learn content-based induction
Before spending on a long run, ask whether the capability exists at all. `attn2` dense, **100%**
tiled random-period bursts, `P ~ U[42,128]`, 30k steps (`attn2-s30k-mix100-rp-dense-seed0`):

| trained periods | 128 | 110 | 96 | 85 | 64 | 43 |
|---|---|---|---|---|---|---|
| P(copy) | 0.9690 | 0.9905 | 0.9855 | 0.9916 | 0.9873 | 0.9591 |

| **untrained** | 32 | 16 | 150 | 180 | 200 |
|---|---|---|---|---|---|
| P(copy) | **0.8787** | **0.9161** | 0.5470 | 0.1253 | 0.0241 |

Near-perfect on every trained period and it **generalises to P=16 and P=32, far outside the trained
range** ⇒ genuine period-agnostic content matching. (Decay at P≥150 is expected: a 256-token context
leaves few positions with any prior occurrence at long periods.)

**So the 10%/30% failure is not architectural capacity — the signal is too sparse at that budget.**

### (2) The attention-offset diagnostic, validated then applied
Last tick's diagnostic was inconclusive (it used `|attn|`, discarding the sign that determines what
the OV writes). Re-run **signed**, with the known content matcher as a positive control:

| model | P | fixed key `q−127` | content keys `q−kP` | induct keys `q−kP+1` | background |
|---|---|---|---|---|---|
| mix100-rp (**known content matcher**) | 96 | −0.083 | −0.982 | **−2.664** | 0.071 |
| mix100-rp | 43 | −0.898 | −1.158 | **−3.043** | 0.110 |
| mix10 (fixed period) | 96 | **+17.445** | −0.020 | +0.145 | 1.557 |
| mix10 | 43 | +16.216 | +16.657 | +19.709 | 1.466 |

The diagnostic **has power** (32× induct-over-fixed on the known matcher) and shows mix10 is
dominated by a **fixed-offset** component.

### (3) ⚠ CORRECTION to my own tick-5/6 claims: the mechanism IS positional
Tick 5 said "positional copier". Tick 6 said that was **falsified** (P=43 → 0.2144, a non-divisor of
128) and the mechanism was "unresolved". **Both were wrong in the same way: I assumed a single
attended offset.** The head has mass at **−127 *and* −128**. A positional copier attending offset δ
copies correctly iff `δ ≡ 1 (mod P)`, so it succeeds iff **`P | 128` or `P | 129`**.

| prediction | P | P(copy) |
|---|---|---|
| copy (`P\|128`) | 128, 64, 32, 16 | 0.8995, 0.5836, 0.4820, 0.4046 |
| copy (`P\|129`) | 43, **129 (untrained, outside range)** | 0.2144, **0.4013** |
| chance (neither) | 126, 63, 44, 96, 85 | 0.0003, 0.0002, 0.0011, 0.0001, 0.0002 |

129 = 3·43 — which is exactly why P=43 copied and broke the naive rule. **P=129 was never trained and
lies outside [42,128], yet copies at 0.4013 as predicted.** Weak residual lifts at 127 (0.0586),
130 (0.0430), 65 (0.0313) are the attention peak's *width* (rotary makes the offset profile smooth,
so periods dividing 127 or 130 get partial credit).

⇒ **Resolved: a positional copier attending a narrow band of offsets near −127/−128.** Not content
matching. Program A's retraction stands, now with a mechanism that predicts an untrained period.

### Standing lesson
A mechanistic hypothesis must be tested in its *general* form. "Positional copier" was right; "fixed
offset −127" was a stronger claim I invented, and its failure looked like the hypothesis failing.

### (4) Mixture ladder — NON-MONOTONIC, threshold NOT reported (single seed)

Tiled random period, 30k steps, `attn2` dense. Copy probability (uniform baseline 0.00020).
`*` = untrained period, so only period-agnostic **content matching** can score there.

| mix | val CE | P=128 | P=96 | P=43 | P=32* | P=16* |
|---|---|---|---|---|---|---|
| 0.1 | 4.675 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0003 |
| 0.3 | 4.743 | 0.0007 | 0.0004 | 0.0006 | 0.0009 | 0.0033 |
| **0.5** | 5.069 | **0.6155** | **0.7483** | **0.7095** | **0.6742** | **0.4662** |
| 0.7 | 5.125 | 0.0126 | 0.0062 | 0.0068 | 0.0069 | 0.0184 |
| 1.0 | 11.297 | 0.9690 | 0.9855 | 0.9591 | 0.8787 | 0.9161 |

**mix 0.5 learns content-based induction; mix 0.7 nearly fails.** A threshold cannot reverse, so this
is either high formation variance (consistent with the basin-competition dynamics found in Program A)
or a bad seed. **One seed per point cannot distinguish these, so no threshold is claimed.**
Seeds 1 and 2 at mix 0.5/0.7 are running.

### (5) A GENUINE content-based induction circuit — the ground truth Tier 1.5 never had

`attn2-s30k-mix50-rp-dense-seed0`, probed at **P=96** (divides neither 128 nor 129, so a positional
copier scores chance there — this isolates content matching). Base P(copy) = 0.7483.

| ablated | P(copy) | drop |
|---|---|---|
| L1H0 | 0.1782 | **76.2%** |
| L1H3 | 0.1850 | **75.3%** |
| L1H1, L1H2 | 0.7485, 0.7483 | 0.0% |
| L0H1 | 0.0078 | **99.0%** |
| L0H0 | 0.0838 | **88.8%** |
| L0H3 | 0.4190 | 44.0% |
| L0H2 | 0.7529 | −0.6% |

A **redundant pair** of L1 copy heads (H0, H3) reading from **two required** L0 heads (H1, H0).
Not a single edge (L0H1 / L0H0 = 1.11×), so the single-edge gate still does not apply — but the
**joint-Δŝ gate ranks heads and does**. This model is the first circuit in this program that is
induction in the Elhage/Olsson sense, and is the right Tier-1.5 ground truth going forward.

### (6) RESOLVED: there is no mixture threshold — formation is STOCHASTIC

Seed replicates at the same mixture and step budget, scored at P=96 (content-only):

| mix | seed 0 | seed 1 |
|---|---|---|
| 0.5 | **0.7483** (hit) | **0.0011** (miss) |
| 0.7 | 0.0062 (miss) | *(pending)* |

**The same setting gives a hit and a miss.** So the non-monotonic ladder in (4) was formation
variance, not a threshold, and the single-seed table must not be read as one. Content-based induction
formation with tiled random-period bursts is **stochastic at 30k steps** — exactly the basin-competition
signature Program A found for the old (positional) lever, now confirmed for the real capability.

Consequences:
- **No "minimum mixture" claim.** Any such number from one seed per point is noise.
- The 10%/30% failures in (4) are consistent with a low *probability* of formation, not impossibility.
- A proper characterisation needs seeds × mixture × steps, and should report **formation rate**, not
  a copy score. That is a real experiment, not a quick sweep.
- `attn2-s30k-mix50-rp-dense-seed0` remains a genuine content-based induction model (§5) — it is a
  *sample* from the hit branch, and is still the best Tier-1.5 ground truth available.

---

## 2026-07-09 — ⚠ THE TIER-1.5 GATE IS VACUOUS. All its PASSes are retracted as evidence.

Ran the joint-Δŝ gate against the **genuine** content-based induction circuit
(`attn2-s30k-mix50-rp-dense-seed0`, ground truth L0H1 −99.0% > L0H0 −88.8% > L0H3 −44.0% > L0H2 −0.6%,
probed at P=96 where positional copiers score chance). It passed — Δŝ tracked true Δ to 0.002, rank
and argmax both correct. That is *too* good, so I controlled it.

### Two controls, both fatal

| control | result |
|---|---|
| **C1: rank heads by `‖write‖`, no decomposition at all** | reproduces the causal ranking exactly (mix50-rp) |
| **C2: random, unlearned dictionary, same k** | OV R² 0.916–0.993; Δŝ within **0.0003** of learned; **passes** |

Re-audit of the previously reported natural-text gate (`attn2-seed0`, "argmax L0H3 at 8.81×, PASS"):

| dictionary | Δŝ per head (H0…H3) | argmax | margin |
|---|---|---|---|
| learned | +0.0099, −0.0034, +0.0023, **+0.0870** | L0H3 | 8.81× |
| **random (unlearned)** | +0.0031, −0.0020, +0.0001, **+0.0799** | L0H3 | **25.59×** |

**A random dictionary passes with a larger margin than the learned one.**
(C1 does *not* match there — it picks L0H0 — so that gate is not trivially magnitude. C2 kills it anyway.)

⇒ **The joint-Δŝ gate measures write *reconstructability*, not mechanism identification.** Every
Tier-1.5 "PASS" (4.07× random-token, 8.81× natural-text, and today's rank/argmax pass) is **retracted
as evidence about the decomposition**. What still stands: the causal edges themselves (ground truth),
the OV reconstruction R² values, and the finding that §1.5's per-branch contraction is ill-posed.

### A gate that CAN fail — atom localization — and it fails
If the decomposition means anything, its atoms should *concentrate* the mechanism. Ablate atoms in
order of contribution energy from L0H1's write and count how many are needed to halve P(copy):

| atoms removed | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| **learned** | 0.7231 | 0.6984 | 0.6584 | 0.5759 | 0.4402 | 0.2456 | 0.0756 | 0.0073 |
| **random** | 0.7155 | 0.6882 | 0.6364 | 0.5524 | 0.4049 | 0.2190 | 0.0695 | 0.0073 |

Atoms to halve: **learned 32, random 32** — the learned curve is *slightly worse at every m*.
**The decomposition does not localize the mechanism any better than random directions.**

### My explanation for this was also wrong
I hypothesised: a copy head's OV action is a near-isometry on its `d_head` subspace, so no sparse
structure exists and a flat singular spectrum should be the signature. Measured:

| map | rank | effective rank (exp-entropy) | eff/rank |
|---|---|---|---|
| attn2 L0H0-OV | 32 | 8.1 | 0.25 |
| attn2 L0H1-OV | 32 | 13.7 | 0.43 |
| attn2 L0H3-OV | 32 | 3.3 | 0.10 |
| pythia-410m L3 `down_proj` | 1024 | 823.1 | **0.80** |

The OV maps are **far more structured** than `down_proj` — the opposite of the prediction — and
`down_proj` is exactly where the learned dictionary *does* beat PCA by 0.25 R² at matched L0.
**Hypothesis refuted; cause unresolved.** The most likely remaining account: with 128 atoms in a
128-dim space against a ≤32-dim target, `k=8` does not bind — any dictionary spans the target
subspace, so both reconstruction and top-atom ablation depend on the *span*, not on atom identity.
Testable next: push overcompleteness up and `k` down (e.g. 1024 atoms, k=2) so sparsity binds, and
re-run the localization gate.

### Net position
- **Pythia Tier 2 stands**: there the learned dictionary dominates a proper floor (PCA-32 0.3509,
  PCA-64 0.4231, OMP-over-PCA 0.3897 vs learned **0.6020**). That comparison had a control that could
  fail, and the method beat it.
- **Tier 1.5 gives no evidence either way** about the decomposition. Its circuits are real; the gate
  built on them was not discriminative. A random-dictionary control is now **mandatory** for every
  future gate in this program.

---

## 2026-07-09 — WHY the Tier-1.5 gate was vacuous: the OV mechanism is *distributed*, not sparse

Follow-up to the gate retraction. Two questions: (a) does the learned-vs-random tie disappear once
sparsity binds? (b) is there any regime where the decomposition localizes the mechanism?

### (a) Sparsity binding explains the *reconstruction* tie
`L0H1-OV` (99% causal), effective rank 13.7 in `d=128`. Learned vs random unlearned dictionary:

| K | k=8 | k=4 | k=2 | k=1 |
|---|---|---|---|---|
| 128 | +0.0039 | +0.0414 | +0.1268 | +0.1805 |
| 512 | +0.0026 | +0.0213 | +0.1176 | +0.2281 |

(R² gap, learned − random.) At `k=8` the dictionary is irrelevant — which is exactly the regime the
old gate ran in. As `k` falls the learned dictionary pulls away. **Consistent with Pythia**, where
`k=32` against a 1024-dim target *does* bind: random unlearned 0.2587 vs learned 0.6020.

### (b) But there is NO regime that is sparse *and* faithful *and* localizing

`K=512`, L0H1-OV. "faithful" = P(copy) after swapping the true write for its reconstruction.

| k | R² learn | R² rand | gap | faithful P(copy) | halve@ learn / rand |
|---|---|---|---|---|---|
| 16 | 0.9997 | 0.9992 | +0.0004 | 0.7470 (100% of base) | 64 / 64 |
| 8 | 0.9987 | 0.9961 | +0.0026 | 0.7472 (100%) | 128 / 128 |
| 4 | 0.9862 | 0.9642 | +0.0219 | 0.7020 (94%) | 128 / 128 |
| 2 | 0.9200 | 0.8005 | +0.1194 | 0.4453 (60%) | 128 / 128 |
| 1 | 0.7652 | 0.5354 | +0.2299 | 0.2350 (31%) | 128 / 128 |

Where the code is faithful, the dictionary does not matter. Where the dictionary matters, the code
destroys the behaviour. **`halve@learned` never beats `halve@random` at any k.**

### (c) Strongest form of the gate: rank atoms by MEASURED causal effect
512 single-atom ablations per dictionary, at the faithful `k=8`:

| | faithful | live atoms | top single-atom effect | atoms to halve |
|---|---|---|---|---|
| learned | 0.7484 | 512 / 512 | 0.0103 (1.4% of base) | 128 |
| random | 0.7479 | 512 / 512 | 0.0093 | 128 |

Identical curves; random marginally *lower* at most points. **No atom carries more than 1.4% of the
copy behaviour, and every atom is live.**

### Conclusion — a scope limit for the spec, not a solver bug
The induction write is "copy the source token's embedding": a **dense** linear map on a 32-dim
subspace. Its mechanism is genuinely distributed, so **no sparse dictionary can localize it** — the
learned one is not failing, there is nothing sparse to find. This is why every Tier-1.5 gate passed
with random atoms.

**Corrects my previous explanation.** I attributed the tie to a flat OV spectrum (isometry). Measured,
the OV maps are *more* structured (eff/rank 0.10–0.43) than Pythia `down_proj` (0.80) — and
`down_proj` is where the method works. **The map's spectrum does not predict decomposability.** What
matters is whether the map's *action on the data* is sparsely structured, and the spectrum is silent
about that.

### A usable diagnostic (proposed for the spec)
Before decomposing a map, sweep `k` and record (i) the learned−random R² gap and (ii) faithfulness of
the reconstructed output. **If no `k` gives both a meaningful gap and preserved behaviour, the map has
no sparse mechanism and the decomposition cannot say anything about it.** Attention OV of a copy head:
fails this test. Pythia `down_proj`: passes (gap 0.34 at faithful sparsity, and beats PCA-32 by 0.25).

---

## 2026-07-09 — AUDIT of the surviving result: Pythia `down_proj` PASSES the gates that killed Tier 1.5

Tier 1.5 died to two controls: a random unlearned dictionary passed every gate, and the learned
dictionary localized no better than random. Pythia's result had never faced either. It does now.
(`mechdecomp/tier2_audit.py`, `logs/tier2_audit.log`, `logs/ce_scale.log`.)

Pythia-410m L3 `down_proj` (1024×4096), K=2048 atoms, held-out, D frozen.
**Faithfulness is measured behaviourally** — splice the atom-reconstructed output back into the model
and read LM cross-entropy — because R² is not behaviour.

### (1) Decomposability pre-test: PASS

| k | R² learn | R² rand | gap | spliced CE | ΔCE vs base (2.8011) |
|---|---|---|---|---|---|
| 8 | 0.5393 | 0.0441 | **+0.4952** | 2.8818 | +0.0807 |
| 16 | 0.5773 | 0.1444 | +0.4329 | 2.8663 | +0.0652 |
| 32 | 0.6397 | 0.2823 | +0.3574 | 2.8497 | +0.0486 |
| 64 | 0.7261 | 0.4585 | +0.2675 | 2.8405 | +0.0394 |

There **is** a k that is sparse, faithful, and dictionary-dependent — precisely what copy-head OV
lacked at every k. At k=8 the dictionary is worth +0.4952 R²; on OV maps at k=8 it was worth +0.0026.

### CE calibration (without this, "+0.049" is meaningless)

| intervention | CE | ΔCE | recovers |
|---|---|---|---|
| baseline | 2.8011 | — | — |
| **down_proj output zeroed** | 3.0258 | **+0.2247** | the layer is worth 0.225 nats total |
| mean-ablated | 3.0131 | +0.2120 | |
| random dict spliced, k=32 | 2.8910 | +0.0899 | 60.0% |
| **learned dict spliced, k=32** | 2.8497 | **+0.0486** | **78.4%** |

A 32-of-2048 sparse code recovers 78.4% of the layer's entire causal contribution; random recovers 60%.

### (2) Localization gate, causal-ranked: PASS

Atoms ranked by **measured** single-atom ΔCE (128 sampled atoms per dictionary), k=32:

| | spliced CE | top single-atom ΔCE | drop 1 | drop 4 | drop 16 | drop 64 | drop 128 |
|---|---|---|---|---|---|---|---|
| learned | 2.8497 | **+0.0112** | 2.8609 | 2.8628 | 2.8680 | 2.8728 | 2.8736 |
| random | 2.8910 | +0.0009 | 2.8920 | 2.8926 | 2.8933 | 2.8998 | 2.8967 |

**A single learned atom carries 12× the causal effect of the best random atom.** Normalised by each
dictionary's own explained CE (learned 0.1761 nats, random 0.1348): one learned atom is 6.4% of what
its dictionary explains, one random atom 0.67% — still ~10×.

### Verdict
**Pythia `down_proj` is the first and only place in this program where the learned decomposition does
something a random dictionary cannot** — on reconstruction (+0.50 R² at k=8), on behaviour (78.4% vs
60.0% CE recovery), and on causal localization (12× single-atom effect). It passes both controls that
Tier 1.5 failed.

Caveats, stated plainly:
- **Modest stakes.** Destroying this MLP entirely costs only 0.2247 nats, so a single atom's 0.0112 is
  5.0% of the layer, not of the model.
- **Effects saturate.** Dropping the top 128 learned atoms adds only +0.0239 CE versus +0.0112 for the
  top 1 — the code is highly redundant, so "one atom = one mechanism" is not supported.
- Single layer, single model. Nothing here says the atoms are *interpretable*, only that they are
  causally concentrated relative to a random basis.

### Where the program now stands
- **Tier 0/1**: theory verified; recovery of a known dictionary confirmed with measured chance levels.
- **Tier 1.5**: real circuits, but the OV maps are **not decomposable** (dense mechanism) — no evidence
  either way about the method. The gate built there was vacuous.
- **Tier 2 (Pythia)**: the method works, and demonstrably beats random and PCA on reconstruction,
  behaviour, and localization.
- **Gemma**: hard-blocked on the licence.

The **decomposability pre-test** (sweep k; require a learned−random gap *and* behavioural faithfulness)
cleanly separates the two regimes and should gate every future target.

---

## 2026-07-09 — THE PRE-TEST IS CONFOUNDED, and my "attention vs MLP" story is REFUTED

Yesterday's proposed **decomposability pre-test** (sweep k; require a learned−random R² gap plus
behavioural faithfulness) was only ever applied to the two maps whose answer I already knew. Applied
to two maps I had *not* tested, its story collapses.

### (a) Predictive test — "attention ⇒ small gap" is FALSE

k=8, K=1024 for all maps (`logs/pretest_predict.log`):

| map | type | rank | K/rank | learn R² | rand R² | gap |
|---|---|---|---|---|---|---|
| pythia-410m L3 `down_proj` | MLP | 1024 | 1.00 | 0.703 | −0.000 | **+0.703** |
| pythia-410m L3 `attention.dense` | attention | 1024 | 1.00 | 0.820 | 0.214 | **+0.606** |
| GPT-2 L6 full attention OV | attention | 768 | 1.33 | 0.728 | 0.362 | **+0.365** |

**Attention maps show large gaps.** My tick-9/10 framing — "the method works on MLP
down-projections and has nothing to say about attention OV maps" (LOG 41) — is **retracted**.

### (b) The gap is partly a function of K/rank(W), not of mechanism structure

attn2 L0H1-OV (d_in 128, rank 32), k=8 (`logs/krank_confound.log`):

| K | K/rank | gap | faithful P(copy) | atoms to halve, learned / random |
|---|---|---|---|---|
| 32 | 1.00 | +0.0271 | 0.7263 | 8 / 8 |
| 64 | 2.00 | +0.0068 | 0.7391 | 16 / 16 |
| 128 | 4.00 | +0.0040 | 0.7448 | 32 / 32 |
| 512 | 16.00 | +0.0026 | 0.7472 | 128 / 128 |

The gap grows 10× as K/rank falls from 16 to 1. **The pre-test as I proposed it is confounded by
overcompleteness relative to `rank(W)`** — a map tested with K < rank would be rated "decomposable"
for free. Any comparison across maps must be at **matched K/rank**.

*(Why the earlier "prediction" looked plausible: attn2 was run at K/rank=16, the other maps at ≈1.)*

### (c) What survives, and what does not

**Survives — attn2 OV really is non-localizable.** At *matched* K/rank=1 its gap is +0.0271 versus
+0.606 for Pythia's attention map at the same ratio, so the difference is not overcompleteness. And
`atoms-to-halve` is exactly **K/4 for learned and random at every K** — the cleanest possible signature
that no atom subset carries the mechanism. The tick-9 conclusion holds **for this map**.

**Retracted — the generalisation.** "Attention OV of a copy head is dense, so the method has nothing to
say about attention" was drawn from one 128-dimensional, rank-32 map in a 1.5M-parameter model. Two
larger attention maps show gaps of +0.365 and +0.606. Their *localization* has not been tested.

**Amended — the pre-test.** It is still useful, but must be specified as: *sweep k at matched K/rank
across maps, and require a learned−random gap AND behavioural faithfulness AND a localization
advantage over a random dictionary.* The gap alone is not sufficient — Pythia's attention map has a
large gap and its localization is untested; attn2's OV has a (small) gap at K/rank=1 and localizes
nothing.

### Corrected program summary
- **Tier 0/1**: theory verified; dictionary recovery confirmed against measured chance.
- **Tier 1.5**: real circuits; the one OV map tested is not localizable at any K. No claim about
  attention maps in general.
- **Tier 2 (Pythia `down_proj`)**: passes gap, behavioural faithfulness (78.4% CE recovery vs 60.0%
  random), and causal localization (12× single-atom effect). Still the only map where all three hold.
- **Untested and interesting**: Pythia `attention.dense` and GPT-2 OV both have large gaps. Do they
  localize? That is the next experiment, and it decides whether the method is MLP-specific at all.

---

## 2026-07-09 — RECONSTRUCTION AND LOCALIZATION ARE INDEPENDENT: attention.dense reconstructs, but localizes nothing

The scope question from last tick: Pythia `attention.dense` has a large learned−random R² gap, so is
the method not MLP-specific? Ran all three gates at **matched K/rank = 2.00** (K=2048, rank 1023),
identical to the `down_proj` audit. `mechdecomp/localize_attn.py`, `logs/localize_attn.log`.

Module stakes are comparable: zeroing `attention.dense` costs **+0.2213** CE (down_proj: +0.2247).

### Gates (1) and (2): PASS, and better than down_proj

| k | R² learn | R² rand | gap | spliced ΔCE (learned) | % recovered | % recovered (random) |
|---|---|---|---|---|---|---|
| 8 | 0.6962 | 0.2479 | +0.4483 | +0.0161 | **92.7%** | 56.9% |
| 32 | 0.8130 | 0.5558 | +0.2572 | +0.0143 | **93.5%** | 83.6% |

A 32-of-2048 sparse code recovers **93.5%** of the module's entire causal contribution
(`down_proj`: 78.4%).

### Gate (3), localization: FAIL — and it is a clean dissociation

Causal-ranked, k=32, 128 sampled atoms per dictionary:

| | spliced CE | top single-atom ΔCE | drop 1 | drop 4 | drop 16 | drop 64 | drop 128 |
|---|---|---|---|---|---|---|---|
| learned | 2.8154 | **+0.0009** | 2.8163 | 2.8161 | 2.8175 | 2.8186 | 2.8186 |
| random | 2.8373 | **+0.0009** | 2.8383 | 2.8396 | 2.8397 | 2.8427 | 2.8438 |

**Identical.** For reference, `down_proj` gave learned +0.0112 vs random +0.0009 (12×).

### The finding
**Reconstruction quality and mechanism localization are independent properties of a map.**

| map | gap | behavioural faithfulness | localization |
|---|---|---|---|
| pythia `down_proj` | +0.4952 | 78.4% recovered | **12× over random** |
| pythia `attention.dense` | +0.4483 | **93.5% recovered** | **1× — none** |
| attn2 L0H1-OV (K/rank 1) | +0.0271 | 97% of base P(copy) | 1× — none |

`attention.dense` reconstructs *better* than `down_proj` and localizes *worse*. So:

- **The gap + faithfulness criteria are NOT sufficient.** Last tick I amended the pre-test to require a
  localization advantage as well; this shows the amendment is load-bearing, not belt-and-braces —
  `attention.dense` passes both of the other criteria and fails this one.
- **The method is not "MLP vs attention".** It is: *does this map's action on data decompose into
  causally-concentrated atoms?* For `down_proj`, yes. For two different attention maps — one in a
  1.5M-param toy, one in Pythia-410m — no.
- Corollary: a decomposition can be an excellent *compression* of a module's behaviour (93.5% of its
  causal effect in 32 atoms/token) while none of its atoms is individually a mechanism.

**Pending:** the `down_proj` 12× is now the program's ONLY localization evidence. A usage-skew control
is running — if the top learned atom simply fires on more tokens than a random atom, the 12× is an
artifact and there is no surviving positive localization result anywhere.

---

## 2026-07-09 — ⚠ THE LOCALIZATION CRITERION HAS NO POWER. Every localization result is retracted.

### (a) The `down_proj` "12×" does not replicate
The 12× compared **max-over-128-sampled-atoms ΔCE** across two runs with different training subsets.
A follow-up on a 40-atom *prefix* of the same sample gave learned +0.00119 vs random +0.00129 — which
is impossible if random's max over the superset were 0.0009. The runs were not comparable.

Clean re-test (`logs/localize_clean.log`): identical sequences, identical atom sample, full
distribution instead of a max, k=32, K=2048:

| dict | spliced CE | mean ΔCE | median | top1 | top5 mean | frac > 1e-3 |
|---|---|---|---|---|---|---|
| learned | 2.7264 | **−0.00005** | −0.00005 | +0.00098 | +0.00077 | 0.00 |
| random | 2.7792 | **+0.00030** | +0.00024 | +0.00123 | +0.00116 | 0.09 |

The learned distribution is **not shifted above random** — random is marginally higher on every
statistic. **The 12× is retracted.**

### (b) The criterion itself cannot detect localization where it provably exists
Power control on the Tier-1 toy, where `D_true` **is** the generator:

| criterion | TRUE dict | RANDOM dict | ratio |
|---|---|---|---|
| marginal single-atom ΔR² (mean) | 0.00392 | 0.00254 | **1.54×** |
| marginal single-atom ΔR² (top1) | 0.00669 | 0.00667 | **1.00×** |
| conditional ΔR² (only where the atom fires) | 0.2502 | 0.1597 | **1.57×** |
| *(reference)* reconstruction R² | **0.9918** | **0.5948** | separates hugely |

**Single-atom ablation — marginal or conditional — barely distinguishes the true generator from a
random dictionary.** With codes fixed, deleting any atom a datapoint uses removes ~1/k of its
reconstruction whether or not that atom is a generative factor. A dictionary's quality lives in *which
atoms it selects*, not in per-atom effect size.

### (c) Therefore, retracted as uninformative (not as false — as untestable by this method)
1. `down_proj` "atoms are 12× more causal than random" — **refuted** (a).
2. Tier 1.5 "the attn2 OV mechanism is distributed; nothing sparse to find" (atoms-to-halve = K/4 for
   learned and random) — the criterion has no power, so this shows nothing.
3. `attention.dense` "reconstructs but localizes nothing" — same criterion, same problem.
4. The tick-12 claim that "reconstruction and localization are independent properties" — the
   localization axis was never measured.

### (d) A criterion that DOES have power: irreplaceability
Remove atom *j* from the **dictionary** and let OMP **re-select** codes from the rest. A true
generative factor cannot be substituted; a random direction can be.

| criterion | TRUE dict | RANDOM dict | ratio |
|---|---|---|---|
| irreplaceability, mean R² loss | 0.00257 | 0.00032 | **8.01×** |
| irreplaceability, top1 | 0.00483 | 0.00114 | **4.24×** |

Compare single-atom ablation: 1.54× / 1.00×. **Irreplaceability separates; ablation does not.**

### Where the program actually stands
**Valid, replicated:** the learned−random *reconstruction* gap and *behavioural CE recovery*. These
separate true from random in the toy (R² 0.99 vs 0.59) and learned from random on real maps
(down_proj 78.4% vs 60.0% CE recovery; attention.dense 93.5% vs 83.6%; attn2 OV gap only +0.027 even
at K/rank=1). By this criterion the method demonstrably learns something on Pythia's maps.

**Unresolved:** whether any atom is a *mechanism*. Every test I ran for that used a criterion that
cannot detect a known-true dictionary. The question is open, and `irreplaceability` is the tool.

**Next:** run the irreplaceability criterion on `down_proj`, `attention.dense`, and attn2 OV. That,
not single-atom ablation, decides whether this method finds mechanisms.

---

## 2026-07-10 — IRREPLACEABILITY on real maps: the first valid answer to "are atoms mechanisms?"

Single-atom ablation was shown to have no power (it cannot distinguish a known-true dictionary from a
random one). Irreplaceability — drop an atom from the **dictionary** and let OMP **re-select** codes —
does. First, the confound check on the toy:

| | base R² | mean loss | top1 |
|---|---|---|---|
| TRUE dict, k=4 | 0.9921 | 0.00253 | 0.00368 |
| RANDOM, same k | 0.5951 | 0.00030 | 0.00084 |
| RANDOM, k=24 (**matched R² 0.9769**) | 0.9769 | 0.00007 | 0.00017 |

Unmatched ratio 8.4×; **matched-R² ratio 34× mean / 21× top1**. The confound works *against* the
criterion — random dictionaries become *more* replaceable as k rises — so a matched comparison is
conservative. Criterion validated.

### Real maps (`mechdecomp/irreplaceability.py`, `logs/irreplaceability.log`), K=1024, k=16, held-out

| map | learned base R² | mean loss | top1 | matched-R² ratio (mean / top1) |
|---|---|---|---|---|
| pythia L3 `down_proj` | 0.5234 | 0.000400 | **0.007924** | **1.84× / 5.18×** |
| pythia L3 `attention.dense` | 0.7406 | 0.000152 | 0.000490 | **1.46× / 2.27×** |
| *(toy true dict, reference)* | 0.9921 | 0.00253 | 0.00368 | *34× / 21×* |

Matched controls: random reaches R² 0.5498 at k=128 (vs learned 0.5234) for `down_proj`, and 0.8145 at
k=128 (vs 0.7406) for `attention.dense`.

### What this says
**The learned atoms on real maps are only mildly irreplaceable — far closer to an arbitrary basis than
to generative factors.** Against a matched random dictionary the mean advantage is 1.84× (`down_proj`)
and 1.46× (`attention.dense`), where a true generator scores 34×.

But the **tail is where the signal is**: `down_proj`'s most irreplaceable atom costs 0.0079 R², *20×
its own mean* and 5.18× the matched-random top atom. `attention.dense`'s tail is much thinner (2.27×).
So a minority of `down_proj` atoms behave meaningfully like mechanisms; the bulk do not.

Ranking is consistent with the reconstruction gap (down_proj > attention.dense > attn2 OV), which is
mild evidence that both measure something real about the map rather than about the solver.

### Honest calibration
The toy is an **exactly sparse generative model** — an upper bound no real activation distribution can
be expected to hit. So "1.84× vs 34×" does not mean "the method fails"; it means real `down_proj`
activations are not generated by a small set of irreplaceable atoms to anywhere near that degree. The
defensible claim is the one against the matched random baseline: **> 1×, modestly, with a heavy tail.**

### Program status (all claims now made with criteria of demonstrated power)
- **Reconstruction / behavioural recovery**: learned ≫ random, replicated. Solid.
- **Mechanism-likeness (irreplaceability)**: learned > matched-random by 1.5–1.8× mean, up to 5× in
  the tail. Real but weak. **This is the first valid measurement; every earlier localization number
  used a criterion that could not detect a known-true dictionary.**
- **Next**: inspect `down_proj`'s tail atoms (top1 = 20× its own mean). If those few atoms are
  interpretable, the method's value is in the tail, not the bulk — which would be a usable finding.

---

## 2026-07-10 — DO THE IRREPLACEABLE ATOMS MEAN ANYTHING? Weakly yes.

`down_proj`'s learned dictionary is only mildly irreplaceable on average but has a heavy tail (top
probed atom = **27× its own mean**). If the method has value it should live there. Tested with a
control, because "interpretable" without one is worthless.

**Metric.** `purity(atom)` = fraction of an atom's top-40 activating positions sharing the same token.
Chance level **measured, not assumed**, by scoring random (non-activation-selected) positions: **0.081**.
Atoms fire on ~1.6% of tokens (k/K = 16/1024), so purity is computed on a separate 8000-point set —
the first attempt used 1000 points and every atom fired < 40 times, which the guard caught.

### First look (n=5 per group) — and why it overstates the effect

| group | mean purity | notable |
|---|---|---|
| TAIL (most irreplaceable) | **0.600** | atoms 264, 537: purity **1.000**, entropy 0, pure `'\n'` detectors; 634: 0.725 on `' a'/' an'` |
| BULK (median irreplaceability) | 0.150 | |
| RANDOM dictionary | 0.090 | |
| chance | 0.081 | |

### Correlation over all 96 probed atoms — the number to quote

| statistic | value |
|---|---|
| **Spearman(irreplaceability, purity)** | **+0.231** |
| mean purity, top-10 irreplaceable | 0.347 |
| mean purity, bottom-10 | 0.120 |
| chance | 0.081 |

**The 0.600-vs-0.150 table was inflated by a 5-atom selection** that happened to contain two
purity-1.0 newline detectors. The honest effect is Spearman +0.231: irreplaceable atoms are *modestly*
more token-pure. Both tails sit well above chance.

Note the relationship is **not monotone**: the single most irreplaceable atom (66, 27× the mean) has
purity only 0.225 (`'Q','Tag','Sen','J'`). Irreplaceability ≠ monosemanticity.

### Internal consistency check (this could have broken the criterion)
Atoms 264 and 537 are *both* pure `'\n'` detectors. If they were near-duplicates, each would substitute
for the other and **neither could be irreplaceable** — a contradiction that would invalidate the metric.

    cos(d_264, d_537) = -0.2475      (each is the other's nearest neighbour, in a very incoherent dict)

They are **not** duplicates. Two distinct directions both select newline tokens, and each is genuinely
irreplaceable. The criterion survives its own consistency test.

### Verdict
The first interpretability-relevant result in this program that has controls:
**irreplaceability weakly predicts token purity** (Spearman +0.231; top-decile purity 4.3× chance),
and the most irreplaceable atoms include some crisply monosemantic ones (two pure newline detectors,
one article detector). But the bulk of the dictionary is not monosemantic, and the single most
irreplaceable atom is not the purest.

**Caveats.** Purity is a crude proxy — it uses *token identity only* and ignores context, so it cannot
see context-dependent features (which is most of what an SAE would call a feature). 96 atoms, one
layer, one model. The effect is real, small, and should not be described as "the method finds
interpretable features".

---

## 2026-07-10 — THE CENTRAL CLAIM, TESTED: weight-aware does NOT beat an activation-only SAE

The spec proposes a *weight-activation* method: atoms chosen for how they decompose `W`'s action
(`Wx`), not for how they reconstruct `x`. The obvious baseline — a plain SAE on the same activations,
whose decoder directions are used as the dictionary — was never run in this program. It is now.
Everything matched: same activations, same K=1024, same k=16, same held-out evaluation.
(`mechdecomp/vs_sae.py`, `logs/vs_sae.log`.)

**Guard G1** — the SAE must actually train, or it is not a fair baseline. x-reconstruction R²:
SAE **0.2733**, random −0.1540, masked-projector −0.0306. Passes (and note the masked projector is
*worse than random* at reconstructing `x`, exactly as its objective predicts).

| dictionary | R²(`Wx`) | irrepl mean | irrepl top1 | purity of top-10 |
|---|---|---|---|---|
| masked-projector (weight-aware) | **0.4919** | 0.000227 | 0.006116 | 0.347 |
| **SAE decoder (x-only)** | 0.4654 | **0.000246** | **0.009319** | **0.385** |
| random | 0.0893 | 0.000046 | 0.000353 | 0.347 |

**An SAE that never sees `W` nearly matches the weight-aware dictionary on the map's own objective
(0.4654 vs 0.4919) and beats it on both mechanism criteria.** The SAE here is deliberately weak
(3000 steps, x-R² 0.2733), so the comparison is *generous* to the masked projector, and it still loses.

⇒ **The spec's central premise — that optimising for `Wx` yields better atoms than optimising for `x` —
is unsupported at this setting.** Both dictionaries crush random on R²(`Wx`) (0.49 / 0.47 vs 0.09), so
the objective is learnable; it just isn't better than plain sparse coding of the activations.

## …and tick 14's "the value is in the tail" is RETRACTED

The vs-SAE run showed the **random** dictionary's top-irreplaceable atoms reaching purity 0.347 —
identical to the masked projector's. Tick 14 compared learned *tail* atoms (ranked by irreplaceability)
against *unranked* random atoms (0.090). Apples to oranges. Proper 2×2 (`logs/purity_2x2.log`):

| dictionary | top-10 irreplaceable | 10 random atoms | all 96 probed |
|---|---|---|---|
| masked-projector | 0.347 | 0.100 | 0.175 |
| SAE decoder | 0.285 | 0.072 | 0.127 |
| **random** | **0.347** | 0.160 | 0.116 |
| *(measured chance)* | | | 0.081 |

**Purity tracks the SELECTION RULE, not the dictionary.** Ranking *any* dictionary — including random
directions — by irreplaceability surfaces token-pure atoms. Mechanism: a few token types (newline,
articles) occupy distinctive, isolated activation directions; whichever atom points that way becomes
both irreplaceable and token-pure. That is a property of the **data**, not of the method.

So: "irreplaceable atoms are modestly more token-pure" is true and reproducible, but it is **not
evidence about this decomposition**. Tick 14's framing ("the method's value lives in the tail") is
withdrawn.

### Corrected standing of the program
| claim | status |
|---|---|
| Closed-form theorem (§1.1), §1.2 negative result | verified |
| Dictionary recovery on toys with known ground truth | verified, chance measured |
| Objective is learnable on real maps (R²(Wx) ≫ random) | **holds** (0.49 vs 0.09) |
| Weight-aware beats activation-only SAE | **FAILS** — SAE matches on `Wx`, wins on irreplaceability + purity |
| Atoms are mechanisms (irreplaceability vs matched random) | weak (1.84× mean), and **not method-specific** |
| Irreplaceable atoms are interpretable | true of **any** dictionary — a selection effect, not a result |
| Spec §1.3 free-ablatability; §1.5 contraction read-off | already retracted |

**The honest summary of the method as specified: it learns a better-than-random basis for a map's
action, is no better than a plain SAE at doing so, and its atoms are not more mechanism-like than an
SAE's.** The single strongest remaining use for the `Wx` objective would be a case where one needs a
basis *for the map's output* specifically — but no such advantage over the SAE has been demonstrated.

**Caveats.** One layer, one model, one (K, k). A better-trained SAE would likely widen its lead. The
irreplaceability differences (0.000246 vs 0.000227) are small; the top1 difference (0.0093 vs 0.0061)
is larger but is a max statistic over 96 atoms — exactly the kind I have already been burned by once.
A seeded replication with a distribution-level test is the right next step before this is final.

---

## 2026-07-10 — ⚠ REVERSAL: the seeded replication overturns tick 15. The premise IS supported.

Tick 15 concluded "weight-aware does NOT beat an activation-only SAE" from **one seed**, using the
**mean** and the **max** of 96 per-atom irreplaceability losses. I flagged at the time that a
max-over-sample statistic had already burned this program once and queued a replication. It reverses
the verdict. (`mechdecomp/vs_sae_replicate.py`, `logs/vs_sae_rep.log`.)

Design: 3 seeds; **identical probe indices, eval points, K=1024, k=16** for every dictionary; a
**stronger** SAE (8k steps, not 3k) so the baseline is not handicapped; full distributions and a
Mann-Whitney U on the pooled per-atom losses instead of a max.

| dictionary | R²(`Wx`) mean±sd | irrepl **median** | mean | p90 | max |
|---|---|---|---|---|---|
| **masked-projector** | **0.4591 ± 0.0003** | **0.000134** | 0.000223 | 0.000387 | 0.008001 |
| SAE-8k (x-only) | 0.4291 ± 0.0042 | 0.000016 | 0.000155 | **0.000513** | 0.002629 |
| random | 0.0887 ± 0.0018 | 0.000028 | 0.000043 | 0.000086 | 0.000950 |

Mann-Whitney U on pooled per-atom losses, vs masked-projector:
`SAE-8k z = −8.88, P(SAE > MP) = 0.238` → **significantly lower**. `random z = −12.05`.

### Why tick 15 got it backwards
The SAE's distribution is **heavy-tailed and low-median**: its typical atom (median 0.000016) is *more
replaceable than a random atom* (0.000028), while a handful are very irreplaceable (p90 0.000513,
above MP's 0.000387). On one seed those few atoms pulled the SAE's **mean** (0.000246) above the
masked projector's (0.000227), and its **max** to 0.0093. Across seeds the max swings 0.0026–0.0093.
The masked projector's atoms are far more *uniformly* irreplaceable.

**Both statistics I used in tick 15 were outlier-driven, on a single seed.** Third time a
max-over-sample number has misled me here (after the retracted "12×" and the retracted
"positional copier" prediction).

### Corrected verdict
**The spec's central premise is supported, modestly but robustly:**
- On the map's own objective, `R²(Wx)`: masked-projector 0.459 vs SAE 0.429 (Δ 0.030; seed sd 0.0003).
- On irreplaceability: median **8× higher** than the SAE's, z = −8.88. The SAE's median atom is more
  replaceable than a *random* direction.

Optimising for `Wx` does buy something over optimising for `x` — a basis whose atoms are more
uniformly load-bearing for the map's action.

### What does NOT change
- Tick 15's **other** finding stands and is untouched by this: **purity tracks the selection rule, not
  the dictionary** (a random dictionary's top-irreplaceable atoms reach purity 0.347, same as the
  masked projector's). "Irreplaceable atoms are interpretable" remains a statement about ranking.
- §1.3 free-ablatability and §1.5 contraction read-off remain retracted.
- Atoms-are-mechanisms remains weak in absolute terms (1.84× a matched-random basis, vs 34× for a
  known-true dictionary on the toy).

### Still open
The **purity** comparison (SAE 0.385 vs MP 0.347) was also single-seed, top-10, and therefore the same
class of statistic that just failed. It is **not** to be cited until replicated with seeds and a
distribution-level test. Queued.

### Purity, replicated: tick 15's SECOND claim also reverses

Same discipline (3 seeds, identical probes, distribution + rank test). `logs/purity_rep.log`.
Measured chance purity **0.070**.

**(a) Dictionary effect — purity over ALL probed atoms, pooled:**

| dictionary | median | mean | Mann-Whitney vs MP |
|---|---|---|---|
| **masked-projector** | **0.100** | **0.159** | — |
| SAE-8k | 0.050 | 0.145 | z = −6.16 → **lower** |
| random | 0.050 | 0.097 | z = −7.60 → **lower** |

**The dictionary does matter.** Masked-projector atoms are more token-pure than the SAE's, which beat
random. Tick 15's "SAE 0.385 vs MP 0.347" was a single-seed **top-10** statistic and inverts.

**(b) Selection effect — top-10-irreplaceable vs 10 random atoms, within each dictionary:**

| dictionary | top-10 | random-10 | ratio |
|---|---|---|---|
| masked-projector | 0.192 | 0.140 | 1.4× |
| **SAE-8k** | **0.477** | 0.110 | **4.3×** |
| random | 0.205 | 0.152 | 1.35× |

**The selection effect is NOT universal.** It is strong only for the SAE. Tick 15's claim that a random
dictionary's top-irreplaceable atoms reach purity 0.347 (≈ the method's) was one seed's noise: across
seeds it is 0.205 against its own 0.152 baseline.

⇒ **Tick 15's "purity tracks the selection rule, not the dictionary" is RETRACTED.** Both of that
tick's conclusions came from single-seed top-10 numbers, and both inverted under replication.

### Coherent picture across the two replicated experiments
- **masked-projector**: atoms *uniformly* load-bearing (irrepl median 0.000134) and *uniformly*
  mildly pure (purity median 0.100). Weak selection effect (1.4×).
- **SAE**: atoms mostly replaceable (irrepl median 0.000016, *below random*) and mostly impure
  (median 0.050), but with a **heavy tail** of atoms that are both highly irreplaceable (p90 0.000513)
  and highly pure (top-10 purity 0.477). Strong selection effect (4.3×).
- **random**: low on everything.

The two dictionaries are doing different things: the `Wx` objective spreads the map's action across
atoms; the `x` objective concentrates interpretable structure in a minority of atoms. **Neither is
strictly better** — which of the two you want depends on whether you need a faithful basis for the
map's action or a few crisp features.

### Standing rule, now enforced
**No single-seed, top-k, or max-over-sample statistic may be reported as a finding in this program.**
Medians, full distributions, rank tests, ≥3 seeds. Three claims have now been reversed by this rule
(the "12×", tick 15's irreplaceability verdict, tick 15's purity verdict).

---

## 2026-07-10 — The "atoms are weakly mechanism-like (1.84×)" headline collapses under the standing rule

That number was a **single-seed mean** — the statistic class that has now reversed three claims. Re-run
with 3 seeds, medians, rank tests, and the matched-R² control (`logs/matched_irrepl_rep.log`).

| dictionary | R²(`Wx`) | irrepl **median** | mean | vs MP |
|---|---|---|---|---|
| masked-projector, k=16 | 0.4591 | **0.000134** | 0.000223 | — |
| random, k=16 (equal sparsity) | 0.0887 | 0.000028 | 0.000043 | **4.71×**, MW z = −12.05 |
| random, k=96 (**matched R² 0.4649**) | 0.4649 | 0.000123 | 0.000147 | **1.09×**, MW z = −1.00 → **n.s.** |

**At matched reconstruction quality, the learned atoms are statistically indistinguishable from random
directions.** The toy's *true* dictionary passes exactly this control at **34×**. The masked projector
does not pass it at all.

### The control is not clean either — stated, not hidden
Matching R² forces the random dictionary to `k=96`: 6× denser codes, 6× higher per-atom usage, and
irreplaceability grows with usage. Three defensible views, all reported:

| comparison | MP vs random | toy true dict vs random |
|---|---|---|
| equal sparsity (k=16 vs k=16) | 4.71× (z = −12.05) | 8.4× |
| matched R² (k=16 vs k=96) | **1.09× (n.s.)** | **34×** |
| per unit usage (loss / usage) | 6.6× | 216× |

**Under every control the masked projector is one to two orders of magnitude short of a genuine
generator.** Whether it beats "random" at all depends on which quantity you hold fixed — and that
ambiguity is the honest result, not a number to be picked.

### What this does and does not overturn
- **Overturned:** "the method's atoms are weakly mechanism-like." At matched R² they are not
  distinguishable from random. The 1.84× was single-seed mean.
- **Not overturned:** the masked projector still beats the SAE on irreplaceability *median* at
  comparable R² (0.000134 vs 0.000016, z = −8.88, 3 seeds) and on purity median (0.100 vs 0.050). Those
  comparisons are between two dictionaries at similar R² and identical k, so they are not affected by
  the k=96 confound.
- **Not overturned:** the objective is learnable (R²(`Wx`) 0.459 ± 0.0003 vs random 0.089).

### Corrected bottom line
The `Wx` objective yields a basis that (i) reconstructs the map's action far better than random,
(ii) is better than an activation-only SAE on the map's action and on atom-level uniformity, and
(iii) whose atoms are **not** demonstrably mechanisms — at matched reconstruction quality they behave
like random directions. The method is a better *basis*, not a feature finder.

---

## 2026-07-10 — OPEN PROBLEM #1 RESOLVED: a usage- and R²-invariant criterion. The atoms are not mechanisms.

The matched-R² control was confounded by usage (random needed k=96 ⇒ 6× per-atom usage). Resolution:
per-atom usage is `k/K` **for any dictionary** (each datapoint selects exactly k atoms), so the
equal-`k` comparison is already usage-matched. That suggests a scale-free statistic:

> **uniqueness** = `irreplaceability_loss / (base_R² / K)` ∈ [0, 1]-ish
> = the fraction of an atom's "fair share" of explained variance that is **uniquely** its own
> (0 = fully compensable by other atoms; 1 = irreplaceable).

### The metric is NOT K/d-invariant — checked, not assumed
Random's uniqueness is 0.129 at K/d=4 (overcomplete ⇒ redundant) but 0.299 at K/d=0.25 (undercomplete
⇒ near-orthogonal). So the toy reference cannot be transferred blind. Re-validated at the **real**
proportions (K/d = 0.25, rank/d_in = 0.25 — exactly Pythia's 1024×4096, K=1024):

| setting | TRUE generator | random | ratio |
|---|---|---|---|
| K/d = 4.00 (overcomplete) | 0.0521 | 0.0363 | 1.44× |
| K/d = 1.00 | 0.7933 | 0.1860 | 4.26× |
| **K/d = 0.25 (real-like)** | **0.9558** | **0.1961** | **4.88×** |

The criterion has power at the setting that matters.

### Result on Pythia `down_proj` (3 seeds, medians, Mann-Whitney on pooled per-atom uniqueness)

| dictionary | median uniqueness | p90 | vs masked-projector |
|---|---|---|---|
| **masked-projector, k=16** | **0.2990** | 0.8630 | — |
| random, k=16 (equal sparsity) | 0.3280 | 0.9992 | z = +0.95 → **no significant difference** |
| random, k=96 (matched R²) | 0.2717 | 0.5520 | z = −1.12 → **no significant difference** |
| SAE-8k, k=16 | 0.0448 | 1.2086 | z = −8.59 → **significantly lower** |
| *(reference: TRUE generator at this K/d)* | *0.9558* | | |

**The masked projector's atoms are statistically indistinguishable from random directions on
uniqueness, under both controls.** A true generator sits at 0.956; MP sits at 0.299, which is where
random directions sit. Its higher raw irreplaceability (4.71× random at equal k) is fully explained by
its 5.2× larger base R² — per unit of explained variance, its atoms are no more unique than random ones.

The SAE is *more* redundant than random (0.0448): its decoder directions are correlated, which is why
its median atom is more replaceable than a random direction.

### Confound chain, now closed
- **usage**: matched by construction at equal k (`usage = k/K` for any dictionary).
- **base R²**: normalised out by the metric; also checked against a matched-R² random dictionary.
- **K/d**: matched (same K, same d); metric re-validated with a known generator at the real K/d.

### Consequence for the program
§2.5 is upgraded from "not demonstrably mechanisms" to a **clean negative**:
> On Pythia `down_proj`, the `Wx` objective produces atoms whose uniqueness is that of a random basis.
> It reconstructs the map's action far better than random and better than an activation-only SAE, but
> it does not carve that action into independent mechanisms.

**The method is a better basis. It is not a feature finder.** That statement now rests on a criterion
whose power was demonstrated at the exact K/d, rank ratio, sparsity, and seed count of the real
experiment — the strongest evidential footing anything in this program has had.

---

## 2026-07-10 — ⚠ CORRECTION: last tick's "clean negative" was regime-specific AND underpowered

Yesterday I wrote: *"the learned atoms have exactly the uniqueness of a random basis"* (§2.5),
measured at K=1024 (**K/d = 0.25**, undercomplete) with N_EVAL=800. Two problems, both mine.

### (1) A measurement failure that masqueraded as a result
Per-atom usage is `k/K`. At K=8192, k=16, N_EVAL=600, an atom fires **1.2 times** — most probed atoms
are never selected, so their loss is exactly zero. The K=8192 run returned medians of 0.0000 and
seed-to-seed swings of 0.053 / 0.002 / 0.000. **It measured dead atoms.** Its "ratio 0.04×" is void.
The same defect biased the whole first sweep (K=4096: 2.3 firings/atom; K=2048: 4.7).

### (2) Redone with adequate firings (N_EVAL=4000, probes restricted to atoms with ≥5 firings)

| K | K/d | firings/atom | % dead probes (MP / rnd) | MP uniq | rnd uniq | ratio | MW z |
|---|---|---|---|---|---|---|---|
| 1024 | 0.25 | 62.5 | 0.0 / 0.0 | 0.2597 | 0.2248 | 1.16 | −1.11 (n.s.) |
| 2048 | 0.50 | 31.2 | 0.0 / 0.2 | 0.3347 | 0.2193 | 1.53 | −1.81 (marginal) |
| **4096** | **1.00** | 15.6 | 4.3 / 11.9 | **0.2711** | 0.1519 | **1.79** | **−3.33 (significant)** |

**At K/d = 1 the masked projector's atoms ARE significantly more unique than random.** The usage
filter is conservative here: it removes random's dead (zero-loss) atoms, raising random's median and
*shrinking* the measured ratio.

Toy reference, recomputed: a TRUE generator scores **4.06× at K/d=1** and **6.02× at K/d=2**.
(My earlier claim that the true/random ratio *shrinks* with overcompleteness was also wrong — that
came from a toy with a different `W` shape.)

### Corrected §2.5
| claim | status |
|---|---|
| "MP atoms have the uniqueness of a random basis" | **retracted** — true only at K/d=0.25, and measured with 12.5 firings/atom |
| MP vs random uniqueness at K/d=0.25 | 1.16×, **not significant** |
| MP vs random uniqueness at K/d=1.0 | **1.79×, significant (z = −3.33)** |
| MP vs a true generator at K/d=1.0 | 1.79× vs **4.06×** — less than half |

**The honest statement:** the `Wx` objective produces atoms that are **modestly but significantly more
irreplaceable than a random basis once the dictionary is at least critically complete** — and still
less than half as unique as a known generator. Below critical completeness the advantage vanishes,
which is why it was invisible at K=1024.

### Lesson (added to the standing rule)
The rule "≥3 seeds, medians, rank tests" is not sufficient. **Also check that the statistic is being
computed on live data**: with sparse codes, `usage = k/N_atoms`, so eval-set size must scale with K or
the medians are taken over atoms that never fire. Report firings/atom and dead-probe fraction beside
any per-atom statistic.

*This is the fifth reversal in this program — and the first of a NEGATIVE result. The controls are not
biased toward pessimism; they are biased toward whatever the data says.*
