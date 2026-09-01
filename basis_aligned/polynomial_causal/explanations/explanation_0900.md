# Plain-English update — 2026-09-01 09:00 UTC

**Our goal.** We are trying to turn the 546-million-parameter bilin18 checkpoint into a smaller executable tensor
program that still predicts fresh text, composes when several pieces are replaced together, and responds correctly
to named causal interventions. A compact fit is not enough: every router, factor, index, and stored dtype must be
counted. Generic token-dependent top-k is therefore a compute trick, not a tensor-network decomposition; a small
finite MoE router counts only when its states and fixed expert subsets are explicit and priced.

**What we achieved overnight.** The best fully gated program now has **511,758,646 semantic scalars and
1,023,517,292 stored bytes**. It adds `.012329` cross entropy above the original model, retains 43 of 62 circuit
certificates, transfers to held-out WikiText, and matches the original model's signed a16 intervention with cosine
`.986524` and collateral Spearman `.995306`. Relative to native, that is 34,144,256 fewer semantic scalars (6.25%)
and 1,044,152,320 fewer bytes (50.50%). Computation is still fp32 after dequantization, so this is not yet a speed
claim. A high-fidelity source-aware BF16 anchor uses 1.0918GB, changes census CE by only `+.000009`, and retains all
62 certificates.

**The important structural result.** Raw MLP bilinear factors do not expose stable reusable atoms, blocks, a tree,
or a DAG. The correct object is the basis-invariant contraction tensor, and the correct geometry is the covariance
of inputs that the layer actually receives. Under that contextual metric, every one of the 18 MLP input maps has a
good low-rank description. A rule written before the physical build selected mild p768 cuts at layers 4 and 0 and
combined them with Q/K rank64. That one-shot design beat the prior frontier on scalar count, CE, and signed causal
agreement.

For MLP0 specifically, folding in the full embedding is exact for the one-token/position-zero regime: every token
input is known, so the quadratic bilinear function can be evaluated over the whole vocabulary without sampling.
Planted blocks are recoverable when the correct prior is supplied. But incompatible priors fit nearly the same toy
function, and the real contraction algebra has no robust nontrivial reducing blocks. A hierarchy would require a
nested family of invariant subspaces; a DAG additionally needs an asymmetric intervention or time relation, because
a single symmetric bilinear map cannot orient an arrow. This is why more factor optimization cannot identify a real
DAG from the weights alone.

**Finite routers versus top-k.** Four-state token and live-context MoE routers were roughly 10x and 25x worse than
one equal-price shared subspace. A behavior-named morphology router stopped before performance scoring because its
digit state had only 356/206 examples across the two fit halves, below the frozen 300-per-half support rule. We did
not merge states or lower the bar. The evidence says MLP0 is primarily one shared contextual spectral object, not a
small partition of token or context states.

**Why the search is now mathematical.** Omitted context-metric singular-tail energy predicts physical CE damage
with log-space R2 around `.995`. A 62-dimensional vector of circuit damage is almost a single ray: the original
Q/K fit has R2 `.99945`. Without refitting it, that ray predicts the three new distributed-MLP programs with cosine
`.9988–.9994`, vector R2 `.9917–.9959`, and certificate-count error only 1–2. This explains the discrete frontier:
two MLP cuts retain 43 certificates, while adding the next selected layer reaches a smaller 509.10M program but
drops to 38. CE and OOD alone would have promoted it; the causal battery correctly rejects it.

The reverse trade gives a useful middle tier. Q/K rank72 with the same two MLP cuts has 516,264,246 scalars,
`.009227` census damage, and 50/62 certificates. Its shifted WikiText mean/p95/max is
`.002260/.024200/.036493`. The final two-byte rebuild and signed gate also pass: **1,032,528,492 bytes**, census
`+.009301`, signed cosine/error/norm `.990600/.153655/1.060392`, and collateral Spearman `.996633`. Its rebuilt
receipt uses corrected rung/status/claim labels. The fully gated fidelity dial is therefore 62 certificates at
1.0918GB, 50 at 1.0325GB, and 43 at 1.0235GB.

**A final alternative failed informatively.** We computed exact contextual tail curves for every MLP layer at
ranks 512/640/768/896, fixed the independently learned tail exponent, calibrated layer gains on fit A, and predicted
fit B. The model is excellent: median multiplicative error `1.223`, Spearman `.952`, and correct rank ordering at
all 18 layers. But water-filling at the current price chooses exactly the existing `{0,4}@p768` pair. Spreading the
same saving over more p896 cuts is not better, and the next cheaper allocation predicts 1.369x the damage. This
closes both manual layer-prefix and variable-rank variants around the current frontier.

One last vector-valued alternative found a real but non-specific second mode. After subtracting the universal ray,
three MLP-bearing programs share a residual direction (rank-one R2 `.935`), and that direction predicts a held-out
program well enough to improve full-vector R2 `.99316 -> .99910` and certificate count `49 -> 50` (actual 50).
However, the value-family control aligns at `.526`, just above the frozen `.50` specificity limit. We therefore map
this as curvature of the universal damage path rather than claim an MLP-specific causal axis or tune another mode.

**How the six independent directions rank now.** The winning route is the executable error-contract machinery,
because it became a reliable rank-and-certificate calculator. Second is contextual bilinear MLP input compression,
which supplied the actual new frontier. Joint input/output vocabulary geometry is real but remains too damaging,
including after sparse rare-row and higher-rank repairs. Causal-response eigenvectors lose to ordinary activation
PCA. Predictive finite-state/Hankel structure remains a circuit classifier, not a compiler. Exact embedding-folded
MLP0 structure supplied strong identifiability limits and redirected us to the contextual spectral object, but did
not itself earn a block/tree/DAG program.

**Different paths worth pursuing next.** Further progress must change the object rather than tune this local grid:

1. Preserve a vector-valued suffix-Jacobian or several named intervention responses, instead of another scalar
   consequence weighting of the same covariance.
2. Fit a genuinely new joint CP/tensor-train factorization under the live activation metric; native atom reuse and
   cross-layer Grassmann sharing are already negative controls.
3. Keep value ranks64–112 closed: the exact tail screen is now done, and even the most optimistic rank104 estimate
   costs `3.6838x` the adopted MLP CE per saved scalar (value96 costs `4.0898x`). A new value representation—not
   another rank in this grid—would be required to reopen the family.
4. Reopen finite-state structure only when an independently named behavior supplies stable supported states and a
   causal consumer. Do not cluster first and name the states afterward.
5. Once semantic structure moves again, implement fused BF16/FP16 execution and measure load time, peak memory, and
   latency separately from storage bytes.

The main lesson is pleasantly concrete: the checkpoint is not yielding a clean symbolic module graph in raw MLP0
weights, but it is yielding a metric-dependent spectral compiler with surprisingly regular error and causal laws.
That program is already predictive, composable, manipulable, and literally smaller—and the local alternatives
around its current frontier are now closed by prospective tests rather than by taste.

The exact price formulas, controls, and kill criteria for the five representation-changing directions above are in
`NEXT_REPRESENTATION_DIRECTIONS_2026-09-01.md` one directory above this explanation.
