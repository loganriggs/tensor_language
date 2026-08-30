# Hourly strategic review — 2026-08-30 02:35 UTC

## Bottom line

The strict amount of the native model that is explained did **not** increase this
hour.  It remains:

- **29,196,288 / 545,904,054 = 5.348245316%** of stored values in certified simpler
  native-component programs;
- **0.57968 / 5.30682 = 10.923302467%** of deletion CE assigned to named causal
  mechanisms;
- **4.72714 nat = 89.076697533%** of deletion CE still unnamed;
- **0 / 68** terminal circuit actions that jointly pass extraction, selective removal,
  and OOD transport.

The important strategic change is negative but useful: the recently “converged”
compiled allocation was fitted to its evaluation window.  On 512 fresh rows (98,304
scored positions, zero overlap), it loses to the older S1959 program by **11.770
milli-nats**, despite winning by 3.064 milli-nats on the repeatedly inspected rows.
The loss is almost entirely the decision to restore larger MLP tables.  Every tested
untruncation depth reverses sign out of sample:

| late MLP tables restored | old-window gain | fresh-window effect |
|---|---:|---:|
| 16--17 | +0.962 mn | **-10.371 mn** |
| 14--17 | +2.132 mn | **-6.006 mn** |
| 12--17 | +2.768 mn | **-5.777 mn** |
| 10--17 | +3.300 mn | **-11.578 mn** |

Thus this was not the wrong cutoff on a good axis; the whole selected table-rank axis
was overfit.  The older S1959 choices largely do transport: attention rank 384 beats
256 by 16.561 mn (`t=38.96`), map rank 640 beats 512 by 0.696 mn (`t=6.45`), and blend
0.30 beats 0.10 by 34.213 mn (`t=27.03`).  Blend 0.30 versus 0.50 is only 1.321 mn at
`t=1.38`, so that particular ordering is not established on fresh rows.  This sharply
separates a result that was **selected on** a window from one that was **tested on** a
window.

## Largest remaining gaps

1. **No independently simpler MLP1 program has yet composed with simplified MLP0 and
   MLP2.**  The historical apparent rank-64 MLP1 correction secretly requires a
   60,553,728-real token/ridge producer and costs 3.812 times native MLP1 in total.
2. **MLP0's low-dimensional continuous code is still not semantically factored.**  We
   can predict substantial writes but cannot yet name sparse lexical/contextual atoms
   with stable downstream readers and edits.
3. **Local geometry does not predict finite suffix loss.**  The held-out MLP2
   Rayleigh/Fisher experiment had 45.27% tangent disagreement, Fisher/KL ratio 1.8268,
   and essentially no gain over local error.
4. **Sparse descriptions still retain dense native computation.**  The current MLP1
   proposal simplifies `Down`, but still evaluates all 4,608 native bilinear products.
5. **Fresh/OOD selection is now a first-class missing interface.**  Repeated paired
   significance on one row family cannot certify a small model-selection margin.

## Candidate actions considered and pruned

- **Another late-table cutoff:** pruned.  Four depths all reverse on fresh rows, and
  the ordering is not monotone.
- **Historical compiler-v2.1 rank-64 MLP1 in the early-layer cube:** pruned as a
  simplicity claim because its complete producer is larger than native MLP1.
- **Another scalar Rayleigh/Fisher weighting:** pruned because held-out finite effects
  were not predicted.
- **Raw HOSVD, matrix rank, or local MSE alone:** pruned because smooth full-rank tails
  and reconstruction scores have repeatedly failed to identify composable causal
  boundaries.
- **Promoting the shallow compiled map cut:** retained only as a compiler diagnostic.
  It transports out of sample but selectively taxes uncovered inputs and does little
  to explain the native early-layer program.

## Top five next actions

### 1. Finish the standalone sparse MLP1 `Down` FIT/SELECT test, then the early-layer cube

For the native MLP1 product vector $g\in\mathbb R^{4608}$, fit

$$
\widehat D_1(g)=c+A\,\operatorname{TopK}_{32}(Eg).
$$

`E` is a bank of 512 product detectors, only the 32 largest positive detector scores
execute per token, `A` maps those sparse coefficients into the 1,152-dimensional
residual write, and `c` is one constant bias.  This stores 2,950,272 reals versus
5,308,416 in native `Down`, saving 2,358,144 values, or 14.8% of complete MLP1.  It is
the highest-information action because it gives a real executable component and then
directly measures whether independently simpler MLP0, MLP1, and MLP2 programs compose.

Success is not output MSE alone.  On unseen SELECT documents the required gate is

$$
R_{CE}=\frac{CE_{zero}-CE_{sparse}}{CE_{zero}-CE_{native}}\ge 0.90,
$$

where `zero` removes only the bias-free MLP1 `Down` action.  If admitted, FINAL opens
once for all eight C512 × sparse-MLP1 × CONTINUE512 combinations and their pairwise and
three-way interaction terms.

**Expected information:** very high.  **Causal/compositional relevance:** direct.
**Falsifiability:** explicit 0.90 gate and factorial contrasts.  **GPU cost:** moderate.

### 2. Compute the oracle bound for sparse routed bilinear products

The Down-only program still pays for all 4,608 products.  Let an oracle choose a small
number of complete bilinear atoms per position and measure the best achievable CE and
composition frontier.  If even the oracle cannot pay for its routing/storage price,
flat, hierarchical, and DAG routers should all be pruned before training.  If it can,
the gap becomes a concrete system-identification problem rather than a vague SAE hope.

### 3. Build common downstream-consumer blocks

Estimate several signed downstream pullback forms
$G_c=\mathbb E[J_c^T W_c J_c]$ and find their common reducing subspaces (equivalently,
the nontrivial commutant shared by the consumers).  This is gauge-invariant and tests
the user's proposed joint writer/reader decomposition: a useful block should predict
unseen consumers and low cross-block edit interactions.  The planted 2/3/2 block toy
already recovers exactly; bilin18 remains unrun.

### 4. Verify several distinct late causal endpoints

Use capitalization, induction/copy, lexical/BPE, and question-like endpoints only after
specificity controls.  They provide multiple independent equations for what early MLP
atoms write and downstream components read.  This is higher value than naming atoms
from nearest tokens alone, but lower than action 1 because the endpoint bank is not yet
certified.

### 5. Repair the executable compiled fallback under a fresh-window rule

Keep S1959 as the build of record; certify each proposed parameter change on rows that
did not select it and report covered, uncovered, and unseen-target groups separately.
This can improve executable compression quickly, but it is fifth because it is a
surrogate compiler rather than a mechanistic decomposition of the native network.

## Action executed this hour

The standalone MLP1 path moved from an unrun idea to a frozen experiment boundary:

- corrected capture to the preregistered positions 64--255;
- enforced exact native/replacement/attention call censuses;
- added finite CE and positive-deletion-denominator gates;
- bound FIT/SELECT tensors by dtype, shape, file hash, tensor-content hash, provenance,
  and document disjointness;
- added receipt-last source/audit/row/checkpoint transaction replay, including
  adversarial rival-terminal tests;
- expanded the source closure through `jacclust/tt_model.py`;
- passed **34/34** focused tests and an independent outcome-blind audit;
- froze **96 FIT + 96 SELECT + 96 FINAL** distinct FineWeb documents in **25.0 s**,
  with all eight registry/full-row/prefix/document disjointness checks true and no model
  or training access.  FINAL remains sealed.

The receipt is
`basis_aligned/bilinear_quotient/mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json`.

## Exact launch blocker and continuation

A concurrent, source-equivalent S2041 commit landed between the audit and row-freezer
snapshot.  Consequently the immutable receipt records source commit `236ae134`, while
its exact audit records `15ed37b9`; all 21 audited source hashes are identical.  The
current FIT runner deliberately requires those commit labels to be equal, so launching
it now would fail before data/model access and permanently spend the namespace.

Do **not** weaken or edit the receipt and do not launch the known-failing v1 runner.
The next safe action is a source-closed v2 lineage-only recovery that proves both commits
resolve to the same 21 hashes, binds the v1 receipt/audit bytes, changes no scientific
setting, and receives a fresh outcome-blind audit.  Then FIT/SELECT may run.  This is a
precise infrastructure blocker, not a data, model, or GPU blocker; the row evidence and
failed audit attempts are preserved.
