# Block-3 shared native-gate subset v1: preregistration

Frozen before any block-3 fit statistic, selected gate, decoder, validation outcome, or
final outcome was computed.  This protocol supersedes typed RRR as the primary pilot;
typed RRR remains a nonpromotive linear-factor control because it retains all 4,608
bilinear products and all `Down` columns.

## Scientific question

Can attention 3 plus MLP3 be replaced by a much smaller executable bilinear program
that uses one shared subset of native product gates across all four exact polarized
paths, predicts untouched documents, and transports typed edits through the remaining
network?

The assay separates:

1. a locally accurate and downstream-faithful typed interface;
2. a locally inaccurate direction that downstream computation happens to cancel;
3. failure of this particular native-gate-subset grammar.

It does not claim that a failed native subset rules out learned CP/block-term gates,
context-conditioned mixtures, or other tensor programs.

## Exact object and grammar

At block 3 define

\[
u=\gamma(h+a)h,
\qquad
v=\gamma(h+a)a,
\qquad
u+v=\operatorname{RMSNorm}(h+a).
\]

After exact positive minimum-norm balancing of every product-gate scale, write the
four bias-free native terms as

\[
W_{pq}=D[(\widetilde Lp)\odot(\widetilde Rq)],
\qquad
(p,q)\in\{u,v\}^2.
\]

A candidate selects one gate set

\[
S\subset\{1,\ldots,4608\},
\qquad |S|=K,
\]

shared by all four paths and fits one shared decoder

\[
\widehat D_S\in\mathbb R^{1152\times K}.
\]

Thus

\[
\widehat W_{pq}=\widehat D_S[
 (\widetilde L_Sp)\odot(\widetilde R_Sq)].
\]

The deployed all-term program computes RMSNorm once, the selected left/right maps once,
exactly (K) products, one selected decoder, and the native bias.  It stores

\[
3\cdot1152\cdot K+1152
\]

floating values plus (K) integer indices, versus

\[
3\cdot1152\cdot4608+1152
\]

floating values and 4,608 products for native MLP3.  Candidate budgets are frozen to

\[
K\in\{256,512\}.
\]

All price reports include literal float bytes, index bytes, products/token, and linear
multiplies/token.  Partial-mask diagnostics may compute native terms but receive no
deployed-price credit.

## Candidate families

### A: activation-fitted shared subset

On fit documents only:

1. compute every native gate's four-path contribution-energy score;
2. retain the top 1,024 gates as a deterministic prefilter;
3. accumulate the exact feature Gram matrix and feature-to-four-write cross moment;
4. run deterministic batch simultaneous OMP in groups of 16;
5. take the nested first 256 and 512 selected gates;
6. fit one joint ridge decoder to all four typed writes, with ridge
   (10^{-6}\operatorname{tr}(G)/K).

This family tests an empirically used local typed interface.  Prefiltering is part of
the registered grammar and bounds any negative conclusion.

### F: finite-suffix consequence-fitted shared subset

Using the same fit documents, budgets, initialization, batching, and selected-gate
grammar, optimize continuous gate scores against detached native teacher logits after
frozen blocks 4--17, then deterministically retain the top (K) scores and refit the
joint decoder.  Fit labels/logits may never cross into validation or final roles.

F is necessary before concluding that a large local write error means no useful
compression: downstream computation might ignore or cancel it.  The optimizer, seed,
step count, learning-rate schedule, and discretization tie rule must be committed in an
implementation amendment before its first teacher-logit fit.  If F is not implemented,
the result may reject A but may not claim the grammar itself failed.

### Controls

- a matched random-gate (K) subset drawn without replacement from the same top-1,024
  prefilter with seed 2026082907;
- a fit-label permutation control at the same selected gates, using reversal of the
  stacked typed-write rows inside every fixed eight-row physical batch;
- exact (K=4608) polarization replay;
- full-write zeroing as the causal stake;
- typed RRR at matched factor-matrix storage as a nonpromotive control, explicitly
  charged for all 4,608 products and the full `Down` map.

## Data lifecycle

- fit: cached collision-separated FineWeb `n480_skip80`, 480 rows;
- validation: `n192_skip7000`, 192 rows;
- fixed final replication: `n192_skip11000`, 192 rows.

Only token positions 64--255 are used.  Counts are 92,160 fit positions and 36,864
positions in each evaluation role.  Bootstrap units are source documents, never tokens
or overlapping rows.  These roles are held out for this fit but have been used elsewhere
in the project, so success is not new-distribution OOD evidence.

Collection streams sufficient statistics and document aggregates.  It must not retain
the full 92,160 by 4,608 gate matrix or final logits.

## Physical typed action cube

Let (Q=\{uu,uv,vu,vv\}).  For all sixteen subsets (M\subseteq Q), run

\[
y_M=b+\sum_{q\notin M}W_q+\sum_{q\in M}\widehat W_q.
\]

Only (M=Q) is a deployable fully compressed MLP3 and must make zero native MLP3 calls.
The other fifteen arms diagnose which typed pathways are safely replaceable and must be
labelled original-factor-using.  Run the matched omission stake

\[
z_M=b+\sum_{q\notin M}W_q
\]

once for every nonempty mask.  Also run the mirror-error all-term arm

\[
y_{\mathrm{mirror}}=2y_{\mathrm{native}}-y_Q.
\]

Native attention 3 and blocks 4--17 remain live.  Call ledgers separately count teacher,
student, partial-mask, omission, mirror, and outer forwards.

## Measurements

### Local

- per-term and summed-write NRMSE;
- per-document q90 NRMSE;
- error norm after cuts 3, 4, 8, and 17;
- exact polarization replay error;
- real program price.

### Final consequence

- \(\mathrm{KL}(p_{native}\Vert p_{arm})\);
- centered-logit NRMSE and cosine;
- signed CE difference;
- top-1 agreement;
- document-bootstrap point, q05, q50, and q95.

For every nonempty typed mask, define causal recovery relative to its omission stake:

\[
\operatorname{recovery}(M)=1-
\frac{\mathrm{KL}(y_M,y_{native})}
     {\mathrm{KL}(z_M,y_{native})}.
\]

Any omission stake below 5% of the full-write-zero KL stake is declared behaviorally
null rather than silently removed.  Report complete-mask Walsh/Mobius interaction and
error closure; global low-degree energy cannot substitute for material-mask recovery.

## Frozen gates

A (K\le512) candidate is useful only if, on final replication:

1. summed-write NRMSE is at most 0.20 and every material term is at most 0.30;
2. all-term KL/full-zero-KL is at most 0.20 point and 0.35 at document-bootstrap q95;
3. the one-sided 95% upper bound on CE difference is at most 0.01 nat;
4. recovery is positive for every material typed mask;
5. it beats both matched random-gate and label-permutation controls;
6. the mirror arm's KL/full-zero-KL is at most 0.35 before the omitted direction may be
   called downstream-null rather than one-sided compensation.

Validation selects the smallest passing (K) separately for A and F.  At most one
candidate per family opens on final.  No rank, subset, decoder, materiality rule, or
threshold changes after validation.

## Interpretation

- Local and final gates plus material-mask recovery pass: a small empirically used,
  editable typed interface exists in this grammar.
- Local summed error fails, candidate and mirror pass, and error decays after block 3:
  downstream-null/cancellation only; behaviorally benign but not a composable block-3
  port.
- Term errors are large but their summed write is accurate: intra-block typed
  cancellation, not downstream cancellation.
- Candidate passes but mirror fails: one-sided downstream compensation, not a null
  direction.
- A fails but F is absent: activation fitting failed; no broader conclusion allowed.
- A and F both fail against controls at (K\le512): no useful compression only in the
  frozen prefiltered native-gate-subset grammar.

No result from this pilot alone earns OOD, semantic circuit, selective-removal,
whole-model storage, or whole-model causal credit.
