# Hourly strategic review — 2026-08-29 06:30 UTC

## The update in one paragraph

The universal-output-basis experiment is finished, and it gives a useful mixed
answer.  The 36 context-free maps do reuse output directions: at global ranks 64 and
128, sharing one output basis beats the strongest set of independent maps that can be
stored for exactly the same price by 0.022--0.036 nat on every document role.  But a
single shared basis discards important site-specific directions: it loses to 36
independent maps at the same rank by 0.038--0.070 nat, and at rank 512 it loses both
comparisons.  Splitting the basis into attention and MLP types barely helps.  The next
factorization is therefore a **shared trunk plus site-private residual directions**,
whose exact CPU mathematics and endpoint controls now pass 38 combined tests.  In
parallel, the finite-composition experiment's replacement cache is now real: it holds
96 basis, 96 fit, and 192 evaluation documents with 384/384 distinct source documents
and a receipt.  This removes the row-independence blocker, although the triangle
runner still needs source-closure and control hardening before it may run.

## What the completed shared-basis computation was

For site \(j\), fit data define a positive-semidefinite output-merit matrix

\[
M_j=C_j^\top(G_j+\lambda I)^{-1}C_j.
\]

Here \(G_j=X_j^\top X_j\) describes which input directions occur, and
\(C_j=X_j^\top Y_j\) connects those inputs to the site's true residual-stream write.
The top eigendirections of \(\sum_j M_j\) are output directions that jointly explain
the most regularized fit signal across all 36 sites.  The experiment stored that one
output dictionary once and stored one input map per site.  It then replaced all 36
writes and measured whole-model cross-entropy (CE), not merely local squared error.

At rank 512, the shared maps cost 21,823,488 floats versus 42,467,328 floats for 36
independent rank-512 maps: a 48.61% map-storage reduction.  Once the common token table
is included, the complete-program saving is 7.73% at this operating point.  This is a
price measurement, not a removal certificate, because the CE conditions failed.

The exact-price result matters more than a visually low-rank spectrum.  At ranks 64
and 128, the shared basis's CE advantage over an exactly equal-price independent
allocation proves that recurring directions are genuinely useful when storage is
tight.  The same-rank loss proves that those recurring directions are not the whole
language.  A two-type attention/MLP split at equal storage improved CE by only
0.00250, 0.00237, and 0.00004 nat on the three roles, below the registered 0.01-nat
bar.  Architectural type is therefore not the missing partition.

## The new hierarchical object

Fix a shared projector \(P=V_0V_0^\top\), and let \(Q=I-P\).  For every site, take
private output directions from the leading eigenvectors of

\[
Q M_j Q.
\]

These are precisely the valuable directions that remain after removing the shared
trunk.  The deployed site map is

\[
\widehat Y_j
=X_jA_j^{(0)}V_0^\top+X_jA_j^{(p)}U_j^\top,
\qquad V_0^\top U_j=0.
\]

The first term speaks a shared output language; the second is a site-private residual
language.  If the total number of stored floats is fixed, every private rank slot has
the same literal storage cost.  We therefore allocate slots by the largest fit-only
residual eigenvalues across all sites.  No held-out CE is allowed to choose the ranks.

This construction has two essential controls:

- shared rank zero is exactly the strongest independent exact-price allocation;
- zero private slots is exactly the already measured global shared map.

An interior point can therefore win only if combining reuse with specialization is
better than either endpoint in held-out CE.  The primary registered criterion is at least 0.01 nat
better than both endpoints on each of the three held-out document roles, at identical
map storage.  The pure CPU core, price accounting, allocation, orthogonality, and both
endpoint identities are implemented.

The first independent math audit nevertheless returned **NO-GO for a real launch**.
It caught four repairable specification problems: the draft falsely claimed an
independent grammar must maximize local merit even though sharing can buy more total
site-directions; it computed factors in float64 while pricing them as deployed
float32; it checked symmetry but not positive semidefiniteness for every merit matrix;
and it overstated identifiability at tied eigenvalues.  The real runner must also
freeze casts, calls, controls, and receipt replay.  Those corrections are in progress.
This is not yet a model result, and no hierarchical authority may open before a
second independent GO.

## How much of the model is explained

The strict fractions did not rise during this hour:

| Currency | Explained | Remaining gap |
|---|---:|---:|
| Structural write interfaces captured and replaceable | 36/36 | Semantics and autonomous upstream state are not thereby explained |
| Original storage with a whole-program consequence certificate for removal | 5.3481% | 94.6519% lacks that certificate |
| Strict named causal CE headroom recovered | 10.923% | 4.72714 nat, or 89.077%, remains unnamed |
| Terminal extraction/removal/OOD actions completed | 0/68 | All 68 behavior-by-causal-path cells remain open |

The shared-basis result improves our model class and prunes a wrong one, but discovery
CE on three exposed roles cannot legitimately increase these strict ledgers.

## Largest remaining gaps and confusing results

1. **Autonomous state remains the largest missing interface.**  Maps supplied with
   native one-token residual streams can be good, while the same maps supplied with
   recursively compiled streams lose about 1.09--1.27 nat.  Refitting on those closed
   streams was much worse.  We still lack a compact state variable that composes
   through the residual/RMSNorm interfaces.
2. **Local reconstruction and downstream behavior disagree.**  In MLP3 Family F,
   refitting Down improved local write NRMSE but harmed downstream KL.  Native Down had
   worse NRMSE and better KL.  This may be real downstream null-space compensation, or
   a fit-distribution accident; the prospective fresh-document, two-sided edit test
   has not run.
3. **Sharing is real but not flat.**  One basis is useful only at tight rank, and the
   obvious attention/MLP hierarchy is too weak.  The unresolved question is whether a
   shared trunk plus unequal private suffixes closes CE efficiently or merely
   interpolates smoothly between two losing endpoints.
4. **Finite predictive state has no outcome yet.**  The new 384-document cache fixes
   pseudoreplication.  It does not itself show that a learned L8-to-L11 map and an
   L11-to-L14 map compose on sealed finite interventions; the runner remains closed
   until its lifecycle and null controls are hardened.
5. **Behavioral usefulness is still untested at the end of the pipeline.**  No one of
   the 68 terminal cells has jointly passed extraction, selective removal, collateral
   damage, and OOD transport.

## Candidate actions considered, then pruned

- **More ranks in one universal basis:** pruned.  Rank 512 already loses the
  equal-price comparator; extra points do not test a new structure.
- **Rotate the failed universal projector with an SAE/dictionary objective:** pruned
  for now.  Rotation can improve coordinate sparsity but cannot restore discarded
  private directions.
- **Use attention versus MLP as the hierarchy:** pruned as the main split because its
  exact-price gain was at most 0.00250 nat and nearly zero on one role.
- **Repeat local MLP3 decoder refits:** pruned.  The existing result already shows that
  optimizing this local metric can move downstream behavior in the wrong direction.
- **Run the old finite triangle on duplicated chunks:** permanently pruned.  It would
  overstate sample size.  Only the new 384-distinct-document receipt is admissible.
- **Name individual eigenvectors now:** deferred.  Eigenvector columns are gauge
  dependent; only projectors and deployed coefficient maps are identified until a
  stable sparse or causal coordinate test succeeds.

## Revised top five

1. **Run the hierarchical shared-plus-private whole-model CE test.**  It follows
   directly from the measured E2 pattern, has exact-price endpoint controls, tests a
   composable replacement of all 36 writes, and costs roughly one discovery GPU pass.
   A failed interior sweep would sharply prune hierarchical output dictionaries.
2. **Run the prospective native-Down behavioral-port test.**  This is the cheapest
   way to decide whether the MLP3 KL/NRMSE reversal is transferable causal structure
   or compensation on exposed data.  It directly measures ordinary substitution and
   finite edits on fresh documents.
3. **Source-close and run the finite transport triangle.**  The row blocker is now
   removed.  A pass would supply the first state defined by its ability to predict an
   unseen finite composition rather than reconstruct a local activation; a failure
   would close a major alternative entry point.
4. **Close one behavior-anchored terminal circuit.**  Capitalization, number format,
   or copying should be screened near the output, then tested for extraction,
   selective removal, collateral CE, and OOD transfer.  This is the necessary external
   validator for every proposed simplicity metric.
5. **Conditionally fit a multilevel sparse dictionary.**  Only if the hierarchical
   CE test succeeds, seek global, group, and site coordinates with an MDL price and
   stable causal supports.  This is the point at which SAE/dictionary learning can add
   semantic/editable structure without pretending a failed flat projector is enough.

## Safe actions executed this hour

- Completed and receipt-bound the real shared-output RRR sweep; E2.1 and E2.2 failed,
  E2.3 was prospectively pruned, and the shared-plus-private successor was registered.
- Implemented the first hierarchical CPU core and exact endpoint controls; the
  combined relevant suites pass 38 tests.  Independent audit then found the launch
  blockers described above.  The core/specification and real source-closed runner are
  being repaired, and neither code nor runner is counted as an outcome.
- Froze and materialized 384 distinct FineWeb documents for the finite triangle:
  rows SHA256 `102b79726b7132a6438b4080272fee1774499ac4fc83c4aa025fa86439b4074d`,
  receipt-file SHA256
  `3f92d8b3aa5e89e6059a010338521bffa0cf440e0815d9d67e1b65aa58a8e102`.
  The receipt explicitly does **not** authorize the triangle runner; source-closure and
  controls are being audited separately.
- Preserved the earlier v1 row-materialization hash-currency failure rather than
  overwriting it.  The v2 recovery changes only raw-byte versus composite hash use.
- Checked running work: the shared GPU lane is owned by the independent quotient
  queue; no competing Codex GPU experiment was launched.  CPU implementation and
  evidence consolidation continued during that interval.

## Update at 07:08 UTC — hierarchy v1 terminal failure and exact v2 recovery

After independent GO, hierarchy v1 ran all seven arms in 398.48 seconds with a peak
allocated CUDA footprint of 4,217,080,320 bytes.  It wrote a result and then failed
the mandatory immediate JSON-reload equality check before receipt publication.  V1
therefore has authority, result, and terminal failure but no receipt, and none of its
numerical values is a scientific outcome.

The cause is fully localized: `dataclasses.asdict` retained two tuple-valued price
fields, `private_ranks` and `dense_multiplies_by_site`, in every arm.  JSON reload
converted them to lists.  Those are exactly 14 container-type mismatches—two paths in
each of seven arms—with no value mismatch anywhere else.  V1 result SHA256 is
`86315dcc855e9a27958b6abfd50ed5c6b7bb7108f00fe3684bfbf624405a772d`;
failure SHA256 is
`054db06c03525b3f78eefdd9ed8e0fa3daf3868175460c76a95e39b875ebc35c`.

A fresh v2 recovery is now source-closed and independently audited.  Its sole change
is to normalize each unchanged program diagnostic through an idempotent JSON round
trip before result assembly.  It reruns all rows and all seven arms and binds the exact
v1 authority/result/failure plus receipt absence at every terminal boundary.  The
applicable audit suite passes 101 tests.  V2 has not opened its authority because the
shared GPU is presently occupied; it is next when the device releases.
