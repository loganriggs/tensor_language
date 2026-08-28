# Hourly strategic review: the shared-linear early-MLP knee is absent

Date: 2026-08-28 11:39 UTC

## Outcome first

The authoritative finite-horizon tangent run completed and independently passed its
artifact audit. It is a useful negative result:

> In the frozen 32-direction covariance-shaped coefficient gauge, the measured
> MLP0--2-to-final-output response does **not** have a stable compressed shared-linear
> interface across the two document-disjoint splits.

This is narrower than “the early MLPs have no simple state.” A context-conditioned
response bundle, a nonlinear finite-edit interface, an interface outside the 32 sampled
directions, and dimensions above the 16-probe per-context ceiling all remain possible.
The factorization \(H_c=D_cE_c\) also means final-output response alone cannot identify
whether context variation belongs to the encoder, the downstream decoder, or both.

The result artifact is
`tensor_bilin18_tangent_pilot_results.json`, SHA-256
`efd788fa0089008c4a2b0767244f1759453f02dd6e98b31aceae3847b26bc9d4`.
It binds program authority
`1dc6fa711803e6d7ac1c7958e8507fec66c8dab983c7562c605331ee46adaadd`,
geometry artifact
`5f8aeac18fef087b9217eedfde4fff254275e94f2b1b9716c03a3a1bcd5a40be`,
and geometry authority
`2b96c001db6053934dd1aa8f33a5cbbcac3e81b59b2525e089baf8e89e7f0e1b`.
All 24 batches and 96 rows completed; no raw logits, VJPs, writes, response matrices,
or live graph aliases were published.

## What the run measured

For context \(c\), source site \(i\), 16 full-vocabulary categorical-Fisher probes,
and 32 unit-RMS covariance-shaped write directions, let

\[
H_{c,i}\in\mathbb{R}^{16\times 32}
\]

be the derivative of the summed future log-probability scores with respect to the
registered MLP write edits. Stacking contexts and concatenating sites before a cut
produces the measured cut operator. Its singular spectrum is an optimal linear
compression curve by Eckart--Young:

\[
\min_{\operatorname{rank}(\widehat H)\le r}
\lVert H-\widehat H\rVert_F^2
=\sum_{j>r}\sigma_j(H)^2.
\]

Thus the experiment tested a causally weighted, finite-horizon interface rather than
local activation MSE.

## Exact result

| Cut | primary \(r_{95}\) | replication \(r_{95}\) | effective ranks | exposure-normalized trace difference | spectrum \(L^1\) |
|---|---:|---:|---:|---:|---:|
| MLP0 | 24/32 | 22/32 | 11.23 / 8.19 | 71.77% | 0.18785 |
| MLP0+1 | 27/64 | 26/64 | 17.63 / 13.50 | 60.92% | 0.16928 |
| MLP0+1+2 | 31/96 | 29/96 | 18.96 / 14.21 | 60.33% | 0.17641 |

No cut has the preregistered 95%-energy, gap-2 knee. All six stacked matrices have full
column rank. The only gap above two is mechanically after direction 32 at cut 2, not at
the 95%-energy rank, so it is not an admitted compression knee.

For 94 of 96 contexts, all three per-context cut matrices attain rank 16, the maximum
observable with 16 probes. The two exceptions have rank 8 and 9 and are the shortest
one-/two-output horizons. Therefore the registered experiment supports a local lower
bound of at least 16 on almost every measured context, but it cannot resolve 16 versus
32 or higher.

## The positive localization

Because the three source direction banks are separately normalized to unit write RMS
and the cut trace is additive across concatenated columns, its increment localizes
equal-RMS final Fisher sensitivity:

| source block | primary energy | replication energy | fraction of cut-3 energy |
|---|---:|---:|---:|
| MLP0 | 0.001306 | 0.002666 | 0.62% / 0.70% |
| MLP1 added | 0.200819 | 0.362559 | 94.90% / 95.44% |
| MLP2 added | 0.009482 | 0.014668 | 4.48% / 3.86% |

MLP1 is therefore the dominant early-write causal observation point for the next
assay. This does **not** establish that MLP1 contains more semantic information: the
response combines encoding, downstream amplification, suffix observability, and the
off-manifold effect of editing MLP1 independently of its native producer.

## Honest fraction explained

The tangent result narrows the hypothesis space but does not increase semantic or
causal coverage, so the project balance sheet stays:

| denominator | coverage | interpretation |
|---|---:|---|
| structural inventory | 36/36 sites | executable tensor ownership, not semantics |
| exact standalone ownership | 545,904,054 / 545,904,054 values | checkpoint-independent reconstruction |
| prospectively certified storage simplification | 29,196,288 values, 5.3481% | rank-640 shared-QK passed OOD CE and the frozen causal bank |
| named semantic behavior | 32.1% ± 6.4% | minority of measured behavior |
| named strict causal recovery | 10.923% | largest honest unexplained denominator |
| dense exact MLP storage untouched | 286,675,200 values, 52.51% | principal remaining compression target |

The largest gaps are the early producer/consumer interface, unexplained residual CE,
finite-edit behavior beyond tangents, interaction/composition failures across
RMSNorm/residual/attention, and validation that any simplicity currency buys useful
prediction, extraction, selective removal, OOD transport, certification, or runtime.

## Candidate pruning and priority

Candidates were judged by information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and redundancy.

1. **MLP1-only same-context split-probe response-bundle assay.** Use 8--16 documents,
   one row per document, common injection position, the frozen 32 MLP1 directions, and
   two independent 32-probe halves per context. This separates Fisher Monte Carlo noise
   from document variation and breaks the current rank-16 ceiling at lower cost than
   repeating the full three-site pilot. Compare fixed ranks 8/16/24 in physical write
   space by mapping coefficient frames through the nonorthogonal direction matrix.
2. **Conditional MLP1 physical compilation.** If the assay approaches rank 32 without
   a knee, stop seeking a small tangent state and directly factor the MLP1 write map
   conditional on its natural residual input. If stable local rank \(\le16\) exists,
   fit a small context-to-chart transport and test it on held-out documents.
3. **Finite replacement/consequence harness.** Any admitted MLP1 dictionary or
   conditional compiler must face CE, OOD transport, extraction, selective removal,
   collateral edits, and executable cost. Tangent fit alone cannot certify it.
4. **Preregistered Möbius interaction cube.** Intervene on MLP0 producer state, MLP1
   write, selected attention routing/value components, and later restoration to
   localize non-additive residual CE and explain why independently good substitutions
   fail to compose.
5. **Shared-dictionary factorization of later MLP consumers.** Compile MLP2/3 and then
   later MLPs jointly relative to whichever early interface survives. This directly
   attacks the 52.51% dense-MLP storage term without imposing independent gauges.

Token clustering, another local-MSE PCA, another attention rank sweep, and more
position-wise lexical refinement are pruned for this cycle: they do not distinguish
the remaining response hypotheses or address whole-model composition.

## Executed highest-priority safe action

The GPU is occupied by `ops/settled_frontier_restated.py`, so the useful CPU interval
was used to implement `finite_horizon_tangent_bundle.py` and nine tests. The analyzer
forms the context-normalized causal density

\[
D_c=\frac{H_c^\top H_c}{\lVert H_c\rVert_F^2},
\]

learns a primary-split intervention dictionary, measures held-out replication energy
capture, reports local-versus-pooled rank as a descriptive rotation gap, and returns
only scalars/eigenvalues. It rejects incomplete/zero/nonfinite inputs and is invariant
to orthogonal coefficient gauge and arbitrary per-context scale. The same module now
also implements the registered paired-probe discriminator: it maps right frames into
physical 1,152-dimensional write space as

\[
U_{c,r}=\operatorname{orth}(D_1^\top V_{c,r}),
\]

compares within-context independent-probe distance against cross-context distance at
fixed ranks, and supplies a deterministic document-paired bootstrap lower bound while
returning no frames or projectors. The focused test is 12/12; the full protected tangent
suite is 53/53.

This implementation is a sufficient-statistic primitive, not yet a positive result.
The red-team correction is incorporated into the next collection design: document
variation cannot be inferred until same-context independent probe halves establish the
noise floor, and physical projector claims must map through the frozen direction
matrix. No further GPU launch is authorized from the old result.

## Preserved failures and current blockers

The receipt-schema rejection, padded-vocabulary rejection, and unrelated-HEAD drift
rejection all published no outcome and remain recorded in `AGENT_BOARD.md`. The
padded-vocabulary authority/geometry pair is archived intact under
`failed_tangent_lifecycle_20260828_vocab_contract/`; its geometry bytes equal the
current geometry hash, but its old receipts remain versioned evidence.

There is no data, checkpoint, RSPD, FineWeb, or authority blocker. The only operational
serialization is that another safe GPU job currently owns about 5.5 GiB. CPU design,
tests, receipts, and consolidation remain unblocked.
