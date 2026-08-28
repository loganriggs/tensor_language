# Hourly strategic review: freeze the MLP1 noise-versus-bundle discriminator

Date: 2026-08-28 11:51 UTC

## Outcome

The global explanation balance did not move this hour. The newly completed
position-wise frontier corrects a local design choice but cannot express contextual
transport; meanwhile the prior tangent result leaves a precise ambiguity at the
highest-value unexplained interface. The highest-priority safe action was therefore to
freeze an outcome-blind MLP1 same-context, independent-probe assay. Its CPU plan,
preregistration, and tests are now complete; GPU execution remains closed pending a
create-only collector and independent lifecycle audit.

Frozen plan fingerprint:
`236d83c6779b064e266a51594edaab2bf4c961006c4ab7905f0e946aa48e16c6`.
Serialized plan SHA-256:
`ff802543b9e3a7a7ddabc427679059c6404b83abdec49e1bf565b98ab878d518`.

## What changed since the previous review

The `settled_frontier_restated` discovery run took 1,424.8 seconds and falsified its
three substantive predictions. The corrected all-position context-free design point is
table rank 64, not rank 4:

| role | rank-64 all-position CE | stored reals |
|---|---:|---:|
| skip7000 | 6.17330 | 20.5309M |
| skip11000 | 6.15261 | 20.5309M |
| skip1200 | 6.14463 | 20.5309M |

Rank 4 had been selected on covered positions only and is worse than the fit-mean
all-tabled all-position baseline by 0.198/0.289/0.297 nat. The learned fallback is not
uniformly better: it loses on skip1200 at table ranks 256 and 16 by 0.00848 and 0.00408
nat. Its maximum benefit is 0.03006 nat at rank 64. This family is still a current-token
function with exactly zero cross-position dependence, so it neither changes the
whole-model frontier nor explains the missing contextual computation.

The completed `program_agreement_beyond_ce` assay addresses a real validation gap,
but it evaluates the **context-free position-wise table family**, not the admitted
rank-640 complete program. For its full-table arm, all-position top-1 agreement with
the live model is only 0.236/0.227/0.242 and program top-1 accuracy is
0.136/0.143/0.136 versus live 0.393/0.423/0.389. Its
$\mathrm{KL}(p_{\rm live}\|p_{\rm table})$ is 2.880/3.049/2.755 nats. Thus this
lexical family retains about one third of live top-1 accuracy while containing
exactly zero cross-position computation. A target-frequency audit sharpens the
failure: it retains 62.9--63.5% of live accuracy for targets seen at least 125 times
in the fit data, but only 2.7--6.2% on unseen targets. This is a frequent-target
lexical predictor, not a contextual model. An analogous agreement/KL assay for the
rank-640 complete program remains open.

## Explanation balance sheet

There remains no scientifically honest single completion percentage:

| denominator | explained | remaining gap |
|---|---:|---:|
| structural inventory | 36/36 sites | none at the ownership level |
| exact standalone executable ownership | 545,904,054 / 545,904,054 | semantics and minimality not implied |
| prospectively certified storage simplification | 29,196,288 values, 5.3481% | 94.6519% not certified removable |
| named semantic behavior | 32.1% ± 6.4% | majority unnamed |
| named strict causal recovery | 10.923% | 89.077%, or 4.72714 nat, unexplained |
| dense exact MLP storage | 0 / 286,675,200 simplified | 52.51% of whole-model storage |

The largest gap is therefore not token clustering. It is the contextual, cross-depth
producer/consumer interface centered on MLP1. Equal-RMS tangent edits localize
94.90--95.44% of early cut-3 Fisher response energy to MLP1, but the shared-linear
tangent knee failed and 94/96 contexts saturated the 16-probe rank ceiling.

## Missing interfaces and failure modes

1. **Noise versus context variation.** The previous primary/replication mismatch
   combines document differences with Monte-Carlo Fisher noise; it cannot yet establish
   a rotating response bundle.
2. **Encoder versus decoder gauge.** Final response factors as \(H_c=D_cE_c\). Varying
   right geometry does not identify whether the producer, downstream consumer, or both
   rotate.
3. **Finite-edit interface.** A stable tangent subspace would still not certify a
   finite replacement, selective removal, or collateral isolation.
4. **Composition.** Independently accurate module substitutions can interact through
   RMSNorm, residual scaling, attention, and later MLPs; no current compiler predicts
   those interaction terms.
5. **Distributional OOD fidelity.** Rank640 passed CE on two prospective roles and a
   16-intervention causal bank, but top-1/KL behavior and broader circuit transport are
   not yet certified.
6. **Residual CE allocation.** 4.72714 nat on the strict causal denominator has no
   composable circuit ownership.

## Candidate pruning and top-five priority

1. **Finish and audit the MLP1 same-context split-probe collector, then run it.** This
   directly resolves the probe-noise confound and rank ceiling at 256 backward passes,
   two thirds of the prior response collection. It is causal, cut-compatible,
   falsifiable, and targets the dominant early response site.
2. **Choose the MLP1 compiler conditional on that result.** If both halves approach
   rank 32, prune \(\le16\)-dimensional tangent compression and factor the physical
   MLP1 map conditional on its natural residual input. If stable local rank \(\le16\)
   with cross-context variation passes, fit and OOD-test a tiny context-to-chart
   transport. This avoids committing GPU time before the branch condition is known.
3. **Run an agreement/KL assay on rank640 and expand the consequence harness.** Require
   top-1 agreement, KL, OOD CE, extraction, selective removal, collateral effects, and
   executable cost. This validates whether storage simplicity buys behavioral control,
   not merely average loss.
4. **Measure a preregistered interaction cube around MLP0 producer state, MLP1 write,
   attention routing/value, and later restoration.** Möbius terms identify the missing
   non-additive interface and explain composition failures.
5. **Jointly factor downstream MLP consumers relative to the surviving interface.** A
   simultaneous tensor dictionary attacks the 52.51% dense-MLP storage block while
   avoiding arbitrary independent gauges.

More context-free table sweeps, token clustering, local-MSE PCA, isolated semantic
correlations, and another general attention rank sweep are pruned: the first four
cannot represent the missing sequence interface, and shared-QK attention already has a
prospective whole-program frontier.

## Executed action: frozen MLP1 paired-probe plan

The plan selects 16 documents by the smallest stateless hashes of
`2026082804:document_id`, then one row within each document by a second stateless hash.
The selected row indices are

`[3, 78, 7, 8, 31, 5, 11, 89, 16, 10, 76, 55, 35, 13, 50, 77]`.

The ledger binds both the full selected `[16,513]` tensor hash and the exact
`[16,256]` model-input tensor hash in addition to row/document identity and uniqueness.
Because these rows appeared in the parent tangent result, this is explicitly a
conditional historical-row follow-up rather than fresh-document confirmation. The
first 12 contexts in frozen hash order are the fixed promotion cohort; the last four
are diagnostic and cannot enter the promotion contrast.

Every context uses injection position 128 and exactly 128 future output positions. The
source is only MLP1, with the already frozen 32-by-1152 direction matrix. Two disjoint
32-probe halves have fingerprints
`6ac4ad53e5bbe5a1b4d5c522c68b562c3150f5759aa6234174f69a41d07d7a21`
and
`6122f89038c7be4c4da6326dd02087a47aa8664b3fadf408063e8958da3a617e`.

The registered outcomes are:

- both halves rank at least 24 and \(r_{95}>16\) in at least 75% of contexts: prune any
  local tangent-state story of dimension at most 16;
- numerical support at least 16, stable energy-plus-gap rank at most 16, and
  same-context **fixed-rank-16** physical projector distance at most 0.15 in every
  member of the fixed 12-document promotion cohort, with the same cohort's rank-16
  cross-minus-same bootstrap LCB at least 0.05: admit a context-varying **response
  bundle** candidate;
- otherwise: no admitted local bundle and no context-to-chart fit.

Physical frames use \(D_1^\top=QR\), \(\widetilde H=HR^{-1}\), and
\(U_{c,r}=QV_{c,r}(\widetilde H)\) at ranks 8/16/24. Raw logits, responses, frames,
and projectors remain forbidden. No branch
licenses an encoder-gauge claim or finite replacement. The plan tests pass 4/4 and the
full tangent suite passes 62/62.

## Operational state and blockers

The prior 1,424.8-second position-wise run and its canary have finished; the GPU is
currently free and the queue is empty. FineWeb, row provenance, checkpoint, CUDA,
rank640 program authority, and frozen site-1 geometry are present.

The exact blocker to GPU launch is scientific authority, not compute: no create-only
paired collector yet binds both plan fingerprints, replays the rank640 program and
parent site-1 geometry, prevents raw response escape, and publishes an aggregate-only
result under source closure. Implementation and independent audit of that collector are
the next safe steps.

Independent audit also caught a pre-launch mathematical defect: the first physical
projector implementation took an ordinary coefficient-space SVD before mapping through
the nonorthogonal direction bank. The actual bank has condition number 2.686, so this
was material. The analyzer now QR-whitens direction coordinates before SVD and has an
adversarial invariance test. No GPU outcome was opened; the failure is preserved on the
board rather than hidden.

Two further red-team cases are also closed. Same-context reproducibility and
cross-context variation are evaluated at the identical fixed promotive rank 16; a
fixed-rank summary is unevaluable unless both halves have numerical support at least
that rank. This prevents unstable tails or arbitrary zero-singular-vector completions
from manufacturing promotion. The two halves now bind a new literal 32-probe protocol
rather than reusing the previous protocol whose text specified 16 probes.

Selecting the 75% that happened to look stable would invalidate the bootstrap. The
promotion estimand is now the first 12 documents in the frozen hash order, and all 12
must pass. A 12-stable/4-heterogeneous no-free-rider regression confirms that the four
diagnostic contexts cannot supply the bundle signal. Row authority, parent-plan bytes
and semantics, subset/input hashes, serialized-artifact equality, and self-fingerprint
are now exact tests.
