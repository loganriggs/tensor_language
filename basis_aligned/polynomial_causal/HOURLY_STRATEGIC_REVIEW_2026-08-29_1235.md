# Hourly strategic review — 2026-08-29 12:35 UTC

## Bottom line

The project has exact structural coverage of the model, but only a small fraction of
its behavior has a causal explanation.  The most valuable open experiment is now the
E4 terminal-copy selection run.  It asks a sharper question than local reconstruction:
whether one of eight small, named sets of attention heads has a repeatable and selective
effect on copy behavior when its native write is replaced by a fit-only mean write.

The selection lifecycle is implemented and its complete CPU assurance suite passes
`60/60`.  A fresh nonauthorizing draft binds 28 source files and the exact data, fit-bank,
adapter, checkpoint, intervention, batching, and bootstrap protocol.  Independent
outcome-blind re-audit is in progress.  The shared GPU is currently occupied by S1929's
matched-cost MLP-versus-attention table-rank sweep, so E4 has not opened its one-shot
authority or read any selection values/model outcomes.

## How much is actually explained?

These percentages answer different questions and should not be conflated:

- **Structural coverage: 36/36 sites.**  We can replace every attention/MLP site with
  an executable compiled component.  This says the interface is complete, not that we
  understand its semantics.
- **Certified parameter storage removed: 5.348245316%.**  This is literal storage that
  has passed the project's removal bookkeeping.
- **Named causal cross-entropy: 0.57968 / 5.30682 nat = 10.923302467%.**  This is the
  part of the model's predictive loss that currently has a named causal accounting.
- **Unexplained cross-entropy: 4.72714 nat = 89.076697533%.**
- **Terminal practical circuits: 0/68.**  None of the 68 possible terminal actions yet
  has a receipt-backed extraction/removal/transplant result.

Thus “the whole network is represented by our compiler” is true structurally, while
“the whole network is reverse engineered” is not.  The principal gap is behavioral
and causal, not tensor bookkeeping.

## What the completed entry points taught us

1. **Closed stream maps do not compose.**  A rank-512 map fitted on native streams is
   strong when given native inputs, but recursive deployment accumulates state drift.
   Its held-out deficit is roughly `1.09--1.27` nat, and refitting on its own generated
   streams is much worse (`5.50--5.62` nat).  This rules out treating native-stream
   compression as a standalone program.
2. **A single global low-rank output language is incomplete.**  Shared ranks 64/128 can
   beat equal-storage independent maps, so some sharing is real.  They still lose to
   same-rank independent maps, and the rank-512 shared/private hierarchy does not repay
   its price.  The evidence favors a shared trunk plus important site-private residuals,
   especially at tight budgets, not one universal semantic basis.
3. **The tested predictive-state interfaces do not close.**  The infinitesimal Fisher
   response panel had full registered rank and poor split stability.  The finite
   rank-64 gauge-transport triangle also failed destination sufficiency and unseen
   composition.  “Low local rank” is therefore not enough to define a downstream API.
4. **Local reconstruction can optimize the wrong decoder.**  Family F's refitted Down
   map improved local NRMSE but produced worse downstream KL than retaining native
   Down.  This is strong evidence that simplicity must be judged at downstream causal
   interfaces, not by local MSE alone.
5. **Storage should currently favor MLP tables.**  At exactly equal storage, table ranks
   attention/MLP = `128/384` improve CE over uniform `256/256` by
   `0.019/0.018/0.017` nat across the three discovery roles, while `384/128` is worse.
   Attention sites are behaviorally important, but their missing context cannot be
   bought back by a larger token table.  S1929 is measuring the optimum along this
   cost-flat one-dimensional family; its partial log is not yet evidence.
6. **A reliability statistic is useful only as a bucket.**  Its extreme quartiles
   separate CE by about one nat after controlling token frequency and at two coverages,
   but the middle quartiles invert reliably.  It may support thresholded selective
   compilation, not a monotone ranking or scalar confidence model.

## Largest remaining gaps and confusing results

- We have no receipt-backed terminal circuit, so we have not yet demonstrated that a
  compressed object enables extraction, selective removal, or OOD transport.
- MLP2 can compensate for upstream compression, but we do not yet possess a small joint
  program that explains that compensation or predicts when it fails.
- Behavioral localization and storage localization point in opposite directions:
  attention restoration changes behavior most, while MLP token tables benefit most
  from extra rank.  This is not a contradiction; storage rank cannot restore deleted
  attention context.
- Family F exposes an objective mismatch: local decoder fit and downstream consequence
  disagree.  A consequence-aware or native-Down decoder is now better motivated than
  another ordinary least-squares refit.
- The E3 failures show that a candidate state can be low-rank in one view yet fail to
  transport across suffixes or finite compositions.  We lack an interface that is both
  reachable from upstream and sufficient for downstream computation.

## Candidate actions, pruned and ranked

The ranking uses expected information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and redundancy with completed work.

1. **Execute the audited E4 terminal-copy selection transaction.**  This has the best
   causal and practical return: one run can produce the first named selective circuit
   or a clean negative for the entire eight-candidate mean-ablation family.  It is
   tightly falsifiable, bounded to 576 forwards, and directly unlocks or forbids final
   and OOD extraction/removal tests.  Current dependency: independent GO and a free GPU.
2. **Finish S1929's cost-flat allocation sweep.**  It is already running and cheaply
   measures whether the free `-0.018` nat MLP-heavy improvement has a stable optimum.
   It advances executable compression, though it is less semantic and causal than E4.
3. **Run a prospective finite native-Down Family-F successor.**  Hold the native Down
   decoder fixed and select/evaluate gate support by finite downstream consequence.
   This directly tests the surprising Family-F result that locally worse reconstruction
   can be behaviorally better.  It should be opened only after E4 because both require
   GPU behavioral measurements and E4 is closer to a practical circuit.
4. **Build a thresholded selective compiler plus exact hybrid telescope.**  Use the
   stable extreme reliability bucket to choose where compilation is permitted, and
   certify the sum/interactions of replacements with the exact 37-arm telescope.  This
   could turn a nonmonotone statistic into a real risk guarantee.  It is deferred until
   S1929 fixes the actual MLP-heavy build and must not treat the statistic as a rank.
5. **Test tight-budget shared-trunk/private-residual factorization.**  Flat sharing was
   useful only around global ranks 64/128, whereas the rank-512 hierarchy failed.
   A small preregistered budget study there is the only nonredundant remaining joint-RRR
   branch.  It ranks below the causal actions because it may improve compression without
   producing semantic or editable coordinates.

Pruned for now: more native-stream-map refits, another rank-512 shared hierarchy,
rank-64 linear gauge transport, direct-sum/HOSVD with the failed fixed projectors, and
SAE/dictionary fits scored only by weight or activation reconstruction.  Each duplicates
a decisive failure or lacks a downstream, compositional success test.

## Highest-priority action executed this hour

The E4 lifecycle was strengthened against the previous independent audit's failures:

- synthetic token banks must be disjoint across items and absent from the base row;
- the row receipt's cross-role-disjoint integrity gate is bound;
- all seven assurance-test files are part of the exact source closure;
- mocked valid-passer, valid-negative, partial-forward failure, protected-input mutation,
  lock replacement, and output-publication failure paths are exercised;
- failure artifacts hash-join all partial outputs and preserve the protected snapshot;
- the natural and synthetic schedules are checked as exactly `48 + 16` batches and
  `576` outer forwards.

The focused lifecycle suite passes `17/17`; the complete adapter/dispatcher/owner/
fit-parent/statistics/lifecycle suite passes `60/60` in 131.70 seconds.  The verified
nonauthorizing draft has SHA256
`ac98bf561e07ade0d48ee1440d094d24a983b67b594b1d842da2f8b42e4377ac`
and source-closure digest
`4de86968d0d2c1a899befc4d6a089045b4879a7f40c0cf74bf0b9afe495bbef4`.
It authorizes no execution.  The next safe state transition is independent GO, then
freezing the exact authority before any selection values or model outcomes are read.

## Immediate plan

1. Receive the independent audit of exact committed bytes and fix any remaining issue.
2. If and only if it returns GO, publish the canonical audit and freeze the one-shot
   E4 selection authority while the output namespace is still pristine.
3. Let S1929 finish; do not contend for the GPU or alter its queue/log.
4. Execute all E4 natural and synthetic batches without adaptive stopping.
5. Reload and recompute the result from per-document sufficient statistics, then publish
   exactly one passer or scientific-negative receipt last.  A failure spends authority
   and is not relabeled as a negative.
6. Open final/OOD roles only if the passer receipt licenses the single selected
   candidate.  Otherwise preserve the negative and move to the native-Down successor.
