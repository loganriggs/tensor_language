# Hourly strategic review — 2026-08-29 02:46 UTC

## Update since the previous review

The previous highest-priority action is closed.  Block-3 family A was evaluated on
held-out validation with autonomous Blocks 4--17, exact physical call accounting, and
source-document bootstrap.  It failed as an all-term MLP3 replacement at both K=256
and K=512.  This is not “no structure”: selected K=512 beats matched random and every
typed singleton replacement has positive causal recovery.  The failure is specifically
**composition under the nonlinear suffix**.

The registered branch is now family F: select the same finite native product gates by
their downstream consequences rather than stacked local MSE.  In this review, family F
was prospectively specified and its gauge-fixed mathematical primitives were
implemented and tested before any family-F teacher logit was opened.

## Honest fraction explained

| Ledger | Current credit | Remaining gap |
|---|---:|---:|
| Structural module inventory | 36/36 | structural representation is not semantic explanation |
| Whole-program storage consequence-certified removable | 5.3481% | 94.6519% lacks that certificate |
| Older named behavior | 32.1% ± 6.4% | not additive with the other ledgers |
| Strict named causal CE recovery | 10.923% | 4.72714 nats / 89.077% remains unnamed |
| Final extraction/removal/OOD actions | 0/68 | entire final action ledger remains open |

No global ledger moves.  Family A's validation is a negative compression decision plus
a positive pathway diagnostic, not whole-model credit.

## Largest gaps and confusing results

1. **A good local tensor approximation is not a composable interface.**  K=512 local
   NRMSE is 0.684 and final KL/bias ratio is 0.712, yet individual typed replacements
   recover 44--85% of their omission consequences.  Four acceptable partial actions
   interact badly when performed together.
2. **The suffix suppresses and then re-amplifies error.**  K=512 state NRMSE falls from
   0.429 at cut 3 to 0.128 at cut 8, then rises to 0.164 at cut 17; its normalized
   cut17/cut3 error is 1.061.  A middle-cut probe would falsely report repair.
3. **Error sign matters.**  K=512 candidate KL/bias is 0.712, while the equal-magnitude
   mirror is 0.198.  Symmetric local MSE cannot model this asymmetry.
4. **Top-1 extraction and distributional faithfulness diverge.**  K=512 agrees with the
   native top token 82.27% of the time while adding 0.0914 CE nat and retaining 71.2%
   of the all-term omission KL.  It might extract many decisions, but it is not a
   faithful or safely removable replacement.
5. **The context-free fallback object was mislabeled.**  The parallel program's
   uncovered path is the linear embedding map alone; a computed nearest-neighbor row is
   discarded.  Increasing map rank 64→128 is a cheap real gain, but most residual loss
   remains at rank 512, so functional form—not merely rank—is now the open question.
6. **The 68-action semantic interface remains absent.**  We still cannot use a common
   terminal ledger to decide whether a simpler representation buys extraction,
   selective removal, or OOD transport.

## Candidate actions considered

### 1. Complete the consequence-fitted native-gate family F

This has the highest expected information gain because it tests the exact hypothesis
raised by family A's failure: downstream consequence, not local write energy, defines
the useful quotient.  It preserves the finite K-product executable grammar, balanced
gauge, literal cost, matched controls, and autonomous suffix.  Failure is sharp: if
consequence-selected gates do not beat calibrated random/permuted controls on the
already registered validation metrics, the native-subset grammar has little remaining
room at K<=512.

The prospective design uses a fixed native decoder during continuous gate scoring, so
gate scores cannot be hidden by reciprocal decoder scaling.  Scores live on the capped
simplex `0<=s<=1, sum(s)=512`; top-256/top-512 supports are nested, then the common
four-term decoder is refit.  A scalar plus constant-vector calibration is included and
folds into the already stored decoder/bias at zero extra executable cost.

### 2. Close the 68-action extraction/removal/OOD scorer

This is the most important whole-project interface.  It turns simplicity definitions
into predictions: a metric is useful only if lower complexity at matched faithfulness
improves action extraction, selective removal, OOD transport, or certificate tightness.
The scorer exists in pieces but has not produced final actions.

### 3. Measure MLP0/MLP1/MLP2 replacement composition factorially

Run each replacement, every pair, and the triple on identical rows.  The interaction
residual

$$
\Delta_{ij}=E_{ij}-E_i-E_j
$$

measures whether apparent downstream compensation provides a reusable interface or
only hides isolated error.  Block 3 shows that singleton success cannot be extrapolated
to joint success, raising this priority.

### 4. Replace the uncovered-token linear map with a nonlinear but priced program

Parallel work establishes map rank 64→128 as unusually cost-effective, but 73% of the
fallback-specific loss survives at rank 512 and the nearest-neighbor output is not
actually consumed.  The next nonredundant candidate is a small nonlinear/shared lexical
map, priced against the rank-128 baseline and checked to leave covered rows bit-identical.
This remains Claude's active branch, so no duplicate GPU job is launched here.

### 5. Fit a consumer-aligned sparse/hierarchical dictionary

A weight SAE or DAG becomes meaningful only if the same atoms sparsely parameterize an
upstream write and the downstream reads that consume it.  Optimize joint description
length plus causal action loss, and compare with rotated/equal-rank controls.  This is
lower priority until family F and the 68-action scorer define the consumers and utility.

## Pruned moves

- **Increase family-A K or collect more validation rows:** pruned.  K=512 is factors of
  3--9 outside the registered gates with narrow document intervals; this is objective
  mismatch, not sampling uncertainty.
- **Complete family A's 16/15 cube:** forbidden by its registered failed branch.
- **More coefficient HOSVD or norm minimization:** pruned for now.  Gauge balancing is
  already exact; another local factorization cannot explain suffix sign asymmetry.
- **Weight-only SAE:** pruned until downstream sparsity/action benefit is part of the
  objective.  Basis-dependent sparsity alone cannot compose.
- **Map-rank beyond 128 as the primary fallback move:** deprioritized by diminishing
  returns and the demonstrated functional-form floor.
- **Middle-cut error decay as a certificate:** rejected; Block 3 error regrows by cut 17.

## Ranked top five

1. **Finish, audit, and run family-F fit** — direct causal test of the new composition
   hypothesis; finite and falsifiable; GPU cost bounded to 45 minutes.
2. **Make the 68-action scorer terminally executable** — largest missing interface for
   usefulness of any simplicity metric.
3. **Run early-MLP factorial composition** — tests the newly demonstrated interaction
   failure at the next most important module boundary.
4. **Evaluate a priced nonlinear uncovered-token map** — attacks a measured residual CE
   term with a bit-identical covered-row control; avoid duplicating the active agent.
5. **Consumer-aligned sparse dictionary/DAG** — promising semantic compression, but it
   needs the interfaces from priorities 1--3.

## Highest-priority action executed

Created the prospective family-F implementation amendment and a reusable mathematical
core.  CPU tests now establish:

- deterministic Euclidean projection onto the capped simplex;
- global-gate tie breaking and nested K=256/512 supports;
- equal total training weight for each of the 209 source documents;
- native-to-student KL invariant to per-token logit translation;
- consequence-score invariance under reciprocal product-factor gauge;
- analytic joint decoder refit on a selected global support;
- exact folding of scalar/vector affine calibration into existing program arrays with
  unchanged bytes and operations.

The focused family-F/native/validation suite passes 32/32 after audit hardening.
Independent mathematical and lifecycle audits both returned **GO to preserve this
outcome-blind design**.  Their findings are now executable requirements: four and only
four microbatches per Adam step; exact optimizer/clipped-parameter identity; per-step
float64 projection KKT checks; 4,612 fitted affine coordinates reported across the four
nonpromotive diagnostic fits; and donor-row reuse multiplicities receipted for the
many-to-one document null.

This is genuine forward progress, but not a family-F result: no family-F fit row or
teacher consequence has been opened.  The precise remaining blocker to numerical
execution is a separate source-closed authority/runner/result/receipt transaction that
enforces this amendment.  It is not missing data, caching, or GPU access.  The shared
GPU is occupied by the nonduplicative uncovered-token branch, so this interval was
spent on CPU preregistration, mathematical primitives, tests, and two independent
audits rather than duplicating that job.
