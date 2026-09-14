# September 14: continuous research restored; latest results and earlier idle gap

## Latest decision — 10:22 UTC / 06:22 EDT

**Do not adopt the fitted operator.** It passed independent synthetic
replication but failed transfer to the historical native-text circuit cache.

- [Independent 1024-sequence replication](../../UNIFORM_PRODUCER_REPLICATION_V1_RESULT.json):
  41.39% lower normalized error energy, 3.39 paired standard errors, positive
  gains in both halves, unchanged storage; all replication criteria pass.
  Error is heavily concentrated: the top 11 sequences contribute 66.9% of
  baseline energy. The original fit's confidence/gradient failures remain.
- [Historical native-cache comparison](../../PRODUCER_FIT_CACHED_COMPARISON_V1_RESULT.json):
  mixed-effect squared error worsens 79.04%, with degradation in all five
  groups. One group's own-effect error reaches 11.26%, failing the 10% bar.
  Numerical replay and whole-compact-effect preservation pass, but neither
  rescues the failed mixed-term criterion. The original sparse operator stays
  the reference. These 120 cached prefixes are not fresh or OOD validation.
- [Score-boundary countercheck](../../PRODUCER_SCORE_BOUNDARY_V1_RESULT.json):
  degradation is already 34.59% on all twelve normalized raw outputs, before
  selecting the task contrast or applying softcap. Different scoring cannot
  fully explain the transfer failure; input/background mismatch remains.

The next [length-transport screen](../../UNIFORM_LENGTH_TRANSPORT_V1_PREREGISTRATION.md)
is executing on frozen programs, testing lengths 5, 32 and 128. It is a small
descriptive synthetic test, not another fit or proof of a good input law.

## Earlier progression — 10:15 UTC / 06:15 EDT

The [coupled suffix test](../../COUPLED_WRITER_TAIL_V1_RESULT.json) passed in
2.68 seconds: bit-exact native FP32 replay on two synthetic contexts, and local
derivative replay within 1.47e-7. The full suffix weights remain present; this
validates an instrument, not a smaller circuit.

The next [256-sequence comparison](../../UNIFORM_PRODUCER_RANKING_V1_RESULT.json)
used actual uniformly sampled embedding rows, native upstream computation,
and the existing child/remainder scalar fields. Both edits share their actual
upstream write direction. These synthetic sequences are not natural-language
or OOD validation. The supplied final background also now matches the previous
conditional composed-last-block construction, rather than an arbitrary vector.

That changed the ranking: sparse error relative to the mixed contribution was
5.17%, ordinary shared-output 3.89%, and balanced shared-output 7.37%.
Ordinary shared-output was better in both sample halves, but only by 1.67
paired standard errors overall; the preregistered sparse-superiority criterion
failed. This is suggestive, not a reliable shared-operator win. It also cannot
isolate which changed aspect of the input/background construction caused it.
No candidate is adopted. Absolute error energies are tiny, around 1e-12,
because the selected mixed contribution is small on these synthetic inputs.

A [precision countercheck](../../UNIFORM_PRODUCER_PRECISION_V1_RESULT.json)
on 32 original sequences changed to FP64 suffix arithmetic and separately
matched the native attention-output subtraction order. Relative changes in
the mixed and candidate-error vectors stayed below 0.0374%, passing the 1% bar.
This does not explain away the observed ranking reversal as a large numerical
artifact on the tested rows; limited statistical precision remains.

A [fixed-support fitting test](../../UNIFORM_PRODUCER_GRADIENT_V1_PREREGISTRATION.md)
completed in 46.50 seconds: independent 256-context training gradients, an exact scalar
line solve, and 512 fresh validation contexts. Mask, adapters and storage are
fixed. Fresh energy fell 37.24%, and contrast energy fell 40.13%, but the run
**failed its registered criteria**: improvement was 2.971 paired standard
errors (required >3), and gradient cosine was 0.424 (required >=0.5).
The numerical and storage checks passed. These results justify a separate
[frozen-candidate replication](../../UNIFORM_PRODUCER_REPLICATION_V1_PREREGISTRATION.md),
not adoption or rewriting the original failure. The new panel has 1024 fresh
synthetic sequences and performs no further fitting.

## Runtime snapshot — 10:06 UTC / 06:06 EDT

The durable research goal is active. Both queue runners and the hourly and
three-hour review timers are active. The latest experiment has finished; the
next producer-constraint check is being implemented. At the 10:03 service
check, the next review invocations were 10:51:50 and 11:43:01 UTC.

The precision audit and frozen-candidate comparison succeeded, but neither
subsequent fit improved the operator meaningfully:

- [Scalar refit](../../MASKED_DIRECTION_REFIT_V1_RESULT.json), 22.23 seconds:
  only 0.000253% held-out improvement, 0.36 paired standard errors.
- [Full-support gradient screen](../../MASKED_GRADIENT_SCREEN_V1_RESULT.json),
  12.40 seconds: independent training gradients had cosine 0.001; fresh-sample
  error worsened 0.456%, about 6.5 paired standard errors. Numerical replay and
  storage checks passed; improvement and gradient-reproducibility criteria failed.

Neither replaces the original sparse operator. The gradient result demonstrates
overfitting at this sample budget, not that every possible refit must fail.
Next is the [coupled-writer tail control](../../COUPLED_WRITER_TAIL_V1_PREREGISTRATION.md):
the actual child and remainder edits initially share one upstream write
direction, whereas independent full-width Gaussian edits omit that coupling.
The functional suffix implementation has begun; independent native replay
must pass before it supplies scientific evidence. This is still a weights-only
synthetic control, not native-text validation or a global low-rank claim.

The timestamped sections below preserve the earlier status and progression.

## Original 09:45 UTC snapshot

Status checked at **09:45 UTC / 05:45 EDT, September 14, 2026**.

Both queue runners have been alive for seven hours, but **no research experiment is currently running or queued**. The GPU has no active compute process. Hourly and three-hour review timers are enabled; neither review is executing at this snapshot. The GPU runner has been running its periodic regression canary, most recently completing at 09:20:24 UTC with all checks green and a stable fingerprint relative to this machine's previous run.

My earlier update was accurate about the services being running, but it did not adequately distinguish that from continuous scientific work. I configured bounded review jobs, not a continuous research driver. Several hourly reviews consequently repeated that the queues were empty and progress had stalled. There were useful CPU controls during the mathematical reviews, but no new compression fit or native-text experiment overnight. The experiment runner executes queued work; it does not choose or create the next experiment itself.

At this snapshot, the next scheduled hourly invocation is 10:44 UTC / 06:44 EDT, and the next mathematical invocation is 11:43 UTC / 07:43 EDT. These are scheduler times, not evidence that a research experiment is pending. The 09:45 hourly invocation skipped writing a duplicate because the previous review was less than 55 minutes old.

## What scientific work actually changed

We are still working on **setting2: head17.2 → MLP17 → twelve selected output readers**. The retained attention predictor has three contributions to a 128-dimensional head write. That write interacts with a 1152-dimensional residual background through MLP17. The existing folded mixed operator is

$$
T\in\mathbb R^{12\times1152\times128},\qquad
M_o(z,a)=\sum_{i,h}T_{oih}z_i a_h.
$$

Here, \(z\) is the supplied residual background and \(a=a_1+a_2+a_3\) is the sum of the three retained attention contractions. For a compressed operator \(\widehat T\), define \(E=\widehat T-T\). With the same supplied normalization denominator \(d\) for reference and candidate, each branch contributes error

$$
e_k=\frac{E(z,a_k)}{d},\qquad
J=\mathbb E\left\|\sum_{k=1}^3e_k\right\|^2
=\sum_{k,l=1}^3\mathbb E\langle e_k,e_l\rangle.
$$

The nine entries matter because errors can reinforce or cancel. This is a conditional raw-logit error objective; it does not include a freshly recomputed compressed suffix or the final token softcap. Using real weights on synthetic Gaussian states is still a modeling choice, not evidence about native text.

| Completed control | Result | What it establishes |
|---|---|---|
| Native-weight, 128-context instrument | Direct error and full nine-term calculation agree within 1.97e-16 relative; native-factor replay within 1.14e-15 | The implemented contraction identity works. Dropping cross terms changes estimated energy by 1.16–12.66% across these small panels. |
| Source-position sampling control | Correct inclusion weighting replays expected Gram entries within 2.31e-16; naively squaring a sampled write biases energy upward 46.57% | An unbiased sampled write does not automatically give an unbiased squared-error objective. Recomputing its normalization from sampled writes introduces another bias. |
| 1024-context integration audit | Independent halves differ 2.13%; estimated relative standard error is 3.27%; both sample-doubling changes exceed the registered 1% threshold | Algebraic correctness does not yet provide the requested statistical precision. The integration criterion failed. |
| Global-sign paired sampling | Paired energies have correlation 0.999999963; approximately twice the variance at equal evaluation count | Negating the entire Gaussian input supplies almost the same squared error, so this attempted variance reduction failed. |

Primary receipts: [native-weight instrument](../../NATIVE_RETAINED_ERROR_V1_RESULT.json), [source-sampling control](../../NORMALIZED_PAIR_HT_CONTROL_20260914_0256_RESULT.json), [outer-context and sign-pair audit](../../OUTER_CONTEXT_ANTITHETIC_20260914_0844_RESULT.json). The controls use different declared synthetic constructions; their absolute energy values should not be compared as one common benchmark.

The 1024-context audit used exact source sums and reported a descriptive mixed-error ratio of 8.25% for the existing sparse candidate. **That is not a new compression improvement or native-behavior result.** Its uncertainty estimates are diagnostics, not finite-sample guarantees. The [mathematical review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-14_0846.md) derives the sampling and sign-symmetry consequences and records their limits.

## What remains unchanged, and the next decision

No new candidate has replaced the previous setting2 sparse operator. Its earlier conditional regional result and broader full-head failures remain in force. The previous setting1 result also remains conditional, retains all 64 reader directions, saves only 1.806% of its declared parent interface, and is not faster. The [September 13 full report](research_update_2026-09-13_final_compression.md) remains the reference for those compression results.

The latest strategic review recommends a **preregistered, fixed-budget 32,768-context fresh-IID precision audit** under the unchanged synthetic law, with exact source sums and all nine error terms. It must test independent-sample agreement and uncertainty before this objective is used to rank or fit candidates. That audit has **not been started or queued**. Even a pass would establish precision for the chosen synthetic measure, not its adequacy for real circuit interventions or native text.

The immediate operational gap is execution continuity, not checkpoint access or GPU availability. This status update does not launch experiments or change the review schedules. [Latest strategic review](../../HOURLY_STRATEGIC_REVIEW_2026-09-14_0851.md) · [Local runtime and service instructions](../../../../session_recovery/LOCAL_START_2026-09-14.md).

## Follow-up: elapsed time and the missing continuation mechanism

Relative to the 09:45:17 UTC status check, the last mathematical-review CPU
experiment finished at 08:45:44 UTC: about **59 minutes 33 seconds earlier**.
Its internal execution took **1.76 seconds**. The last research job I put through
the managed CPU lane finished at 02:48:38 UTC: **6 hours 56 minutes 39 seconds
earlier**, with **0.59 seconds** of measured calculation. Periodic GPU canaries
continued during that gap; they were regression checks, not new research.

After Logan pointed out the prior session's fix, I checked this thread's durable
goal state and found it empty. The recovered NEXT_CODEX_PROMPT explicitly says
to create that goal when absent. I had missed that instruction while restoring
the runners and timers. The durable research goal is now **active**, restoring
the mechanism for continuing research across turns. It is distinct from the
bounded scheduled reviews, which remain enabled. The next precision audit is
now [preregistered](../../OUTER_CONTEXT_PRECISION_V1_PREREGISTRATION.md), with
32,768 fresh contexts and fixed precision criteria. No result is claimed yet.

### 09:52 UTC: continuation produced two new results

The [32,768-context audit](../../OUTER_CONTEXT_PRECISION_V1_RESULT.json) now
passes all registered criteria in27.12seconds: estimated relativeSE0.53%,
independent-half gap0.45%, and doubling changes0.33%/0.22%. This repairs the
precision limitation under the fixed synthetic law; it does not retroactively
pass the earlier1024-context test or establish a native-text objective.

A [fresh paired comparison](../../PAIR_RANKING_V1_RESULT.json),8192contexts in
8.41seconds, ranks the existing sparse candidate ahead of both shared-output
candidates in total normalized error at approximately equal4.9MBstorage.
Ordinary shared error energy is0.02736, balanced0.06223, sparse0.01579.
The paired gaps are43and75standard errors. Balanced improves the selected paired
contrast energy slightly (0.01020versus0.01086), but its much larger total error
remains a failure to replace the sparse operator. These are error energies under
one synthetic measure, not CE or native effect-preservation scores.

Next is a [fixed-support scalar refit](../../MASKED_DIRECTION_REFIT_V1_PREREGISTRATION.md):
test whether a weight-derived direction can improve the sparse coefficients while
keeping its mask, adapters and storage budget. Separate training and validation
randomness is frozen. No fitted result or new compressed circuit is claimed yet.
