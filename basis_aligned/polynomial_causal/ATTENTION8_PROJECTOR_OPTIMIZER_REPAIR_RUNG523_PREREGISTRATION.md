# Rung 523 preregistration: diagnose and repair the attention8 projector optimizer

**Frozen:** 2026-09-03 08:18 UTC, after the complete rung-522 frame archive and scheduler-aligned spike analysis,
before implementing or running any rung-523 model computation.

## Decision this experiment makes

Rung 522 attempted to learn a four-dimensional part of attention8 whose physical swaps reproduce three known
circuit effects on their examples while avoiding matched non-examples. All 103 registered fits finished, but none
of the 15 required leave-one-circuit-out fits passed the frozen optimizer health checks. The failure is numerical,
not a scientific result about whether the shared computation exists: the learned frames remained orthonormal and
moved, but rare training updates produced enormous normalized losses.

This rung asks which of two specific choices caused that failure:

1. **row-specific normalization:** every update divides by the squared full-attention response on that update's
   one selected member row; or
2. **step size:** Adam's learning rate of `.03` may be too large when the four-dimensional frame is rebuilt by QR
   decomposition on every update.

This is an optimizer-instrument calibration only. It does not test a circuit, does not change rank, does not open
TEST, does not evaluate the omitted circuit identity, and cannot contribute adoption evidence. Its only output is
whether one already-motivated optimizer definition is healthy enough to repeat the unchanged scientific experiment.

## Evidence available before freezing this rung

The immutable rung-522 archive contains 103 frames and 20,600 exact training losses. Using the frozen scheduler to
join each loss to the member row, control row, circuit target, and donor map used on that update gives:

- 379/20,600 losses above 100 (`1.8398%`);
- 84/103 fits with at least one such loss;
- 110 exact target/member/control/map combinations that spike in more than one fitted frame; and
- 87 combinations that spike in every fitted frame in which they occur.

The recurrence across fits is evidence for data-dependent scaling or donor effects. It is not decisive because
fits sharing a seed also share their initial frame and row schedule, so optimizer state can still contribute.

The source archive has file SHA-256
`2b8d3709714903890c4ae935a07da7284ac3253b7b2242d055023b33adeca2bb`. The derived spike receipt is
`basis_aligned/bilinear_quotient/attention8_selective_shared_projector_rung522_spike_analysis.json`.

## Frozen 2-by-2 comparison

The two factors are:

| training scale | Adam learning rate |
|---|---:|
| row-specific | `.03` |
| row-specific | `.003` |
| fixed target/map scale | `.03` |
| fixed target/map scale | `.003` |

The row-specific/`.03` cell is the already archived rung-522 result and will not be rerun. The other three cells
will be run prospectively. Each contains exactly the same 15 real leave-one-target-out specifications: three omitted
targets times seeds `52200..52204`.

Everything other than the two factors remains fixed from rung 522:

- rank 4 in attention8's 1,152-dimensional output write;
- the physical projector action `P = Q Q^T`;
- FIT folds 0--5 for gradients and VALIDATION folds 6--7 for health;
- the same target membership, matched controls, donor maps, initial frames, and row schedules;
- 200 Adam updates with betas `.9/.999` and epsilon `1e-8`;
- control coefficient 24;
- the exact maximum across the two included targets on each update; and
- differentiable QR to turn the optimized 1,152-by-4 raw matrix into an orthonormal frame.

The omitted third target is neither included in the loss nor evaluated. TEST folds 8--9 are inaccessible.

## Exact two normalization rules

For target `t`, donor map `m`, and scheduled member positions on update `u`, let:

- `f_tmu` be the signed per-token CE change caused by swapping the complete attention8 output;
- `p_tmu(Q)` be the signed per-token CE change caused by swapping only projector `Q Q^T`; and
- `c_tmu(Q)` be the corresponding projected swap response on the scheduled matched-control positions.

The member error and control penalty have numerators

`N_member = mean((p_tmu(Q) - f_tmu)^2)`

and

`N_control = mean(c_tmu(Q)^2)`.

The existing row-specific scale is

`d_row(t,m,u) = mean(f_tmu^2) + 1e-12`.

The fixed scale is computed once, before optimization, from every eligible FIT member position for the same target
and donor map:

`d_fixed(t,m) = mean_all_eligible_FIT_member_positions(f_tm^2) + 1e-12`.

It is never recomputed from the learned projector and never uses VALIDATION or TEST. The per-target training loss is

`N_member / d + 24 * N_control / d`,

using either `d_row` or `d_fixed`; the update loss is the maximum across the two included targets.

All candidates are compared on one common validation objective that always uses the FIT-derived fixed scale. Thus a
candidate cannot win merely because its own denominator makes its reported number smaller. The common objective is
evaluated on the unchanged rung-522 fixed VALIDATION batch at the initial and final frame.

## Opposing predictions

### Scaling explanation

If small row-specific full responses create the spikes, the fixed-scale/`.03` cell passes every adoption rule below.
The row-specific/`.003` cell either remains unhealthy or retains much more tail instability. Fixed-scale/`.003`
may also pass. The adopted repair is fixed-scale/`.03` because it changes only the implicated normalization.

### Step-size explanation

If differentiating through QR at learning rate `.03` is the main problem, row-specific/`.003` passes while
fixed-scale/`.03` fails. The adopted repair is row-specific/`.003`.

### Joint explanation

If both choices matter, only fixed-scale/`.003` passes. That combined change is adopted.

### Optimizer-family failure

If none of the three candidates passes, Adam on a raw frame followed by differentiable QR is closed for this
projector. The next method must optimize directly on the space of four-dimensional subspaces with an explicit
tangent-space update and retraction; it may not tune more Adam learning rates after seeing these results.

If both single-change candidates pass, both explanations remain viable. Fixed-scale/`.03` is adopted by the
predeclared minimal-change order because it directly removes the repeated row dependence while preserving the old
step size. Fixed-scale/`.003` is adopted only when neither single change passes.

## Health and adoption rules

A candidate cell passes only if all 15 fits satisfy all of these conditions:

1. all losses and raw-frame gradients are finite, and frozen model parameters receive no gradients;
2. `max|Q^T Q-I| <= 1e-5` and projector distance from initialization is above `.02`;
3. mean training loss over updates 180--199 is below the mean over updates 0--19;
4. the common fixed-scale VALIDATION objective is lower at the final frame than at initialization; and
5. across the cell's 3,000 updates, at most three losses are strictly above 100 and no loss is strictly above
   1,000.

The final two limits prevent an apparently improved endpoint from hiding the same rare explosions. The normalized
objective has a natural order-one baseline: zero projector gives member error near one and zero control response.
These bounds are therefore optimizer-health checks, not circuit-effect thresholds.

The baseline row-specific/`.03` cell is scored from its immutable archive under the same rules. It is already known
to fail and is retained to measure the size and recurrence of the repair, not to select a threshold.

## Registered outcomes

- **Exactly one single-change cell passes:** adopt that repair for a new, fully sealed repeat of rung 522.
- **Both single-change cells pass:** adopt fixed-scale/`.03`; report that either change is sufficient and the cause
  is not uniquely identified.
- **No single-change cell passes but fixed-scale/`.003` passes:** adopt the combined repair.
- **No prospective cell passes:** close raw-Adam-through-QR and preregister a direct subspace optimizer.
- **A cell crashes or violates its exact call ledger:** that cell is invalid, not a failure of its mathematical
  hypothesis. No healthy cell may be silently substituted or added.

Passing R523 licenses only a repeat of rung 522 with its original A--D scientific gates unchanged. It does not make
the four-dimensional projector a circuit and does not license changing any circuit threshold.

## Circuit-goal relevance and anti-rank check

R523 can change no circuit-interpretation ledger by itself. Its value is narrower and necessary: it decides whether
the existing intervention can validly test held-out circuit grouping, selective manipulation, and stability. Rank 4
is a matched constant in every cell. No result based only on rank, reconstruction, CE, or variance is accepted.

If the repaired scientific repeat later succeeds, it would address cross-head/within-module grouping, held-out
prediction, selective intervention, reuse on a fourth circuit, and stable identification. If it fails with a healthy
instrument, that is a real null for this particular downstream-use-defined attention8 subspace.

## Literal price and sealing

The prospective workload is exactly 45 fits times 200 updates = 9,000 optimization forwards and 9,000 backwards.
It may additionally use at most 1,000 inference-only forwards to capture native responses, compute fixed scales, and
evaluate the 90 initial/final common VALIDATION health calls. The implementation receipt must replace that inference
ceiling with an exact named-bucket ledger before enqueueing.

The runner must have an import-free dry run, frozen dependency hashes, CPU tests for both denominators and the
decision table, create-only result files, and no code path that can read TEST. GPU work must use the managed queue.

## Frozen dependencies

- rung-522 scientific runner commit: `2836dac0ae20817dc268f120f8e28be3fedc38a0`
- rung-522 pre-outcome implementation receipt commit: `a855e40d9`
- rung-522 archive file SHA-256:
  `2b8d3709714903890c4ae935a07da7284ac3253b7b2242d055023b33adeca2bb`
- rung-522 spike-analysis commit: `402d1ea20`
- `census_state_diverse.pt`:
  `c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b`
- `curated_rows.pt`:
  `faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd`
- `circuits/BATTERY.json`:
  `86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030`
