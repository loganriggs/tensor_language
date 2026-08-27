# MLP0 C512 -> MLP1 physical interchange: registered interface discriminator

## Why this experiment exists

The native-Down hierarchy v1 result selected one diagnostic fact, not a passing
program: C512 has small pointwise final KL/CE on the exposed v1 rows, but a large
physical MLP1-output error. This experiment asks whether that mismatch is
suffix-null on tested backgrounds or whether large causal terms cancel only in the
ordinary observational composition.

C512, its serialized bytes, its fit, every cell, and every threshold are frozen by
the v1 authority. No rank, lexical table, refit, or new candidate may enter this
experiment. The old 384 evaluation documents are forbidden. This is an oracle
diagnostic and earns zero executable-description credit regardless of outcome.

## Gauge-independent physical 2x2

For one frozen input, run the exact-native path `O` and the serialized C512 path `C`
through block-1 attention. Capture the physical pre-MLP1 residual streams `s_O` and
`s_C`, including their own upstream attention states, and compute

```text
m_O = MLP1(RMSNorm(s_O))
m_C = MLP1(RMSNorm(s_C)).
```

Replay an identical suffix from each physical post-MLP1 residual:

```text
OO = s_O + m_O       exact observational parent
CC = s_C + m_C       C512 observational parent
CO = s_C + m_O       C512 state, exact-state MLP1 write
OC = s_O + m_C       exact state, C512-state MLP1 write.
```

These arms use no learned coordinate system and are invariant to ordinary internal
factor gauges. They decompose the suffix effect into upstream state, MLP1 write, and
their interaction. They do not assert that a physical residual difference is a
parameter gauge or exact network symmetry.

The full-forward OO and CC parents are run separately. Suffix OO and CC replays must
match their parents at logits and CE to numerical tolerance before any candidate
outcome is admitted. C512 candidate evaluation poisons the original MLP0 `Down` and
asserts zero calls. MLP1 teacher calls are counted and licensed only for this oracle
diagnostic.

## Registered suffix backgrounds

The complete 2x2 is evaluated under exactly two backgrounds:

1. `live`: the unmodified block-2-through-unembed suffix;
2. `mlp2_omit`: the same suffix with the block-2 MLP residual write set to zero for
   every arm, motivated before this experiment by the registered early-MLP
   superadditivity cube.

The second background is a stress test for downstream compensation, not a candidate
model. A pass licenses statements only about these two backgrounds, never a universal
worst-background claim.

## Alignment and sensitivity controls

Let `Delta_m = m_C - m_O`. Add two controls on the OO state in both suffix
backgrounds:

- `shuffle`: add a deterministic within-cell, across-source-document permutation of
  `Delta_m`. The permutation seed and algorithm are frozen before model evaluation,
  have no fixed source-document points, and never move a vector across one of the 16
  registered cells. It preserves the empirical vector and norm multiset while
  breaking state alignment.
- `native_write`: add the native write direction `m_O`, scaled separately at every
  position to the norm of `Delta_m`. This is a label-free, norm-matched sensitivity
  control; zero-norm positions remain zero.

If aligned `Delta_m` is null but shuffled `Delta_m` is harmful, the null is
state-conditional and a later compiler must preserve that alignment. If both are
null while the norm-matched native-write control is harmful, the discarded direction
is specifically suffix-insensitive. If the native-write control is not detectably
harmful, no broad downstream-null conclusion is licensed; the assay is locally
uninformative at this intervention scale.

## New row authority and effective sample size

FineWeb confirmation uses the pinned local parquet and begins at dataset-document
index 43,000. Select exactly 384 eligible source documents in ordered dataset order,
with one to three non-overlapping 513-token chunks per document. Score each chunk as
the two windows `[0:257]` and `[256:513]`. Wave A is the first 192 eligible documents
and wave B the next 192; a source document never crosses waves. The freezer must
exclude every source-document id, dataset index, full token row, and 32-token prefix
in every prior registered row receipt, including native-Down v1 and compiler v2.1.

Each wave must retain 192 independent documents, at least 98,304 raw prediction
positions before masks, at least 60 source documents in every registered cell, and at
least 90% evaluated coverage. The source document is the bootstrap unit. Mechanical
row failure is inconclusive and must be repaired without any model forward.

As a separately reported OOD diagnostic, evaluate the already-frozen
`code_oracle_corpus_v2.pt` heldout split `[288:480]`. Its effective resampling unit is
the source file (48 files), never the token row. This corpus was not used to fit or
select C512, but it has been examined by earlier project experiments; therefore it is
called a frozen repository-Python OOD register, not a pristine confirmatory dataset.
It is never pooled with FineWeb. A pass supports only that narrow code register; a
failure blocks a broad OOD-interface claim.

The result serializes the ordered 384 FineWeb document ids, ordered 48 code-file ids,
and complete row-to-source-unit mappings. FineWeb has exactly 1,170 windows with two
to six per document; code has exactly 192 rows with four per file. Every ledger has
exactly 16 cells, integer nonnegative counts, and identical counts across all arms
and consumers. The scorer fails closed if any identity, occupancy, cell, or count
invariant changes. Code requires at least 12 independent source files in every cell.

## Estimands

Use the same fit-frozen 16 cells and margins as native-Down v1. For every source
document, wave, background, cell, and arm retain paired sums and counts for:

- final `KL(p_reference || p_arm)` with margin 0.01;
- signed next-token CE difference, tested symmetrically as absolute difference with
  margin 0.0075;
- centered-logit RMS response, normalized by the fit-frozen native centered-logit
  RMS, with margin 0.05.

The primary causal-write comparisons are `OC versus OO` and `CC versus CO`. The
upstream-state comparison is `CO versus OO`. The observational comparison is `CC
versus OO`.

For interaction, form the additive centered-logit prediction

```text
z_add = center(z_CO + z_OC - z_OO)
```

and use the observational teacher as reference: compare `softmax(z_CC)` with
`softmax(z_add)` by `KL(p_CC || p_add)`, symmetric CE difference,
and centered-logit nRMSE using the same margins. Also report the scalar CE
interaction `CE_CC - CE_CO - CE_OC + CE_OO`, but do not use a cancellation-prone
mean as the sole gate.

For FineWeb, the shuffle is performed separately inside each preregistered wave and
cell, so wave A never consumes a wave-B donor; for code it is within cell. The shuffle
and native-write controls are compared with OO at the same margins. No
outcome-defined cell, direction, sign, amplitude, or statistic may be added.

## Simultaneous inference

Use at least 20,000 paired source-document bootstrap draws with common resample
indices across every arm, contrast, cell, consumer, background, and wave. Confidence
bounds are computed from coordinatewise-centered bootstrap deviations and a single
family maximum; centering only after taking an arm or family maximum is forbidden.
For the code register, resample source files and report a separate simultaneous
family.

An equivalence claim requires, for each FineWeb wave independently and pooled:

1. the simultaneous two-sided 95% UCB on every standardized absolute effect below
   1.0 in each wave and below 0.8 pooled;
2. every support and coverage gate;
3. unchanged pass/fail conclusion and over-margin family identities after wave B;
4. exact parent replay and call-integrity gates.

A positive-control claim requires a simultaneous one-sided 95% LCB above 1.0 in at
least one final-output consumer under each background independently in wave A, wave
B, and pooled FineWeb. The code register must pass its own file-bootstrap positive
control and may not borrow sensitivity from FineWeb. A practical rejection of
equivalence requires the corresponding pooled simultaneous LCB above 1.0; otherwise
the outcome is an ordinary failure or inconclusive rather than a powered rejection.

Signed CE coordinates are retained through resampling. Their simultaneous error is
`abs(theta*_CE - theta_CE)` and their UCB is `abs(theta_CE) + c`; centering bootstrap
absolute estimates is forbidden because it is invalid when resamples cross zero.
KL and nRMSE retain their registered one-sided centered deviations.

Every promotive decision is conjoined with the complete integrity contract: frozen
source/row/program hashes, exact native and C512 parent replay at raw logits,
softcapped logits and CE in both backgrounds, zero poisoned-original-Down calls, and
exact registered C512-proxy and MLP1-teacher call counts. A scientific point estimate
cannot override failed integrity.

## Frozen decision tree

- **Fresh observational break:** if CC versus OO fails, C512 does not reproduce its
  v1 behavioral proximity on new rows. Reject promotion and do not fit a joint MLP1
  adapter from this assay.
- **Cancellation/interface break:** if CC versus OO passes but an MLP1-write contrast
  or interaction is poweredly rejected after every mechanical gate passes, the
  ordinary suffix hides causal terms that do not compose modularly. Missing an
  equivalence UCB, losing support, or failing integrity is merely practical failure
  or inconclusive, not a promoted causal-break claim.
- **Downstream-null on tested backgrounds:** only if CC, both write contrasts, and
  interaction all pass in both backgrounds and the native-write sensitivity control
  is positive. This certifies a physical suffix-null effect on the tested rows and
  backgrounds, not an internal gauge symmetry.
- **Conditional null:** if aligned writes pass and shuffle is poweredly rejected with
  valid support in both waves and pooled, the suffix null is alignment-dependent. An
  ordinary, mechanical, or underpowered shuffle failure is inconclusive.
- **MLP1-repair license:** if CO improves CC versus OO by a positive simultaneous
  lower bound in both waves under the live background, is pointwise no worse in every
  registered consumer/cell in both waves, fresh observational CC passes, and all
  integrity gates pass, a conditional MLP1 compiler under live C512 states is the
  next licensed rung. MLP2-omit rescue remains diagnostic. Otherwise joint
  MLP0-to-MLP1 fitting is not licensed by this assay.

The OOD register can veto a broad generalization statement but cannot rescue a
FineWeb failure.

## Price boundary for a later executable compiler

This 2x2 reads paired live teacher states and therefore adds no program and no
compression credit. Continue to report C512's 5,904,640-byte program plus the exact
checkpoint-float32 Left/Right common cost of 42,467,328 bytes.

If licensed, a later joint compiler must poison both original MLP0 `Down` and MLP1,
and charge

```text
P(total) = P(C512) + P(g1) + P(bases/codebooks/routing/interfaces/decoder).
```

No amortized basis discount is allowed without preregistered reuse. The joint
compiler must be selected by causal suffix KL/CE under live C512 states; Euclidean
MLP1-target fit is only a control. A matched-price higher-rank direct MLP0 map and an
equal-byte unstructured joint map are mandatory baselines.

## Execution authority

This specification authorizes only outcome-blind row freezing, implementation, and
CPU tests. A separate committed authority must bind the checkpoint, model and suffix
source, C512 bundle hash, row receipt, code-register hash, cell scales, permutation
seed, exact arms, control construction, replay tolerances, inference code, and result
path before the first model forward.
