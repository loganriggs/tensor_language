# Broad-MLP suffix de-alias v1 preregistration

**Frozen:** 2026-08-29, before any outcome for the MLP-only layers-3--8 suffix

**Status:** prospective new-mask design; **NO-GO for execution** until a source-closed
collector/backend/scorer and independent audit exist

**Claim boundary:** same-corpus, new-mask causal-composition prediction only. This is
not OOD, a semantic circuit, a zero-native-call program, or a whole-model compression
claim.

## Why this is the next experiment

The completed early-MLP/context grid contains these broad suffixes:

- $E$: replace nothing downstream;
- $A$: replace attention outputs at layers 3--8;
- $AM$: replace attention and MLP outputs at layers 3--8.

It does **not** contain:

- $M$: replace only MLP outputs at layers 3--8.

Therefore the old contrast “$AM$ beyond $A$” aliases the effect of replacing the
broad MLP band with its non-additive interaction with replacing the attention band.
More specifically, once early-prefix interactions are formed, it aliases the
early-prefix×MLP interaction with the three-way early-prefix×attention×MLP contrast.
The descriptive Möbius analysis cannot distinguish them.

## Frozen new mask family

Use the unchanged eight early-prefix masks $P_i$ from
`EARLY_MLP_CONTEXT_CROSS_V1_PREREGISTRATION.md`. For every $i\in\{0,\ldots,7\}$,
measure exactly one new suffix:

$$
M=\{\operatorname{MLP}_3,\operatorname{MLP}_4,\ldots,\operatorname{MLP}_8\}.
$$

This is eight new program masks on each of the two existing roles, `skip7000` and
`skip11000`: 16 role-cell evaluations. Use the exact same context-free rank-64
replacement program, row ordering, scored positions, model realization, and
native-before-substitution semantics as the completed context-cross transaction. No
mask-specific fitting, gain fitting, pivot selection, or program change is allowed.

The two roles have already appeared in earlier work. This is prospective only with
respect to the new physical masks, not a new-data or OOD test.

## Frozen estimands

For suffix $S\in\{A,M,AM\}$ and nonempty early prefix $P_i$, define its interaction
with the early replacement by

$$
D_i^S
=C(P_i\cup S)-C(P_i)-C(S)+C(E),
$$

where every $C$ is token-weighted CE on identical rows and scored positions. The
existing sealed transaction supplies $D_i^A$ and $D_i^{AM}$; the new measurement
supplies $D_i^M$.

The **attention-invariance prediction**, frozen before observing $M$, is

$$
\widehat D_i^M=D_i^{AM}-D_i^A.
$$

It says that compiling the broad attention band does not materially change the
interaction between an early-prefix replacement and the broad MLP replacement. It
does **not** say that the standalone MLP suffix cost is the old contrast, nor does it
require the attention-by-MLP suffix interaction itself to be small.

After the new outcomes close, report the actual attention-by-MLP suffix synergy on
every early-prefix background, including the empty background:

$$
R_i=C(P_i\cup A\cup M)-C(P_i\cup A)-C(P_i\cup M)+C(P_i),
$$

with $P_0=E$, and report the three-way contrast

$$
Q_i=R_i-R_0=D_i^{AM}-D_i^A-D_i^M.
$$

Thus the prediction error $\widehat D_i^M-D_i^M$ equals $Q_i$. A zero $Q_i$ means
attention×MLP suffix synergy is invariant to the early prefix; it does not mean the
synergy $R_i$ is zero. Both $R_i$ and $Q_i$ are identified finite-replacement
interactions for the registered masks, not native tensor coefficients or per-site
mechanisms.

Also report the standalone broad-MLP marginal

$$
L_M=C(M)-C(E).
$$

No gate promotes $L_M$ as a simple main effect; it remains visible so the interaction
profiles cannot be mislabeled as the standalone MLP cost.

Top-1 percentage-point effects are mandatory secondary outcomes, but CE alone decides
the registered claim. No post-outcome top-1 gate may be invented.

## Frozen baselines and metrics

The null/additive predictor is $D_i^M=0$. For the seven nonempty prefixes, report:

- RMSE of $\widehat D^M$;
- zero-interaction RMSE $\sqrt{7^{-1}\sum_i(D_i^M)^2}$;
- NRE = prediction RMSE / zero-interaction RMSE;
- $R^2$ relative to the mean of the seven observed $D_i^M$ values;
- maximum absolute error;
- sign agreement count, treating exact zero as disagreement unless both values are
  exactly zero;
- RMSE/NRE separately for early singletons $\{1,3,5\}$, pairs $\{2,6,7\}$, and the
  triple $\{4\}$;
- norms of $D^M$, $R$, and $Q$; and descriptive cosines between them.

If either vector in a descriptive cosine has zero norm, serialize the cosine as the
string `undefined_zero_norm`. Descriptive cosines are not decision metrics and cannot
fail an otherwise ideal zero-$Q$ result. A zero denominator in registered NRE or
$R^2$ is, by contrast, retained as a gate failure.

No coefficient-norm “explained energy” is permitted because the Möbius/zeta basis is
not orthogonal.

## Frozen uncertainty

Within each role, perform exactly 2,000 document bootstraps with seeds `2026082903`
(`skip7000`) and `2026082904` (`skip11000`). Resample source documents with
replacement; carry all rows belonging to each sampled document together; and use the
same document multiplicity vector for $E$, $A$, $M$, and $AM$ in a draw. Aggregate
CE sums and token counts before division. Do not resample tokens or rows independently.

Use type-7 empirical 2.5%, 5%, 95%, and 97.5% quantiles. Singular/zero-denominator
draws are retained as failures, not dropped or repaired. Roles are bootstrapped
independently and may not pool for a pass.

## Frozen attention-invariance-law gates

The attention-invariance interpretation passes only if **all** of the following hold separately
on both roles:

1. every registered decision metric at the point estimate and in all 2,000 bootstrap
   draws is finite; descriptive zero-norm cosines are exempt as specified above;
2. point NRE $<0.5$;
3. the 95th percentile of NRE is $<1$;
4. point $R^2>0.5$ and the 2.5th percentile of $R^2>0$;
5. sign agreement is at least 6 of 7 at the point estimate;
6. the predictor is no worse than zero interaction at the point estimate separately
   for the singleton, pair, and triple early-prefix groups;
7. maximum absolute error is smaller than the maximum absolute observed $D_i^M$.

These thresholds mean the frozen formula must remove at least half the zero-model RMS
error at the point estimate, remain better than zero under 95% document uncertainty,
and predict meaningful variation rather than only overall scale.

## Frozen cross-role transport

Also apply the seven predictions computed from one role's old $A/AM$ cells directly
to the other role's new $M$ cells, without coefficient refitting, bias correction, or
scalar calibration. Report both directions. This is same-corpus population transport.
It passes only if point NRE $<0.5$ and the 95th percentile is $<1$ in both directions.
For each direction, freeze the source-role prediction at its token-weighted point
estimate and bootstrap only the target role's documents with that target role's seed.
The resulting interval is explicitly conditional on the source estimate; source and
target multiplicities are never paired or shared. Cross-role transport is a required
robustness gate for promotion of the attention-invariance law, but it is not OOD
credit.

## Decision rule

- **All within-role and cross-role gates pass:** the early-prefix×broad-MLP
  interaction is provisionally invariant to whether the broad attention suffix is
  also replaced, on these masks. The old broad-block contrast can be reused to predict
  that interaction, but it is not a standalone MLP main effect. Next test the frozen
  sparse hierarchy at an adjacent physical cut and add vector-valued responses.
- **Any gate fails:** prune attention-invariant transport of the old broad-block
  contrast. Report $L_M$, $D^M$, $R$, and $Q$ descriptively, but do not retune
  thresholds or add a bias/scale correction on these outcomes.
- A large, stable $Q$ after failure may nominate a three-way early×attention×MLP
  grammar, but it requires a separately frozen new-mask test before promotion.

## Required implementation and publication boundary

Before GPU execution, the amendment must freeze exact row/model/program/source hashes,
cell order, existing-artifact hashes, output namespaces, scorer source, and the full
call-ledger schema. The join to old $E/A/AM$ sufficient statistics must bind exact
ordered document IDs, row-to-document mapping, per-document token denominators,
common scored-target/support hash, program-tensor hash, and physical cell
materialization hashes. It must verify that $M$ contains exactly MLP3--MLP8 and no
attention site. Canonical execution must be fresh, receipt-last, overwrite-refusing,
and incapable of scoring injected or synthetic outcomes as authoritative. No partial
cell, role, bootstrap, or scientific result may be published before both roles close.
