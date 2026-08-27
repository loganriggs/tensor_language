# MLP0 C512 → MLP2 compensation factorial: frozen protocol

## Status and zero-credit boundary

This protocol is written before any evaluation forward on its fresh FineWeb role.
It is a **causal localization assay**, not an executable compiler. Crossed teacher
writes receive zero recovery or compression credit.

The preceding C512 → MLP1 experiment established that C512 changes the physical MLP1
write, omitting MLP2 exposes a powered failure, restoring the exact MLP1 write repairs
that exposed failure, and deployed MLP2 suppresses most of it. The remaining question
is whether suppression is carried by the state reaching MLP2, MLP2's state-conditioned
write, or their interaction.

## Physical factorial

Let $r_O$ be the natural post-MLP1 residual under exact MLP0 and $r_C$ the natural
post-MLP1 residual under C512, including C512's naturally induced MLP1 write. Run the
exact block-2 residual mixing and attention to obtain the physical pre-MLP2 states

$$
u_O,\qquad u_C,
$$

then evaluate the exact native MLP2 on each state:

$$
w_O=\operatorname{MLP2}(\operatorname{RMSNorm}(u_O)),
\qquad
w_C=\operatorname{MLP2}(\operatorname{RMSNorm}(u_C)).
$$

The four factorial cells are

$$
OO=u_O+w_O,\qquad CC=u_C+w_C,
$$

$$
CO=u_C+w_O,\qquad OC=u_O+w_C.
$$

Every cell replays the identical unchanged suffix from block 3 through the final
readout. Crossed arms are assembled from one physical realization; they do not call
MLP1 or MLP2 again.

Two omitted-write parents are also replayed:

$$
O0=u_O,\qquad C0=u_C.
$$

These reproduce the prior MLP2-omission contrast on wholly fresh rows and provide the
required exposure check.

## Controls

Let

$$
\Delta w=w_C-w_O.
$$

Within each frozen FineWeb wave and difficulty cell, source-document-derange
$\Delta w$ without changing its vector multiset and replay

$$
CS=u_C+w_O+\Pi(\Delta w).
$$

Donor and recipient documents must never match and wave membership must never cross.
This is the alignment null.

The suffix-sensitivity positive control is

$$
ON=OO+\alpha w_O,
\qquad
\|\alpha w_O\|_2=\|\Delta w\|_2
$$

at every position. It tests whether the registered suffix metrics can detect a native
write-scale perturbation.

Thus the common eight-arm family is

$$
\{OO,CC,CO,OC,O0,C0,CS,ON\}.
$$

## Registered contrasts

The reference and KL direction of every registered contrast are fixed here. Signed CE
is always `candidate CE - reference CE` in the same orientation.

1. **Observational:** $KL(p_{OO}\|p_{CC})$.
2. **Prewrite state:** $KL(p_{OO}\|p_{CO})$.
3. **Write on exact state:** $KL(p_{OO}\|p_{OC})$.
4. **Write on candidate state:** $KL(p_{CO}\|p_{CC})$.
5. **State-by-write interaction:**

   $$
   \operatorname{softmax}(\ell_{CO}+\ell_{OC}-\ell_{OO}),
   $$

   after fixing the per-token additive-logit gauge, scored as
   $KL(p_{CC}\|p_{\mathrm{add}})$.
6. **Omission exposure:** $KL(p_{O0}\|p_{C0})$.
7. **Alignment null:** $KL(p_{OO}\|p_{CS})$.
8. **Sensitivity:** $KL(p_{OO}\|p_{ON})$.

The direct divergence between $CC$ and $CS$ may be retained descriptively, but it
cannot decide alignment. The registered alignment rescue compares both arms with the
common $OO$ reference.

No Euclidean activation norm can decide a causal label. Norms of $u_C-u_O$ and
$w_C-w_O$ are retained only as descriptive interface measurements.

## Metrics, cells, and inference

Each contrast is measured by:

- final capped-logit KL, margin $0.01$;
- signed final CE change, margin $0.0075$; and
- centered-logit nRMSE using the inherited fit-frozen centered **capped-logit** RMS,
  margin $0.05$.

Signed CE remains signed in source-unit ledgers and bootstrap replicates. If the point
estimate is $\theta$ and a bootstrap replicate is $\theta^*$, its deviation is
$|\theta^*-\theta|$; the two-sided UCB is $|\theta|+c$. An absolute CE estimate is
never bootstrapped directly.

The 16 cells are inherited unchanged from the C512 → MLP1 authority:

- early versus later prediction position;
- common versus rarer current token;
- punctuation versus non-punctuation predecessor; and
- lower versus higher pre-MLP0 residual norm.

The exact inherited fit currency is the centered capped-logit RMS, the token-count
vector and its hash, the frequency and pre-MLP0-norm cutoffs, and
`valid = fit_count[current_token] > 0`. Each is hash- or numerically bound before the
first evaluation forward. No evaluation outcome may choose a threshold, arm, margin,
or subset.

FineWeb supplies 384 wholly new source documents, frozen prospectively as two
disjoint waves of 192. They must be disjoint from **every prior registered row receipt
or manifest**—including Stage0, native-Down, C512 fit/eval, compiler v2/v2.1, exact
oracle, and composition roles—by source id, dataset index, full row, and 32-token
prefix. Source document is the resampling unit. This minimal assay does not repeat the
underpowered 86.4%-coverage code register.

Every arm and metric must have identical integer support counts, at least 60 source
documents per cell per wave, and at least 90% valid-token coverage in each wave. One
coordinatewise-centered source-document bootstrap with 20,000 replicates supplies one
joint arm-by-metric-by-cell two-sided 95% event **within each scope**: wave A, wave B,
and pooled. The three scope decisions are conjoined, but this is not claimed to be one
95% event across scopes. Max-arm comparisons in scope $s$ use its own conservative
common-event lower bound

$$
\widehat D_s-2c_{\mathrm{joint},s},
$$

never a bootstrap that reselects maxima inside replicates.

## Registered outcome labels

### `mlp2_suppression_replicates`

Every outcome label below is conjoined with full source/model/program integrity,
finite-ledger, identical-support, coverage, cell-support, call-count, carried-state,
and parent-replay gates. If any common gate fails, all labels fail closed. Subject to
those common gates, `mlp2_suppression_replicates` requires:

1. omission $C0/O0$ powered outside the margin in waves A, B, and pooled
   (simultaneous lower bound $>1$);
2. the live observational family maximum smaller than the omission maximum with a
   positive conservative reduction lower bound in A, B, and pooled; and
3. live observational coordinates pointwise no worse than omission coordinates.

`complete_compensation` additionally requires live observational equivalence: wave A
and B simultaneous UCB $<1$ and pooled UCB $<0.8$.

### Component statuses

Prewrite state, write-on-exact-state, write-on-candidate-state, and interaction each
receive exactly one of:

- **equivalent:** wave A/B UCB $<1$ and pooled UCB $<0.8$, with prewrite equivalence
  additionally requiring powered omission exposure, both write-equivalence labels
  requiring powered $ON/OO$, and interaction equivalence requiring both powered
  omission exposure and powered $ON/OO$;
- **powered non-null:** wave A/B/pooled LCB $>1$; or
- **inconclusive.**

A missed UCB is never called evidence of absence.

### `aligned_mlp2_write_compensates`

This requires:

1. the suffix-sensitivity control $ON/OO$ powered in waves A, B, and pooled;
2. the common-reference reduction

   $$
   \max D(OO,CS)-\max D(OO,CC)-2c_{\mathrm{joint},s}>0
   $$

   in waves A, B, and pooled; and
3. $|\operatorname{effect}(CC/OO)|\le
   |\operatorname{effect}(CS/OO)|$ pointwise on every registered coordinate.

This licenses the statement that MLP2's **state-aligned changed write** contributes to
compensation. It still licenses no executable compensator because both $w_O$ and $w_C$
come from native MLP2.

## Authority and integrity

Before any evaluation forward, a committed authority receipt must bind:

- synchronized source commit and complete source closure;
- fresh-row receipt/tensor, source-document identities, wave assignment, and every
  disjointness check;
- exact C512 program and its original fit-authority chain;
- inherited capped-logit RMS, token-count hash, cutoffs, and valid-mask definition;
- exact config and checkpoint blobs, loaded directly rather than through a mutable
  Hub resolver;
- eight arms, three metrics, 16 cells, bootstrap constants, call counts, and replay
  tolerances.

The evaluator must:

- poison candidate access to original MLP0 `Down` and observe zero candidate calls;
- run one poison canary;
- count exact C512, MLP1-teacher, MLP2-teacher, and suffix-site calls;
- permit teacher calls only while capturing $O$ and $C$, never in crossed replays;
- independently replay ordinary exact/C512 live parents and exact/C512 MLP2-omitted
  parents at raw logits, capped logits, and CE before scoring;
- assert $x_0$ and the carried block-0 value state are identical at the registered
  tolerance;
- require finite tensors and exact ordered unit ids/mappings/occupancies; and
- fail closed with atomic artifacts while retaining raw source-unit sufficient
  statistics for exact scorer replay.

A later executable compiler must generate its own MLP1/MLP2 states and writes with
both native modules poisoned, beat equal-price continuous and deranged controls, and
pass the exact early composition cube on fresh rows.
