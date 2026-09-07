# Full two-day research update: circuits, DAS, and tensor-weight programs

**Window covered:** 2026-09-05 16:14 UTC through 2026-09-07 16:14 UTC  
**Model:** `bilin18`, an 18-layer transformer with residual width 1,152, nine attention
heads per layer, and bilinear MLPs  
**Program goal:** replace selected model computations with a smaller transparent tensor
program that is predictive on held-out text, composable under joint edits, manipulable by
removal/swap/dose interventions, and literally simpler to store and execute

## Executive summary

The work moved from locating whole residual, attention, and MLP sites to identifying executable
read-compute-write programs and translating learned causal subspaces into actual model-weight
factors.

The strongest mature result in this window is the controlled-domain Task14/bracket answer-margin
program. It uses 22 stored FP32 scalars (88 bytes), compiles its task selectors from text, supports
typed counterfactual composition, and needs no model or Torch dependency for execution. Task14 is
standalone in its tested domain; the bracket side still requires one native unedited baseline
margin. It is a transparent answer-margin program, not a whole language model.

The aspectual `since/by -> has/had` line progressed from a residual-stream screen into an explicit
operational quotient. A fixed local reader and four affine coefficients drive a rank-one write;
the executable release stores 1,157 fitted scalars. It transfers across multiple disjoint lexical
families and supports continuous dose manipulation. It does not establish that one small set of
native heads implements the entire behavior: the early four-head path is real but explains only
about 5% of the full residual effect, while a later three-head set explains about 61%.

The largest circuit effort built a joint temporal-auxiliary / `is`-versus-`was` program. A stable
eight-dimensional downstream response interface was made literal through attention and MLP weights.
A recursively executable upstream graph was reduced from 110 candidate components to a frozen
46-component pooled program that survives forward/reverse OOD transfer and 10% source noise. Inside
that shared physical graph, the two tasks use distinct rank-four response modes and distinct local
weight functions. A separate five-MLP, rank-16 source interface is also bidirectional and selective.
Its exact hidden-to-write maps are `A_s = Down_s^T Q_s`; the maps close to numerical precision and
show that the physical hidden support is broad but nearly additive.

The user's shared-subspace-to-weight proposal is now part of the live method, not merely a future
idea. We pull each causal response basis through `Down`, `W_V`, and `W_O`, then use those exact
factors to rank and patch upstream writers and downstream readers. This has already exposed explicit
edges such as `MLP0 -> MLP1`, `MLP1/2 -> MLP3`, `MLP4/5 -> MLP6`, `MLP6 -> L8H1`, and
`MLP7/8 -> L9H4`.

Constrained DAS now has a more precise verdict. Same-family complement optimization can memorize a
task-specific solution. Full-vocabulary KL is genuinely useful and can outperform the starting
axis on a sealed family, while small tangent noise alone has usually done little. But one fixed
regularizer reverses across construction folds. Therefore the live test uses nested
construction-level selection, hard target-retention constraints, an all-six-loss score, pooled
difference-in-means and nested no-regularization comparators, and a capability-qualified sealed v15
family. This is a test of whether regularization prevents memorization, not another coefficient
sweep on an already opened task.

The breadth campaign simultaneously expanded the behavior inventory. As of the end of the window,
44 of 86 behavior specifications had reached the amended four-row screen, and the stricter
within-family separability accounting supported 27 provisional distinct circuits. These are not 27
finished transparent programs. Most are causal screens; only a smaller set has OOD, composition,
selective manipulation, and weight-level dossiers comparable to the deep temporal lines.

## 1. What the project is trying to build

The target is not “find a low-rank activation” or “compress the model while preserving average
loss.” The desired output is a smaller executable program with seven properties:

1. **Computational specification:** say what information is read, what operation is performed, what
   is written, and which later components consume the write.
2. **Correct circuit grain:** merge pieces of different native modules when they implement one
   variable, and split one head or MLP when it implements multiple variables.
3. **Held-out prediction:** predict behavior and internal causal response on unseen lexical items,
   templates, directions, and construction families.
4. **Sufficiency or extraction:** an isolated interface plus explicitly stated background reproduces
   the target computation or signed causal effect.
5. **Selective manipulation:** removal, interchange, editing, and dose changes affect the intended
   behavior without silently changing unrelated behaviors.
6. **Composition and reuse:** shared subcomputations can be combined with task-specific branches and
   their joint effect is predictable.
7. **Literal simplicity:** storage, arithmetic, states, and edges are counted for the executable
   object, rather than inferred from an activation rank alone.

The evidence vocabulary used throughout is deliberately strict:

- A **screen** is a causal or correlational result that nominates an object.
- **Identification** requires held-out prediction and survival under relevant controls, gauges,
  directions, and construction shifts.
- **Adoption** requires a legal executable replacement that also improves the priced program
  frontier. No new result in this two-day window is a whole-model adoption.

## 2. Core terms and computations

### 2.1 Bilinear MLP as an explicit tensor

For residual state `x in R^d`, a bilinear MLP has left and right hidden factors and a down map:

```text
l = Left x
r = Right x
h = l ⊙ r
y = Down h
```

In indices,

```text
y_o = sum_k Down[o,k] (sum_i Left[k,i] x_i) (sum_j Right[k,j] x_j).
```

The equivalent third-order tensor is

```text
T[o,i,j] = sum_k Down[o,k] Left[k,i] Right[k,j].
```

This matters because a causal output direction can be pulled back into the exact quadratic
function of the MLP input, rather than approximated by weight norm or activation variance.

### 2.2 Interchange intervention and recovery

For a base prompt `b`, donor prompt `d`, and site state `x`, an interchange replaces a declared
part of `x_b` with its value from `x_d`. If `m` is an answer-minus-foil margin, normalized recovery
is

```text
rho = (m_patch - m_base) / (m_donor - m_base).
```

`rho = 1` means the intervention reproduced the full donorward margin change; `rho = 0` means no
donorward movement. Signed projection is the vector analogue:

```text
signed_projection = <delta_patch, delta_target> / ||delta_target||^2.
```

Relative squared error is

```text
RSE = ||delta_patch - delta_target||^2 / ||delta_target||^2.
```

We report both because a projection can have the correct amplitude while still contain a large
orthogonal error. Controls also report full-vocabulary KL divergence, margin RMS relative to target
scale, and top-1 answer flips.

### 2.3 Subspace patching

For an orthonormal basis `Q in R^(d x r)`, the projector is `P = Q Q^T`. A donor-difference patch is

```text
x_patch = x_base + P (x_donor - x_base).
```

The orthogonal-complement patch uses `I - P`. A claimed causal subspace should be sufficient when
`P` is patched and inert or weak when `I-P` is patched. Exact closure checks that the two pieces
reconstruct the declared parent intervention.

### 2.4 Difference in means and DAS

**Difference in means (DIM)** forms a direction from averaged signed donor-minus-base differences,
then normalizes it. It is simple, stable, and often a strong baseline.

**Distributed Alignment Search (DAS)** optimizes a direction or subspace so an intervention has a
desired causal effect. In the constrained form used here, the learned subspace should reproduce the
target while its complement remains inert. The danger is under-identification: on a finite family,
many directions can satisfy the observed scalar readout while encoding construction-specific
nuisance information.

The six normalized loss terms used in the current DAS work are:

```text
margin_match   = error between subspace-patched and full target margins
l15_match      = error at the registered layer-15 reader
margin_inert   = error between complement-patched and base margins
l15_inert      = complement disturbance at the layer-15 reader
vocab_match    = full-vocabulary subspace-to-target discrepancy
vocab_inert    = full-vocabulary complement-to-base discrepancy
```

The first two are also hard feasibility constraints. Current optimization uses a robust maximum
over environments, optional full-vocabulary KL weight, optional antithetic tangent noise on the
unit sphere, and a fixed cross-environment variance penalty. Crucially, hyperparameters and stopping
time are selected on inner construction splits; complete outer families and sealed v15 do not select
the optimizer.

### 2.5 Causal-response basis

A source intervention can affect many later heads, MLPs, residual coordinates, and output logits.
Its **causal-response fingerprint** is the vector of those downstream changes. Two source states are
operationally equivalent for the circuit when downstream readers cannot distinguish their response
within the tested domain.

This gives a response quotient: instead of preserving the largest activation variance, preserve the
directions that span distinct downstream causal effects. It is why a small raw activation rank is not
automatically a useful circuit rank.

### 2.6 Translating a response subspace into weights

Suppose source MLP `s` writes through `Down_s` and `Q_s` is a causal output basis. Define

```text
A_s = Down_s^T Q_s.
```

If `delta_h` is the bilinear hidden change, the exact projected residual write is

```text
delta_y_Q = (delta_h A_s) Q_s^T.
```

This separates the physical hidden writer coefficients `delta_h A_s` from the residual output
directions `Q_s`. It is invariant to a rotation `Q_s -> Q_s R` when the coordinate map is rotated
consistently.

For an MLP output covector `a`, the exact symmetric quadratic reader on the MLP input is

```text
S(a) = 1/2 [Left^T diag(a) Right + Right^T diag(a) Left].
```

For attention, a residual response basis can be pulled through the output and value weights using
the corresponding `W_O Q` and `W_V` contractions. These maps tell us which components can physically
write to or read from the subspace. Weight contraction proposes edges; complete-model patching
decides whether those edges are causal.

### 2.7 Shared versus task-specific subspaces

Given task bases `Q_1` and `Q_2`, compute

```text
Q_1^T Q_2 = U Sigma V^T.
```

The singular values are principal cosines. Aligned pairs `Q_1 U` and `Q_2 V` can be combined into
canonical mean and contrast directions, schematically

```text
q_mean     proportional to q_1 + q_2
q_contrast proportional to q_1 - q_2.
```

This construction is gauge-safe at the subspace level. Stable singular gaps and crossfit projector
agreement are required before individual axes receive names; otherwise only the whole block is
identifiable.

### 2.8 Factorials, Möbius interactions, and greedy composition

For `n` candidate components, a complete intervention lattice evaluates every subset `S`. The exact
Möbius coefficient is

```text
mu(S) = sum_{T subseteq S} (-1)^(|S|-|T|) f(T).
```

Degree-one mass indicates additive singleton effects; large degree-two or higher terms indicate
conditional computation. Greedy composition is licensed only after singleton semantics transfer
and interaction energy is small. It is not licensed merely because a low-rank reconstruction looks
good.

## 3. Day one: from Task14 sites to an executable margin program

### 3.1 MLP8 and the distributed MLP6/7 source

At the beginning of the window, the Task14 grammatical-number route was already centered on an
MLP8 bilinear response feeding the L11H3 answer route. Full residual-history factorials split the
pre-MLP8 state into embedding/skip history `E`, attention history `A`, early-MLP history `U`, and
MLP4-7 history `W`. `W` was the dominant stable source, but a complete MLP6-versus-MLP7 test showed
that neither native layer was a stable semantic unit: their effects were distributed,
direction-dependent, and interacting.

This phase also exposed a serious systems failure. An initial 6,064-row intervention OOMed, and a
two-chunk repair then stalled for 99 minutes before exiting without evidence. The run was treated as
an invalid instrument, not a causal null. Later experiments used bounded batches and live price
accounting.

### 3.2 Quadratic manipulation and OOD composition

The grouped MLP6/7 source acquired a midpoint-quadratic readout law. Endpoint linearization failed,
but a midpoint derivative predicted finite intervention effects. Frozen coefficients then predicted
new source gains at `-0.5`, `0.5`, and `1.5` after native-tail execution.

The normalized `E/A/U/W` gate transferred across matched and fronted syntax. Its contributions were
distributed—roughly 29-31% each for `E` and `A`, about 24% for `W`, and about 16% for `U`—with signs
reversing by grammatical direction. Frozen profiles predicted large intervention lattices with
normalized errors around `.04-.06`. Continuous scaling moved the target effect monotonically
through zero while preserving low lexical collateral.

A pristine zero-anchor test supplied an important null: normalized factor shape transferred, but
absolute amplitudes did not. That failure motivated a downstream Jacobian-vector product rather than
more fitted scalar tables. A target-free central downstream gradient then predicted the full
512-condition lattice at cosine `.999961` and relative L2 error `.009332`.

### 3.3 Fixed readers, upstream prototypes, and editing

Two fixed 1,152-dimensional direction readers transferred to a disjoint corpus with cosine `.96983`,
relative L2 `.29219`, and perfect signs. Swapping them reversed the effect, showing that direction
assignment was causal rather than a generic magnitude predictor.

Ten direction-by-cardinality upstream prototypes were frozen before another corpus was opened. They
substituted native displacements at cosine `.86301`; the existing readers predicted the installed
effects at cosine `.96927` with perfect signs. A reader-guided edit then chose an intervention gain
that achieved a requested signed `.04` margin change. This moved the work from passive description
to manipulation.

MLP15 and MLP17 were identified as an approximately additive but cancelling downstream mediator
pair. A six-scalar direction-only gain law was preferred to a larger direction-by-cardinality table,
but its later fourth-corpus transfer failed in one singular-to-plural cell. The upstream program and
fixed readers survived independently; the mediator-gain law was not promoted.

### 3.4 Task14 plus bracket compilation

The Task14 and bracket work was compiled into a typed counterfactual-margin API. Composition was
tested exhaustively: identity/no-edit, idempotent same-slot overwrite, right-biased overwrite from
an immutable baseline, and commuting independent slots all held.

A fully standalone 27-scalar candidate failed because bracket baseline prediction was inaccurate
near cancellation. Attribution showed that baseline error explained 95.07% of total error, with
large cancellation amplification. The honest hybrid therefore keeps one native bracket baseline
margin while making Task14 standalone.

The final controlled-domain release stores 22 FP32 scalars (88 bytes), compiles Task14 direction and
bracket source/closer selectors from text, passes 9,336 selector/equation checks, and imports no
model, Torch, transformer, training, or network library. It is predictive, composable, and
manipulable at the answer-margin boundary. It does not generate free-form logits or replace the
whole model.

## 4. Day one into day two: the aspectual-anchor circuit

### 4.1 Behavior and initial localization

The aspectual task contrasts prompts such as `Since last survey the leader ... has` with
`By last survey the leader ... had`. Whole-site interchange localized the complete state to
`resid:10` through `resid:18`. No single MLP or attention block carried the full effect.

An explicit causal path was then found:

```text
since/by context
  -> MLP4 bilinear writer at last/period/determiner positions
  -> attention5 carrier heads
  -> accumulated residual state
  -> attention9 H1/H4 and later readers
  -> has/had output
```

MLP4's response was well approximated by its two linearized bilinear terms; the mixed interaction
was small. Attention5 heads H7/H1/H6/H8 formed the early transporter, and their effect was almost
entirely reconstructed from the contextual `last + period + the` source bank rather than the raw
`since/by` cue. This was a useful conceptual correction: the heads read an already contextualized
state, not the surface cue directly.

### 4.2 Honest scale of the localized paths

The first four-head path was real and prospectively replicated, but it recovered only about 5% of
the complete residual effect. A later greedy full-head screen found `L8H1`, `L9H4`, and `L9H1`, whose
joint held-out recovery was about `.61-.63`. One DIM direction per head retained about `.98` of that
set effect on held-out A1 and `.89` on A2; the complement was near zero. The remaining effect is still
distributed outside the three-head set.

### 4.3 Operational quotient and executable release

A target-guided rank-one actuator was converted into a donor-free program. A local `has`/`had`
contrast at `resid:10` is read, four affine coefficients compute a gain, and a fixed rank-one vector
is written at `resid:18`. Prospective lexicon tests and a three-dose experiment showed ordered,
approximately linear causal control with very small control movement.

The current executable aspectual release stores 1,157 fitted scalars: one 1,152-vector plus five
small parameters inherited by the packaged interface. It is a valid upstream-predictive operational
quotient in the tested families. A syntax-OOD attempt that failed native capability remains invalid
and does not support a broader construction claim.

## 5. Day two: temporal auxiliary and is/was shared-circuit program

### 5.1 Different readouts do not imply one shared direction

The `will/had` temporal auxiliary and `is/was` tense tasks can use the same native locations while
encoding different variables. Early cross-task tests showed that their selective rank-one writers
had low cosine and asymmetric transfer. This separated **where** a task is computed from **which
subspace or weight function** it uses.

At L9H1/H4, the task bases had principal cosines only about `.648/.406/.349`, with no shared rank at
the frozen `.80` threshold. Exact projected-response audits showed that controllability and
observability are distinct: a direction can be visible to another readout without being a reusable
writer for it.

### 5.2 From an H3 operation to a literal downstream program

For the temporal H3 route, a rank-eight operation transferred zero-fit across text-disjoint
families. Weight contraction exposed a shared is/was component, but the orthogonal remainder was
also necessary. Module removal and factorization traced the route through L9H1/H4, MLP9,
L11H1/H3, L15H5, and a broader MLP suffix.

A complete downstream response lattice over ten sites showed that cached response interventions
were almost additive: `99.9994%` of nonconstant Möbius energy was degree one. A three-MLP cached
program (`MLP13 + MLP15 + MLP16`) was compact on its selection bank, but failed one important
fresh-transfer cell. Expanding to a five-MLP response program restored transfer on temporal-v13.

The five MLPs were then expanded through their literal bilinear tensors, and a complete upstream
atlas patched all earlier heads and MLPs. Weight-derived coordinate incidence predicted causal
magnitude well, but a top-20 graph was not recursively closed. Full layer-band addback, an all-110
deletion atlas, and frozen greedy pruning produced a 48-component source graph and then a minimal
46-component pooled graph under the registered order and `.80` boundary.

The pooled graph passes forward and reverse OOD tests and three 10%-source-noise seeds. A clean
summary point is behavior around `.81-.83`, coordinate projection around `.94`, small direction
gaps, and zero control flips on the successful pooled tests. Rank 45 failed at the next deletion,
freezing the current physical support under that search order.

### 5.3 Task-specific response modes inside the shared graph

Inside the 46-component physical support, a pooled rank-eight response projector is the robust
executor. Each task also has a clean bidirectional rank-four mode. Exact MLP quadratic and complete
attention OV comparisons show weak cross-task overlap: the tasks reuse sites but not the same local
weight functions.

The rank-four coordinates remain stable under source noise, but their worst behavior was `.79708`,
just below the frozen `.80` noise bar. Therefore clean bidirectional rank-four task programs are
licensed, while the pooled rank-eight program remains the robust executor.

The corrected weight/Jacobian atlas and source factorial revealed another important interaction
lesson. Eight shared MLP sources jointly reproduce both task modes, but singleton and pairwise
effects overcount badly. Ordered conditional deletion, not a degree-two approximation, reduced the
source set to `MLP0, MLP1, MLP2, MLP3, MLP6`.

### 5.4 Five-MLP rank-16 source interface and exact weights

Per-row DIM and per-row-mean SVD were too weak because position averaging erased contextual writes.
An all-position source basis with frozen gain `1.15` yielded a bidirectional five-MLP rank-16
interface. On reverse OOD it achieved temporal/is-was behavior `.7575/.9174`, signed response
`1.0630/1.0997`, RSE `.01463/.01454`, and no control flips.

For each source MLP, `A_s = Down_s^T Q_s` exactly compiles the intervention. Fit/fresh closure is
`8.11e-14`, all five `4608 x 16` maps have rank 16, and a float64 gauge replay closes at
`2.96e-15`.

Static hidden weight support is not sparse. Effective participation widths range from 1,462 to
3,782 hidden units, and only MLP6 concentrates at least 25% of energy in its top 10%. Activation-
conditioned top-25 support beats equal-size static weight support, but even top-50 at every site is
not fully functional.

A complete 32-arm five-site lattice found one safe within-module deletion: use activation-top-50 at
MLP0 and full hidden support at MLP1/2/3/6. This mask transfers to a different construction family
and is functionally indistinguishable from the full parent. Higher-order interaction energy is only
`.00138`, so this particular broad support is distributed but nearly additive. The transferred mask
inherits the parent's two control flips; it does not add collateral beyond the parent.

### 5.5 Shared/contrast physical weight blocks and the current residual

Canonical-angle decomposition of the two task source subspaces produced mean/shared and
task-contrast physical weight blocks at all five MLPs. The corrected transferred test is valid:

- all five mean/contrast blocks are stable across even/odd fits;
- own-task blocks strongly beat cross-task blocks;
- the mean+contrast union is functional, with temporal/is-was behavior `.8527/.8744`;
- signed causal response is `1.0734/1.0870`;
- the union reduces the parent control flips from two to one.

It is not full-equivalent. Full behavior is `.8642/.9233`, leaving an is/was gap of `.04891`, above
the frozen `.03` bar. The orthogonal complement has only `.01181` signed linear response but
`.04693` behavior. A small residual is therefore being amplified nonlinearly downstream. The
registered result is `probe_basis_only`, not a complete task-pair weight circuit.

The active successor adds the exact complement back at MLP0/1/2/3/6 one site at a time. It tests
whether one site explains most of the gap, whether behavior amplification exceeds linear-response
gain, whether collateral remains parent-bounded, and whether singleton effects are additive enough
to license a frozen greedy repair.

## 6. Constrained DAS: what failed, what worked, and what is now being tested

### 6.1 The original loophole

The complement objective was originally used both to find and justify a subspace. That creates a
memorization channel: optimization can discover a direction that makes the complement inert on the
specific prompts and readers it sees without representing the reusable causal variable.

This explains why a simple DIM direction sometimes looked better than constrained DAS. It does not
prove optimization is intrinsically inferior; it shows that the optimization target and validation
split were under-observed.

### 6.2 Evidence for regularization

Full-vocabulary KL materially improves the bad solution. In earlier tests it reduced held-out
full-vocabulary error from roughly `.4485` to `.2445`, close to DIM's `.2385`; small tangent noise
alone stayed near `.4486`. On a later whole-family tournament, KL produced geometrically stable
axes (`|cos|=.8827`) and a refit reduced sealed-v12 mean/worst loss from `.9803/1.0095` to
`.3979/.4803`, within all hard target limits.

The same KL setting did not win both construction folds. It improved v8-to-v10 worst loss from
`.7261` to `.6968`, but worsened v10-to-v8 from `.3486` to `.4007`. The correct verdict is
`regularization_fold_heterogeneity`: KL has real anti-overfitting value, but one global recipe does
not yet define the stable causal subspace.

### 6.3 Instrument corrections

The first tournament receipt was quarantined because it counted native model calls but omitted
differentiable manual-reader evaluations and did not persist sealed reader closure. The fully
instrumented replay counted 24 native plus 1,208 manual evaluations and 170 updates, and passed all
atomic checks.

Before the next runner was executed, another audit caught that the inherited `secondary` score
ranked only four losses while the preregistration required all six. The pending job was removed,
the evaluator was corrected to retain hard feasibility while ranking the full six-term sum, and
the new hash-bound runner was requeued. The sealed v15 family is not constructed until configuration,
step, and final axis are frozen from v8/v10 evidence.

### 6.4 Current nested test

The current configurations are no regularization, KL weights `.25/1/4`, tangent noise `.10`, and
noise combined with KL `.25/1`. Within each outer training construction, one cue direction trains
and the opposite direction selects, then they reverse. Complete v8 and v10 families are untouched
outer tests. A final configuration is chosen only from aggregate inner evidence, refit on v8+v10,
and evaluated once on capability-qualified v15. V15 passed native capability at 31/32 rows in both
panels; the failed v14 family remains recorded and excluded.

Success requires a moved regularized checkpoint to beat both nested no-reg and pooled DIM-like step
zero on both outer folds, then beat pooled on v15 without target sacrifice. Failure would mean one
of three different things:

- inner selection cannot predict construction transfer, so the objective remains non-identifying;
- regularization helps but must be construction-adaptive rather than global;
- rank one cannot jointly satisfy sufficiency and complement selectivity, so the object must become
  a higher-rank or multi-reader causal operator.

## 7. Breadth campaign: circuit number versus circuit quality

The parallel breadth lane uses a common four-row battery and then within-family separability tests.
It has screened agreement, polarity, complement type, inversion, person, number, voice, modality,
animacy, dative, quantifier, possessive, and other controlled behaviors.

At the end of the window:

- the amended battery standing is 44 of 86 behavior specifications;
- the stricter full-family separability ledger supports 27 provisional distinct circuits;
- several apparent circuits were fused or retracted when the control family became larger;
- one-protocol masks, route components, and full transparent programs remain separate evidence
  tiers.

Three canonical breadth dossiers illustrate the range:

- **Possessive number:** a number direction transfers across several intervening structures, while
  particular disruptors reveal where the shared account breaks.
- **Both/neither correlatives:** a strong one-direction result was later reinterpreted as primarily a
  `neither` code rather than a generic correlative state.
- **Aspectual anchor:** a full residual behavior coexists with smaller explicit head paths and a
  donor-free rank-one operational program.

The main lesson is that circuit count is control-set dependent. A behavior counts as distinct only
when it is separable under the full relevant sibling family, ideally symmetrically. The current 27
is therefore a provisional controlled count, not a claim of 27 completed mechanistic explanations.

## 8. Important nulls and corrections preserved during the window

The following failures materially changed the route:

- The 99-minute Task14 OOM/stall was an invalid instrument, not negative circuit evidence.
- Several construction attempts failed native capability and were closed before causal outcomes.
- The standalone bracket baseline failed near cancellation; the released program honestly retains
  one native baseline scalar.
- The Task14 MLP15/17 scalar mediator law failed a fourth-corpus cell and was not promoted.
- The aspectual four-head path is only about 5% of the full effect; later head sets improve coverage
  but do not make the whole route a single-head circuit.
- Raw-response SVD and per-row-mean source SVD preserved the wrong objects despite high fit energy.
- Rank 32 MLP input readers were too small; rank 64 transferred.
- A top-20 donor-ranked source graph was not recursively closed; complete component addback and
  deletion were required.
- Rank 45 failed the frozen physical-support deletion boundary.
- Task-rank-four source-noise behavior missed by roughly `.003`; pooled rank eight remains the robust
  executor.
- Static hidden-unit sparsity failed despite exact `Down^T Q` compilation.
- The source shared/contrast union is functional but not full-equivalent because of a small
  behaviorally amplified complement.
- Fixed KL regularization helps one construction direction and hurts the other.
- A source-weight v1 result used the wrong evaluation population and was quarantined; v2 replayed
  the exact registered transferred rows with zero full-replay error.
- Absolute floating-point and price/counting mistakes were corrected through hash-bound audits or
  prospective reruns, never by silently relaxing scientific outcomes.

These nulls are not bookkeeping clutter. They constrain what a valid circuit object must include:
contextual positions rather than row means, nonlinear downstream amplification rather than only
linear response size, construction-level validation rather than row-level splits, and actual
weight/readout execution rather than activation reconstruction.

## 9. Current status at the end of the window

### Strongest executable objects

1. **Task14/bracket controlled margin program:** 22 FP32 scalars, text-compiled selectors, exact
   typed composition; Task14 standalone, bracket baseline-conditioned.
2. **Aspectual has/had operational quotient:** 1,157 fitted scalars, local read, affine gain, fixed
   rank-one write, prospective lexical and dose evidence.
3. **Temporal/is-was downstream response program:** eight literal weight-executed response sites,
   pooled rank-eight robust executor, task-specific rank-four modes, and a frozen 46-component
   recursively executable source graph.
4. **Five-MLP source interface:** bidirectional rank-16 source program with exact `Down^T Q`
   compilation and one transferable within-MLP deletion; physical task-pair grouping remains a
   functional probe pending complement repair.

### What is still missing

- None of these replaces the whole transformer or generates unrestricted next-token distributions.
- The 46-component temporal/is-was graph is still large, and its full parent has a small selectivity
  caveat on some controls.
- The task-pair source blocks need sitewise residual localization and prospective greedy repair.
- The DAS axis needs construction-nested and sealed-family validation before it can be called a
  stable causal subspace.
- Shared subspaces across more than two tasks need to be compiled into common physical reader/writer
  factors and tested under joint composition.
- Literal storage/compute pricing should be applied only after the remaining identification gates,
  not used to substitute for them.

### Live queue and immediate trajectory

At the cutoff, the managed queue contains:

1. continuing breadth/family circuit screens;
2. the five-site source-complement localization screen;
3. the nested construction-adaptive DAS tournament.

The next branch is already determined prospectively:

- If one complement site explains most of the nonlinear gap and singleton effects are additive,
  freeze a greedy minimal addback and validate it on a new family.
- If singleton addbacks interact materially, run the smallest required pairwise factorial instead
  of a greedy search.
- Once the repaired source program is stable, split shared and contrast factors site by site and
  map their exact upstream writers/downstream readers through the weight tensors.
- For DAS, promote only a regularized direction that beats DIM and no-reg on both outer
  constructions and sealed v15. Otherwise switch to a multi-reader causal-response quotient rather
  than adding more coefficients to the same complement loss.

## 10. Bottom-line trajectory

The last two days improved circuit quality more than raw compression. The work now has multiple
examples of the full chain

```text
behavior
  -> causal site
  -> within-module subspace or factor
  -> exact weight pullback
  -> upstream writer/downstream reader edges
  -> held-out complete-model execution
  -> selective manipulation and composition
  -> literal executable interface.
```

The central methodological change is that subspaces are no longer treated as endpoints. A DAS or
DIM basis is translated into weight coordinates, tested at native module boundaries, decomposed by
causal response, and rejected if its complement, controls, or construction transfer reveal a
shortcut. That is the route most likely to turn the growing circuit inventory into reusable shared
subprograms rather than a collection of task-specific probes.

## Primary local references

- [Temporal/is-was canonical dossier](CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md)
- [Aspectual-anchor canonical dossier](CIRCUIT_aspectual_anchor_has_vs_had_2026-09-06.md)
- [Possessive-number canonical dossier](CIRCUIT_possessive_number_agreement_2026-09-06.md)
- [Correlative-pair canonical dossier](CIRCUIT_correlative_pair_both_vs_neither_2026-09-06.md)
- [Constrained-DAS regularization red-team](../CONSTRAINED_DAS_REGULARIZATION_REDTEAM_2026-09-07.md)
- [15:14 hourly strategic review](../HOURLY_STRATEGIC_REVIEW_2026-09-07_1514.md)
- [14:26 mathematical review of task-pair weight directions](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_1426.md)
